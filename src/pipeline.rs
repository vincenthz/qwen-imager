use crate::{Event, Observer, Request, Stage, dit::Dit, text, vae::Vae, weights::Weights};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use image::{RgbaImage, imageops::FilterType};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::{sync::Arc, time::Instant};

pub fn validate(args: &Request) -> Result<(u32, u32)> {
    ensure!(!args.prompt.trim().is_empty(), "prompt must not be empty");
    ensure!(
        args.images.len() <= 10,
        "at most 10 reference images are supported"
    );
    ensure!(
        args.images.iter().all(|i| i.width() > 0 && i.height() > 0),
        "reference images must not be empty"
    );
    ensure!(args.steps > 0, "steps must be positive");
    ensure!(
        args.scale.is_finite() && args.scale > 0.0 && args.scale <= 1.0,
        "scale must be between 0 and 1"
    );
    dimensions(
        args.ratio.as_deref(),
        args.images.last().map(|i| i.dimensions()),
        args.scale,
    )
}

pub fn run(
    args: &Request,
    weights: &Weights,
    observer: &mut Observer<'_>,
) -> Result<Arc<RgbaImage>> {
    let (width, height) = validate(args)?;
    observer.emit(Event::Started {
        width,
        height,
        steps: args.steps,
        seed: args.seed,
    })?;
    observer.progress(Stage::Loading, 0, 1)?;
    let images = &args.images;
    let references: Vec<_> = images
        .iter()
        .map(|img| {
            let (w, h) = reference_dimensions(img.width(), img.height());
            image::imageops::resize(img, w, h, FilterType::Lanczos3)
        })
        .collect();
    ensure!(
        objc2_metal::MTLCreateSystemDefaultDevice().is_some(),
        "no Metal GPU is visible; run on an Apple Silicon Mac with GPU access"
    );
    let device = Device::new_metal(0).context("could not initialize Metal")?;
    let config: serde_json::Value = weights.config("transformer/config.json")?;
    ensure!(
        config["_class_name"] == "QwenImage21Transformer2DModel"
            && config["num_layers"] == 32
            && config["causal_condition"] == true
            && config["in_channels"] == 64,
        "checkpoint must be the original Qwen/Qwen-Image-2.1 architecture"
    );
    observer.progress(Stage::Loading, 1, 1)?;
    let encoded = text::encode(weights, &args.prompt, &references, &device, observer)?;
    device.synchronize()?;
    let mut reference_latents = Vec::new();
    let mut shapes = Vec::new();
    if !references.is_empty() {
        let vae = Vae::new(
            weights.builder("vae", DType::BF16, &device)?,
            weights.config("vae/config.json")?,
        )?;
        for (i, img) in references.iter().enumerate() {
            observer.progress(Stage::ReferenceEncoding, i, references.len())?;
            let (h, w) = (img.height() as usize / 16, img.width() as usize / 16);
            let rgba: Vec<f32> = img
                .as_raw()
                .iter()
                .map(|v| *v as f32 / 127.5 - 1.0)
                .collect();
            let rgba = Tensor::from_vec(
                rgba,
                (1, img.height() as usize, img.width() as usize, 4),
                &device,
            )?
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .to_dtype(DType::BF16)?;
            reference_latents.push(
                vae.encode(&rgba)?
                    .reshape((1, 64, h * w))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
            shapes.push((h, w));
            observer.progress(Stage::ReferenceEncoding, i + 1, references.len())?;
        }
        device.synchronize()?;
    }
    let (h, w) = (height as usize / 16, width as usize / 16);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let noise: Vec<f32> = (0..64 * h * w)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let mut latents = Tensor::from_vec(noise, (1, 64, h * w), &device)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::BF16)?;
    observer.progress(Stage::DenoiserLoading, 0, 32)?;
    let mut dit = Dit::load(
        weights.builder("transformer", DType::BF16, &device)?,
        &encoded,
        &reference_latents,
        &shapes,
        (h, w),
        observer,
    )?;
    drop(encoded);
    drop(reference_latents);
    let sigmas = schedule(args.steps, h * w);
    for (i, pair) in sigmas.windows(2).enumerate() {
        let step_start = Instant::now();
        observer.check()?;
        let predicted = dit.forward(&latents, pair[0], observer)?;
        // Flow matching: x_sigma = x_clean + sigma * velocity.
        let preview =
            if args.preview_every.is_some_and(|n| (i + 1) % n.get() == 0) && i + 1 < args.steps {
                Some(
                    (latents.to_dtype(DType::F32)? - (predicted.to_dtype(DType::F32)? * pair[0])?)?
                        .to_dtype(DType::BF16)?,
                )
            } else {
                None
            };
        latents = (latents.to_dtype(DType::F32)?
            + (predicted.to_dtype(DType::F32)? * (pair[1] - pair[0]))?)?
            .to_dtype(DType::BF16)?;
        device.synchronize()?;
        ensure!(
            latents
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?
                .is_finite(),
            "denoiser produced non-finite latents at step {}",
            i + 1
        );
        observer.emit(Event::StepFinished {
            step: i + 1,
            total: args.steps,
            duration: step_start.elapsed(),
        })?;
        observer.check()?;
        if let Some(clean) = preview {
            let image = decode(&clean, weights, &device, width, height)?;
            observer.emit(Event::Preview {
                step: i + 1,
                total: args.steps,
                image,
            })?;
            observer.check()?;
        }
    }
    drop(dit);
    observer.progress(Stage::Decoding, 0, 1)?;
    let image = decode(&latents, weights, &device, width, height)?;
    observer.progress(Stage::Decoding, 1, 1)?;
    if args.preview_every.is_some() {
        observer.emit(Event::Preview {
            step: args.steps,
            total: args.steps,
            image: image.clone(),
        })?;
        observer.check()?;
    }
    Ok(image)
}

fn decode(
    latents: &Tensor,
    weights: &Weights,
    device: &Device,
    width: u32,
    height: u32,
) -> Result<Arc<RgbaImage>> {
    let (h, w) = (height as usize / 16, width as usize / 16);
    let vae = Vae::new(
        weights.builder("vae", DType::BF16, device)?,
        weights.config("vae/config.json")?,
    )?;
    let z = latents
        .transpose(1, 2)?
        .reshape((1, 64, h, w))?
        .contiguous()?;
    let decoded = vae
        .decode(&z)?
        .to_dtype(DType::F32)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        decoded.iter().all(|v| v.is_finite()),
        "model produced non-finite pixels"
    );
    let bytes = decoded
        .iter()
        .map(|x| ((x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)
        .collect();
    Ok(Arc::new(
        RgbaImage::from_raw(width, height, bytes).context("decoder output size mismatch")?,
    ))
}

fn dimensions(
    ratio: Option<&str>,
    reference: Option<(u32, u32)>,
    scale: f64,
) -> Result<(u32, u32)> {
    let (w, h) = match ratio {
        Some("1:1") => (2048., 2048.),
        Some("4:3") => (2400., 1792.),
        Some("3:4") => (1792., 2400.),
        Some("3:2") => (2528., 1696.),
        Some("2:3") => (1696., 2528.),
        Some("16:9") => (2752., 1536.),
        Some("9:16") => (1536., 2752.),
        None => {
            if let Some((w, h)) = reference {
                let r = w as f64 / h as f64;
                ((4194304. * r).sqrt(), (4194304. / r).sqrt())
            } else {
                (2048., 2048.)
            }
        }
        Some(other) => anyhow::bail!("unsupported aspect ratio: {other}"),
    };
    // Both VLM slots and the DiT layout require even latent grids (32 pixels).
    let (w, h) = (
        (w * scale / 32.).floor() as u32 * 32,
        (h * scale / 32.).floor() as u32 * 32,
    );
    ensure!(
        w >= 32 && h >= 32 && w <= 8192 && h <= 8192,
        "scaled dimensions must be between 32 and 8192 pixels per side"
    );
    Ok((w, h))
}

fn reference_dimensions(w: u32, h: u32) -> (u32, u32) {
    let ratio = w as f64 / h as f64;
    (
        ((1048576. * ratio).sqrt() / 32.).round_ties_even().max(1.) as u32 * 32,
        ((1048576. / ratio).sqrt() / 32.).round_ties_even().max(1.) as u32 * 32,
    )
}

/// Fixed FlowMatchEulerDiscreteScheduler settings from the pinned checkpoint.
fn schedule(steps: usize, tokens: usize) -> Vec<f64> {
    if steps == 1 {
        return vec![1., 0.];
    }
    let mu = 0.5 + (tokens as f64 - 256.) * (0.9 - 0.5) / (8192. - 256.);
    let exp = mu.exp();
    let mut sigmas: Vec<f64> = (0..steps)
        .map(|i| {
            let t = 1. - i as f64 / steps as f64;
            exp / (exp + 1. / t - 1.)
        })
        .collect();
    let scale = (1. - sigmas[steps - 1]) / (1. - 0.02);
    for sigma in &mut sigmas {
        *sigma = (1. - (1. - *sigma) / scale) as f32 as f64;
    }
    sigmas.push(0.);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_follow_last_reference_and_align_latents() -> Result<()> {
        assert_eq!(dimensions(None, None, 1.)?, (2048, 2048));
        assert_eq!(dimensions(Some("16:9"), None, 0.5)?, (1376, 768));
        assert_eq!(dimensions(None, Some((1600, 900)), 1.)?, (2720, 1536));
        assert!(dimensions(None, None, 0.001).is_err());
        Ok(())
    }
    #[test]
    fn flow_schedule_has_exact_endpoints_and_descends() {
        for steps in [1, 2, 40] {
            for tokens in [256, 4096, 16384] {
                let s = schedule(steps, tokens);
                assert_eq!(s.len(), steps + 1);
                assert_eq!(s[0], 1.);
                assert_eq!(s[steps], 0.);
                assert!(s.windows(2).all(|p| p[0] > p[1]));
                if steps > 1 {
                    assert!((s[steps - 1] - 0.02).abs() < 1e-7);
                }
            }
        }
    }
}
