use crate::{
    AttentionPrecision, Event, Observer, Request, Stage,
    dit::Dit,
    noise,
    preview::{DecoderWorker, Snapshot},
    text::{self, Encoded},
    vae::Vae,
    vision::{self, Visual},
    weights::{Weights, cached_builder},
};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use image::{RgbaImage, imageops::FilterType};
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
        args.preview_control.is_none() || args.preview_every.is_none(),
        "choose either automatic or manual previews"
    );
    ensure!(
        args.scale.is_finite() && args.scale > 0.0 && args.scale <= 1.0,
        "scale must be between 0 and 1"
    );
    let (width, height) = dimensions(
        args.ratio.as_deref(),
        args.images.last().map(|i| i.dimensions()),
        args.scale,
    )?;
    if let Some(source) = args.noise_source_size {
        ensure!(
            width == height && (32..=8192).contains(&source) && source % width == 0,
            "noise source size must be 32–8192 pixels and an integer multiple of the square output size"
        );
    }
    Ok((width, height))
}

/// Results that depend only on the prompt and reference images, kept between
/// requests so re-rolling the seed or changing steps/size skips the encoders.
/// Encoder results are held on the CPU. Model weights and their Metal device
/// persist until the generator is dropped or its model cache is unloaded.
#[derive(Default)]
pub struct Cache {
    /// The prompt and encoder output for exactly `references`, in order.
    encoded: Option<(String, Encoded)>,
    references: Vec<CachedReference>,
    device: Option<Device>,
    denoiser: Option<candle_nn::VarBuilder<'static>>,
    vae: Option<Vae>,
    decoder: Option<DecoderWorker>,
}

impl Cache {
    fn decoder(&mut self, weights: &Weights) -> Result<&DecoderWorker> {
        if self.decoder.is_none() {
            self.decoder = Some(DecoderWorker::new(weights.clone())?);
        }
        Ok(self.decoder.as_ref().unwrap())
    }

    pub fn unload_models(&mut self) {
        self.denoiser = None;
        self.vae = None;
        self.decoder = None;
        self.device = None;
    }
}

pub(crate) fn load_vae<'a>(
    slot: &'a mut Option<Vae>,
    weights: &Weights,
    device: &Device,
) -> Result<&'a Vae> {
    if slot.is_none() {
        *slot = Some(Vae::new(
            cached_builder(weights.builder("vae", DType::BF16, device)?),
            weights.config("vae/config.json")?,
        )?);
    }
    Ok(slot.as_ref().unwrap())
}

struct CachedReference {
    /// The resized reference, compared exactly.
    image: RgbaImage,
    visual: Option<Visual>,
    latents: Option<Tensor>,
}

pub fn run(
    args: &Request,
    weights: &Weights,
    cache: &mut Cache,
    observer: &mut Observer<'_>,
) -> Result<Arc<RgbaImage>> {
    let (width, height) = validate(args)?;
    if let Some(control) = &args.preview_control {
        observer.previews = Some(
            cache
                .decoder(weights)?
                .session(control.clone(), args.steps)?,
        );
    }
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
    if cache.device.is_none() {
        ensure!(
            objc2_metal::MTLCreateSystemDefaultDevice().is_some(),
            "no Metal GPU is visible; run on an Apple Silicon Mac with GPU access"
        );
        cache.device = Some(Device::new_metal(0).context("could not initialize Metal")?);
    }
    let device = cache.device.as_ref().unwrap().clone();
    let config: serde_json::Value = weights.config("transformer/config.json")?;
    ensure!(
        config["_class_name"] == "QwenImage21Transformer2DModel"
            && config["num_layers"] == 32
            && config["causal_condition"] == true
            && config["in_channels"] == 64,
        "checkpoint must have the Qwen/Qwen-Image-2.1 architecture"
    );
    observer.progress(Stage::Loading, 1, 1)?;
    let cpu = Device::Cpu;
    // Keep entries for the references still in use, in request order.
    let mut previous = std::mem::take(&mut cache.references);
    let same_references = previous.len() == references.len()
        && previous.iter().zip(&references).all(|(c, r)| c.image == *r);
    if !same_references
        || cache
            .encoded
            .as_ref()
            .is_some_and(|(p, _)| *p != args.prompt)
    {
        cache.encoded = None;
    }
    for img in &references {
        let entry = match previous.iter().position(|c| c.image == *img) {
            Some(i) => previous.swap_remove(i),
            None => CachedReference {
                image: img.clone(),
                visual: None,
                latents: None,
            },
        };
        cache.references.push(entry);
    }
    drop(previous);
    let encoded = match &cache.encoded {
        Some((_, encoded)) => encoded.to_device(&device)?,
        None => {
            let entries = &mut cache.references;
            let encoded = text::encode(
                weights,
                &args.prompt,
                &references,
                &device,
                observer,
                |i, vb, observer| {
                    if let Some(visual) = &entries[i].visual {
                        return Ok(visual.to_device(&device)?);
                    }
                    let visual = vision::encode(&references[i], vb, i, observer)?;
                    entries[i].visual = Some(visual.to_device(&cpu)?);
                    Ok(visual)
                },
            )?;
            cache.encoded = Some((args.prompt.clone(), encoded.to_device(&cpu)?));
            encoded
        }
    };
    device.synchronize()?;
    let mut reference_latents = Vec::new();
    let mut shapes = Vec::new();
    for (i, entry) in cache.references.iter_mut().enumerate() {
        let img = &entry.image;
        let (h, w) = (img.height() as usize / 16, img.width() as usize / 16);
        shapes.push((h, w));
        if let Some(latents) = &entry.latents {
            reference_latents.push(latents.to_device(&device)?);
            continue;
        }
        observer.progress(Stage::ReferenceEncoding, i, references.len())?;
        let vae = load_vae(&mut cache.vae, weights, &device)?;
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
        let latents = vae
            .encode(&rgba)?
            .reshape((1, 64, h * w))?
            .transpose(1, 2)?
            .contiguous()?;
        entry.latents = Some(latents.to_device(&cpu)?);
        reference_latents.push(latents);
        observer.progress(Stage::ReferenceEncoding, i + 1, references.len())?;
    }
    device.synchronize()?;
    let (h, w) = (height as usize / 16, width as usize / 16);
    let noise = noise::sample(
        args.seed,
        h,
        w,
        args.noise_source_size.map(|size| size as usize / 16),
    );
    let mut latents = Tensor::from_vec(noise, (1, 64, h * w), &device)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::BF16)?;
    observer.progress(Stage::DenoiserLoading, 0, 32)?;
    if cache.denoiser.is_none() {
        cache.denoiser = Some(cached_builder(weights.builder(
            "transformer",
            DType::BF16,
            &device,
        )?));
    }
    // Layers borrow cached immutable weights. Prefix KV and rotary state are
    // rebuilt for this request, and never retained after cancellation or failure.
    let mut dit = Dit::load(
        cache.denoiser.as_ref().unwrap().clone(),
        &encoded,
        &reference_latents,
        &shapes,
        (h, w),
        observer,
    )?;
    drop(encoded);
    drop(reference_latents);
    let sigmas = schedule(args.steps, h * w);
    let mut attention = match args.attention {
        AttentionPrecision::Float32 => DType::F32,
        AttentionPrecision::BFloat16 => DType::BF16,
    };
    for (i, pair) in sigmas.windows(2).enumerate() {
        let step_start = Instant::now();
        observer.check()?;
        let want_preview = (args.preview_control.is_some()
            || args.preview_every.is_some_and(|n| (i + 1) % n.get() == 0))
            && i + 1 < args.steps;
        let mut step = sample(&mut dit, &latents, pair, want_preview, attention, observer)?;
        if !step.finite && attention != DType::F32 {
            // Reduced-precision attention overflowed: redo this step, and the
            // rest of the generation, on the float32 reference path.
            attention = DType::F32;
            step = sample(&mut dit, &latents, pair, want_preview, attention, observer)?;
        }
        ensure!(
            step.finite,
            "denoiser produced non-finite latents at step {}",
            i + 1
        );
        latents = step.latents;
        let preview = step.preview;
        observer.poll_previews()?;
        if let (Some(session), Some(clean)) = (&observer.previews, &preview) {
            // Publish a compact, immutable CPU snapshot before announcing the
            // completed step. The Preview button never touches the sampler queue.
            session.publish(Snapshot {
                latents: clean.to_device(&Device::Cpu)?,
                step: i + 1,
                width,
                height,
            })?;
        }
        observer.emit(Event::StepFinished {
            step: i + 1,
            total: args.steps,
            duration: step_start.elapsed(),
        })?;
        observer.check()?;
        // The final decode supersedes a preview requested during the last step.
        if i + 1 < args.steps {
            observer.wait_previews()?;
        }
        if args.preview_control.is_none()
            && let Some(clean) = preview
        {
            let image = cache.decoder(weights)?.decode(
                Snapshot {
                    latents: clean.to_device(&Device::Cpu)?,
                    step: i + 1,
                    width,
                    height,
                },
                observer.cancellation,
            )?;
            observer.emit(Event::Preview {
                step: i + 1,
                total: args.steps,
                image,
            })?;
            observer.check()?;
        }
    }
    drop(dit);
    observer.poll_previews()?;
    // Retire the session before final decoding. Late preview replies cannot
    // replace the final image or escape into the next generation.
    observer.previews = None;
    observer.progress(Stage::Decoding, 0, 1)?;
    let image = cache.decoder(weights)?.decode(
        Snapshot {
            latents: latents.to_device(&Device::Cpu)?,
            step: args.steps,
            width,
            height,
        },
        observer.cancellation,
    )?;
    observer.progress(Stage::Decoding, 1, 1)?;
    if args.preview_every.is_some() || args.preview_control.is_some() {
        observer.emit(Event::Preview {
            step: args.steps,
            total: args.steps,
            image: image.clone(),
        })?;
        observer.check()?;
    }
    Ok(image)
}

pub(crate) fn decode(
    latents: &Tensor,
    vae: &Vae,
    width: u32,
    height: u32,
) -> Result<Arc<RgbaImage>> {
    let (h, w) = (height as usize / 16, width as usize / 16);
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
struct Sampled {
    latents: Tensor,
    /// Estimated clean latents for a preview, when requested.
    preview: Option<Tensor>,
    finite: bool,
}

/// One Euler step from `sigma[0]` to `sigma[1]`.
fn sample(
    dit: &mut Dit,
    latents: &Tensor,
    sigma: &[f64],
    preview: bool,
    attention: DType,
    observer: &mut Observer<'_>,
) -> Result<Sampled> {
    let predicted = dit.forward(latents, sigma[0], attention, observer)?;
    // Flow matching: x_sigma = x_clean + sigma * velocity.
    let preview = if preview {
        Some(
            (latents.to_dtype(DType::F32)? - (predicted.to_dtype(DType::F32)? * sigma[0])?)?
                .to_dtype(DType::BF16)?,
        )
    } else {
        None
    };
    let latents = (latents.to_dtype(DType::F32)?
        + (predicted.to_dtype(DType::F32)? * (sigma[1] - sigma[0]))?)?
        .to_dtype(DType::BF16)?;
    latents.device().synchronize()?;
    let finite = latents
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?
        .is_finite();
    Ok(Sampled {
        latents,
        preview,
        finite,
    })
}

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
    fn shared_noise_requires_a_compatible_square_grid() -> Result<()> {
        let mut request = Request::new("test");
        request.scale = 0.25;
        request.noise_source_size = Some(2048);
        assert_eq!(validate(&request)?, (512, 512));
        request.scale = 1.;
        assert_eq!(validate(&request)?, (2048, 2048));
        request.noise_source_size = Some(512);
        assert!(validate(&request).is_err());
        request.scale = 0.25;
        request.noise_source_size = Some(2000);
        assert!(validate(&request).is_err());
        request.noise_source_size = Some(2048);
        request.ratio = Some("16:9".into());
        assert!(validate(&request).is_err());
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
