//! Still-image specialization of the Qwen 2.1 RGBA autoencoder.
//! Temporal convolution caches are unused for one frame. Temporal factors in
//! the residual shortcuts still affect channel layout and must be preserved.
use crate::{ops, vae_kernels};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

pub struct Vae {
    vb: VarBuilder<'static>,
    mean: Tensor,
    std: Tensor,
}

impl Vae {
    pub fn new(vb: VarBuilder<'static>, config: Config) -> Result<Self> {
        let mean = Tensor::from_vec(config.latents_mean, (1, 64, 1, 1), vb.device())?
            .to_dtype(vb.dtype())?;
        let std = Tensor::from_vec(config.latents_std, (1, 64, 1, 1), vb.device())?
            .to_dtype(vb.dtype())?;
        Ok(Self { vb, mean, std })
    }

    pub fn encode(&self, rgba: &Tensor) -> Result<Tensor> {
        let vb = self.vb.pp("encoder");
        let mut x = conv(rgba, vb.pp("conv_in"), 4, 96, 3, 1, 1)?;
        let dims = [96, 96, 192, 384, 768, 768];
        for i in 0..5 {
            let block = vb.pp(format!("down_blocks.{i}"));
            let residual = x.clone();
            x = resnet(&x, block.pp("resnets.0"), dims[i], dims[i + 1])?;
            x = resnet(&x, block.pp("resnets.1"), dims[i + 1], dims[i + 1])?;
            let spatial = if i < 4 { 2 } else { 1 };
            let temporal = if (1..4).contains(&i) { 2 } else { 1 };
            if i < 4 {
                x = conv(
                    &x.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?,
                    block.pp("downsampler.resample.1"),
                    dims[i + 1],
                    dims[i + 1],
                    3,
                    0,
                    2,
                )?;
            }
            x = (x + down_shortcut(&residual, dims[i + 1], temporal, spatial)?)?;
            self.vb.device().synchronize()?;
        }
        x = mid(&x, vb.pp("mid_block"), 768)?;
        x = conv(
            &candle_nn::ops::silu(&channel_norm(&x, vb.pp("norm_out"), true)?)?,
            vb.pp("conv_out"),
            768,
            128,
            3,
            1,
            1,
        )?;
        x = conv(&x, self.vb.pp("quant_conv"), 128, 128, 1, 0, 1)?.narrow(1, 0, 64)?;
        x.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = latents
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)?;
        let x = conv(&z, self.vb.pp("post_quant_conv"), 64, 64, 1, 0, 1)?;
        let vb = self.vb.pp("decoder");
        let mut x = conv(&x, vb.pp("conv_in"), 64, 1152, 3, 1, 1)?;
        x = mid(&x, vb.pp("mid_block"), 1152)?;
        let dims = [1152, 1152, 1152, 576, 288, 144];
        for i in 0..5 {
            let block = vb.pp(format!("up_blocks.{i}"));
            let residual = x.clone();
            x = resnet(&x, block.pp("resnets.0"), dims[i], dims[i + 1])?;
            for j in 1..3 {
                x = resnet(
                    &x,
                    block.pp(format!("resnets.{j}")),
                    dims[i + 1],
                    dims[i + 1],
                )?;
            }
            if i < 4 {
                let (_, _, h, w) = x.dims4()?;
                x = conv(
                    &x.upsample_nearest2d(h * 2, w * 2)?,
                    block.pp("upsampler.resample.1"),
                    dims[i + 1],
                    dims[i + 1],
                    3,
                    1,
                    1,
                )?;
                x = (x + up_shortcut(&residual, dims[i + 1], if i < 3 { 2 } else { 1 })?)?;
            }
            self.vb.device().synchronize()?;
        }
        x = conv(
            &candle_nn::ops::silu(&channel_norm(&x, vb.pp("norm_out"), true)?)?,
            vb.pp("conv_out"),
            144,
            4,
            3,
            1,
            1,
        )?;
        if !x
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?
            .is_finite()
        {
            candle_core::bail!("VAE decoder produced non-finite pixels");
        }
        x.clamp(-1.0, 1.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn conv(
    x: &Tensor,
    vb: VarBuilder,
    input: usize,
    output: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
) -> Result<Tensor> {
    let (b, _, h, w) = x.dims4()?;
    if !x.device().is_metal() || b != 1 {
        let config = candle_nn::Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        };
        return candle_nn::conv2d(input, output, kernel, config, vb)?.forward(x);
    }
    let oh = (h + 2 * padding - kernel) / stride + 1;
    let ow = (w + 2 * padding - kernel) / stride + 1;
    let weight = vb
        .get((output, input, kernel, kernel), "weight")?
        .reshape((output, input * kernel * kernel))?;
    let bias = vb.get(output, "bias")?.reshape((output, 1))?;
    if kernel == 1 && stride == 1 && padding == 0 {
        return weight
            .matmul(&x.reshape((input, h * w))?)?
            .broadcast_add(&bias)?
            .reshape((1, output, h, w));
    }
    // weight (out, in*k*k) @ patches (in*k*k, rows*ow) is directly NCHW.
    // Bound the patch matrix to 256 MiB by computing bands of output rows;
    // the result is exact, not a blend of independently decoded tiles.
    let rows =
        (256 * 1024 * 1024 / (ow * input * kernel * kernel * x.dtype().size_in_bytes())).max(1);
    let mut parts = Vec::new();
    for start in (0..oh).step_by(rows) {
        let len = (oh - start).min(rows);
        let patches = vae_kernels::im2col(x, kernel, stride, padding, start, len, ow)?;
        parts.push(
            weight
                .matmul(&patches)?
                .broadcast_add(&bias)?
                .reshape((1, output, len, ow))?,
        );
    }
    Tensor::cat(&parts, 2)
}

fn channel_norm(x: &Tensor, vb: VarBuilder, temporal: bool) -> Result<Tensor> {
    let c = x.dim(1)?;
    let weight = if temporal {
        vb.get((c, 1, 1, 1), "gamma")?.reshape((1, c, 1, 1))?
    } else {
        vb.get((c, 1, 1), "gamma")?.unsqueeze(0)?
    };
    if x.device().is_metal() && x.dim(0)? == 1 {
        return vae_kernels::channel_norm(x, &weight);
    }
    let xf = x.to_dtype(DType::F32)?;
    let l2 = xf
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX as f64)?;
    (xf.broadcast_div(&l2)?.to_dtype(x.dtype())? * (c as f64).sqrt())?.broadcast_mul(&weight)
}

fn resnet(x: &Tensor, vb: VarBuilder, input: usize, output: usize) -> Result<Tensor> {
    let residual = if input != output {
        conv(x, vb.pp("conv_shortcut"), input, output, 1, 0, 1)?
    } else {
        x.clone()
    };
    let x = conv(
        &candle_nn::ops::silu(&channel_norm(x, vb.pp("norm1"), true)?)?,
        vb.pp("conv1"),
        input,
        output,
        3,
        1,
        1,
    )?;
    let x = conv(
        &candle_nn::ops::silu(&channel_norm(&x, vb.pp("norm2"), true)?)?,
        vb.pp("conv2"),
        output,
        output,
        3,
        1,
        1,
    )?;
    x + residual
}

fn mid(x: &Tensor, vb: VarBuilder, c: usize) -> Result<Tensor> {
    let x = resnet(x, vb.pp("resnets.0"), c, c)?;
    let a = vb.pp("attentions.0");
    let (_, _, h, w) = x.dims4()?;
    let qkv = conv(
        &channel_norm(&x, a.pp("norm"), false)?,
        a.pp("to_qkv"),
        c,
        c * 3,
        1,
        0,
        1,
    )?
    .reshape((1, 1, c * 3, h * w))?
    .transpose(2, 3)?;
    let attended = ops::attention(
        &qkv.narrow(3, 0, c)?,
        &qkv.narrow(3, c, c)?,
        &qkv.narrow(3, c * 2, c)?,
        false,
    )?;
    let attended = attended
        .squeeze(1)?
        .transpose(1, 2)?
        .reshape((1, c, h, w))?;
    let x = (x + conv(&attended, a.pp("proj"), c, c, 1, 0, 1)?)?;
    resnet(&x, vb.pp("resnets.1"), c, c)
}

fn down_shortcut(x: &Tensor, output: usize, temporal: usize, spatial: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x = x.unsqueeze(2)?.pad_with_zeros(2, temporal - 1, 0)?;
    let x = x
        .reshape(
            &[
                b,
                c,
                1,
                temporal,
                h / spatial,
                spatial,
                w / spatial,
                spatial,
            ][..],
        )?
        .permute([0, 1, 3, 5, 7, 2, 4, 6])?
        .contiguous()?;
    let group = c * temporal * spatial * spatial / output;
    x.reshape((b, output, group, (h / spatial) * (w / spatial)))?
        .transpose(2, 3)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .mean(3)?
        .reshape((b, output, h / spatial, w / spatial))?
        .to_dtype(x.dtype())
}

fn up_shortcut(x: &Tensor, output: usize, temporal: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let repeats = output * temporal * 4 / c;
    let x = x
        .unsqueeze(2)?
        .broadcast_as((b, c, repeats, h, w))?
        .contiguous()?;
    x.reshape(&[b, output, temporal, 2, 2, 1, h, w][..])?
        .permute([0, 1, 5, 2, 6, 3, 7, 4])?
        .contiguous()?
        .reshape((b, output, temporal, h * 2, w * 2))?
        .narrow(2, temporal - 1, 1)?
        .squeeze(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn temporal_shortcuts_keep_single_frame_layout() -> Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &Device::Cpu)?;
        let down = down_shortcut(&x, 2, 2, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(down, [0., 2.5]);
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4, 1, 1), &Device::Cpu)?;
        assert_eq!(
            up_shortcut(&x, 1, 2)?.flatten_all()?.to_vec1::<f32>()?,
            [3., 3., 4., 4.]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the cached Qwen 2.1 checkpoint and Metal GPU access"]
    fn checkpoint_matches_diffusers_reference() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let weights = crate::weights::Weights::new(None, true);
        let vae = Vae::new(
            weights.builder("vae", DType::F32, &device)?,
            weights.config("vae/config.json")?,
        )?;
        let fixtures = candle_core::safetensors::load(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/vae.safetensors"
            ),
            &device,
        )?;
        for (label, actual) in [
            ("encoded", vae.encode(&fixtures["input"])?),
            ("decoded", vae.decode(&fixtures["latent"])?),
        ] {
            let delta = (actual - &fixtures[label])?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(delta.iter().all(|v| v.is_finite()), "non-finite {label}");
            let max = delta.into_iter().fold(0f32, f32::max);
            assert!(max < 0.002, "{label} max absolute error {max}");
        }
        Ok(())
    }
}
