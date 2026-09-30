//! Tensor operations shared only by the three Qwen 2.1 components.
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub fn linear(vb: VarBuilder, input: usize, output: usize, bias: bool) -> Result<Linear> {
    candle_nn::linear_b(input, output, bias, vb)
}

pub fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps as f32)
}

/// Unscaled LayerNorm, computed in float32 by the fused kernel.
pub fn norm(x: &Tensor) -> Result<Tensor> {
    let n = x.dim(D::Minus1)?;
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let ones = Tensor::ones(n, DType::F32, x.device())?;
    let zeros = Tensor::zeros(n, DType::F32, x.device())?;
    candle_nn::ops::layer_norm(&xf, &ones, &zeros, 1e-6)?.to_dtype(x.dtype())
}

pub fn affine_norm(x: &Tensor, vb: VarBuilder) -> Result<Tensor> {
    let n = x.dim(D::Minus1)?;
    candle_nn::ops::layer_norm(
        &x.contiguous()?,
        &vb.get(n, "weight")?,
        &vb.get(n, "bias")?,
        1e-6,
    )
}

pub fn heads(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    x.reshape((b, s, n, d / n))?.transpose(1, 2)?.contiguous()
}

pub fn unheads(x: &Tensor) -> Result<Tensor> {
    let (b, n, s, d) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, s, n * d))
}

/// Angles contain half a head's dimensions. DiT uses adjacent complex pairs;
/// Qwen3-VL rotates the first and second half of a head against each other.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
    let (_, _, s, d) = x.dims4()?;
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let cos = cos.reshape((s, d / 2))?.to_dtype(DType::F32)?.contiguous()?;
    let sin = sin.reshape((s, d / 2))?.to_dtype(DType::F32)?.contiguous()?;
    let out = if interleaved {
        candle_nn::rotary_emb::rope_i(&xf, &cos, &sin)?
    } else {
        candle_nn::rotary_emb::rope(&xf, &cos, &sin)?
    };
    out.to_dtype(x.dtype())
}

/// Fused Metal attention for encoder/DiT heads. VAE heads exceed the Metal
/// kernel's supported size; split query rows to bound its score-buffer memory.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let d = q.dim(3)?;
    let scale = (d as f32).sqrt().recip();
    // Candle's vector kernel ignores causal masking, and the partial-tile
    // fused path can produce NaNs for short sequences. These small attention
    // matrices are cheap to compute directly (also used between references).
    if q.device().is_metal() && q.dim(2)? > 32 && [32, 64, 72, 80, 96, 128, 256].contains(&d) {
        // Real Qwen3-VL keys can reach magnitudes above 200. The BF16 fused
        // kernel produces NaNs for some such inputs; use FP32 attention and
        // cast its output back, retaining BF16 model weights and activations.
        return candle_nn::ops::sdpa(
            &q.to_dtype(DType::F32)?.contiguous()?,
            &k.to_dtype(DType::F32)?.contiguous()?,
            &v.to_dtype(DType::F32)?.contiguous()?,
            None,
            causal,
            scale,
            1.0,
        )?
        .to_dtype(q.dtype());
    }
    let (b, h, seq, _) = q.dims4()?;
    let kv_seq = k.dim(2)?;
    let groups = h / k.dim(1)?;
    let expand = |x: &Tensor| -> Result<Tensor> {
        let (_, kh, ks, kd) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((b, kh, groups, ks, kd))?
            .reshape((b, h, ks, kd))?
            .to_dtype(DType::F32)
    };
    let kt = expand(k)?.transpose(2, 3)?.contiguous()?;
    let vf = expand(v)?.contiguous()?;
    let mut parts = Vec::new();
    for start in (0..seq).step_by(128) {
        let len = (seq - start).min(128);
        let mut scores = (q
            .narrow(2, start, len)?
            .to_dtype(DType::F32)?
            .contiguous()?
            .matmul(&kt)?
            * scale as f64)?;
        if causal {
            let mask: Vec<f32> = (0..len)
                .flat_map(|i| {
                    (0..kv_seq).map(move |j| {
                        if j <= kv_seq - seq + start + i {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            scores = scores.broadcast_add(&Tensor::from_vec(mask, (len, kv_seq), q.device())?)?;
        }
        parts.push(
            candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&vf)?
                .to_dtype(q.dtype())?,
        );
    }
    Tensor::cat(&parts, 2)
}

pub struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl SwiGlu {
    pub fn new(vb: VarBuilder, dim: usize, hidden: usize, names: [&str; 3]) -> Result<Self> {
        Ok(Self {
            gate: linear(vb.pp(names[0]), dim, hidden, false)?,
            up: linear(vb.pp(names[1]), dim, hidden, false)?,
            down: linear(vb.pp(names[2]), hidden, dim, false)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Chunk MLP activations at 2K resolutions; token-wise operations are
        // independent. Small chunks underutilise the GPU's matmul kernels.
        let mut chunks = Vec::new();
        for start in (0..x.dim(1)?).step_by(4096) {
            let xx = x.narrow(1, start, (x.dim(1)? - start).min(4096))?;
            chunks.push(self.down.forward(
                &(candle_nn::ops::silu(&self.gate.forward(&xx)?)? * self.up.forward(&xx)?)?,
            )?);
        }
        Tensor::cat(&chunks, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires Metal GPU access"]
    fn metal_attention_matches_dense_reference() -> Result<()> {
        let gpu = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::BF16] {
            // Cover grouped encoder heads, right-aligned causal prefix segments,
            // the vector kernel, full target attention, and the vision head size.
            for (seq, kv, head, groups, causal) in [
                (12, 12, 128, 4, true),
                (42, 42, 128, 4, true),
                (28, 28, 128, 1, true),
                (70, 70, 128, 4, true),
                (11, 19, 128, 1, true),
                (3, 19, 128, 1, true),
                (16, 23, 128, 1, false),
                (16, 16, 72, 1, false),
                (1024, 1066, 128, 1, false),
            ] {
                let make = |heads: usize, len: usize, shift: f32| -> Result<Tensor> {
                    let data: Vec<f32> = (0..heads * len * head)
                        .map(|i| (i as f32 * 0.03 + shift).sin())
                        .collect();
                    Tensor::from_vec(data, (1, heads, len, head), &Device::Cpu)?.to_dtype(dtype)
                };
                let (q, k, v) = (
                    make(4, seq, 0.)?,
                    make(4 / groups, kv, 0.2)?,
                    make(4 / groups, kv, 0.7)?,
                );
                let expected = attention(&q, &k, &v, causal)?;
                let actual = attention(
                    &q.to_device(&gpu)?,
                    &k.to_device(&gpu)?,
                    &v.to_device(&gpu)?,
                    causal,
                )?
                .to_device(&Device::Cpu)?;
                assert!(
                    expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|v| v.is_finite()),
                    "non-finite CPU reference {dtype:?} seq={seq}"
                );
                assert!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|v| v.is_finite()),
                    "non-finite Metal output {dtype:?} seq={seq}"
                );
                let delta = (expected.to_dtype(DType::F32)? - actual.to_dtype(DType::F32)?)?
                    .abs()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(
                    delta.iter().all(|v| v.is_finite()),
                    "non-finite {dtype:?} attention seq={seq}"
                );
                let max = delta.into_iter().fold(0f32, f32::max);
                assert!(
                    max < if dtype == DType::F32 { 1e-4 } else { 0.01 },
                    "{dtype:?} attention seq={seq} kv={kv} causal={causal}: max error {max}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires Metal GPU access"]
    fn attention_handles_qwen_key_magnitudes() -> Result<()> {
        let data = candle_core::safetensors::load(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/attention.safetensors"
            ),
            &Device::Cpu,
        )?;
        let gpu = Device::new_metal(0)?;
        let expected = attention(&data["q"], &data["k"], &data["v"], true)?.to_dtype(DType::F32)?;
        let actual = attention(
            &data["q"].to_device(&gpu)?,
            &data["k"].to_device(&gpu)?,
            &data["v"].to_device(&gpu)?,
            true,
        )?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
        let error = (expected - actual)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(error.iter().all(|v| v.is_finite()));
        assert!(error.into_iter().fold(0f32, f32::max) < 0.002);
        Ok(())
    }
}
