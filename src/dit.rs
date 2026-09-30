//! Qwen Image 2.1's 32-layer single-stream diffusion transformer.
use crate::{
    ops::{self, SwiGlu},
    text::Encoded,
};
use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    mlp: SwiGlu,
    cache: Option<(Tensor, Tensor)>,
}

impl Block {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let a = vb.pp("attn");
        Ok(Self {
            q: ops::linear(a.pp("to_q"), 4096, 4096, false)?,
            k: ops::linear(a.pp("to_k"), 4096, 4096, false)?,
            v: ops::linear(a.pp("to_v"), 4096, 4096, false)?,
            out: ops::linear(a.pp("to_out.0"), 4096, 4096, false)?,
            q_norm: a.get(128, "norm_q.weight")?,
            k_norm: a.get(128, "norm_k.weight")?,
            mlp: SwiGlu::new(vb.pp("img_mlp"), 4096, 12288, ["gate_layer", "proj", "out"])?,
            cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        modulation: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        segments: Option<&[Segment]>,
    ) -> candle_core::Result<Tensor> {
        let scale1 = (modulation.narrow(2, 0, 4096)? + 1.0)?;
        let gate1 = modulation.narrow(2, 4096, 4096)?.tanh()?;
        let scale2 = (modulation.narrow(2, 8192, 4096)? + 1.0)?;
        let gate2 = modulation.narrow(2, 12288, 4096)?.tanh()?;
        let n = ops::norm(x)?.broadcast_mul(&scale1)?;
        let q = ops::rope(
            &ops::rms(&ops::heads(&self.q.forward(&n)?, 32)?, &self.q_norm, 1e-6)?,
            cos,
            sin,
            true,
        )?;
        let k = ops::rope(
            &ops::rms(&ops::heads(&self.k.forward(&n)?, 32)?, &self.k_norm, 1e-6)?,
            cos,
            sin,
            true,
        )?;
        let v = ops::heads(&self.v.forward(&n)?, 32)?;
        let a = if let Some(segments) = segments {
            let mut outputs = Vec::new();
            for seg in segments {
                outputs.push(ops::attention(
                    &q.narrow(2, seg.start, seg.len)?,
                    &k.narrow(2, 0, seg.start + seg.len)?,
                    &v.narrow(2, 0, seg.start + seg.len)?,
                    seg.text,
                )?);
            }
            self.cache = Some((k, v));
            Tensor::cat(&outputs, 2)?
        } else {
            let (pk, pv) = self
                .cache
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("prefix was not encoded".into()))?;
            ops::attention(
                &q,
                &Tensor::cat(&[pk, &k], 2)?,
                &Tensor::cat(&[pv, &v], 2)?,
                false,
            )?
        };
        let x = (x + self
            .out
            .forward(&ops::unheads(&a)?)?
            .broadcast_mul(&gate1)?)?;
        let n = ops::norm(&x)?.broadcast_mul(&scale2)?;
        &x + self.mlp.forward(&n)?.broadcast_mul(&gate2)?
    }
}

#[derive(Debug)]
struct Segment {
    start: usize,
    len: usize,
    text: bool,
}

pub struct Dit {
    img_in: Linear,
    time1: Linear,
    time2: Linear,
    modulation: Linear,
    final_scale: Linear,
    out: Linear,
    blocks: Vec<Block>,
    cos: Tensor,
    sin: Tensor,
}

impl Dit {
    pub fn load(
        vb: VarBuilder,
        encoded: &Encoded,
        references: &[Tensor],
        shapes: &[(usize, usize)],
        target: (usize, usize),
        observer: &mut crate::Observer<'_>,
    ) -> Result<Self> {
        ensure!(
            references.len() == shapes.len(),
            "reference latent count mismatch"
        );
        let img_in = ops::linear(vb.pp("img_in"), 64, 4096, false)?;
        let txt = vb.pp("txt_in");
        let weight = (txt.get(4096, "text_norm.weight")?.to_dtype(DType::F32)? + 1.0)?;
        // Zero-centered RMS scale is added in float32, before casting the product.
        let xf = encoded.hidden.to_dtype(DType::F32)?;
        let norm = xf
            .broadcast_div(&(xf.sqr()?.mean_keepdim(2)? + 1e-6)?.sqrt()?)?
            .broadcast_mul(&weight)?
            .to_dtype(vb.dtype())?;
        let text = ops::linear(txt.pp("out_layer"), 4096, 4096, false)?.forward(
            &ops::linear(txt.pp("in_layer"), 4096, 4096, false)?
                .forward(&norm)?
                .gelu()?,
        )?;
        let mut parts = Vec::new();
        let mut segments = Vec::new();
        let mut positions = Vec::new();
        let mut cursor = 0;
        let mut joint = 0;
        let mut position = 0;
        for (idx, (&(h, w), latents)) in shapes.iter().zip(references).enumerate() {
            let start = encoded.image_mask[cursor..]
                .iter()
                .position(|v| *v)
                .map(|v| cursor + v)
                .ok_or_else(|| {
                    anyhow::anyhow!("missing reference image {idx} in encoder output")
                })?;
            append_text(
                &text,
                cursor,
                start,
                &mut parts,
                &mut segments,
                &mut positions,
                &mut joint,
                &mut position,
            )?;
            let slots = h * w / 4;
            ensure!(
                encoded
                    .image_mask
                    .get(start..start + slots)
                    .is_some_and(|s| s.iter().all(|v| *v)),
                "reference image latent grid does not match encoder slots"
            );
            parts.push(img_in.forward(latents)?);
            segments.push(Segment {
                start: joint,
                len: h * w,
                text: false,
            });
            image_positions(&mut positions, &mut position, h, w);
            joint += h * w;
            cursor = start + slots;
        }
        ensure!(
            encoded.image_mask[cursor..].iter().all(|v| !v),
            "unmatched reference slots"
        );
        append_text(
            &text,
            cursor,
            encoded.image_mask.len(),
            &mut parts,
            &mut segments,
            &mut positions,
            &mut joint,
            &mut position,
        )?;
        let prefix_len = positions.len();
        image_positions(&mut positions, &mut position, target.0, target.1);
        let (cos, sin) = rotary(&positions, vb.device())?;
        let mut model = Self {
            img_in,
            time1: ops::linear(
                vb.pp("time_text_embed.timestep_embedder.linear_1"),
                256,
                4096,
                false,
            )?,
            time2: ops::linear(
                vb.pp("time_text_embed.timestep_embedder.linear_2"),
                4096,
                4096,
                false,
            )?,
            modulation: ops::linear(vb.pp("modulation.1"), 4096, 16384, false)?,
            final_scale: ops::linear(vb.pp("norm_out.linear"), 4096, 4096, false)?,
            out: ops::linear(vb.pp("proj_out"), 4096, 64, false)?,
            blocks: Vec::with_capacity(32),
            cos: cos
                .narrow(0, prefix_len, target.0 * target.1)?
                .contiguous()?,
            sin: sin
                .narrow(0, prefix_len, target.0 * target.1)?
                .contiguous()?,
        };
        let pc = cos.narrow(0, 0, prefix_len)?;
        let ps = sin.narrow(0, 0, prefix_len)?;
        let mut x = Tensor::cat(&parts, 1)?;
        let t0 = model.time(0.0)?;
        let modulation = model
            .modulation
            .forward(&candle_nn::ops::silu(&t0)?)?
            .unsqueeze(1)?;
        for i in 0..32 {
            let mut block = Block::load(vb.pp(format!("transformer_blocks.{i}")))?;
            x = block.forward(&x, &modulation, &pc, &ps, Some(&segments))?;
            ensure!(
                x.to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .is_finite(),
                "non-finite prefix block {i}"
            );
            model.blocks.push(block);
            vb.device().synchronize()?;
            observer.progress(crate::Stage::DenoiserLoading, i + 1, 32)?;
        }
        Ok(model)
    }

    fn time(&self, sigma: f64) -> candle_core::Result<Tensor> {
        let w = self.time1.weight();
        let t = Tensor::new(&[sigma as f32], w.device())?
            .to_dtype(w.dtype())?
            .to_dtype(DType::F32)?;
        let freqs: Vec<f32> = (0..128)
            .map(|i| (-10000_f32.ln() * i as f32 / 128.0).exp())
            .collect();
        let args =
            (t.unsqueeze(1)?
                .broadcast_mul(&Tensor::from_vec(freqs, (1, 128), w.device())?)?
                * 1000.0)?;
        let sinusoidal = Tensor::cat(&[args.cos()?, args.sin()?], 1)?.to_dtype(w.dtype())?;
        self.time2
            .forward(&candle_nn::ops::silu(&self.time1.forward(&sinusoidal)?)?)
    }

    pub fn forward(
        &mut self,
        latents: &Tensor,
        sigma: f64,
        observer: &mut crate::Observer<'_>,
    ) -> Result<Tensor> {
        let time = self.time(sigma)?;
        let activated = candle_nn::ops::silu(&time)?;
        let modulation = self.modulation.forward(&activated)?.unsqueeze(1)?;
        let mut x = self.img_in.forward(latents)?;
        // No per-block finiteness readback: it would stall the Metal queue 32
        // times per step. The caller checks the latents once per step.
        for block in &mut self.blocks {
            observer.poll_previews()?;
            x = block.forward(&x, &modulation, &self.cos, &self.sin, None)?;
        }
        let scale = (self.final_scale.forward(&activated)?.unsqueeze(1)? + 1.0)?;
        Ok(self.out.forward(&ops::norm(&x)?.broadcast_mul(&scale)?)?)
    }
}

#[allow(clippy::too_many_arguments)]
fn append_text(
    text: &Tensor,
    from: usize,
    to: usize,
    parts: &mut Vec<Tensor>,
    segments: &mut Vec<Segment>,
    positions: &mut Vec<[i32; 3]>,
    joint: &mut usize,
    position: &mut i32,
) -> candle_core::Result<()> {
    if to > from {
        parts.push(text.narrow(1, from, to - from)?);
        segments.push(Segment {
            start: *joint,
            len: to - from,
            text: true,
        });
        for _ in from..to {
            positions.push([*position; 3]);
            *position += 1;
        }
        *joint += to - from;
    }
    Ok(())
}

fn image_positions(out: &mut Vec<[i32; 3]>, position: &mut i32, h: usize, w: usize) {
    for y in 0..h {
        for x in 0..w {
            out.push([
                *position,
                y as i32 - h.div_ceil(2) as i32,
                x as i32 - w.div_ceil(2) as i32,
            ]);
        }
    }
    *position += h.max(w) as i32;
}

fn rotary(
    positions: &[[i32; 3]],
    device: &candle_core::Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let mut angles = Vec::with_capacity(positions.len() * 64);
    for p in positions {
        for (axis, half) in [8, 28, 28].into_iter().enumerate() {
            for i in 0..half {
                angles.push(p[axis] as f32 / 10000_f32.powf(i as f32 / half as f32));
            }
        }
    }
    let angles = Tensor::from_vec(angles, (positions.len(), 64), device)?;
    Ok((angles.cos()?, angles.sin()?))
}
