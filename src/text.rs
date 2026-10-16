//! The Qwen3-VL-8B encoder used by Qwen Image 2.1. Its layers are shared with
//! text generation in `chat`, which adds the final norm and LM head.
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use tokenizers::Tokenizer;

use crate::{
    ops::{self, SwiGlu},
    vision,
    weights::Weights,
};

const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const IMAGE_TOKEN: u32 = 151655;

#[derive(Clone)]
pub struct Encoded {
    pub hidden: Tensor,
    pub image_mask: Vec<bool>,
}

impl Encoded {
    pub fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            hidden: self.hidden.to_device(device)?,
            image_mask: self.image_mask.clone(),
        })
    }
}

/// `visual` supplies each reference's vision features, given its index and the
/// vision tower's weights, so the caller can reuse them across requests.
pub fn encode(
    weights: &Weights,
    prompt: &str,
    images: &[image::RgbaImage],
    device: &Device,
    observer: &mut crate::Observer<'_>,
    mut visual: impl FnMut(usize, VarBuilder, &mut crate::Observer<'_>) -> Result<vision::Visual>,
) -> Result<Encoded> {
    let tokenizer = tokenizer(weights)?;
    let mut input = format!("{SYSTEM}<|im_start|>user\n");
    for i in 0..images.len() {
        if i > 0 {
            input.push(' ');
        }
        input.push_str(&format!(
            "<image{}><|vision_start|><|image_pad|><|vision_end|>",
            i + 1
        ));
    }
    input.push_str(prompt);
    input.push_str("<|im_end|>\n<|im_start|>assistant\n");
    let tokens = tokenize(&tokenizer, &input, images)?;
    let dtype = DType::BF16;
    let vb = weights.builder("text_encoder", dtype, device)?.pp("model");
    let visual = (0..images.len())
        .map(|i| visual(i, vb.pp("visual"), observer))
        .collect::<Result<Vec<_>>>()?;
    observer.progress(crate::Stage::TextEncoding, 0, LAYERS)?;
    let language = vb.pp("language_model");
    let mut x = embed(&language, &tokens.ids, device)?;
    let (cos, sin) = rotary(&tokens.positions, device)?;
    for i in 0..LAYERS {
        x = layer(&language, i, &x, &cos, &sin, None, &tokens.spans, &visual)?;
        device.synchronize()?;
        ensure!(
            x.to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?
                .is_finite(),
            "text encoder produced non-finite values at layer {i}"
        );
        observer.progress(crate::Stage::TextEncoding, i + 1, LAYERS)?;
    }
    // The diffusion transformer expects the last decoder output BEFORE final RMSNorm.
    let drop = tokenizer
        .encode(SYSTEM, false)
        .map_err(anyhow::Error::msg)?
        .len();
    let ids = &tokens.ids;
    Ok(Encoded {
        hidden: x.narrow(1, drop, ids.len() - drop)?.contiguous()?,
        image_mask: ids[drop..].iter().map(|id| *id == IMAGE_TOKEN).collect(),
    })
}

pub(crate) const LAYERS: usize = 36;

pub(crate) fn tokenizer(weights: &Weights) -> Result<Tokenizer> {
    Tokenizer::from_file(weights.file("processor/tokenizer.json")?).map_err(anyhow::Error::msg)
}

/// A tokenized chat, with each image placeholder expanded to one token per
/// merged vision patch, and its multimodal rotary positions.
pub(crate) struct Tokens {
    pub ids: Vec<u32>,
    pub positions: Vec<[usize; 3]>,
    /// Start and length of each image's tokens.
    pub spans: Vec<(usize, usize)>,
    /// The rotary position of the next appended text token.
    pub next: usize,
}

/// Tokenizes `input`, whose `<|image_pad|>` placeholders stand for `images` in order.
pub(crate) fn tokenize(
    tokenizer: &Tokenizer,
    input: &str,
    images: &[image::RgbaImage],
) -> Result<Tokens> {
    let raw_ids = tokenizer
        .encode(input, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let mut ids = Vec::new();
    let mut grids = Vec::new();
    let mut image_idx = 0;
    for id in raw_ids {
        if id == IMAGE_TOKEN {
            ensure!(
                image_idx < images.len(),
                "prompt must not contain reserved image tokens"
            );
            let img = &images[image_idx];
            let (h, w) = (img.height() as usize / 32, img.width() as usize / 32);
            grids.push((h, w));
            ids.extend(std::iter::repeat_n(id, h * w));
            image_idx += 1;
        } else {
            ids.push(id);
        }
    }
    ensure!(
        image_idx == images.len(),
        "image token count does not match references"
    );
    ensure!(
        ids.len() <= 32768,
        "prompt and reference images exceed 32768 encoder tokens"
    );
    let (positions, spans, next) = positions(&ids, &grids)?;
    Ok(Tokens {
        ids,
        positions,
        spans,
        next,
    })
}

pub(crate) fn embed(language: &VarBuilder, ids: &[u32], device: &Device) -> Result<Tensor> {
    let embed = candle_nn::embedding(151936, 4096, language.pp("embed_tokens"))?;
    Ok(embed.forward(&Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)?)?)
}

/// Float32 keys and values of the tokens already processed by one layer.
pub(crate) type Cache = candle_nn::kv_cache::KvCache;

/// One decoder layer of the Qwen3-VL language model. Image features replace
/// the image tokens' embeddings before the first layer, and the vision tower's
/// deep features are added after each of the first three. With a `cache`,
/// `x` continues the cached tokens and its keys and values are appended; after
/// the first call, `x` must be a single token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer(
    language: &VarBuilder,
    i: usize,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    cache: Option<&mut Cache>,
    spans: &[(usize, usize)],
    visual: &[vision::Visual],
) -> candle_core::Result<Tensor> {
    let mut x = x.clone();
    if i == 0 && !visual.is_empty() {
        let features: Vec<_> = visual.iter().map(|v| v.hidden.clone()).collect();
        x = replace_images(&x, spans, &features, false)?;
    }
    let layer = language.pp(format!("layers.{i}"));
    let attn = layer.pp("self_attn");
    let n = ops::rms(&x, &layer.get(4096, "input_layernorm.weight")?, 1e-6)?;
    let project = |name: &str, heads: usize| -> candle_core::Result<Tensor> {
        ops::heads(
            &ops::linear(attn.pp(name), 4096, heads * 128, false)?.forward(&n)?,
            heads,
        )
    };
    let q = ops::rope(
        &ops::rms(
            &project("q_proj", 32)?,
            &attn.get(128, "q_norm.weight")?,
            1e-6,
        )?,
        cos,
        sin,
        false,
    )?;
    let k = ops::rope(
        &ops::rms(
            &project("k_proj", 8)?,
            &attn.get(128, "k_norm.weight")?,
            1e-6,
        )?,
        cos,
        sin,
        false,
    )?;
    let v = project("v_proj", 8)?;
    let attended = match cache {
        Some(cache) if cache.current_seq_len() > 0 => {
            let (k, v) = cache.append(&k.to_dtype(DType::F32)?, &v.to_dtype(DType::F32)?)?;
            ops::single_query_attention(&q, &k, &v)?
        }
        Some(cache) => {
            cache.append(&k.to_dtype(DType::F32)?, &v.to_dtype(DType::F32)?)?;
            ops::attention(&q, &k, &v, true)?
        }
        None => ops::attention(&q, &k, &v, true)?,
    };
    x = (&x
        + ops::linear(attn.pp("o_proj"), 4096, 4096, false)?
            .forward(&ops::unheads(&attended)?)?)?;
    let n = ops::rms(
        &x,
        &layer.get(4096, "post_attention_layernorm.weight")?,
        1e-6,
    )?;
    x = (&x
        + SwiGlu::new(
            layer.pp("mlp"),
            4096,
            12288,
            ["gate_proj", "up_proj", "down_proj"],
        )?
        .forward(&n)?)?;
    if i < 3 && !visual.is_empty() {
        let features: Vec<_> = visual.iter().map(|v| v.deep[i].clone()).collect();
        x = replace_images(&x, spans, &features, true)?;
    }
    Ok(x)
}

fn replace_images(
    x: &Tensor,
    spans: &[(usize, usize)],
    images: &[Tensor],
    add: bool,
) -> candle_core::Result<Tensor> {
    let mut parts = Vec::new();
    let mut pos = 0;
    for (&(start, len), image) in spans.iter().zip(images) {
        if start > pos {
            parts.push(x.narrow(1, pos, start - pos)?);
        }
        let image = image.reshape((1, len, 4096))?;
        parts.push(if add {
            (x.narrow(1, start, len)? + image)?
        } else {
            image
        });
        pos = start + len;
    }
    if pos < x.dim(1)? {
        parts.push(x.narrow(1, pos, x.dim(1)? - pos)?);
    }
    Tensor::cat(&parts, 1)
}

type Positions = (Vec<[usize; 3]>, Vec<(usize, usize)>, usize);
fn positions(ids: &[u32], grids: &[(usize, usize)]) -> Result<Positions> {
    let mut positions = Vec::new();
    let mut spans = Vec::new();
    let mut offset = 0;
    let mut i = 0;
    let mut img = 0;
    while i < ids.len() {
        if ids[i] == IMAGE_TOKEN {
            let &(h, w) = grids
                .get(img)
                .ok_or_else(|| anyhow::anyhow!("unexpected image token"))?;
            ensure!(
                ids.get(i..i + h * w)
                    .is_some_and(|s| s.iter().all(|t| *t == IMAGE_TOKEN)),
                "invalid image token span"
            );
            spans.push((i, h * w));
            for y in 0..h {
                for x in 0..w {
                    positions.push([offset, offset + y, offset + x]);
                }
            }
            offset += h.max(w);
            i += h * w;
            img += 1;
        } else {
            positions.push([offset; 3]);
            offset += 1;
            i += 1;
        }
    }
    Ok((positions, spans, offset))
}

pub(crate) fn rotary(positions: &[[usize; 3]], device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    let angles: Vec<f32> = positions
        .iter()
        .flat_map(|p| {
            (0..64).map(move |i| {
                let axis = if i < 60 { i % 3 } else { 0 };
                p[axis] as f32 / 5_000_000_f32.powf(i as f32 / 64.0)
            })
        })
        .collect();
    let angles = Tensor::from_vec(angles, (positions.len(), 64), device)?;
    Ok((angles.cos()?, angles.sin()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_positions_resume_after_spatial_extent() -> Result<()> {
        let (p, spans, next) = positions(
            &[1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2],
            &[(2, 2)],
        )?;
        assert_eq!(
            p,
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3]
            ]
        );
        assert_eq!(spans, [(1, 4)]);
        assert_eq!(next, 4);
        Ok(())
    }
}
