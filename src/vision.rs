//! Image-only Qwen3-VL vision tower; no video processing.
use crate::ops;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use image::RgbaImage;

pub struct Visual {
    pub hidden: Tensor,
    pub deep: Vec<Tensor>,
}

pub fn encode(
    image: &RgbaImage,
    vb: VarBuilder,
    index: usize,
    observer: &mut crate::Observer<'_>,
) -> anyhow::Result<Visual> {
    observer.progress(crate::Stage::ReferenceVision { index }, 0, 27)?;
    let device = vb.device();
    let dtype = vb.dtype();
    let gh = image.height() as usize / 16;
    let gw = image.width() as usize / 16;
    let n = gh * gw;
    // The processor groups 2x2 patches before merging. Each still image is
    // duplicated along the two-frame kernel axis, with channels outermost.
    let mut pixels = Vec::with_capacity(n * 1536);
    let mut coords = Vec::with_capacity(n);
    for by in 0..gh / 2 {
        for bx in 0..gw / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (py, px) = (by * 2 + dy, bx * 2 + dx);
                    coords.push((py, px));
                    for c in 0..3 {
                        for _t in 0..2 {
                            for y in 0..16 {
                                for x in 0..16 {
                                    let rgba = image
                                        .get_pixel((px * 16 + x) as u32, (py * 16 + y) as u32)
                                        .0;
                                    // Match PIL's RGBA-over-white integer compositing for the vision copy.
                                    let rgb = (rgba[c] as u32 * rgba[3] as u32
                                        + 255 * (255 - rgba[3] as u32)
                                        + 127)
                                        / 255;
                                    pixels.push(rgb as f32 / 127.5 - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let patch_weight = vb
        .get((1152, 3, 2, 16, 16), "patch_embed.proj.weight")?
        .reshape((1152, 1536))?;
    let patch = candle_nn::Linear::new(patch_weight, Some(vb.get(1152, "patch_embed.proj.bias")?));
    let mut x = patch.forward(&Tensor::from_vec(pixels, (n, 1536), device)?.to_dtype(dtype)?)?;

    // Learned 48x48 positions, bilinear interpolation with aligned corners.
    let pos = vb
        .get((2304, 1152), "pos_embed.weight")?
        .to_dtype(DType::F32)?;
    let mut indices = Vec::with_capacity(n * 4);
    let mut factors = Vec::with_capacity(n * 4);
    let mut angles = Vec::with_capacity(n * 36);
    for &(y, xx) in &coords {
        let yy = y as f32 * 47.0 / (gh - 1) as f32;
        let xx_f = xx as f32 * 47.0 / (gw - 1) as f32;
        let (y0, x0) = (yy.floor() as usize, xx_f.floor() as usize);
        let (fy, fx) = (yy - y0 as f32, xx_f - x0 as f32);
        indices.extend([
            (y0 * 48 + x0) as u32,
            (y0 * 48 + (x0 + 1).min(47)) as u32,
            ((y0 + 1).min(47) * 48 + x0) as u32,
            ((y0 + 1).min(47) * 48 + (x0 + 1).min(47)) as u32,
        ]);
        factors.extend([
            (1.0 - fy) * (1.0 - fx),
            (1.0 - fy) * fx,
            fy * (1.0 - fx),
            fy * fx,
        ]);
        for p in [y, xx] {
            for j in 0..18 {
                angles.push(p as f32 / 10000_f32.powf(j as f32 / 18.0));
            }
        }
    }
    let pos = pos
        .index_select(&Tensor::from_vec(indices, n * 4, device)?, 0)?
        .reshape((n, 4, 1152))?
        .broadcast_mul(&Tensor::from_vec(factors, (n, 4, 1), device)?)?
        .sum(1)?
        .to_dtype(dtype)?;
    x = (x + pos)?.unsqueeze(0)?;
    let angles = Tensor::from_vec(angles, (n, 36), device)?;
    let (cos, sin) = (angles.cos()?, angles.sin()?);
    let mut deep = Vec::new();
    for i in 0..27 {
        let block = vb.pp(format!("blocks.{i}"));
        let norm = ops::affine_norm(&x, block.pp("norm1"))?;
        let qkv = ops::linear(block.pp("attn.qkv"), 1152, 3456, true)?.forward(&norm)?;
        let q = ops::rope(
            &ops::heads(&qkv.narrow(2, 0, 1152)?, 16)?,
            &cos,
            &sin,
            false,
        )?;
        let k = ops::rope(
            &ops::heads(&qkv.narrow(2, 1152, 1152)?, 16)?,
            &cos,
            &sin,
            false,
        )?;
        let v = ops::heads(&qkv.narrow(2, 2304, 1152)?, 16)?;
        let a = ops::unheads(&ops::attention(&q, &k, &v, false)?)?;
        x = (x + ops::linear(block.pp("attn.proj"), 1152, 1152, true)?.forward(&a)?)?;
        let norm = ops::affine_norm(&x, block.pp("norm2"))?;
        let mlp = ops::linear(block.pp("mlp.linear_fc1"), 1152, 4304, true)?
            .forward(&norm)?
            .gelu()?;
        x = (x + ops::linear(block.pp("mlp.linear_fc2"), 4304, 1152, true)?.forward(&mlp)?)?;
        if [8, 16, 24].contains(&i) {
            deep.push(merge(
                &x,
                vb.pp(format!("deepstack_merger_list.{}", deep.len())),
                true,
            )?);
        }
        device.synchronize()?;
        observer.progress(crate::Stage::ReferenceVision { index }, i + 1, 27)?;
    }
    Ok(Visual {
        hidden: merge(&x, vb.pp("merger"), false)?,
        deep,
    })
}

fn merge(x: &Tensor, vb: VarBuilder, post_shuffle: bool) -> Result<Tensor> {
    let n = x.dim(1)? / 4;
    let x = if post_shuffle {
        ops::affine_norm(&x.reshape((n, 4608))?, vb.pp("norm"))?
    } else {
        ops::affine_norm(x, vb.pp("norm"))?.reshape((n, 4608))?
    };
    let x = ops::linear(vb.pp("linear_fc1"), 4608, 4608, true)?
        .forward(&x)?
        .gelu_erf()?;
    ops::linear(vb.pp("linear_fc2"), 4608, 4096, true)?.forward(&x)
}
