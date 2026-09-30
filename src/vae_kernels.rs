//! Metal kernels for the VAE's NCHW convolutions and channel norms. Candle's
//! conv2d and strided copies are bandwidth-starved at VAE sizes; these kernels
//! read NCHW directly and write the layouts the following matmul needs.
use candle_core::{
    CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::{collections::HashMap, sync::Mutex};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Im2col {
    uint c, h, w, ow, k, stride, pad, row0, rows;
};

// Output row `(ci * k + dy) * k + dx`, column `r * ow + x`: the transposed
// patch matrix, so `weight (out, c*k*k) @ patches` is already NCHW.
template <typename T>
[[kernel]] void im2col(
    constant Im2col &p [[buffer(0)]],
    device const T *src [[buffer(1)]],
    device T *dst [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = p.rows * p.ow;
    if (gid.x >= m || gid.y >= p.c * p.k * p.k) return;
    uint r = gid.x / p.ow, x = gid.x % p.ow;
    uint dx = gid.y % p.k, dy = (gid.y / p.k) % p.k, ci = gid.y / (p.k * p.k);
    int iy = int((p.row0 + r) * p.stride + dy) - int(p.pad);
    int ix = int(x * p.stride + dx) - int(p.pad);
    T v = T(0);
    if (iy >= 0 && iy < int(p.h) && ix >= 0 && ix < int(p.w)) {
        v = src[(ulong(ci) * p.h + uint(iy)) * p.w + uint(ix)];
    }
    dst[ulong(gid.y) * m + gid.x] = v;
}

struct ChannelNorm {
    uint c, pixels;
    float scale;
};

// Per pixel: x / max(|x|, 1e-12) * sqrt(c) * gamma, accumulated in float.
template <typename T>
[[kernel]] void channel_norm(
    constant ChannelNorm &p [[buffer(0)]],
    device const T *src [[buffer(1)]],
    device const T *gamma [[buffer(2)]],
    device T *dst [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.pixels) return;
    float sum = 0;
    for (uint ci = 0; ci < p.c; ci++) {
        float v = float(src[ulong(ci) * p.pixels + gid]);
        sum += v * v;
    }
    float inv = p.scale / max(sqrt(sum), 1e-12f);
    for (uint ci = 0; ci < p.c; ci++) {
        ulong i = ulong(ci) * p.pixels + gid;
        dst[i] = T(float(src[i]) * inv * float(gamma[ci]));
    }
}

template [[host_name("im2col_f32")]] [[kernel]] void im2col<float>(
    constant Im2col &, device const float *, device float *, uint2);
template [[host_name("im2col_bf16")]] [[kernel]] void im2col<bfloat>(
    constant Im2col &, device const bfloat *, device bfloat *, uint2);
template [[host_name("channel_norm_f32")]] [[kernel]] void channel_norm<float>(
    constant ChannelNorm &, device const float *, device const float *, device float *, uint);
template [[host_name("channel_norm_bf16")]] [[kernel]] void channel_norm<bfloat>(
    constant ChannelNorm &, device const bfloat *, device const bfloat *, device bfloat *, uint);
"#;

#[repr(C)]
struct Im2colParams {
    c: u32,
    h: u32,
    w: u32,
    ow: u32,
    k: u32,
    stride: u32,
    pad: u32,
    row0: u32,
    rows: u32,
}

#[repr(C)]
struct ChannelNormParams {
    c: u32,
    pixels: u32,
    scale: f32,
}

type PipelineKey = (u64, &'static str);
static PIPELINES: Mutex<Option<HashMap<PipelineKey, ComputePipeline>>> = Mutex::new(None);

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let metal = device.device();
    let key = (metal.registry_id(), name);
    let mut cache = PIPELINES
        .lock()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(p) = cache.get(&key) {
        return Ok(p.clone());
    }
    let msg =
        |e: candle_metal_kernels::err::MetalKernelError| candle_core::Error::Msg(e.to_string());
    let library = metal.new_library_with_source(SOURCE, None).map_err(msg)?;
    let function = library.get_function(name, None).map_err(msg)?;
    let p = metal
        .new_compute_pipeline_state_with_function(&function)
        .map_err(msg)?;
    cache.insert(key, p.clone());
    Ok(p)
}

fn kernel_name(dtype: DType, f32: &'static str, bf16: &'static str) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok(f32),
        DType::BF16 => Ok(bf16),
        dtype => candle_core::bail!("VAE Metal kernels do not support {dtype:?}"),
    }
}

fn size(x: usize, y: usize) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: 1,
    }
}

struct Im2colOp {
    k: usize,
    stride: usize,
    pad: usize,
    row0: usize,
    rows: usize,
    ow: usize,
}

impl CustomOp1 for Im2colOp {
    fn name(&self) -> &'static str {
        "vae-im2col"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("vae-im2col is Metal-only")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let (_, c, h, w) = layout.shape().dims4()?;
        let Some((start, _)) = layout.contiguous_offsets() else {
            candle_core::bail!("vae-im2col requires a contiguous input");
        };
        let dtype = storage.dtype();
        let device = storage.device();
        let rows_out = c * self.k * self.k;
        let m = self.rows * self.ow;
        let p = pipeline(device, kernel_name(dtype, "im2col_f32", "im2col_bf16")?)?;
        let dst = device.new_buffer(rows_out * m, dtype, "vae-im2col")?;
        let params = Im2colParams {
            c: c as u32,
            h: h as u32,
            w: w as u32,
            ow: self.ow as u32,
            k: self.k as u32,
            stride: self.stride as u32,
            pad: self.pad as u32,
            row0: self.row0 as u32,
            rows: self.rows as u32,
        };
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&p);
        encoder.set_bytes(0, &params);
        encoder.set_buffer(1, Some(storage.buffer()), start * dtype.size_in_bytes());
        encoder.set_buffer(2, Some(&dst), 0);
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*dst, MTLResourceUsage::Write);
        encoder.dispatch_threads(size(m, rows_out), size(64, 4));
        drop(encoder);
        Ok((
            MetalStorage::new(dst, device.clone(), rows_out * m, dtype),
            Shape::from((rows_out, m)),
        ))
    }
}

/// The `(c * k * k, rows * ow)` patch matrix for output rows
/// `row0..row0 + rows` of a `k`x`k` convolution over `x` (1, c, h, w).
pub fn im2col(
    x: &Tensor,
    k: usize,
    stride: usize,
    pad: usize,
    row0: usize,
    rows: usize,
    ow: usize,
) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&Im2colOp {
        k,
        stride,
        pad,
        row0,
        rows,
        ow,
    })
}

struct ChannelNormOp {
    gamma: Tensor,
}

impl CustomOp1 for ChannelNormOp {
    fn name(&self) -> &'static str {
        "vae-channel-norm"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("vae-channel-norm is Metal-only")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let Some((start, _)) = layout.contiguous_offsets() else {
            candle_core::bail!("vae-channel-norm requires a contiguous input");
        };
        if b != 1 {
            candle_core::bail!("vae-channel-norm supports batch size 1");
        }
        let dtype = storage.dtype();
        let device = storage.device();
        let (gamma, gamma_layout) = self.gamma.storage_and_layout();
        let candle_core::Storage::Metal(gamma) = &*gamma else {
            candle_core::bail!("vae-channel-norm gamma must be on Metal");
        };
        let Some((gamma_start, _)) = gamma_layout.contiguous_offsets() else {
            candle_core::bail!("vae-channel-norm requires a contiguous gamma");
        };
        let pixels = h * w;
        let p = pipeline(
            device,
            kernel_name(dtype, "channel_norm_f32", "channel_norm_bf16")?,
        )?;
        let dst = device.new_buffer(c * pixels, dtype, "vae-channel-norm")?;
        let params = ChannelNormParams {
            c: c as u32,
            pixels: pixels as u32,
            scale: (c as f32).sqrt(),
        };
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&p);
        encoder.set_bytes(0, &params);
        encoder.set_buffer(1, Some(storage.buffer()), start * dtype.size_in_bytes());
        encoder.set_buffer(2, Some(gamma.buffer()), gamma_start * dtype.size_in_bytes());
        encoder.set_buffer(3, Some(&dst), 0);
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(gamma.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*dst, MTLResourceUsage::Write);
        encoder.dispatch_threads(size(pixels, 1), size(256, 1));
        drop(encoder);
        Ok((
            MetalStorage::new(dst, device.clone(), c * pixels, dtype),
            layout.shape().clone(),
        ))
    }
}

/// `x / max(|x|, 1e-12) * sqrt(c) * gamma` across the channels of each pixel.
pub fn channel_norm(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let gamma = gamma.flatten_all()?.to_dtype(x.dtype())?.contiguous()?;
    x.contiguous()?.apply_op1_no_bwd(&ChannelNormOp { gamma })
}
