//! Linear layers stored either dense or as MLX affine-quantized weights.
//!
//! MLX packs `bits`-wide unsigned values into little-endian `u32` words along
//! the input dimension (first value in the lowest bits). Each group of
//! `group` inputs has its own BF16 `scale` and offset (`biases`):
//! `w = scale * q + offset`. The layer's own additive bias is `bias`.
//!
//! Quantized weights stay packed in memory. On Metal, a fused kernel unpacks
//! weight tiles into threadgroup memory inside the matmul, so dense weights
//! never exist in device memory. The CPU (and unsupported shapes) dequantize
//! into a temporary dense weight instead.
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Device, Layout, MetalStorage, Module, Result, Shape,
    Tensor, backend::BackendStorage,
};
use candle_metal_kernels::metal::ComputePipeline;
use candle_nn::VarBuilder;
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::{collections::HashMap, sync::Mutex};

pub enum Linear {
    Dense(candle_nn::Linear),
    Affine(Affine),
}

pub struct Affine {
    packed: Tensor,
    scales: Tensor,
    offsets: Tensor,
    bias: Option<Tensor>,
    bits: usize,
    group: usize,
    input: usize,
}

impl Linear {
    /// Loads `weight` (and `bias`), quantized when the checkpoint has `scales`.
    pub fn load(vb: VarBuilder, input: usize, output: usize, bias: bool) -> Result<Self> {
        if !vb.contains_tensor("scales") {
            return Ok(Self::Dense(candle_nn::linear_b(input, output, bias, vb)?));
        }
        // Not `get_unchecked`: in Candle 0.9.2 it ignores `VarBuilder::to_dtype`.
        let scales = vb.get_unchecked_dtype("scales", vb.dtype())?;
        let (rows, groups) = scales.dims2()?;
        if rows != output || groups == 0 || !input.is_multiple_of(groups) {
            candle_core::bail!(
                "{}: scales {:?} do not match a {input}->{output} linear",
                vb.prefix(),
                scales.shape()
            );
        }
        // Packed values are read unconverted; the builder's dtype is for floats.
        let packed = vb.get_unchecked_dtype("weight", DType::U32)?;
        let (rows, words) = packed.dims2()?;
        let bits = words * 32 / input;
        if rows != output || words * 32 != bits * input || ![2, 4, 8].contains(&bits) {
            candle_core::bail!(
                "{}: packed weight {:?} is not 2-, 4- or 8-bit for {input} inputs",
                vb.prefix(),
                packed.shape()
            );
        }
        Ok(Self::Affine(Affine {
            packed,
            scales,
            offsets: vb.get((output, groups), "biases")?,
            bias: bias.then(|| vb.get(output, "bias")).transpose()?,
            bits,
            group: input / groups,
            input,
        }))
    }

    /// The equivalent dense layer. Quantized layers allocate their weights.
    pub fn dense(&self) -> Result<candle_nn::Linear> {
        match self {
            Self::Dense(linear) => Ok(linear.clone()),
            Self::Affine(a) => Ok(candle_nn::Linear::new(a.dequantize()?, a.bias.clone())),
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Dense(linear) => linear.weight().device(),
            Self::Affine(a) => a.scales.device(),
        }
    }

    /// Floating-point dtype of the layer's weights and outputs.
    pub fn dtype(&self) -> DType {
        match self {
            Self::Dense(linear) => linear.weight().dtype(),
            Self::Affine(a) => a.scales.dtype(),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::Affine(a) if a.fused(x) => a.forward(x),
            Self::Affine(_) => self.dense()?.forward(x),
        }
    }
}

impl Affine {
    /// The fused Metal kernel needs 4- or 8-bit weights, whole 32-value K
    /// tiles and 16-value groups, with inputs and scales of one float dtype.
    fn fused(&self, x: &Tensor) -> bool {
        x.device().is_metal()
            && matches!(self.bits, 4 | 8)
            && x.dtype() == self.scales.dtype()
            && matches!(x.dtype(), DType::F32 | DType::BF16)
            && self.input.is_multiple_of(32)
            && self.group.is_multiple_of(16)
            && x.elem_count() > 0
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut dims = x.dims().to_vec();
        let k = dims.pop().unwrap_or(0);
        if k != self.input {
            candle_core::bail!("quantized linear expects {} inputs, got {k}", self.input);
        }
        let m = x.elem_count() / k;
        let y = x.reshape((m, k))?.contiguous()?.apply_op2_no_bwd(
            &self.packed,
            &Qmm {
                scales: self.scales.clone(),
                offsets: self.offsets.clone(),
                bits: self.bits,
                group: self.group,
            },
        )?;
        dims.push(y.dim(1)?);
        let y = y.reshape(dims)?;
        match &self.bias {
            Some(bias) => y.broadcast_add(bias),
            None => Ok(y),
        }
    }

    fn dequantize(&self) -> Result<Tensor> {
        self.packed.apply_op3_no_bwd(
            &self.scales,
            &self.offsets,
            &Dequantize {
                bits: self.bits,
                group: self.group,
                input: self.input,
            },
        )
    }
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Dequantize {
    uint input, output, words, groups, group, bits;
};

// One thread per output weight, so that adjacent threads write adjacent values.
template <typename T>
[[kernel]] void dequantize(
    constant Dequantize &p [[buffer(0)]],
    device const uint *packed [[buffer(1)]],
    device const T *scales [[buffer(2)]],
    device const T *offsets [[buffer(3)]],
    device T *dst [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.input || gid.y >= p.output) return;
    uint per_word = 32 / p.bits;
    uint word = packed[ulong(gid.y) * p.words + gid.x / per_word];
    uint q = (word >> (p.bits * (gid.x % per_word))) & ((1u << p.bits) - 1u);
    ulong g = ulong(gid.y) * p.groups + gid.x / p.group;
    dst[ulong(gid.y) * p.input + gid.x] = T(float(scales[g]) * float(q) + float(offsets[g]));
}

struct Qmm {
    uint m, n, k, groups, group;
};

constant constexpr short BM = 128, BN = 64, BK = 32, LD = BK + 4, THREADS = 256;

// y (m, n) = x (m, k) * w^T, with w (n, k) unpacked tile by tile into
// threadgroup memory; the dense weights never exist in device memory.
// 256 threads per 128x64 output tile: eight simdgroups of 32x32, each as 4x4
// float 8x8 accumulators, so that a core runs enough simdgroups to hide the
// load phase. Requires k % 32 == 0 and group % 8 == 0, so that each thread's
// consecutive weights share one scale and offset.
template <typename T, int BITS>
[[kernel]] void qmm(
    constant Qmm &p [[buffer(0)]],
    device const T *x [[buffer(1)]],
    device const uint *w [[buffer(2)]],
    device const T *scales [[buffer(3)]],
    device const T *offsets [[buffer(4)]],
    device T *y [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    // Each thread loads 16 consecutive activations of one row and 8
    // consecutive weights of one row. Rows past the edge read a valid row and
    // are zeroed by their multiplier.
    constexpr uint PER_WORD = 32 / BITS, WORDS = 8 / PER_WORD, MASK = (1u << BITS) - 1u;
    threadgroup float xs[BM * LD], ws[BN * LD];
    const uint row0 = tg.y * BM, col0 = tg.x * BN;
    const short xr_ = tid / 2, xc = (tid % 2) * 16;
    const short wr_ = tid / 4, wc = (tid % 4) * 8;
    const short sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    const float x_keep = row0 + xr_ < p.m ? 1.f : 0.f, w_keep = col0 + wr_ < p.n ? 1.f : 0.f;
    device const vec<T, 4> *xp =
        (device const vec<T, 4> *)(x + ulong(min(row0 + xr_, p.m - 1)) * p.k + xc);
    const uint wn = min(col0 + wr_, p.n - 1);
    device const uint *wp = w + ulong(wn) * (p.k / PER_WORD) + wc / PER_WORD;
    device const T *sp = scales + ulong(wn) * p.groups;
    device const T *op = offsets + ulong(wn) * p.groups;
    threadgroup float4 *xd = (threadgroup float4 *)(xs + xr_ * LD + xc);
    threadgroup float4 *wd = (threadgroup float4 *)(ws + wr_ * LD + wc);

    simdgroup_matrix<float, 8, 8> acc[4][4];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 4; i++)
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    // Registers for the next tile: its device loads overlap the current MMA.
    vec<T, 4> xr[4];
    uint wv[WORDS];
    float s, b;
    auto fetch = [&](uint k0) {
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < 4; i++) xr[i] = xp[k0 / 4 + i];
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < short(WORDS); i++) wv[i] = wp[k0 / PER_WORD + i];
        uint g = (k0 + wc) / p.group;
        s = float(sp[g]) * w_keep;
        b = float(op[g]) * w_keep;
    };
    fetch(0);
    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < 4; i++) xd[i] = float4(xr[i]) * x_keep;
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < 2; i++) {
            // Values 4i..4i+3 of this thread's 8, from word (4i) / PER_WORD.
            uint word = wv[(4 * i) / PER_WORD] >> (BITS * ((4 * i) % PER_WORD));
            uint4 q = uint4(word, word >> BITS, word >> (2 * BITS), word >> (3 * BITS)) & MASK;
            wd[i] = s * float4(q) + b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < p.k) fetch(k0 + BK);
        _Pragma("clang loop unroll(full)")
        for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> a[4], bt[4];
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 4; i++)
                simdgroup_load(a[i], xs + (sr + i * 8) * LD + kk, LD);
            _Pragma("clang loop unroll(full)")
            for (short j = 0; j < 4; j++)
                simdgroup_load(bt[j], ws + (sc + j * 8) * LD + kk, LD, ulong2(0, 0), true);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 4; i++)
                _Pragma("clang loop unroll(full)")
                for (short j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], bt[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each lane writes its two elements of every fragment: row fm, columns
    // fn and fn + 1 (the simdgroup matrix layout MLX's steel kernels use).
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 4; i++) {
        uint r = row0 + sr + i * 8 + fm;
        if (r >= p.m) continue;
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; j++) {
            uint c = col0 + sc + j * 8 + fn;
            auto v = acc[i][j].thread_elements();
            if (c < p.n) y[ulong(r) * p.n + c] = T(v[0]);
            if (c + 1 < p.n) y[ulong(r) * p.n + c + 1] = T(v[1]);
        }
    }
}

#define QMM(NAME, T, BITS) \
template [[host_name(NAME)]] [[kernel]] void qmm<T, BITS>( \
    constant Qmm &, device const T *, device const uint *, device const T *, \
    device const T *, device T *, uint2, ushort, ushort, ushort);
QMM("qmm_f32_4", float, 4)
QMM("qmm_f32_8", float, 8)
QMM("qmm_bf16_4", bfloat, 4)
QMM("qmm_bf16_8", bfloat, 8)

template [[host_name("dequantize_f32")]] [[kernel]] void dequantize<float>(
    constant Dequantize &, device const uint *, device const float *, device const float *,
    device float *, uint2);
template [[host_name("dequantize_bf16")]] [[kernel]] void dequantize<bfloat>(
    constant Dequantize &, device const uint *, device const bfloat *, device const bfloat *,
    device bfloat *, uint2);
"#;

#[repr(C)]
struct DequantizeParams {
    input: u32,
    output: u32,
    words: u32,
    groups: u32,
    group: u32,
    bits: u32,
}

static PIPELINES: Mutex<Option<HashMap<(u64, &'static str), ComputePipeline>>> = Mutex::new(None);

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

struct Dequantize {
    bits: usize,
    group: usize,
    input: usize,
}

struct Qmm {
    scales: Tensor,
    offsets: Tensor,
    bits: usize,
    group: usize,
}

#[repr(C)]
struct QmmParams {
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    group: u32,
}

/// A contiguous Metal tensor's buffer and byte offset.
fn metal_buffer(t: &Tensor) -> Result<(candle_metal_kernels::metal::Buffer, usize)> {
    let (storage, layout) = t.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        candle_core::bail!("quantized matmul weights must be on Metal");
    };
    Ok((
        storage.buffer().clone(),
        contiguous(layout)? * t.dtype().size_in_bytes(),
    ))
}

impl CustomOp2 for Qmm {
    fn name(&self) -> &'static str {
        "mlx-qmm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("mlx-qmm is Metal-only")
    }

    fn metal_fwd(
        &self,
        x: &MetalStorage,
        x_layout: &Layout,
        packed: &MetalStorage,
        packed_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (m, k) = x_layout.shape().dims2()?;
        let (n, words) = packed_layout.shape().dims2()?;
        let groups = k / self.group;
        let dtype = x.dtype();
        if words * 32 != k * self.bits
            || self.scales.dims2()? != (n, groups)
            || self.offsets.shape() != self.scales.shape()
            || self.scales.dtype() != dtype
            || packed.dtype() != DType::U32
        {
            candle_core::bail!("mlx-qmm: inconsistent quantized weights");
        }
        let name = match (dtype, self.bits) {
            (DType::F32, 4) => "qmm_f32_4",
            (DType::F32, 8) => "qmm_f32_8",
            (DType::BF16, 4) => "qmm_bf16_4",
            (DType::BF16, 8) => "qmm_bf16_8",
            (dtype, bits) => candle_core::bail!("mlx-qmm does not support {bits}-bit {dtype:?}"),
        };
        let device = x.device();
        let p = pipeline(device, name)?;
        let dst = device.new_buffer(m * n, dtype, "mlx-qmm")?;
        let params = QmmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            groups: groups as u32,
            group: self.group as u32,
        };
        let (scales, scales_offset) = metal_buffer(&self.scales)?;
        let (offsets, offsets_offset) = metal_buffer(&self.offsets)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&p);
        encoder.set_bytes(0, &params);
        encoder.set_buffer(
            1,
            Some(x.buffer()),
            contiguous(x_layout)? * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(packed.buffer()),
            contiguous(packed_layout)? * DType::U32.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&scales), scales_offset);
        encoder.set_buffer(4, Some(&offsets), offsets_offset);
        encoder.set_buffer(5, Some(&dst), 0);
        for buffer in [x.buffer(), packed.buffer(), &scales, &offsets] {
            encoder.use_resource(buffer, MTLResourceUsage::Read);
        }
        encoder.use_resource(&*dst, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n.div_ceil(64),
                height: m.div_ceil(128),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(dst, device.clone(), m * n, dtype),
            Shape::from((m, n)),
        ))
    }
}

impl Dequantize {
    fn shapes(&self, packed: &Layout, scales: &Layout, offsets: &Layout) -> Result<(usize, usize)> {
        let (output, words) = packed.shape().dims2()?;
        let groups = self.input / self.group;
        if scales.shape().dims2()? != (output, groups) || offsets.shape() != scales.shape() {
            candle_core::bail!("dequantize: scale and offset shapes do not match the weights");
        }
        Ok((output, words))
    }
}

fn contiguous(layout: &Layout) -> Result<usize> {
    match layout.contiguous_offsets() {
        Some((start, _)) => Ok(start),
        None => candle_core::bail!("dequantize requires contiguous inputs"),
    }
}

impl CustomOp3 for Dequantize {
    fn name(&self) -> &'static str {
        "mlx-dequantize"
    }

    fn cpu_fwd(
        &self,
        packed: &CpuStorage,
        packed_layout: &Layout,
        scales: &CpuStorage,
        scales_layout: &Layout,
        offsets: &CpuStorage,
        offsets_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (output, words) = self.shapes(packed_layout, scales_layout, offsets_layout)?;
        let CpuStorage::U32(packed) = packed else {
            candle_core::bail!("dequantize: packed weights must be u32");
        };
        let packed = &packed[contiguous(packed_layout)?..];
        let floats = |s: &CpuStorage, layout: &Layout| -> Result<Vec<f32>> {
            let start = contiguous(layout)?;
            let n = layout.shape().elem_count();
            Ok(match s {
                CpuStorage::F32(v) => v[start..start + n].to_vec(),
                CpuStorage::BF16(v) => v[start..start + n].iter().map(|x| x.to_f32()).collect(),
                _ => candle_core::bail!("dequantize supports f32 and bf16 scales"),
            })
        };
        let (scales_f, offsets_f) = (
            floats(scales, scales_layout)?,
            floats(offsets, offsets_layout)?,
        );
        let groups = self.input / self.group;
        let per_word = 32 / self.bits;
        let mask = (1u32 << self.bits) - 1;
        let mut values = Vec::with_capacity(output * self.input);
        for row in 0..output {
            for col in 0..self.input {
                let word = packed[row * words + col / per_word];
                let q = (word >> (self.bits * (col % per_word))) & mask;
                let g = row * groups + col / self.group;
                values.push(scales_f[g] * q as f32 + offsets_f[g]);
            }
        }
        let storage = match scales {
            CpuStorage::BF16(_) => {
                CpuStorage::BF16(values.into_iter().map(half::bf16::from_f32).collect())
            }
            _ => CpuStorage::F32(values),
        };
        Ok((storage, Shape::from((output, self.input))))
    }

    fn metal_fwd(
        &self,
        packed: &MetalStorage,
        packed_layout: &Layout,
        scales: &MetalStorage,
        scales_layout: &Layout,
        offsets: &MetalStorage,
        offsets_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (output, words) = self.shapes(packed_layout, scales_layout, offsets_layout)?;
        let dtype = scales.dtype();
        if packed.dtype() != DType::U32 || offsets.dtype() != dtype {
            candle_core::bail!("dequantize: expected u32 weights and matching scale dtypes");
        }
        let name = match dtype {
            DType::F32 => "dequantize_f32",
            DType::BF16 => "dequantize_bf16",
            dtype => candle_core::bail!("dequantize does not support {dtype:?}"),
        };
        let device = packed.device();
        let p = pipeline(device, name)?;
        let n = output * self.input;
        let dst = device.new_buffer(n, dtype, "mlx-dequantize")?;
        let params = DequantizeParams {
            input: self.input as u32,
            output: output as u32,
            words: words as u32,
            groups: (self.input / self.group) as u32,
            group: self.group as u32,
            bits: self.bits as u32,
        };
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&p);
        encoder.set_bytes(0, &params);
        let buffers = [
            (packed, contiguous(packed_layout)?, DType::U32),
            (scales, contiguous(scales_layout)?, dtype),
            (offsets, contiguous(offsets_layout)?, dtype),
        ];
        for (i, (storage, start, dtype)) in buffers.into_iter().enumerate() {
            encoder.set_buffer(i + 1, Some(storage.buffer()), start * dtype.size_in_bytes());
            encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(4, Some(&dst), 0);
        encoder.use_resource(&*dst, MTLResourceUsage::Write);
        encoder.dispatch_threads(
            MTLSize {
                width: self.input,
                height: output,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 4,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(dst, device.clone(), n, dtype),
            Shape::from((output, self.input)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture() -> Result<HashMap<String, Tensor>> {
        candle_core::safetensors::load(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/quantized.safetensors"
            ),
            &Device::Cpu,
        )
    }

    fn max_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let delta = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(delta.into_iter().fold(0f32, f32::max))
    }

    fn check(device: &Device, dtype: DType) -> Result<()> {
        let data = fixture()?;
        let vb = VarBuilder::from_tensors(data.clone(), dtype, device);
        let x = data["x"].to_dtype(dtype)?.to_device(device)?;
        for bits in [4, 8] {
            let p = format!("q{bits}");
            let linear = Linear::load(vb.pp(&p), 192, 96, false)?;
            assert!(matches!(&linear, Linear::Affine(a) if a.bits == bits && a.group == 64));
            // MLX's dequantize rounds the same float expression to BF16.
            let weight = linear.dense()?.weight().to_device(&Device::Cpu)?;
            let error = max_error(&weight, &data[&format!("{p}.dequantized")])?;
            assert!(
                error <= 1e-3,
                "{p} {dtype:?} dequantize on {device:?}: {error}"
            );
            let y = linear.forward(&x)?.to_device(&Device::Cpu)?;
            let error = max_error(&y, &data[&format!("{p}.y")])?;
            assert!(error < 0.02, "{p} {dtype:?} matmul on {device:?}: {error}");
        }
        Ok(())
    }

    #[test]
    fn cpu_dequantization_matches_mlx() -> Result<()> {
        // Candle's CPU backend has no BF16 matmul.
        check(&Device::Cpu, DType::F32)
    }

    #[test]
    #[ignore = "requires Metal GPU access"]
    fn metal_dequantization_matches_mlx() -> Result<()> {
        let device = Device::new_metal(0)?;
        check(&device, DType::BF16)?;
        check(&device, DType::F32)
    }

    /// A random packed layer: words, scales and offsets of plausible magnitude.
    fn random_layer(
        n: usize,
        k: usize,
        bits: usize,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        let groups = k / 64;
        let words: Vec<u32> = (0..n * k * bits / 32).map(|_| rand::random()).collect();
        let small = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|_| (rand::random::<f32>() - 0.5) * scale)
                .collect()
        };
        let tensors = HashMap::from([
            (
                "weight".to_string(),
                Tensor::from_vec(words, (n, k * bits / 32), device)?,
            ),
            (
                "scales".to_string(),
                Tensor::from_vec(small(n * groups, 0.02), (n, groups), device)?,
            ),
            (
                "biases".to_string(),
                Tensor::from_vec(small(n * groups, 0.1), (n, groups), device)?,
            ),
            (
                "bias".to_string(),
                Tensor::from_vec(small(n, 1.), n, device)?,
            ),
        ]);
        Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
    }

    #[test]
    #[ignore = "requires Metal GPU access"]
    fn fused_metal_matmul_matches_dequantized_weights() -> Result<()> {
        let gpu = Device::new_metal(0)?;
        // Model shapes, plus ragged M and N edges and a tile-sized K.
        for (m, k, n) in [
            (37, 4096, 1024),
            (130, 1152, 3456),
            (1, 128, 96),
            (64, 64, 64),
            (200, 4096, 12288),
        ] {
            for bits in [4, 8] {
                for dtype in [DType::F32, DType::BF16] {
                    let vb = random_layer(n, k, bits, &gpu)?.to_dtype(dtype);
                    // Vision layers add a bias after the quantized matmul.
                    let linear = Linear::load(vb, k, n, bits == 8)?;
                    let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
                    let x = Tensor::from_vec(data, (1, m, k), &gpu)?.to_dtype(dtype)?;
                    let Linear::Affine(affine) = &linear else {
                        unreachable!()
                    };
                    assert!(affine.fused(&x));
                    let fused = linear.forward(&x)?;
                    assert_eq!(fused.dims(), [1, m, n]);
                    let expected = linear.dense()?.forward(&x)?;
                    let scale = expected
                        .to_dtype(DType::F32)?
                        .abs()?
                        .flatten_all()?
                        .max(0)?
                        .to_scalar::<f32>()?;
                    let error = max_error(&fused, &expected)? / scale;
                    let limit = if dtype == DType::F32 { 1e-5 } else { 1e-2 };
                    assert!(
                        error < limit,
                        "m={m} k={k} n={n} {bits}-bit {dtype:?}: relative error {error}"
                    );
                }
            }
        }
        // Each fused thread reads 8 values, half a 2-bit word: 2-bit falls back.
        let linear = Linear::load(random_layer(64, 128, 2, &gpu)?, 128, 64, false)?;
        let x = Tensor::ones((3, 128), DType::F32, &gpu)?;
        let Linear::Affine(affine) = &linear else {
            unreachable!()
        };
        assert!(!affine.fused(&x));
        assert_eq!(linear.forward(&x)?.dims(), [3, 64]);
        Ok(())
    }

    #[test]
    fn dense_weights_load_unchanged_and_bad_packing_is_rejected() -> Result<()> {
        let mut data = fixture()?;
        data.insert("dense.weight".into(), data["q4.dequantized"].clone());
        let vb = VarBuilder::from_tensors(data.clone(), DType::BF16, &Device::Cpu);
        assert!(matches!(
            Linear::load(vb.pp("dense"), 192, 96, false)?,
            Linear::Dense(_)
        ));
        assert!(Linear::load(vb.pp("q4"), 128, 96, false).is_err());
        data.insert("q4.weight".into(), data["q4.weight"].narrow(1, 0, 20)?);
        let vb = VarBuilder::from_tensors(data, DType::BF16, &Device::Cpu);
        assert!(Linear::load(vb.pp("q4"), 192, 96, false).is_err());
        Ok(())
    }
}
