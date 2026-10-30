//! The compute device: Metal on macOS, the CPU everywhere else.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Shown in logs and the CLI's summary line.
pub const NAME: &str = if cfg!(target_os = "macos") {
    "Metal"
} else {
    "CPU"
};

#[cfg(target_os = "macos")]
pub fn new() -> Result<Device> {
    use anyhow::{Context, ensure};
    ensure!(
        objc2_metal::MTLCreateSystemDefaultDevice().is_some(),
        "no Metal GPU is visible; run on an Apple Silicon Mac with GPU access"
    );
    Device::new_metal(0).context("could not initialize Metal")
}

#[cfg(not(target_os = "macos"))]
pub fn new() -> Result<Device> {
    Ok(Device::Cpu)
}

/// Candle's CPU backend has no BF16 matmul (or convolution). Such products are
/// computed in float32 there; weights stay BF16 in memory.
pub fn upcast(x: &Tensor) -> bool {
    x.device().is_cpu() && x.dtype() == DType::BF16
}
