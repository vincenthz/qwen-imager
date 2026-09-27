//! Native Qwen Image 2.1 inference on Apple Metal.
//!
//! Run [`Generator::generate`] on a worker thread. Callbacks execute synchronously
//! on that thread; forward owned [`Event`] values to your UI through a channel.
//! No tensor or GPU handle is exposed to the caller. See `examples/progress.rs`.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("img-gen requires an Apple Silicon Mac with Metal.");

mod dit;
mod ops;
mod pipeline;
mod text;
mod vae;
mod vision;
mod weights;

use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

pub use anyhow::Result;
pub use image::RgbaImage;
pub use weights::{MODEL, REVISION};

/// Checkpoint location and download policy. GPU resources are allocated by `generate`.
#[derive(Clone, Debug, Default)]
pub struct ModelOptions {
    /// Original Diffusers snapshot directory, or the Hugging Face cache when absent.
    /// Keep checkpoint files unchanged for the duration of inference (they are mapped).
    pub model_dir: Option<PathBuf>,
    pub offline: bool,
}

/// Download progress for one missing checkpoint file. Existing cached files do
/// not emit events. `total` is unknown while fetching the server's metadata.
#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
}

/// One image generation or edit. Images are owned CPU RGBA buffers, not file paths.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub images: Vec<RgbaImage>,
    /// `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, or `9:16`.
    /// When absent, follows the last reference image, or defaults to square.
    pub ratio: Option<String>,
    /// Fraction of native 2K resolution, in (0, 1].
    pub scale: f64,
    pub steps: usize,
    pub seed: u64,
    /// Decode an estimated clean image every N steps, plus the final image.
    /// Disabled by default. Full-resolution previews add decoding time and memory.
    pub preview_every: Option<NonZeroUsize>,
}

impl Request {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            images: Vec::new(),
            ratio: None,
            scale: 1.0,
            steps: 40,
            seed: 42,
            preview_every: None,
        }
    }

    /// Validate inputs and determine output size without loading weights or Metal.
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        pipeline::validate(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Loading,
    /// Zero-based reference index; progress counts vision layers.
    ReferenceVision {
        index: usize,
    },
    TextEncoding,
    ReferenceEncoding,
    DenoiserLoading,
    Decoding,
}

/// Owned, Send events suitable for passing to a GUI thread. Counts are stage-local.
#[derive(Clone, Debug)]
pub enum Event {
    Started {
        width: u32,
        height: u32,
        steps: usize,
        seed: u64,
    },
    Progress {
        stage: Stage,
        completed: usize,
        total: usize,
    },
    /// One-based completed step. Duration excludes optional preview decoding.
    StepFinished {
        step: usize,
        total: usize,
        duration: Duration,
    },
    /// Early previews are estimated clean images, not the noisy current latent.
    /// The final preview is identical to `Generation::image`.
    Preview {
        step: usize,
        total: usize,
        image: Arc<RgbaImage>,
    },
    Finished {
        elapsed: Duration,
    },
}

#[derive(Debug)]
pub struct Generation {
    pub image: Arc<RgbaImage>,
    pub elapsed: Duration,
}

/// Cooperative cancellation. Clone for the UI; create a fresh token per request.
/// Checked between model layers/stages. An in-flight GPU operation, VAE decode,
/// or file download must finish before cancellation can be observed.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Cancelled.into())
        } else {
            Ok(())
        }
    }
}

/// Downcast the returned error to this type to distinguish cancellation from failure.
#[derive(Debug)]
pub struct Cancelled;
impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("generation cancelled")
    }
}
impl std::error::Error for Cancelled {}

/// Reusable configuration for sequential requests. Model tensors are released
/// between stages/requests to limit memory; this is not a permanently loaded model.
pub struct Generator {
    weights: weights::Weights,
}

impl Generator {
    pub fn new(options: ModelOptions) -> Self {
        Self {
            weights: weights::Weights::new(options.model_dir, options.offline),
        }
    }

    /// Ensure all required checkpoint files are local, without allocating GPU
    /// resources. Blocks; call on a worker thread. Byte progress is per file and
    /// includes resumed bytes. Cancellation takes effect between files; the hub
    /// client cannot interrupt a download already in progress.
    pub fn prepare(
        &mut self,
        cancellation: &CancellationToken,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<()> {
        self.weights.prepare(cancellation, &mut on_progress)
    }

    /// Blocking inference, with no terminal output or image-file writes.
    /// Events are ordered; `Finished` occurs only on success. Errors and cancellation
    /// are returned to the caller. Keep callbacks quick to avoid delaying inference.
    pub fn generate(
        &mut self,
        request: &Request,
        cancellation: &CancellationToken,
        mut on_event: impl FnMut(Event),
    ) -> Result<Generation> {
        let mut observer = Observer {
            cancellation,
            callback: &mut on_event,
        };
        cancellation.check()?;
        let start = Instant::now();
        let image = pipeline::run(request, &self.weights, &mut observer)?;
        let elapsed = start.elapsed();
        observer.emit(Event::Finished { elapsed })?;
        Ok(Generation { image, elapsed })
    }
}

pub(crate) struct Observer<'a> {
    cancellation: &'a CancellationToken,
    callback: &'a mut dyn FnMut(Event),
}

impl Observer<'_> {
    fn check(&self) -> Result<()> {
        self.cancellation.check()
    }
    fn emit(&mut self, event: Event) -> Result<()> {
        self.check()?;
        (self.callback)(event);
        // Finished is a committed success, even if its callback cancels the token.
        Ok(())
    }
    fn progress(&mut self, stage: Stage, completed: usize, total: usize) -> Result<()> {
        self.emit(Event::Progress {
            stage,
            completed,
            total,
        })?;
        self.check()
    }
}
