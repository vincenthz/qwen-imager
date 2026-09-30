//! Native Qwen Image 2.1 inference on Apple Metal.
//!
//! Run [`Generator::generate`] on a worker thread. Callbacks execute synchronously
//! on that thread; forward owned [`Event`] values to your UI through a channel.
//! No tensor or GPU handle is exposed to the caller. See `examples/progress.rs`.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("img-gen requires an Apple Silicon Mac with Metal.");

mod dit;
pub mod image_input;
mod noise;
mod ops;
mod pipeline;
mod preview;
mod shared;
mod text;
mod vae;
mod vision;
mod weights;

use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

pub use anyhow::Result;
pub use image::RgbaImage;
pub use preview::PreviewControl;
pub use weights::{MODEL, REVISION};

/// Checkpoint location and download policy. GPU resources are allocated by `generate`.
#[derive(Clone, Debug, Default)]
pub struct ModelOptions {
    /// Original Diffusers snapshot directory, or the Hugging Face cache when absent.
    /// Keep checkpoint files unchanged while the generator exists (they are mapped).
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
    /// Experimental shared noise: generate at this square pixel size, then use
    /// non-overlapping, variance-preserving pooling to the output latent grid.
    /// Requires square output whose side divides this size. None uses native noise.
    pub noise_source_size: Option<u32>,
    /// Decode an estimated clean image every N steps, plus the final image.
    /// Disabled by default. Full-resolution previews add decoding time and memory.
    pub preview_every: Option<NonZeroUsize>,
    /// On-demand background previews. Clone the control for a UI Preview button.
    /// Use a fresh control for each request; cannot combine with `preview_every`.
    pub preview_control: Option<PreviewControl>,
    /// Cooperative pause. Clone the control for a UI Pause button.
    pub pause: Option<PauseControl>,
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
            noise_source_size: None,
            preview_every: None,
            preview_control: None,
            pause: None,
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
    /// Loads weights on first use and prepares request-specific conditioning on every run.
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
    /// Manual previews may arrive after later steps finish; `step` is the snapshot's step.
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
/// Checked between model layers/stages. GPU operations and downloads cannot be
/// preempted. An in-flight background decode may finish after cancellation, but
/// its result is discarded; unloading the generator waits for its worker to exit.
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

/// Cooperative pause, checked with cancellation between model layers and steps.
/// Clone for the UI. A paused generation keeps its thread and GPU memory; it can
/// still be cancelled. A background preview decode already started continues.
#[derive(Clone, Debug, Default)]
pub struct PauseControl(Arc<(Mutex<bool>, Condvar)>);

impl PauseControl {
    pub fn pause(&self) {
        *self.0.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }
    pub fn resume(&self) {
        *self.0.0.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.0.1.notify_all();
    }
    pub fn is_paused(&self) -> bool {
        *self.0.0.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn wait(&self, cancellation: &CancellationToken) -> Result<()> {
        let (paused, resumed) = &*self.0;
        let mut paused = paused.lock().unwrap_or_else(|e| e.into_inner());
        while *paused {
            cancellation.check()?;
            // Cancellation does not notify; re-check it periodically.
            paused = resumed
                .wait_timeout(paused, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
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

/// Reusable generator for sequential requests. The Metal device and loaded
/// denoiser/VAE weights persist between requests. Drop it or call `unload_models`
/// to release them. Attention state is rebuilt for each request.
/// Encoder results for the latest prompt and references are cached in CPU memory,
/// so reusing one generator for a new seed, step count, or size skips re-encoding.
pub struct Generator {
    weights: weights::Weights,
    cache: pipeline::Cache,
}

/// One lazy model-weight pool for concurrent, independent generation sessions.
/// Sessions use separate Metal queues, latents, conditioning and preview workers.
/// Immutable denoiser/VAE buffers are loaded once and shared without copying.
#[derive(Clone)]
pub struct SharedModel {
    weights: weights::Weights,
}

impl SharedModel {
    pub fn new(options: ModelOptions) -> Self {
        Self {
            weights: weights::Weights::shared(options.model_dir, options.offline),
        }
    }

    /// Create a session for one workspace. Keep it for prompt/reference caching;
    /// run separate sessions on separate threads to generate concurrently.
    pub fn generator(&self) -> Generator {
        Generator {
            weights: self.weights.clone(),
            cache: pipeline::Cache::default(),
        }
    }
}

impl Generator {
    pub fn new(options: ModelOptions) -> Self {
        Self {
            weights: weights::Weights::new(options.model_dir, options.offline),
            cache: pipeline::Cache::default(),
        }
    }

    /// Release cached model weights and the Metal device, retaining CPU encoder
    /// results. The next generation reloads weights lazily. Also safe after cancellation.
    /// SharedModel sessions release their own queue/working state; shared weights
    /// stay resident until the model and all its sessions are dropped.
    pub fn unload_models(&mut self) {
        self.cache.unload_models();
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
            pause: request.pause.as_ref(),
            callback: &mut on_event,
            previews: None,
        };
        cancellation.check()?;
        let start = Instant::now();
        let image = pipeline::run(request, &self.weights, &mut self.cache, &mut observer)?;
        let elapsed = start.elapsed();
        observer.emit(Event::Finished { elapsed })?;
        Ok(Generation { image, elapsed })
    }
}

pub(crate) struct Observer<'a> {
    cancellation: &'a CancellationToken,
    pause: Option<&'a PauseControl>,
    callback: &'a mut dyn FnMut(Event),
    previews: Option<preview::PreviewSession>,
}

impl Observer<'_> {
    fn poll_previews(&mut self) -> Result<()> {
        self.check()?;
        if let Some(previews) = &mut self.previews {
            while let Some(event) = previews.poll()? {
                (self.callback)(event);
            }
        }
        self.check()
    }
    /// Emits blocking previews as they arrive, holding sampling until none remain.
    fn wait_previews(&mut self) -> Result<()> {
        while self
            .previews
            .as_ref()
            .is_some_and(preview::PreviewSession::holds_sampling)
        {
            self.check()?;
            let previews = self.previews.as_mut().unwrap();
            if let Some(event) = previews.wait(Duration::from_millis(25))? {
                (self.callback)(event);
            }
        }
        self.check()
    }
    fn check(&self) -> Result<()> {
        self.cancellation.check()?;
        if let Some(pause) = self.pause {
            pause.wait(self.cancellation)?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_blocks_until_resumed_or_cancelled() {
        let pause = PauseControl::default();
        let cancellation = CancellationToken::default();
        assert!(pause.wait(&cancellation).is_ok());
        pause.pause();
        let resumer = pause.clone();
        let started = Instant::now();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            resumer.resume();
        });
        assert!(pause.wait(&cancellation).is_ok());
        assert!(started.elapsed() >= Duration::from_millis(100));
        worker.join().unwrap();
        pause.pause();
        cancellation.cancel();
        assert!(pause.wait(&cancellation).unwrap_err().is::<Cancelled>());
    }
}
