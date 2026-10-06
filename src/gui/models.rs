//! App-wide, CPU-only cache checks and explicitly requested model downloads.
use std::thread;

use gpui::{Context, Task};
use image_forger::{
    CancellationToken, Cancelled, Checkpoint, DownloadProgress, Generator, ModelOptions,
};

use crate::settings::{Backend, model_label};

#[derive(Clone, Debug)]
pub enum ModelState {
    Checking,
    Missing,
    Downloading {
        progress: Option<DownloadProgress>,
        cancelling: bool,
    },
    Ready,
    Cancelled,
    Failed(String),
}

impl ModelState {
    pub fn can_download(&self) -> bool {
        matches!(self, Self::Missing | Self::Cancelled | Self::Failed(_))
    }

    pub fn is_downloading(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }

    pub fn is_cancelling(&self) -> bool {
        matches!(
            self,
            Self::Downloading {
                cancelling: true,
                ..
            }
        )
    }

    /// The hub reports one file at a time, not a model-wide byte total.
    pub fn fraction(&self) -> Option<f32> {
        match self {
            Self::Downloading {
                progress: Some(progress),
                ..
            } => progress
                .total
                .filter(|&total| total > 0)
                .map(|total| (progress.downloaded as f64 / total as f64).clamp(0., 1.) as f32),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Checking => "Checking local cache…".into(),
            Self::Missing => "Not downloaded on this Mac".into(),
            Self::Ready => "Downloaded on this Mac".into(),
            Self::Cancelled => "Download cancelled; cached files kept".into(),
            Self::Failed(error) => format!("Download failed: {error}"),
            Self::Downloading {
                cancelling: true, ..
            } => "Cancelling after the current file finishes…".into(),
            Self::Downloading { progress: None, .. } => "Preparing download…".into(),
            Self::Downloading {
                progress: Some(progress),
                ..
            } => match self.fraction() {
                Some(fraction) => format!(
                    "{} · {:.1} / {:.1} MB · {:.0}% of file",
                    progress.file,
                    progress.downloaded as f64 / 1_000_000.,
                    progress.total.unwrap() as f64 / 1_000_000.,
                    fraction * 100.
                ),
                None => format!("Connecting to download {}…", progress.file),
            },
        }
    }

    pub fn color(&self) -> u32 {
        match self {
            Self::Ready => 0x75cfb8,
            Self::Downloading { .. } => 0x8aa6ff,
            Self::Failed(_) => 0xf1ae75,
            _ => 0x9da6b5,
        }
    }
}

pub struct ModelEntry {
    pub checkpoint: Checkpoint,
    pub state: ModelState,
    cancellation: CancellationToken,
    receiver: Option<Task<()>>,
}

pub struct Models {
    pub entries: Vec<ModelEntry>,
}

enum Message {
    Progress(DownloadProgress),
    Complete(ModelState),
}

fn prepare(
    checkpoint: Checkpoint,
    offline: bool,
    cancellation: &CancellationToken,
    progress: impl FnMut(DownloadProgress),
) -> image_forger::Result<()> {
    Generator::new(ModelOptions {
        offline,
        checkpoint,
        ..Default::default()
    })
    .prepare(cancellation, progress)
}

/// Remote generation must never depend on the frontend's local cache.
fn generation_blocker(
    backend: &Backend,
    checkpoint: Checkpoint,
    state: &ModelState,
) -> Option<String> {
    if matches!(backend, Backend::Remote { .. }) || matches!(state, ModelState::Ready) {
        return None;
    }
    Some(format!(
        "{}: {}. Download local models in Settings → Models, or select a remote compute host in the title bar.",
        model_label(checkpoint),
        state.label()
    ))
}

impl Models {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let mut models = Self {
            entries: Checkpoint::ALL
                .into_iter()
                .map(|checkpoint| ModelEntry {
                    checkpoint,
                    state: ModelState::Checking,
                    cancellation: CancellationToken::default(),
                    receiver: None,
                })
                .collect(),
        };
        // These never access the network, allocate GPU resources, or block the UI.
        for index in 0..models.entries.len() {
            models.launch(index, false, cx);
        }
        models
    }

    pub fn generation_blocker(&self, backend: &Backend, checkpoint: Checkpoint) -> Option<String> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.checkpoint == checkpoint)
            .unwrap();
        generation_blocker(backend, checkpoint, &entry.state)
    }

    pub fn download(&mut self, index: usize, cx: &mut Context<Self>) {
        if self
            .entries
            .get(index)
            .is_some_and(|entry| entry.state.can_download())
        {
            self.launch(index, true, cx);
        }
    }

    pub fn cancel(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(entry) = self.entries.get_mut(index) {
            if let ModelState::Downloading { cancelling, .. } = &mut entry.state {
                *cancelling = true;
                entry.cancellation.cancel();
                cx.notify();
            }
        }
    }

    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        for index in 0..self.entries.len() {
            if !matches!(
                self.entries[index].state,
                ModelState::Checking | ModelState::Downloading { .. }
            ) {
                self.launch(index, false, cx);
            }
        }
    }

    fn launch(&mut self, index: usize, download: bool, cx: &mut Context<Self>) {
        let entry = &mut self.entries[index];
        entry.state = if download {
            ModelState::Downloading {
                progress: None,
                cancelling: false,
            }
        } else {
            ModelState::Checking
        };
        entry.cancellation = CancellationToken::default();
        let cancellation = entry.cancellation.clone();
        let checkpoint = entry.checkpoint;
        let (sender, receiver) = async_channel::bounded(2);
        entry.receiver = Some(cx.spawn(async move |models, cx| {
            while let Ok(message) = receiver.recv().await {
                if models
                    .update(cx, |models, cx| {
                        let entry = &mut models.entries[index];
                        match message {
                            Message::Progress(value) => {
                                if let ModelState::Downloading { progress, .. } = &mut entry.state {
                                    *progress = Some(value);
                                }
                            }
                            Message::Complete(state) => {
                                if let ModelState::Failed(error) = &state {
                                    crate::errors::report(
                                        format!("Downloading {}", model_label(entry.checkpoint)),
                                        &anyhow::anyhow!(error.clone()),
                                        cx,
                                    );
                                }
                                entry.state = state;
                            }
                        }
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
            }
        }));
        thread::spawn(move || {
            let result = prepare(checkpoint, !download, &cancellation, |progress| {
                if sender.send_blocking(Message::Progress(progress)).is_err() {
                    cancellation.cancel();
                }
            });
            let state = match result {
                Ok(()) => ModelState::Ready,
                Err(_) if !download => ModelState::Missing,
                Err(error) if error.is::<Cancelled>() => {
                    // Cancellation may arrive during the final file's transfer.
                    if prepare(checkpoint, true, &CancellationToken::default(), |_| {}).is_ok() {
                        ModelState::Ready
                    } else {
                        ModelState::Cancelled
                    }
                }
                Err(error) => ModelState::Failed(format!("{error:#}")),
            };
            let _ = sender.send_blocking(Message::Complete(state));
        });
        cx.notify();
    }
}

impl Drop for Models {
    fn drop(&mut self) {
        for entry in &self.entries {
            entry.cancellation.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_generation_never_requires_a_local_model() {
        let remote = Backend::Remote {
            address: "render.local:6996".into(),
        };
        for state in [
            ModelState::Checking,
            ModelState::Missing,
            ModelState::Ready,
            ModelState::Downloading {
                progress: None,
                cancelling: false,
            },
            ModelState::Cancelled,
            ModelState::Failed("offline".into()),
        ] {
            assert!(generation_blocker(&remote, Checkpoint::Original, &state).is_none());
            assert_eq!(
                generation_blocker(&Backend::Local, Checkpoint::Original, &state).is_none(),
                matches!(state, ModelState::Ready)
            );
        }
    }

    #[test]
    fn progress_is_per_file_bounded_and_handles_unknown_totals() {
        let state = |downloaded, total| ModelState::Downloading {
            progress: Some(DownloadProgress {
                file: "weights".into(),
                downloaded,
                total,
            }),
            cancelling: false,
        };
        assert_eq!(state(25, Some(100)).fraction(), Some(0.25));
        assert!(state(25, Some(100)).label().contains("25% of file"));
        assert_eq!(state(150, Some(100)).fraction(), Some(1.));
        assert_eq!(state(25, None).fraction(), None);
        assert_eq!(state(25, Some(0)).fraction(), None);
    }

    #[test]
    fn only_missing_cancelled_or_failed_downloads_can_start() {
        for state in [
            ModelState::Missing,
            ModelState::Cancelled,
            ModelState::Failed("error".into()),
        ] {
            assert!(state.can_download());
        }
        for state in [
            ModelState::Checking,
            ModelState::Ready,
            ModelState::Downloading {
                progress: None,
                cancelling: false,
            },
            ModelState::Downloading {
                progress: None,
                cancelling: true,
            },
        ] {
            assert!(!state.can_download());
        }
    }
}
