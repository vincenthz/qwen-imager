//! A dedicated decoder queue: only CPU latent snapshots cross between Metal devices.
use crate::{CancellationToken, Event, RgbaImage, pipeline, weights::Weights};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use std::{
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

/// Thread-safe Preview button. Requests use the latest completed step, or wait
/// for the first step if none has completed. At most one preview is outstanding,
/// except that a blocking request may queue behind a background decode.
/// Create a fresh control for each generation and put a clone in `Request`.
#[derive(Clone, Default)]
pub struct PreviewControl(Arc<Mutex<Option<Active>>>);

impl std::fmt::Debug for PreviewControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreviewControl").finish_non_exhaustive()
    }
}

struct Active {
    jobs: mpsc::SyncSender<Job>,
    replies: mpsc::Sender<Reply>,
    lifetime: CancellationToken,
    latest: Option<Snapshot>,
    requested: bool,
    pending: Option<Snapshot>,
    busy: bool,
    // The requested preview, or the one decoding, pauses sampling until it arrives.
    blocking: bool,
    busy_blocking: bool,
}

#[derive(Clone)]
pub(crate) struct Snapshot {
    pub latents: Tensor,
    pub step: usize,
    pub width: u32,
    pub height: u32,
}

struct Job {
    snapshot: Snapshot,
    replies: mpsc::Sender<Reply>,
    lifetime: CancellationToken,
}

struct Reply {
    step: usize,
    image: Result<Arc<RgbaImage>>,
}

impl Active {
    fn dispatch(&mut self) -> Result<()> {
        if self.requested && self.pending.is_none() {
            self.pending = self.latest.clone();
        }
        if self.requested
            && !self.busy
            && let Some(snapshot) = &self.pending
        {
            match self.jobs.try_send(Job {
                snapshot: snapshot.clone(),
                replies: self.replies.clone(),
                lifetime: self.lifetime.clone(),
            }) {
                Ok(()) => {
                    self.requested = false;
                    self.pending = None;
                    self.busy = true;
                    self.busy_blocking = std::mem::take(&mut self.blocking);
                }
                // A cancelled run's job can briefly occupy the queue. Retry on
                // the next snapshot; never block the UI or accumulate snapshots.
                Err(mpsc::TrySendError::Full(_)) => {}
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    anyhow::bail!("preview decoder stopped")
                }
            }
        }
        Ok(())
    }

    fn holds_sampling(&self) -> bool {
        (self.requested && self.blocking) || (self.busy && self.busy_blocking)
    }
}

impl PreviewControl {
    /// Returns false when inactive or a preview is already queued/decoding.
    /// Does no GPU work and never waits for a decode.
    pub fn request_preview(&self) -> bool {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(active) = state.as_mut() else {
            return false;
        };
        if active.requested || active.busy {
            return false;
        }
        active.requested = true;
        // The inference thread also checks dispatch errors when publishing.
        let _ = active.dispatch();
        true
    }

    /// Like `request_preview`, but sampling pauses after its current step until
    /// this preview is decoded. Queues behind a background decode already in
    /// progress (sampling then waits for both). Returns false when inactive or a
    /// blocking preview is already outstanding.
    pub fn request_blocking_preview(&self) -> bool {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(active) = state.as_mut() else {
            return false;
        };
        if active.holds_sampling() {
            return false;
        }
        active.requested = true;
        active.blocking = true;
        let _ = active.dispatch();
        true
    }
}

pub(crate) struct PreviewSession {
    control: PreviewControl,
    replies: mpsc::Receiver<Reply>,
    total: usize,
}

impl PreviewSession {
    pub fn publish(&self, snapshot: Snapshot) -> Result<()> {
        let mut state = self.control.0.lock().unwrap_or_else(|e| e.into_inner());
        let active = state.as_mut().context("preview session closed")?;
        active.latest = Some(snapshot);
        active.dispatch()
    }

    pub fn poll(&mut self) -> Result<Option<Event>> {
        match self.replies.try_recv() {
            Ok(reply) => self.received(reply),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => anyhow::bail!("preview decoder disconnected"),
        }
    }

    /// True while a blocking preview is requested or decoding.
    pub fn holds_sampling(&self) -> bool {
        let mut state = self.control.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(active) = state.as_mut() else {
            return false;
        };
        // Retry a dispatch that found the decoder queue briefly occupied.
        let _ = active.dispatch();
        active.holds_sampling()
    }

    /// Waits briefly for a decoded preview.
    pub fn wait(&mut self, timeout: Duration) -> Result<Option<Event>> {
        match self.replies.recv_timeout(timeout) {
            Ok(reply) => self.received(reply),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("preview decoder disconnected")
            }
        }
    }

    fn received(&mut self, reply: Reply) -> Result<Option<Event>> {
        if let Some(active) = self
            .control
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            active.busy = false;
            active.busy_blocking = false;
            // A blocking request may be queued behind the decode that just finished.
            active.dispatch()?;
        }
        Ok(Some(Event::Preview {
            step: reply.step,
            total: self.total,
            image: reply.image?,
        }))
    }
}

impl Drop for PreviewSession {
    fn drop(&mut self) {
        if let Some(active) = self
            .control
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            active.lifetime.cancel();
        }
    }
}

/// Persists across requests, retaining its own Metal queue and cached VAE tensors.
pub(crate) struct DecoderWorker {
    jobs: Option<mpsc::SyncSender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl DecoderWorker {
    pub fn new(weights: Weights) -> Result<Self> {
        let mut device = None;
        let mut vae = None;
        Self::spawn(move |snapshot| {
            if device.is_none() {
                device = Some(Device::new_metal(0)?);
            }
            let device = device.as_ref().unwrap();
            let vae = pipeline::load_vae(&mut vae, &weights, device)?;
            pipeline::decode(
                &snapshot.latents.to_device(device)?,
                vae,
                snapshot.width,
                snapshot.height,
            )
        })
    }

    fn spawn(
        mut decode: impl FnMut(&Snapshot) -> Result<Arc<RgbaImage>> + Send + 'static,
    ) -> Result<Self> {
        let (jobs, receiver) = mpsc::sync_channel::<Job>(1);
        let thread = thread::Builder::new()
            .name("qwen-preview".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    if job.lifetime.is_cancelled() {
                        continue;
                    }
                    let image = decode(&job.snapshot);
                    if !job.lifetime.is_cancelled() {
                        let _ = job.replies.send(Reply {
                            step: job.snapshot.step,
                            image,
                        });
                    }
                }
            })?;
        Ok(Self {
            jobs: Some(jobs),
            thread: Some(thread),
        })
    }

    pub fn session(&self, control: PreviewControl, total: usize) -> Result<PreviewSession> {
        let (replies, receiver) = mpsc::channel();
        {
            let mut state = control.0.lock().unwrap_or_else(|e| e.into_inner());
            ensure!(
                state.is_none(),
                "preview control is already in use by another generation"
            );
            *state = Some(Active {
                jobs: self.jobs.as_ref().unwrap().clone(),
                replies,
                lifetime: CancellationToken::default(),
                latest: None,
                requested: false,
                pending: None,
                busy: false,
                blocking: false,
                busy_blocking: false,
            });
        }
        Ok(PreviewSession {
            control,
            replies: receiver,
            total,
        })
    }

    /// Automatic previews and final decoding share this worker's cached VAE.
    pub fn decode(&self, snapshot: Snapshot, cancel: &CancellationToken) -> Result<Arc<RgbaImage>> {
        let (replies, receiver) = mpsc::channel();
        let mut job = Job {
            snapshot,
            replies,
            lifetime: cancel.clone(),
        };
        loop {
            cancel.check()?;
            match self.jobs.as_ref().unwrap().try_send(job) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(returned)) => {
                    job = returned;
                    thread::sleep(Duration::from_millis(10));
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    anyhow::bail!("preview decoder stopped")
                }
            }
        }
        loop {
            cancel.check()?;
            match receiver.recv_timeout(Duration::from_millis(25)) {
                Ok(reply) => return reply.image,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("preview decoder stopped")
                }
            }
        }
    }
}

impl Drop for DecoderWorker {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn snapshot(step: usize) -> Snapshot {
        Snapshot {
            latents: Tensor::new(&[step as f32], &Device::Cpu).unwrap(),
            step,
            width: 1,
            height: 1,
        }
    }

    fn rendered(snapshot: &Snapshot) -> Result<Arc<RgbaImage>> {
        Ok(Arc::new(RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([snapshot.latents.to_vec1::<f32>()?[0] as u8, 0, 0, 255]),
        )))
    }

    fn next(session: &mut PreviewSession) -> Result<Event> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(event) = session.poll()? {
                return Ok(event);
            }
            ensure!(Instant::now() < deadline, "preview timed out");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn preview_uses_clicked_snapshot_while_sampling_advances() -> Result<()> {
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let worker = DecoderWorker::spawn(move |snapshot| {
            started.send(snapshot.step)?;
            release_rx.recv_timeout(Duration::from_secs(5))?;
            rendered(snapshot)
        })?;
        let control = PreviewControl::default();
        assert!(!control.request_preview());
        let mut session = worker.session(control.clone(), 10)?;
        assert!(worker.session(control.clone(), 10).is_err());
        session.publish(snapshot(1))?;
        session.publish(snapshot(2))?;
        assert!(started_rx.try_recv().is_err()); // No automatic decoding.
        assert!(control.request_preview());
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(5))?, 2);
        // The decoder is deliberately blocked. Publishing later completed steps
        // still succeeds, and repeated clicks cannot queue more work.
        session.publish(snapshot(3))?;
        assert!(!control.request_preview());
        release.send(())?;
        match next(&mut session)? {
            Event::Preview { step, image, .. } => {
                assert_eq!(step, 2);
                assert_eq!(image.get_pixel(0, 0)[0], 2);
            }
            _ => panic!("expected preview"),
        }
        assert!(control.request_preview());
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(5))?, 3);
        release.send(())?;
        assert!(matches!(
            next(&mut session)?,
            Event::Preview { step: 3, .. }
        ));
        drop(session);
        assert!(!control.request_preview());
        Ok(())
    }

    #[test]
    fn early_requests_wait_and_retired_sessions_discard_late_replies() -> Result<()> {
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let worker = DecoderWorker::spawn(move |snapshot| {
            if snapshot.step == 1 {
                started.send(())?;
                release_rx.recv_timeout(Duration::from_secs(5))?;
            }
            rendered(snapshot)
        })?;
        let old = PreviewControl::default();
        let session = worker.session(old.clone(), 10)?;
        assert!(old.request_preview()); // Click before the first completed step.
        assert!(!old.request_preview());
        assert!(started_rx.try_recv().is_err());
        session.publish(snapshot(1))?;
        started_rx.recv_timeout(Duration::from_secs(5))?;
        drop(session); // Cancel or finish while the decode is in flight.
        assert!(!old.request_preview());
        let new = PreviewControl::default();
        let mut session = worker.session(new.clone(), 10)?;
        session.publish(snapshot(7))?;
        assert!(new.request_preview());
        release.send(())?;
        assert!(matches!(
            next(&mut session)?,
            Event::Preview { step: 7, .. }
        ));
        assert!(session.poll()?.is_none());
        drop(session);
        // Final decoding shares the worker and cannot be replaced by stale replies.
        assert_eq!(
            worker
                .decode(snapshot(10), &CancellationToken::default())?
                .get_pixel(0, 0)[0],
            10
        );
        Ok(())
    }

    #[test]
    fn blocking_previews_hold_sampling_and_queue_behind_background_decodes() -> Result<()> {
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let worker = DecoderWorker::spawn(move |snapshot| {
            started.send(snapshot.step)?;
            release_rx.recv_timeout(Duration::from_secs(5))?;
            rendered(snapshot)
        })?;
        let control = PreviewControl::default();
        assert!(!control.request_blocking_preview());
        let mut session = worker.session(control.clone(), 10)?;
        session.publish(snapshot(1))?;
        assert!(control.request_preview()); // Background decode of step 1.
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(5))?, 1);
        assert!(!session.holds_sampling());
        session.publish(snapshot(2))?;
        // A checkpoint queues behind the background decode and holds sampling.
        assert!(control.request_blocking_preview());
        assert!(!control.request_blocking_preview());
        assert!(!control.request_preview());
        assert!(session.holds_sampling());
        release.send(())?;
        assert!(matches!(
            next(&mut session)?,
            Event::Preview { step: 1, .. }
        ));
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(5))?, 2);
        assert!(session.holds_sampling());
        release.send(())?;
        assert!(matches!(
            session.wait(Duration::from_secs(5))?,
            Some(Event::Preview { step: 2, .. })
        ));
        assert!(!session.holds_sampling());
        Ok(())
    }

    #[test]
    fn decoder_errors_are_reported() -> Result<()> {
        let worker = DecoderWorker::spawn(|_| anyhow::bail!("decode failed"))?;
        let control = PreviewControl::default();
        let mut session = worker.session(control.clone(), 2)?;
        session.publish(snapshot(1))?;
        assert!(control.request_preview());
        assert!(
            next(&mut session)
                .unwrap_err()
                .to_string()
                .contains("decode failed")
        );
        drop(session);
        assert!(
            worker
                .decode(snapshot(2), &CancellationToken::default())
                .is_err()
        );
        Ok(())
    }
}
