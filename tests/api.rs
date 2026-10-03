use image_forger::{
    CancellationToken, Cancelled, Event, Generator, ModelOptions, Request, RgbaImage, Stage,
};
use std::sync::Arc;

fn generator() -> Generator {
    Generator::new(ModelOptions {
        offline: true,
        ..Default::default()
    })
}

#[test]
fn invalid_requests_fail_before_events_or_gpu_access() {
    let mut requests = vec![Request::new("  ")];
    let mut request = Request::new("test");
    request.images.push(RgbaImage::new(0, 32));
    requests.push(request);
    let mut request = Request::new("test");
    request.steps = 0;
    requests.push(request);
    let mut request = Request::new("test");
    request.scale = f64::NAN;
    requests.push(request);
    let mut request = Request::new("test");
    request.ratio = Some("unsupported".into());
    requests.push(request);
    for request in requests {
        assert!(
            generator()
                .generate(&request, &CancellationToken::default(), |_| {
                    panic!("invalid request emitted an event")
                })
                .is_err()
        );
    }
}

#[test]
fn cancellation_before_start_and_from_callback_is_typed() {
    let cancellation = CancellationToken::default();
    cancellation.clone().cancel();
    let error = generator()
        .generate(&Request::new("test"), &cancellation, |_| {
            panic!("pre-cancelled request emitted an event")
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());

    let cancellation = CancellationToken::default();
    let mut events = 0;
    let error = generator()
        .generate(&Request::new("test"), &cancellation, |event| {
            assert!(matches!(event, Event::Started { .. }));
            events += 1;
            cancellation.cancel();
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());
    assert_eq!(events, 1);
}

#[test]
fn ui_messages_and_inputs_can_cross_threads() {
    fn send<T: Send>() {}
    send::<Request>();
    send::<Event>();
    send::<image_forger::Generation>();
    send::<CancellationToken>();
    send::<Generator>();
    send::<image_forger::PreviewControl>();
    send::<image_forger::PauseControl>();
}

#[test]
#[ignore = "requires cached Qwen Image 2.1 checkpoint and Metal GPU"]
fn manual_background_previews_preserve_pixels_and_do_not_leak_between_runs()
-> image_forger::Result<()> {
    let mut request = Request::new("A red ceramic teapot on a wooden table.");
    request.scale = 0.0625;
    request.steps = 8;
    request.preview_every = std::num::NonZeroUsize::new(1);
    let mut generator = generator();
    let cancellation = CancellationToken::default();
    let mut first_preview = None;
    let baseline = generator.generate(&request, &cancellation, |event| {
        if let Event::Preview { step: 1, image, .. } = event {
            first_preview = Some(image);
        }
    })?;

    request.preview_every = None;
    let control = image_forger::PreviewControl::default();
    request.preview_control = Some(control.clone());
    let mut previews = Vec::new();
    let manual = generator.generate(&request, &cancellation, |event| match event {
        Event::StepFinished { step: 1, .. } => {
            assert!(control.request_preview());
            assert!(!control.request_preview());
        }
        Event::Preview { step, image, .. } => {
            previews.push(step);
            if step == 1 {
                assert_eq!(image.as_raw(), first_preview.as_ref().unwrap().as_raw());
            }
        }
        _ => {}
    })?;
    assert_eq!(previews, [1, request.steps]);
    assert_eq!(manual.image.as_raw(), baseline.image.as_raw());
    assert!(!control.request_preview());

    // Cancel with a preview in flight, then rerun without clicking Preview.
    let control = image_forger::PreviewControl::default();
    request.preview_control = Some(control.clone());
    let interrupted = CancellationToken::default();
    let error = generator
        .generate(&request, &interrupted, |event| {
            if let Event::StepFinished { step: 1, .. } = event {
                assert!(control.request_preview());
                interrupted.cancel();
            }
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());
    assert!(!control.request_preview());
    request.preview_control = Some(image_forger::PreviewControl::default());
    let mut previews = Vec::new();
    let quiet = generator.generate(&request, &cancellation, |event| {
        if let Event::Preview { step, .. } = event {
            previews.push(step);
        }
    })?;
    assert_eq!(previews, [request.steps]);
    assert_eq!(quiet.image.as_raw(), baseline.image.as_raw());
    Ok(())
}

#[test]
#[ignore = "requires cached Qwen Image 2.1 checkpoint and Metal GPU"]
fn blocking_previews_and_pause_hold_sampling() -> image_forger::Result<()> {
    use std::time::{Duration, Instant};
    let mut request = Request::new("A red ceramic teapot on a wooden table.");
    request.scale = 0.0625;
    request.steps = 6;
    let mut generator = generator();
    let cancellation = CancellationToken::default();
    let baseline = generator.generate(&request, &cancellation, |_| {})?;

    let control = image_forger::PreviewControl::default();
    let pause = image_forger::PauseControl::default();
    request.preview_control = Some(control.clone());
    request.pause = Some(pause.clone());
    let mut order = Vec::new();
    let mut resumed_at = None;
    let mut worker = None;
    let result = generator.generate(&request, &cancellation, |event| match event {
        Event::StepFinished { step, .. } => {
            order.push(format!("step {step}"));
            if step == 2 {
                assert!(control.request_blocking_preview());
                assert!(!control.request_blocking_preview());
            }
            if step == 4 {
                pause.pause();
                resumed_at = Some(Instant::now());
                let pause = pause.clone();
                worker = Some(std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(300));
                    pause.resume();
                }));
            }
        }
        Event::Preview { step, .. } => order.push(format!("preview {step}")),
        _ => {}
    })?;
    worker.unwrap().join().unwrap();
    assert!(resumed_at.unwrap().elapsed() >= Duration::from_millis(300));
    // The step-2 preview arrives before sampling continues to step 3.
    assert_eq!(
        order,
        [
            "step 1",
            "step 2",
            "preview 2",
            "step 3",
            "step 4",
            "step 5",
            "step 6",
            "preview 6"
        ]
    );
    assert_eq!(result.image.as_raw(), baseline.image.as_raw());
    Ok(())
}

#[test]
#[ignore = "requires cached Qwen Image 2.1 checkpoint and Metal GPU"]
fn previews_preserve_output_and_report_ordered_progress() -> image_forger::Result<()> {
    let mut request = Request::new("A red ceramic teapot on a wooden table.");
    request.scale = 0.0625;
    request.steps = 3;
    let cancellation = CancellationToken::default();
    let mut generator = generator();
    generator.prepare(&cancellation, |_| {
        panic!("cached model triggered a download")
    })?;
    request.preview_every = std::num::NonZeroUsize::new(1);
    let mut events = Vec::new();
    let generated = generator.generate(&request, &cancellation, |e| events.push(e))?;
    // The rerun reuses the cached encoder output and must not change pixels.
    request.preview_every = None;
    let baseline = generator.generate(&request, &cancellation, |event| {
        assert!(!matches!(
            event,
            Event::Preview { .. }
                | Event::Progress {
                    stage: Stage::TextEncoding,
                    ..
                }
        ));
    })?;
    assert_eq!(baseline.image.as_raw(), generated.image.as_raw());
    assert_eq!(generated.image.dimensions(), (128, 128));
    assert!(matches!(
        events.first(),
        Some(Event::Started { steps: 3, .. })
    ));
    assert!(matches!(events.last(), Some(Event::Finished { .. })));
    let mut steps = Vec::new();
    let mut previews = Vec::new();
    let mut text_layers = Vec::new();
    let mut prefix_layers = Vec::new();
    for event in &events {
        match event {
            Event::StepFinished { step, total, .. } => {
                assert_eq!(*total, 3);
                steps.push(*step);
            }
            Event::Preview { step, total, image } => {
                assert_eq!(steps.last(), Some(step));
                assert_eq!(*total, 3);
                assert_eq!(image.dimensions(), (128, 128));
                previews.push(*step);
                if *step == 3 {
                    assert!(Arc::ptr_eq(image, &generated.image));
                }
            }
            Event::Progress {
                stage: Stage::TextEncoding,
                completed,
                ..
            } => text_layers.push(*completed),
            Event::Progress {
                stage: Stage::DenoiserLoading,
                completed,
                ..
            } => prefix_layers.push(*completed),
            _ => {}
        }
    }
    assert_eq!(steps, [1, 2, 3]);
    assert_eq!(previews, [1, 2, 3]);
    assert_eq!(text_layers, (0..=36).collect::<Vec<_>>());
    assert_eq!(prefix_layers, (0..=32).collect::<Vec<_>>());
    // Interrupted prefix preparation must not leak attention state into the
    // next request. Immutable cached weights remain usable.
    let interrupted = CancellationToken::default();
    let error = generator
        .generate(&request, &interrupted, |event| {
            if matches!(
                event,
                Event::Progress {
                    stage: Stage::DenoiserLoading,
                    completed: 2,
                    ..
                }
            ) {
                interrupted.cancel();
            }
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());
    let recovered = generator.generate(&request, &cancellation, |_| {})?;
    assert_eq!(recovered.image.as_raw(), baseline.image.as_raw());
    // Explicit unloading releases model resources but preserves CPU encoding.
    generator.unload_models();
    let reloaded = generator.generate(&request, &cancellation, |event| {
        assert!(!matches!(
            event,
            Event::Progress {
                stage: Stage::TextEncoding,
                ..
            }
        ));
    })?;
    assert_eq!(reloaded.image.as_raw(), baseline.image.as_raw());
    // Cancellation during encoder loading must stop before denoising/Finished.
    // A fresh generator has no cached encoder output, so the encoder runs.
    let cancellation = CancellationToken::default();
    let error = self::generator()
        .generate(&request, &cancellation, |event| {
            assert!(!matches!(
                event,
                Event::Finished { .. } | Event::StepFinished { .. }
            ));
            if matches!(
                event,
                Event::Progress {
                    stage: Stage::TextEncoding,
                    completed: 1,
                    ..
                }
            ) {
                cancellation.cancel();
            }
        })
        .unwrap_err();
    assert!(error.is::<Cancelled>());
    Ok(())
}

#[test]
#[ignore = "requires cached Qwen Image 2.1 checkpoint and Metal GPU"]
fn cached_encoder_results_match_a_fresh_generator() -> image_forger::Result<()> {
    let cancellation = CancellationToken::default();
    let mut request = Request::new("A red ceramic teapot on a wooden table.");
    request.scale = 0.0625;
    request.steps = 2;
    let mut generator = generator();
    let reference = generator.generate(&request, &cancellation, |_| {})?.image;
    request.prompt = "Make the teapot blue.".into();
    request.images.push((*reference).clone());
    generator.generate(&request, &cancellation, |_| {})?;
    // A new prompt reuses the reference's vision features and latents; a new
    // seed also reuses the encoder output.
    let mut stages = Vec::new();
    for (prompt, seed) in [
        ("Make the teapot green.", 42),
        ("Make the teapot green.", 7),
    ] {
        request.prompt = prompt.into();
        request.seed = seed;
        let mut events = Vec::new();
        let cached = generator.generate(&request, &cancellation, |e| events.push(e))?;
        let fresh = self::generator().generate(&request, &cancellation, |_| {})?;
        assert_eq!(cached.image.as_raw(), fresh.image.as_raw());
        stages.push(
            events
                .into_iter()
                .filter_map(|e| match e {
                    Event::Progress { stage, .. } if stage != Stage::Loading => Some(stage),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        );
    }
    assert!(stages[0].contains(&Stage::TextEncoding));
    assert!(!stages[0].contains(&Stage::ReferenceEncoding));
    assert!(!stages[0].contains(&Stage::ReferenceVision { index: 0 }));
    assert!(
        stages[1]
            .iter()
            .all(|s| matches!(s, Stage::DenoiserLoading | Stage::Decoding))
    );
    // Removing references and changing the target size must rebuild all rotary
    // and prefix state, even though the same model tensors stay resident.
    request.images.clear();
    request.scale = 0.03125;
    let cached = generator.generate(&request, &cancellation, |_| {})?;
    generator.unload_models();
    let fresh = self::generator().generate(&request, &cancellation, |_| {})?;
    assert_eq!(cached.image.dimensions(), (64, 64));
    assert_eq!(cached.image.as_raw(), fresh.image.as_raw());
    Ok(())
}

#[test]
#[ignore = "requires cached Qwen Image 2.1 checkpoint and Metal GPU"]
fn shared_model_sessions_run_concurrently_and_cancel_independently() -> image_forger::Result<()> {
    use image_forger::{PreviewControl, SharedModel};
    use std::{sync::mpsc, thread, time::Duration};

    let mut first = Request::new("A red ceramic teapot on a wooden table.");
    first.scale = 0.0625;
    first.steps = 6;
    let mut second = Request::new("Turn the square in the reference image blue.");
    second.scale = 0.0625;
    second.steps = 6;
    second.seed = 12345;
    second.images.push(RgbaImage::from_pixel(
        32,
        32,
        image::Rgba([220, 40, 30, 255]),
    ));
    let requests = [first, second];
    let mut baseline = generator();
    let expected: Vec<_> = requests
        .iter()
        .map(|request| {
            baseline
                .generate(request, &CancellationToken::default(), |_| {})
                .map(|g| g.image)
        })
        .collect::<image_forger::Result<_>>()?;
    drop(baseline);

    let model = SharedModel::new(ModelOptions {
        offline: true,
        ..Default::default()
    });
    let mut sessions = vec![model.generator(), model.generator()];
    for cancel_first in [false, true] {
        let (ready, receiver) = mpsc::channel();
        let mut release = Vec::new();
        let mut workers = Vec::new();
        for (index, mut session) in sessions.drain(..).enumerate() {
            let mut request = requests[index].clone();
            let control = PreviewControl::default();
            request.preview_control = Some(control.clone());
            let ready = ready.clone();
            let (sender, start) = mpsc::channel();
            release.push(sender);
            workers.push(thread::spawn(move || {
                let cancel = CancellationToken::default();
                let mut previews = Vec::new();
                let mut requested = false;
                let result = session.generate(&request, &cancel, |event| match event {
                    Event::StepFinished { step: 1, .. } => {
                        requested = control.request_preview();
                        let _ = ready.send((index, true));
                        // Neither session can pass step 1 until BOTH reach it.
                        // A whole-generation lock would fail this handshake.
                        if start.recv_timeout(Duration::from_secs(180)).is_err() {
                            cancel.cancel();
                        }
                    }
                    Event::StepFinished { step: 2, .. } if cancel_first && index == 0 => {
                        cancel.cancel()
                    }
                    Event::Preview { step, .. } => previews.push(step),
                    _ => {}
                });
                let _ = ready.send((index, false));
                (session, result, requested, previews, control)
            }));
        }
        drop(ready);
        let reached = (|| -> image_forger::Result<()> {
            let a = receiver.recv_timeout(Duration::from_secs(180))?;
            let b = receiver.recv_timeout(Duration::from_secs(180))?;
            anyhow::ensure!(
                a.1 && b.1 && a.0 != b.0,
                "both independent sessions must reach step 1: {a:?}, {b:?}"
            );
            Ok(())
        })();
        for sender in release {
            let _ = sender.send(());
        }
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().expect("inference thread panicked"))
            .collect();
        reached?;
        for (index, (session, result, requested, previews, control)) in
            results.into_iter().enumerate()
        {
            assert!(requested);
            assert!(
                !control.request_preview(),
                "finished/cancelled session retained a preview control"
            );
            if cancel_first && index == 0 {
                assert!(result.unwrap_err().is::<Cancelled>());
            } else {
                assert_eq!(result?.image.as_raw(), expected[index].as_raw());
                assert!(
                    previews.contains(&1),
                    "missing independent intermediate preview: {previews:?}"
                );
                assert_eq!(previews.last(), Some(&requests[index].steps));
            }
            sessions.push(session);
        }
    }
    // Unloading one session must not invalidate the other session's shared buffers.
    sessions[0].unload_models();
    let recovered = sessions[1].generate(&requests[1], &CancellationToken::default(), |_| {})?;
    assert_eq!(recovered.image.as_raw(), expected[1].as_raw());
    Ok(())
}

#[test]
#[ignore = "requires the cached MLX 4-bit checkpoint and Metal GPU"]
fn mlx_checkpoint_generates_and_edits_with_shared_weights() -> image_forger::Result<()> {
    use image_forger::{Checkpoint, SharedModel};
    let model = SharedModel::new(ModelOptions {
        offline: true,
        checkpoint: Checkpoint::Mlx4Bit,
        ..Default::default()
    });
    let cancellation = CancellationToken::default();
    let mut request = Request::new("a red teapot on a white table");
    request.scale = 0.125;
    request.steps = 2;
    let first = model
        .generator()
        .generate(&request, &cancellation, |_| {})?;
    let pixels = first.image.as_raw();
    assert_eq!((first.image.width(), first.image.height()), (256, 256));
    assert!(
        pixels.chunks(4).any(|p| p != &pixels[..4]),
        "uniform image from quantized weights"
    );
    // Quantized weights are shared and deterministic across sessions.
    let second = model
        .generator()
        .generate(&request, &cancellation, |_| {})?;
    assert_eq!(first.image.as_raw(), second.image.as_raw());
    // Editing runs the quantized Qwen3-VL vision tower.
    let mut edit = Request::new("make the teapot blue");
    edit.images.push((*first.image).clone());
    edit.scale = 0.125;
    edit.steps = 2;
    let mut vision = 0;
    model.generator().generate(&edit, &cancellation, |event| {
        if let Event::Progress {
            stage: Stage::ReferenceVision { .. },
            ..
        } = event
        {
            vision += 1;
        }
    })?;
    assert!(vision > 0);
    Ok(())
}
