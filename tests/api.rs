use img_gen::{
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
    send::<img_gen::Generation>();
    send::<CancellationToken>();
    send::<Generator>();
}

#[test]
#[ignore = "requires cached Qwen Image 2.1 checkpoint and Metal GPU"]
fn previews_preserve_output_and_report_ordered_progress() -> img_gen::Result<()> {
    let mut request = Request::new("A red ceramic teapot on a wooden table.");
    request.scale = 0.0625;
    request.steps = 3;
    let cancellation = CancellationToken::default();
    let mut generator = generator();
    generator.prepare(&cancellation, |_| {
        panic!("cached model triggered a download")
    })?;
    let baseline = generator.generate(&request, &cancellation, |event| {
        assert!(!matches!(event, Event::Preview { .. }));
    })?;
    request.preview_every = std::num::NonZeroUsize::new(1);
    let mut events = Vec::new();
    let generated = generator.generate(&request, &cancellation, |e| events.push(e))?;
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
    // Cancellation during encoder loading must stop before denoising/Finished.
    let cancellation = CancellationToken::default();
    let error = generator
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
