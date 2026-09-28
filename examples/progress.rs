//! Worker/channel boundary for a future GUI. Run with:
//! cargo run --release --no-default-features --example progress -- "a red teapot"
use qwen_imager::{CancellationToken, Event, Generator, ModelOptions, Request};
use std::{num::NonZeroUsize, sync::mpsc, thread};

fn main() -> qwen_imager::Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a red teapot".into());
    let mut request = Request::new(prompt);
    request.scale = 0.25;
    request.steps = 20;
    request.preview_every = NonZeroUsize::new(5);

    let cancellation = CancellationToken::default();
    let worker_cancellation = cancellation.clone();
    // Bound the queue so slow rendering cannot accumulate full-resolution images.
    let (sender, receiver) = mpsc::sync_channel(2);
    let worker = thread::spawn(move || {
        let mut generator = Generator::new(ModelOptions::default());
        generator.generate(&request, &worker_cancellation, |event| {
            if sender.send(event).is_err() {
                worker_cancellation.cancel();
            }
        })
    });
    // This blocking loop is a stand-in for the UI's background event receiver.
    // A GPUI app should dispatch updates onto its UI thread, not block that thread.
    // Wire a Cancel button to cancellation.cancel(); closing the receiver also stops work.
    for event in receiver {
        match event {
            Event::Preview { step, image, .. } => {
                // image.as_raw() contains RGBA8 bytes for uploading to a UI texture.
                println!("Preview {step}: {}x{}", image.width(), image.height());
            }
            Event::StepFinished { step, total, .. } => println!("Step {step}/{total}"),
            Event::Progress {
                stage,
                completed,
                total,
            } => println!("{stage:?}: {completed}/{total}"),
            _ => {}
        }
    }
    let generated = worker
        .join()
        .map_err(|_| anyhow::anyhow!("inference worker panicked"))??;
    generated.image.save("out.png")?;
    Ok(())
}
