use std::{
    cell::Cell,
    num::NonZeroUsize,
    path::Path,
    rc::Rc,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use anyhow::Context as _;
use image::ImageDecoder;

use crate::timing::{GenerationTiming, format_duration};

use gpui::{
    Bounds, Context, CursorStyle, DevicePixels, Entity, MouseButton, MouseDownEvent,
    MouseMoveEvent, ObjectFit, PathBuilder, PathPromptOptions, Pixels, Point, RenderImage, Task,
    Window, canvas, div, img, point, prelude::*, px, relative, rgb, size,
};
use gpui_component::{
    Disableable, Icon, Selectable,
    button::*,
    input::{Input, InputState},
};
use qwen_imager::{
    CancellationToken, Cancelled, DownloadProgress, Event, Generation, Generator, ModelOptions,
    Request, RgbaImage, Stage,
};

const MAX_REFERENCES: usize = 10;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PaintColor {
    Red,
    Blue,
    Green,
    White,
}

impl PaintColor {
    const ALL: [Self; 4] = [Self::Red, Self::Blue, Self::Green, Self::White];

    fn rgb(self) -> [u8; 3] {
        match self {
            Self::Red => [255, 0, 0],
            Self::Blue => [0, 0, 255],
            Self::Green => [0, 255, 0],
            Self::White => [255, 255, 255],
        }
    }

    fn hex(self) -> u32 {
        let [r, g, b] = self.rgb();
        u32::from_be_bytes([0, r, g, b])
    }

    fn name(self) -> &'static str {
        match self {
            Self::Red => "Red",
            Self::Blue => "Blue",
            Self::Green => "Green",
            Self::White => "White",
        }
    }
}

// Brush widths as a fraction of the image's longer side, so strokes keep their
// look between the on-screen preview and the full-resolution reference.
const BRUSHES: [(&str, f32); 3] = [("S", 0.006), ("M", 0.015), ("L", 0.035)];

/// A freehand stroke in normalized image coordinates (0..1 on both axes).
#[derive(Clone)]
struct Stroke {
    color: PaintColor,
    width: f32,
    points: Vec<(f32, f32)>,
}

enum Message {
    Checked(qwen_imager::Result<()>),
    Download(DownloadProgress),
    Prepared(qwen_imager::Result<()>),
    Inference(Event),
    Complete(qwen_imager::Result<Generation>),
}

struct ReferenceImage {
    name: String,
    image: Arc<RgbaImage>,
    preview: Arc<RenderImage>,
    // Paint not yet applied to `image`; it is baked in when generation starts.
    strokes: Vec<Stroke>,
}

impl ReferenceImage {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let mut decoder = image::ImageReader::open(path)?
            .with_guessed_format()?
            .into_decoder()?;
        let orientation = decoder.orientation()?;
        let mut image = image::DynamicImage::from_decoder(decoder)?;
        image.apply_orientation(orientation);
        anyhow::ensure!(
            image.width() > 0 && image.height() > 0,
            "image must not be empty"
        );
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        Ok(Self::new(name, Arc::new(image.to_rgba8())))
    }

    fn new(name: String, image: Arc<RgbaImage>) -> Self {
        // Bound the display texture; inference still receives the original RGBA pixels.
        let thumbnail = image::imageops::thumbnail(&*image, 1024, 1024);
        Self {
            name,
            preview: render_image(&thumbnail),
            image,
            strokes: Vec::new(),
        }
    }

    /// Applies pending strokes to the full-resolution pixels and refreshes the preview.
    fn bake(&mut self, window: &mut Window) {
        if self.strokes.is_empty() {
            return;
        }
        let mut painted = (*self.image).clone();
        for stroke in &self.strokes {
            rasterize_stroke(&mut painted, stroke);
        }
        let name = std::mem::take(&mut self.name);
        let old = std::mem::replace(self, Self::new(name, Arc::new(painted)));
        let _ = window.drop_image(old.preview);
    }
}

pub struct ImageWindow {
    prompt: Entity<InputState>,
    steps: Entity<InputState>,
    size: Entity<InputState>,
    seed: Entity<InputState>,
    automatic_seed: bool,
    busy: bool,
    ready: bool,
    checking_model: bool,
    loading_image: bool,
    references: Vec<ReferenceImage>,
    // Reference being painted on in the main view, if any.
    painting: Option<usize>,
    paint_color: PaintColor,
    brush: f32,
    // A stroke is in progress (the last stroke of the painted reference).
    drawing: bool,
    // Window-space bounds of the main view, recorded at paint time for mouse mapping.
    canvas_bounds: Rc<Cell<Bounds<Pixels>>>,
    cancellation: CancellationToken,
    status: String,
    progress: f32,
    image: Option<Arc<RgbaImage>>,
    rendered: Option<Arc<RenderImage>>,
    // Reference preview shown by the Before/After toggle once a generation finishes.
    before: Option<Arc<RenderImage>>,
    showing_before: bool,
    // The displayed image is a finished result (not a preview), so it can become the input.
    completed: bool,
    // Dropping the task closes the bounded channel when the window closes.
    receiver: Option<Task<()>>,
    timing: Option<GenerationTiming>,
    ticker: Option<Task<()>>,
    // One generator for the window's lifetime, so reruns with the same prompt
    // or references reuse its cached encoder results. Only one run holds it.
    generator: Arc<Mutex<Generator>>,
}

impl ImageWindow {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let prompt = cx.new(|cx| {
            InputState::new(window, cx)
                .multi_line(true)
                .rows(3)
                .placeholder("Describe the image or the changes you want…")
        });
        let steps = cx.new(|cx| {
            InputState::new(window, cx)
                .default_value("20")
                .validate(|text, _| text.bytes().all(|c| c.is_ascii_digit()))
        });
        let size = cx.new(|cx| {
            InputState::new(window, cx)
                .default_value("512")
                .validate(|text, _| text.bytes().all(|c| c.is_ascii_digit()))
        });
        let seed = cx.new(|cx| {
            InputState::new(window, cx)
                .default_value(rand::random_range(0..(1_u64 << 30)).to_string())
                .validate(|text, _| text.bytes().all(|c| c.is_ascii_digit()))
        });
        let mut view = Self {
            prompt,
            steps,
            size,
            seed,
            automatic_seed: true,
            busy: false,
            ready: false,
            checking_model: true,
            loading_image: false,
            references: Vec::new(),
            painting: None,
            paint_color: PaintColor::Red,
            brush: BRUSHES[1].1,
            drawing: false,
            canvas_bounds: Rc::default(),
            cancellation: CancellationToken::default(),
            status: "Checking model files…".into(),
            progress: 0.,
            image: None,
            rendered: None,
            before: None,
            showing_before: false,
            completed: false,
            receiver: None,
            timing: None,
            ticker: None,
            generator: Arc::new(Mutex::new(Generator::new(ModelOptions {
                offline: true,
                ..Default::default()
            }))),
        };
        view.check_model(window, cx);
        view
    }

    fn check_model(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.busy = true;
        let cancellation = self.cancellation.clone();
        let sender = self.listen(window, cx);
        thread::spawn(move || {
            // Startup only checks local files, including every indexed weight shard.
            // Downloads are authorized exclusively by the Download button below.
            let mut generator = Generator::new(ModelOptions {
                offline: true,
                ..Default::default()
            });
            let result = generator.prepare(&cancellation, |_| {});
            let _ = sender.send_blocking(Message::Checked(result));
        });
    }

    fn prepare(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy || self.ready {
            return;
        }
        self.busy = true;
        self.cancellation = CancellationToken::default();
        self.status = "Preparing download…".into();
        self.progress = 0.;
        let cancellation = self.cancellation.clone();
        let sender = self.listen(window, cx);
        thread::spawn(move || {
            let mut generator = Generator::new(ModelOptions::default());
            let result = generator.prepare(&cancellation, |progress| {
                if sender.send_blocking(Message::Download(progress)).is_err() {
                    cancellation.cancel();
                }
            });
            let _ = sender.send_blocking(Message::Prepared(result));
        });
        cx.notify();
    }

    fn generate(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || !self.ready {
            return;
        }
        let mut request = Request::new(self.prompt.read(cx).value().to_string());
        let size = self.size.read(cx).value().parse::<u32>().ok();
        let Some(size) = size.filter(|size| (32..=2048).contains(size) && size % 32 == 0) else {
            self.status =
                "Size must be a multiple of 32 between 32 and 2048 pixels (square image).".into();
            cx.notify();
            return;
        };
        request.scale = f64::from(size) / 2048.;
        // Keep the Size control's square output dimensions when editing a reference.
        request.ratio = Some("1:1".into());
        let Ok(steps) = self.steps.read(cx).value().parse::<usize>() else {
            self.status = "Enter a positive whole number of steps.".into();
            cx.notify();
            return;
        };
        request.steps = steps;
        if !self.automatic_seed {
            let Ok(seed) = self.seed.read(cx).value().parse::<u64>() else {
                self.status = "Enter a seed from 0 to 18446744073709551615.".into();
                cx.notify();
                return;
            };
            request.seed = seed;
        }
        request.preview_every = NonZeroUsize::new(1);
        if let Err(error) = request.dimensions() {
            self.status = error.to_string();
            cx.notify();
            return;
        }
        if self.automatic_seed {
            request.seed = self.randomize_seed(window, cx);
        }
        self.busy = true;
        self.progress = 0.;
        self.status = "Preparing generation…".into();
        self.timing = Some(GenerationTiming::new(request.steps));
        self.ticker = Some(cx.spawn(async move |view, cx| {
            loop {
                gpui::Timer::after(Duration::from_secs(1)).await;
                if !view
                    .update(cx, |view, cx| {
                        cx.notify();
                        view.timing.is_some()
                    })
                    .unwrap_or(false)
                {
                    break;
                }
            }
        }));
        self.clear_image(window);
        self.stop_painting();
        for reference in &mut self.references {
            reference.bake(window);
        }
        self.cancellation = CancellationToken::default();
        let cancellation = self.cancellation.clone();
        let sender = self.listen(window, cx);
        let generator = self.generator.clone();
        let references: Vec<_> = self
            .references
            .iter()
            .map(|reference| reference.image.clone())
            .collect();
        thread::spawn(move || {
            request.images = references
                .into_iter()
                .map(|reference| (*reference).clone())
                .collect();
            // A panicked run leaves the cache consistent: entries are only
            // stored once complete.
            let mut generator = generator.lock().unwrap_or_else(|e| e.into_inner());
            let result = generator.generate(&request, &cancellation, |event| {
                if sender.send_blocking(Message::Inference(event)).is_err() {
                    cancellation.cancel();
                }
            });
            let _ = sender.send_blocking(Message::Complete(result));
        });
        cx.notify();
    }

    fn load_image(&mut self, window: &Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || !self.ready || self.references.len() >= MAX_REFERENCES
        {
            return;
        }
        self.loading_image = true;
        let available = MAX_REFERENCES - self.references.len();
        let answer = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Add reference images".into()),
        });
        cx.spawn_in(window, async move |view, cx| {
            let result = async {
                let Some(paths) = answer.await?? else {
                    return Ok(None);
                };
                if paths.is_empty() {
                    return Ok(None);
                }
                anyhow::ensure!(paths.len() <= available, "Choose at most {available} more images (10 references maximum).");
                cx.background_executor()
                    .spawn(async move {
                        paths.into_iter().map(|path| {
                            ReferenceImage::load(&path)
                                .with_context(|| format!("opening {}", path.display()))
                        }).collect::<anyhow::Result<Vec<_>>>().map(Some)
                    })
                    .await
            }
            .await;
            let _ = view.update_in(cx, |view, window, cx| {
                view.loading_image = false;
                match result {
                    Ok(Some(references)) => {
                        view.clear_image(window);
                        view.references.extend(references);
                        view.status = format!(
                            "{} reference image(s) loaded. Describe the changes you want, then Generate.",
                            view.references.len()
                        );
                        view.progress = 0.;
                    }
                    Ok(None) => {} // Cancelling the picker preserves existing references.
                    Err(error) => view.status = format!("Could not load image: {error:#}"),
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    fn remove_reference(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || index >= self.references.len() {
            return;
        }
        let reference = self.references.remove(index);
        match self.painting {
            Some(painting) if painting == index => self.stop_painting(),
            Some(painting) if painting > index => self.painting = Some(painting - 1),
            _ => {}
        }
        if self
            .before
            .as_ref()
            .is_some_and(|before| Arc::ptr_eq(before, &reference.preview))
        {
            self.clear_comparison();
        }
        let _ = window.drop_image(reference.preview);
        self.status = if self.references.is_empty() {
            "References removed. Generate from your prompt alone.".into()
        } else {
            format!("{} reference image(s) loaded.", self.references.len())
        };
        cx.notify();
    }

    fn use_as_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || !self.completed {
            return;
        }
        let Some(image) = self.image.clone() else {
            return;
        };
        // Clears the before/after comparison before its reference textures are dropped.
        self.clear_image(window);
        self.stop_painting();
        for reference in self.references.drain(..) {
            let _ = window.drop_image(reference.preview);
        }
        self.references
            .push(ReferenceImage::new("Generated image".into(), image));
        self.progress = 0.;
        self.status =
            "Generated image is now the reference. Describe the changes you want, then Generate."
                .into();
        cx.notify();
    }

    fn toggle_painting(&mut self, index: usize, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || index >= self.references.len() {
            return;
        }
        if self.painting == Some(index) {
            self.stop_painting();
        } else {
            self.drawing = false;
            self.painting = Some(index);
            self.status = format!(
                "Drawing on image {}. Paint over it, then Generate.",
                index + 1
            );
        }
        cx.notify();
    }

    fn stop_painting(&mut self) {
        self.painting = None;
        self.drawing = false;
    }

    fn painted_reference(&mut self) -> Option<&mut ReferenceImage> {
        self.painting
            .and_then(|index| self.references.get_mut(index))
    }

    // Maps a window position to normalized image coordinates, clamped to the image.
    fn image_point(&self, position: Point<Pixels>) -> Option<(f32, f32)> {
        let reference = self.references.get(self.painting?)?;
        let rect = contain(self.canvas_bounds.get(), reference.image.dimensions());
        let width = f32::from(rect.size.width);
        let height = f32::from(rect.size.height);
        if width <= 0. || height <= 0. {
            return None;
        }
        let x = f32::from(position.x - rect.origin.x) / width;
        let y = f32::from(position.y - rect.origin.y) / height;
        Some((x, y))
    }

    fn start_stroke(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let Some((x, y)) = self.image_point(event.position) else {
            return;
        };
        if !(0. ..=1.).contains(&x) || !(0. ..=1.).contains(&y) {
            return;
        }
        let stroke = Stroke {
            color: self.paint_color,
            width: self.brush,
            points: vec![(x, y)],
        };
        if let Some(reference) = self.painted_reference() {
            reference.strokes.push(stroke);
            self.drawing = true;
            cx.notify();
        }
    }

    fn extend_stroke(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        if !self.drawing {
            return;
        }
        if event.pressed_button != Some(MouseButton::Left) {
            self.drawing = false;
            return;
        }
        let Some((x, y)) = self.image_point(event.position) else {
            return;
        };
        let point = (x.clamp(0., 1.), y.clamp(0., 1.));
        if let Some(stroke) = self
            .painted_reference()
            .and_then(|reference| reference.strokes.last_mut())
            && stroke.points.last() != Some(&point)
        {
            stroke.points.push(point);
            cx.notify();
        }
    }

    fn undo_stroke(&mut self, cx: &mut Context<Self>) {
        self.drawing = false;
        if let Some(reference) = self.painted_reference() {
            reference.strokes.pop();
        }
        cx.notify();
    }

    fn clear_strokes(&mut self, cx: &mut Context<Self>) {
        self.drawing = false;
        if let Some(reference) = self.painted_reference() {
            reference.strokes.clear();
        }
        cx.notify();
    }

    fn randomize_seed(&mut self, window: &mut Window, cx: &mut Context<Self>) -> u64 {
        let seed = rand::random_range(0..(1_u64 << 30));
        self.seed.update(cx, |input, cx| {
            input.set_value(seed.to_string(), window, cx)
        });
        seed
    }

    fn toggle_seed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        self.automatic_seed = !self.automatic_seed;
        if self.automatic_seed {
            self.randomize_seed(window, cx);
        }
        // Switching to Manual retains the displayed seed, including the last run's.
        cx.notify();
    }

    fn listen(
        &mut self,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> async_channel::Sender<Message> {
        let (sender, receiver) = async_channel::bounded(2);
        self.receiver = Some(cx.spawn_in(window, async move |view, cx| {
            while let Ok(message) = receiver.recv().await {
                if view
                    .update_in(cx, |view, window, cx| {
                        view.receive(message, window);
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
            }
        }));
        sender
    }

    fn receive(&mut self, message: Message, window: &mut Window) {
        // Keep the cancellation message visible until the worker has stopped.
        if self.cancellation.is_cancelled()
            && matches!(message, Message::Download(_) | Message::Inference(_))
        {
            return;
        }
        match message {
            Message::Checked(result) => {
                self.checking_model = false;
                self.busy = false;
                self.ready = result.is_ok();
                self.status = if self.ready {
                    "Model ready. Enter a prompt to begin.".into()
                } else {
                    String::new()
                };
                self.progress = if self.ready { 1. } else { 0. };
            }
            Message::Download(progress) => {
                self.progress = progress.total.filter(|&n| n > 0).map_or(0., |total| {
                    (progress.downloaded as f64 / total as f64).clamp(0., 1.) as f32
                });
                self.status = match progress.total {
                    Some(total) => format!(
                        "Downloading {} — {:.1} / {:.1} MB ({:.0}%)",
                        progress.file,
                        progress.downloaded as f64 / 1_000_000.,
                        total as f64 / 1_000_000.,
                        self.progress * 100.
                    ),
                    None => format!("Connecting to download {}…", progress.file),
                };
            }
            Message::Prepared(result) => {
                self.busy = false;
                self.ready = result.is_ok();
                self.status = match result {
                    Ok(()) => "Model ready. Enter a prompt to begin.".into(),
                    Err(error) => error_status(error),
                };
                self.progress = if self.ready { 1. } else { 0. };
            }
            Message::Inference(event) => match event {
                Event::Progress {
                    stage,
                    completed,
                    total,
                } => {
                    if stage == Stage::DenoiserLoading
                        && completed == total
                        && let Some(timing) = &mut self.timing
                    {
                        timing.start_steps();
                    }
                    self.progress = completed as f32 / total.max(1) as f32;
                    let name = match stage {
                        Stage::Loading => "Preparing model",
                        Stage::TextEncoding => "Encoding prompt",
                        Stage::ReferenceVision { .. } | Stage::ReferenceEncoding => {
                            "Encoding reference"
                        }
                        Stage::DenoiserLoading => "Loading denoiser",
                        Stage::Decoding => "Decoding final image",
                    };
                    self.status = format!("{name} — {completed}/{total}");
                }
                Event::StepFinished { step, total, .. } => {
                    self.progress = step as f32 / total as f32;
                    self.status = format!("Step {step}/{total} — decoding preview…");
                }
                Event::Preview { step, total, image } => {
                    self.set_image(image, window);
                    if let Some(timing) = &mut self.timing {
                        timing.complete_step(step);
                    }
                    self.status = format!("Step {step}/{total}");
                }
                _ => {}
            },
            Message::Complete(result) => {
                self.busy = false;
                self.timing = None;
                self.ticker = None;
                match result {
                    Ok(generated) => {
                        self.set_image(generated.image, window);
                        // References cannot change while busy, so these are the run's inputs.
                        self.before = self.references.last().map(|r| r.preview.clone());
                        self.completed = true;
                        self.progress = 1.;
                        self.status = format!("Complete — {}", format_duration(generated.elapsed));
                    }
                    Err(error) => self.status = error_status(error),
                }
            }
        }
    }

    fn clear_image(&mut self, window: &mut Window) {
        if let Some(old) = self.rendered.take() {
            let _ = window.drop_image(old);
        }
        self.image = None;
        self.clear_comparison();
    }

    // The before texture is owned by its reference thumbnail, so it is not dropped here.
    fn clear_comparison(&mut self) {
        self.before = None;
        self.showing_before = false;
        self.completed = false;
    }

    fn toggle_before(&mut self, cx: &mut Context<Self>) {
        if self.before.is_none() {
            return;
        }
        self.showing_before = !self.showing_before;
        cx.notify();
    }

    fn set_image(&mut self, image: Arc<RgbaImage>, window: &mut Window) {
        if self
            .image
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, &image))
        {
            return;
        }
        self.clear_image(window);
        self.rendered = Some(render_image(&image));
        self.image = Some(image);
    }

    fn cancel(&mut self, cx: &mut Context<Self>) {
        if !self.busy {
            return;
        }
        self.cancellation.cancel();
        self.timing = None;
        self.ticker = None;
        self.status = if self.ready {
            "Cancelling after the current operation…"
        } else {
            "Cancelling after the current file finishes downloading…"
        }
        .into();
        cx.notify();
    }

    fn save(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.image.clone() else {
            return;
        };
        let directory = std::env::current_dir().unwrap_or_default();
        let answer = cx.prompt_for_new_path(&directory, Some("qwen-image.png"));
        cx.spawn(async move |view, cx| {
            let result = async {
                let Some(path) = answer.await?? else {
                    return Ok(None);
                };
                // Save exactly the path approved by the native dialog; do not change
                // it afterwards and bypass the dialog's overwrite confirmation.
                anyhow::ensure!(
                    path.extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("png")),
                    "Choose a filename ending in .png"
                );
                cx.background_executor()
                    .spawn(async move {
                        image.save_with_format(&path, image::ImageFormat::Png)?;
                        Ok::<_, anyhow::Error>(Some(format!("Saved {}", path.display())))
                    })
                    .await
            }
            .await;
            let _ = view.update(cx, |view, cx| {
                match result {
                    Ok(Some(status)) => view.status = status,
                    Err(error) => view.status = format!("Could not save: {error:#}"),
                    Ok(None) => return,
                }
                cx.notify();
            });
        })
        .detach();
    }
}

impl Drop for ImageWindow {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

impl Render for ImageWindow {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .p_5()
                .bg(rgb(0x15171b))
                .text_color(rgb(0xe4e7ec))
                .child(
                    div().w_full().max_w(px(460.)).flex().flex_col().gap_4()
                        .map(|content| {
                            if self.checking_model {
                                return content.child("Checking local model files…");
                            }
                            content
                                .child("The image model is not fully available on this Mac. Download it to start generating images. The full model needs about 32 GB of disk space; files already downloaded will be reused.")
                                .child(div().child(
                                    Button::new("download")
                                        .primary()
                                        .label(if self.busy {
                                            if self.cancellation.is_cancelled() { "Cancelling…" } else { "Cancel" }
                                        } else { "Download" })
                                        .disabled(self.busy && self.cancellation.is_cancelled())
                                        .on_click(cx.listener(|view, _, window, cx| {
                                            if view.busy { view.cancel(cx); } else { view.prepare(window, cx); }
                                        }))))
                                .when(self.busy, |content| content.child(
                                    div().h(px(6.)).w_full().rounded_md().overflow_hidden().bg(rgb(0x303640))
                                        .child(div().h_full().w(relative(self.progress.clamp(0., 1.))).bg(rgb(0x8aa6ff)))))
                                .when(!self.status.is_empty(), |content| content.child(
                                    div().text_sm().text_color(rgb(0x9da6b5)).child(self.status.clone())))
                        }),
                );
        }
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_3()
            .p_5()
            .bg(rgb(0x15171b))
            .text_color(rgb(0xe4e7ec))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().flex_1().min_w_0().child(Input::new(&self.prompt).disabled(self.busy)))
                    .child(
                        Button::new("generate")
                            .primary()
                            .flex_shrink_0()
                            .label(if self.busy {
                                if self.cancellation.is_cancelled() { "Cancelling…" } else { "Cancel" }
                            } else {
                                "Generate"
                            })
                            .disabled(self.loading_image || (self.busy && self.cancellation.is_cancelled()))
                            .on_click(cx.listener(|view, _, window, cx| {
                                if view.busy { view.cancel(cx); } else { view.generate(window, cx); }
                            })),
                    )
                    .child(
                        Button::new("save")
                            .flex_shrink_0()
                            .icon(Icon::default().path("icons/save.svg"))
                            .tooltip("Save PNG")
                            .disabled(self.busy || self.loading_image || self.image.is_none())
                            .on_click(cx.listener(|view, _, _, cx| view.save(cx))),
                    )
            )
            .child(div().flex().flex_wrap().items_center().gap_3()
                .child(div().flex().items_center().gap_2()
                    .child("Steps")
                    .child(Input::new(&self.steps).w(px(72.)).disabled(self.busy)))
                .child(div().flex().items_center().gap_2()
                    .child("Size (px)")
                    .child(Input::new(&self.size).w(px(80.)).disabled(self.busy)))
                .child(div().flex().items_center().gap_2()
                    .child(Button::new("seed-mode")
                        .label(if self.automatic_seed { "Auto seed" } else { "Manual seed" })
                        .selected(self.automatic_seed)
                        .tooltip(if self.automatic_seed {
                            "A new random seed each generation. Click to use the displayed seed manually."
                        } else { "Use this seed each generation. Click to switch to automatic random seeds." })
                        .disabled(self.busy)
                        .on_click(cx.listener(|view, _, window, cx| view.toggle_seed(window, cx))))
                    .child(Input::new(&self.seed).w(px(200.)).disabled(self.busy || self.automatic_seed))))
            .child(div().id("reference-images").flex().items_center().gap_2()
                .flex_shrink_0().overflow_x_scroll().py_1()
                .children(self.references.iter().enumerate().map(|(index, reference)| {
                    let selected = self.painting == Some(index);
                    div().id(("reference", index)).relative().w(px(72.)).h(px(72.)).flex_shrink_0()
                        .rounded_md().border_1()
                        .border_color(if selected { rgb(0x8aa6ff) } else { rgb(0x303640) })
                        .bg(rgb(0x22262d))
                        .when(!self.busy && !self.loading_image, |tile| tile.cursor_pointer())
                        .on_click(cx.listener(move |view, _, _, cx| view.toggle_painting(index, cx)))
                        .child(img(reference.preview.clone()).size_full().object_fit(ObjectFit::Contain))
                        .when(!reference.strokes.is_empty(), |tile| tile.child(
                            stroke_overlay(reference, None)))
                        .child(div().absolute().bottom_0().left_0().px_1().text_xs()
                            .bg(rgb(0x15171b)).child((index + 1).to_string()))
                        .child(Button::new(("remove-image", index))
                            .absolute().top(px(2.)).right(px(2.)).w(px(22.)).h(px(22.)).p_0()
                            .label("−").tooltip(format!("Remove {}", reference.name))
                            .disabled(self.busy || self.loading_image)
                            .on_click(cx.listener(move |view, _, window, cx| {
                                view.remove_reference(index, window, cx);
                            })))
                }))
                .child(Button::new("add-images").w(px(72.)).h(px(72.)).flex_shrink_0()
                    .icon(Icon::default().path("icons/image.svg"))
                    .label("+")
                    .tooltip(if self.loading_image { "Loading images…" }
                        else if self.references.len() >= MAX_REFERENCES { "Maximum of 10 reference images" }
                        else { "Add reference images" })
                    .disabled(self.busy || self.loading_image || self.references.len() >= MAX_REFERENCES)
                    .on_click(cx.listener(|view, _, window, cx| view.load_image(window, cx)))))
            .child(
                div()
                    .h(px(6.))
                    .w_full()
                    .rounded_md()
                    .overflow_hidden()
                    .bg(rgb(0x303640))
                    .child(
                        div()
                            .h_full()
                            .w(relative(self.progress.clamp(0., 1.)))
                            .bg(rgb(0x8aa6ff)),
                    ),
            )
            .child(div().flex().flex_wrap().items_center().justify_between().gap_2().text_sm()
                .child(self.status.clone())
                .map(|row| match &self.timing {
                    Some(timing) => row.child(div().text_color(rgb(0x9da6b5)).child(timing.label())),
                    None => row,
                })
                .when(self.completed, |row| row.child(
                    div().flex().items_center().gap_2().ml_auto()
                    .child(
                        Button::new("use-as-input")
                            .flex_shrink_0()
                            .label("Use as input")
                            .tooltip("Replace all reference images with this result to keep editing it")
                            .disabled(self.loading_image)
                            .on_click(cx.listener(|view, _, window, cx| view.use_as_input(window, cx))),
                    )
                    .when(self.before.is_some(), |row| row.child(
                        Button::new("before-after")
                            .flex_shrink_0()
                            .label("Before")
                            .selected(self.showing_before)
                            .tooltip(if self.showing_before {
                                "Showing the reference image. Click to show the generated image."
                            } else { "Show the reference image before generation" })
                            .on_click(cx.listener(|view, _, _, cx| view.toggle_before(cx))),
                    )))))
            .when_some(self.painting.filter(|_| !self.busy), |view_root, index| {
                let has_strokes = !self.references[index].strokes.is_empty();
                view_root.child(div().flex().flex_wrap().items_center().gap_2().text_sm()
                    .child(format!("Draw on image {}", index + 1))
                    .children(PaintColor::ALL.into_iter().map(|color| {
                        let selected = self.paint_color == color;
                        div().id(color.name()).w(px(24.)).h(px(24.)).rounded_full()
                            .border_2()
                            .border_color(if selected { rgb(0x8aa6ff) } else { rgb(0x303640) })
                            .bg(rgb(color.hex()))
                            .cursor_pointer()
                            .tooltip(move |window, cx| gpui_component::tooltip::Tooltip::new(color.name()).build(window, cx))
                            .on_click(cx.listener(move |view, _, _, cx| {
                                view.paint_color = color;
                                cx.notify();
                            }))
                    }))
                    .child(div().w(px(8.)))
                    .children(BRUSHES.into_iter().map(|(label, width)| {
                        Button::new(("brush", (width * 1000.) as usize))
                            .label(label)
                            .selected(self.brush == width)
                            .tooltip("Brush size")
                            .on_click(cx.listener(move |view, _, _, cx| {
                                view.brush = width;
                                cx.notify();
                            }))
                    }))
                    .child(div().w(px(8.)))
                    .child(Button::new("undo-stroke").label("Undo").disabled(!has_strokes)
                        .on_click(cx.listener(|view, _, _, cx| view.undo_stroke(cx))))
                    .child(Button::new("clear-strokes").label("Clear").disabled(!has_strokes)
                        .on_click(cx.listener(|view, _, _, cx| view.clear_strokes(cx))))
                    .child(Button::new("done-painting").label("Done").ml_auto()
                        .tooltip("Stop drawing. Paint is applied when you Generate.")
                        .on_click(cx.listener(move |view, _, _, cx| view.toggle_painting(index, cx)))))
            })
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_md()
                    .overflow_hidden()
                    .bg(rgb(0x22262d))
                    .relative()
                    .map(|container| match self.painting.and_then(|index| self.references.get(index)) {
                        Some(reference) => {
                            let bounds = self.canvas_bounds.clone();
                            container
                                .cursor(CursorStyle::Crosshair)
                                .on_mouse_down(MouseButton::Left, cx.listener(|view, event, _, cx| view.start_stroke(event, cx)))
                                .on_mouse_move(cx.listener(|view, event, _, cx| view.extend_stroke(event, cx)))
                                .on_mouse_up(MouseButton::Left, cx.listener(|view, _, _, _| view.drawing = false))
                                .child(img(reference.preview.clone()).size_full().object_fit(ObjectFit::Contain))
                                .child(stroke_overlay(reference, Some(bounds)))
                        }
                        None => container.map(|container| match self.before.as_ref().filter(|_| self.showing_before).or(self.rendered.as_ref()).or_else(|| self.references.last().map(|reference| &reference.preview)) {
                        Some(image) => container.child(
                            img(image.clone())
                                .size_full()
                                .object_fit(ObjectFit::Contain),
                        ),
                        None => container.child(
                            div()
                                .text_color(rgb(0x9da6b5))
                                .child("Your image will appear here"),
                        ),
                    })})
                    .when(self.before.is_some() && self.painting.is_none(), |container| container.child(
                        div().absolute().top_2().left_2().px_2().py_1().rounded_md().text_xs()
                            .bg(rgb(0x15171b)).text_color(rgb(0xe4e7ec))
                            .child(if self.showing_before { "Before" } else { "After" }))),
            )
    }
}

fn error_status(error: anyhow::Error) -> String {
    if error.is::<Cancelled>() {
        "Cancelled. You can try again.".into()
    } else {
        format!("Error: {error:#}")
    }
}

fn contain(bounds: Bounds<Pixels>, (width, height): (u32, u32)) -> Bounds<Pixels> {
    let image = size(
        DevicePixels::from(width as i32),
        DevicePixels::from(height as i32),
    );
    ObjectFit::Contain.get_bounds(bounds, image)
}

// Draws a reference's pending strokes over its contained preview; optionally
// records the element bounds so mouse positions can be mapped back.
fn stroke_overlay(
    reference: &ReferenceImage,
    record: Option<Rc<Cell<Bounds<Pixels>>>>,
) -> impl IntoElement {
    let strokes = reference.strokes.clone();
    let dimensions = reference.image.dimensions();
    canvas(
        move |bounds, _, _| {
            if let Some(record) = record {
                record.set(bounds);
            }
        },
        move |bounds, (), window, _| {
            let rect = contain(bounds, dimensions);
            let scale = f32::from(rect.size.width.max(rect.size.height));
            let to_window = |(x, y): (f32, f32)| {
                point(
                    rect.origin.x + rect.size.width * x,
                    rect.origin.y + rect.size.height * y,
                )
            };
            window.with_content_mask(Some(gpui::ContentMask { bounds: rect }), |window| {
                for stroke in &strokes {
                    let width = (stroke.width * scale).max(1.);
                    let color = rgb(stroke.color.hex());
                    // Round caps and joins: a dot at every vertex plus straight segments.
                    for &p in &stroke.points {
                        let center = to_window(p);
                        let radius = px(width / 2.);
                        window.paint_quad(
                            gpui::fill(
                                Bounds::new(
                                    point(center.x - radius, center.y - radius),
                                    size(radius * 2., radius * 2.),
                                ),
                                color,
                            )
                            .corner_radii(radius),
                        );
                    }
                    if stroke.points.len() > 1 {
                        let mut path = PathBuilder::stroke(px(width));
                        path.move_to(to_window(stroke.points[0]));
                        for &p in &stroke.points[1..] {
                            path.line_to(to_window(p));
                        }
                        if let Ok(path) = path.build() {
                            window.paint_path(path, color);
                        }
                    }
                }
            });
        },
    )
    .absolute()
    .size_full()
}

/// Paints an opaque round-capped polyline onto the image.
fn rasterize_stroke(image: &mut RgbaImage, stroke: &Stroke) {
    let (width, height) = image.dimensions();
    let radius = (stroke.width * width.max(height) as f32 / 2.).max(0.5);
    let [r, g, b] = stroke.color.rgb();
    let points: Vec<(f32, f32)> = stroke
        .points
        .iter()
        .map(|&(x, y)| (x * width as f32, y * height as f32))
        .collect();
    let segments = points
        .windows(2)
        .map(|w| (w[0], w[1]))
        .chain(points.first().map(|&p| (p, p)));
    for (a, b_) in segments {
        let x0 = (a.0.min(b_.0) - radius).floor().max(0.) as u32;
        let y0 = (a.1.min(b_.1) - radius).floor().max(0.) as u32;
        let x1 = ((a.0.max(b_.0) + radius).ceil() as u32).min(width);
        let y1 = ((a.1.max(b_.1) + radius).ceil() as u32).min(height);
        let (dx, dy) = (b_.0 - a.0, b_.1 - a.1);
        let length = dx * dx + dy * dy;
        for y in y0..y1 {
            for x in x0..x1 {
                // Distance from the pixel centre to the segment.
                let (px_, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let t = if length > 0. {
                    (((px_ - a.0) * dx + (py - a.1) * dy) / length).clamp(0., 1.)
                } else {
                    0.
                };
                let (ex, ey) = (px_ - (a.0 + t * dx), py - (a.1 + t * dy));
                if ex * ex + ey * ey <= radius * radius {
                    image.put_pixel(x, y, image::Rgba([r, g, b, 255]));
                }
            }
        }
    }
}

fn render_image(rgba: &RgbaImage) -> Arc<RenderImage> {
    // GPUI's RenderImage expects BGRA bytes. Preserve alpha for the preview.
    let mut bgra = rgba.clone();
    for pixel in bgra.pixels_mut() {
        pixel.0.swap(0, 2);
    }
    Arc::new(RenderImage::new(vec![image::Frame::new(bgra)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preview_swaps_red_blue_and_preserves_alpha() {
        let rgba = RgbaImage::from_raw(1, 1, vec![200, 30, 10, 128]).unwrap();
        assert_eq!(
            render_image(&rgba).as_bytes(0).unwrap(),
            &[10, 30, 200, 128]
        );
        assert_eq!(rgba.as_raw(), &[200, 30, 10, 128]);
    }

    #[test]
    fn stroke_paints_the_segment_and_nothing_else() {
        let mut image = RgbaImage::from_pixel(100, 100, image::Rgba([0, 0, 0, 255]));
        let stroke = Stroke {
            color: PaintColor::Green,
            width: 0.04,
            points: vec![(0.1, 0.5), (0.9, 0.5)],
        };
        rasterize_stroke(&mut image, &stroke);
        assert_eq!(image.get_pixel(50, 50).0, [0, 255, 0, 255]);
        assert_eq!(image.get_pixel(10, 51).0, [0, 255, 0, 255]);
        assert_eq!(image.get_pixel(50, 45).0, [0, 0, 0, 255]);
        assert_eq!(image.get_pixel(2, 50).0, [0, 0, 0, 255]);
    }
}
