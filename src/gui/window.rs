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

use crate::{
    settings::{Backend, Settings, digits_input, parse_size, parse_steps, server_url},
    timing::{GenerationTiming, format_duration},
};

use crate::palette::{Palette, palette};
use gpui::{
    Bounds, Context, CursorStyle, DevicePixels, Entity, MouseButton, MouseDownEvent,
    MouseMoveEvent, ObjectFit, PathBuilder, Pixels, Point, RenderImage, Task,
    Window, canvas, div, img, point, prelude::*, px, relative, rgb, size,
};
use gpui_component::{
    Disableable, Icon, Selectable, Sizable as _,
    button::*,
    input::{Input, InputState, Textarea, TextareaState},
};
use image_forger::{
    AttentionPrecision, CancellationToken, Cancelled, Event, Generation,
    Generator, PauseControl, PreviewControl, Request, RgbaImage, Stage,
};

gpui::actions!(image_window, [Generate]);

/// Key context of the prompt. Enter inserts a newline; Cmd-Enter generates.
pub const PROMPT_CONTEXT: &str = "Prompt";

const MAX_REFERENCES: usize = 10;
const DECODING_PREVIEW: &str = " — decoding preview…";

/// A newly available image or a completed generation, for the workspace tab.
pub struct WorkspaceActivity;

#[derive(Default)]
struct ManualPreviewSchedule {
    due_step: Option<usize>,
}

impl ManualPreviewSchedule {
    fn on_event(&mut self, event: &Event, request: impl FnOnce() -> bool) -> bool {
        match event {
            Event::StepFinished { step, total, .. } => {
                if step == total {
                    self.due_step = None; // Final decoding supplies this image.
                } else if *step == 5 || *step == total.div_ceil(2) {
                    self.due_step = Some(*step);
                }
            }
            Event::Preview { step, .. } => {
                if self.due_step.is_some_and(|due| *step >= due) {
                    self.due_step = None;
                }
            }
            _ => return false,
        }
        // Coalesce milestones while a decode is busy. Retry with the latest
        // completed snapshot as soon as the decoder becomes available.
        self.due_step.is_some() && request()
    }
}

/// Preview settings captured when a generation starts.
#[derive(Clone, Copy)]
struct PreviewMode {
    automatic: bool,
    sequential: bool,
}

impl PreviewMode {
    fn from_settings(settings: &Settings) -> Self {
        Self {
            automatic: settings.automatic_previews,
            sequential: settings.sequential_previews,
        }
    }

    /// Every step decodes its preview before sampling continues.
    fn inline(self) -> bool {
        self.automatic && self.sequential
    }
}

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

// Smallest crop, in reference pixels, on either side.
const MIN_CROP: u32 = 16;

/// How the pointer edits the reference shown in the main view.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Draw,
    Crop,
}

/// The workspace layout. Simple keeps only a prompt; Advanced exposes the full
/// editing controls; Workflow provides a node-based canvas.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Mode {
    #[default]
    Advanced,
    Simple,
    Workflow,
}

impl Mode {
    const ALL: [Self; 3] = [Self::Simple, Self::Advanced, Self::Workflow];

    fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Advanced => "Advanced",
            Self::Workflow => "Workflow",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Self::Simple => "icons/sparkles.svg",
            Self::Advanced => "icons/sliders.svg",
            Self::Workflow => "icons/workflow.svg",
        }
    }
}

/// A dragged crop rectangle in normalized image coordinates (0..1 on both axes).
#[derive(Clone, Copy)]
struct CropSelection {
    anchor: (f32, f32),
    corner: (f32, f32),
}

/// A freehand stroke in normalized image coordinates (0..1 on both axes).
#[derive(Clone)]
struct Stroke {
    color: PaintColor,
    width: f32,
    points: Vec<(f32, f32)>,
}

enum Message {
    PreviewRequested,
    Inference(Event),
    Complete(image_forger::Result<Generation>),
}

struct ReferenceImage {
    name: String,
    image: Arc<RgbaImage>,
    preview: Arc<RenderImage>,
    // Paint not yet applied to `image`; it is baked in when generation starts.
    strokes: Vec<Stroke>,
}

/// A completed generation kept in the workspace history.
struct HistoryItem {
    image: Arc<RgbaImage>,
    thumbnail: Arc<RenderImage>,
}

impl HistoryItem {
    fn new(image: Arc<RgbaImage>) -> Self {
        let thumbnail = image::imageops::thumbnail(&*image, 256, 256);
        Self {
            image,
            thumbnail: render_image(&thumbnail),
        }
    }
}

impl ReferenceImage {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let image = image_forger::image_input::open(path)?;
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        Ok(Self::new(name, Arc::new(image)))
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

    /// Keeps only `region` (x, y, width, height) of the full-resolution pixels.
    /// Pending strokes stay where they were drawn and remain undoable.
    fn crop(&mut self, region: [u32; 4], window: &mut Window) {
        let [x, y, width, height] = region;
        let strokes = crop_strokes(&self.strokes, region, self.image.dimensions());
        let cropped = image::imageops::crop_imm(&*self.image, x, y, width, height).to_image();
        let name = std::mem::take(&mut self.name);
        let old = std::mem::replace(self, Self::new(name, Arc::new(cropped)));
        self.strokes = strokes;
        let _ = window.drop_image(old.preview);
    }
}

pub struct ImageWindow {
    mode: Mode,
    workflow: Entity<crate::workflow::WorkflowCanvas>,
    prompt: Entity<TextareaState>,
    steps: Entity<InputState>,
    size: Entity<InputState>,
    seed: Entity<InputState>,
    automatic_seed: bool,
    previews: PreviewMode,
    preview_control: Option<PreviewControl>,
    pause: Option<PauseControl>,
    paused: bool,
    preview_pending: bool,
    preview_step: Option<usize>,
    busy: bool,
    models: Entity<crate::models::Models>,
    loading_image: bool,
    references: Vec<ReferenceImage>,
    // Completed generations for this workspace, newest last.
    history: Vec<HistoryItem>,
    // Reference being painted on or cropped in the main view, if any.
    painting: Option<usize>,
    tool: Tool,
    paint_color: PaintColor,
    brush: f32,
    // A drag is in progress: the last stroke of the painted reference, or `crop`.
    drawing: bool,
    crop: Option<CropSelection>,
    // Constrain the crop selection to a square, matching the square output.
    square_crop: bool,
    // Window-space bounds of the main view, recorded at paint time for mouse mapping.
    canvas_bounds: Rc<Cell<Bounds<Pixels>>>,
    cancellation: CancellationToken,
    status: String,
    progress: f32,
    image: Option<Arc<RgbaImage>>,
    rendered: Option<Arc<RenderImage>>,
    // A reference or history image the user clicked to view in the preview.
    viewing: Option<Arc<RenderImage>>,
    // Reference preview shown by the Before/After toggle once a generation finishes.
    before: Option<Arc<RenderImage>>,
    showing_before: bool,
    // The displayed image is a finished result (not a preview), so it can become the input.
    completed: bool,
    // Dropping the task closes the bounded channel when the window closes.
    receiver: Option<Task<()>>,
    timing: Option<GenerationTiming>,
    ticker: Option<Task<()>>,
    // This workspace owns its generation state and queues. Its immutable model
    // buffers are shared with the other workspaces through SharedModel.
    generator: Arc<Mutex<Generator>>,
}

impl gpui::EventEmitter<WorkspaceActivity> for ImageWindow {}

impl ImageWindow {
    pub fn new(
        window: &mut Window,
        cx: &mut Context<Self>,
        generator: Arc<Mutex<Generator>>,
        models: Entity<crate::models::Models>,
    ) -> Self {
        let prompt = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(3, 12)
                .placeholder("Describe the image or the changes you want… (⌘↩ to generate)")
        });
        let settings = cx.global::<Settings>().clone();
        let steps = cx.new(|cx| digits_input(window, cx, settings.steps.to_string()));
        let size = cx.new(|cx| digits_input(window, cx, settings.size.to_string()));
        let seed = cx.new(|cx| {
            digits_input(window, cx, rand::random_range(0..(1_u64 << 30)).to_string())
        });
        Self {
            mode: Mode::Advanced,
            workflow: cx.new(crate::workflow::WorkflowCanvas::new),
            prompt,
            steps,
            size,
            seed,
            automatic_seed: true,
            previews: PreviewMode::from_settings(&settings),
            preview_control: None,
            pause: None,
            paused: false,
            preview_pending: false,
            preview_step: None,
            busy: false,
            models,
            loading_image: false,
            references: Vec::new(),
            history: Vec::new(),
            painting: None,
            tool: Tool::Draw,
            paint_color: PaintColor::Red,
            brush: BRUSHES[1].1,
            drawing: false,
            crop: None,
            square_crop: false,
            canvas_bounds: Rc::default(),
            cancellation: CancellationToken::default(),
            status: "Enter a prompt. Select a remote host or download a local model in Settings.".into(),
            progress: 0.,
            image: None,
            rendered: None,
            viewing: None,
            before: None,
            showing_before: false,
            completed: false,
            receiver: None,
            timing: None,
            ticker: None,
            generator,
        }
    }

    pub fn activate(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.busy {
            self.prompt.update(cx, |input, cx| input.focus(window, cx));
        }
        cx.notify();
    }

    /// Switches to a session of a newly selected model once this workspace is idle.
    /// Cached encoder results belong to the previous model and are dropped.
    pub fn set_generator(
        &mut self,
        generator: Arc<Mutex<Generator>>,
        cx: &mut Context<Self>,
    ) {
        // An in-flight run owns a clone of the old generator until it completes.
        self.generator = generator;
        cx.notify();
    }

    pub fn deactivate(&mut self) {
        self.drawing = false;
    }

    /// Adopts new default steps and size where the old defaults are unchanged.
    pub fn apply_defaults(
        &mut self,
        previous: &Settings,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.busy {
            cx.notify();
            return;
        }
        let settings = cx.global::<Settings>().clone();
        for (input, old, new) in [
            (&self.steps, previous.steps.to_string(), settings.steps.to_string()),
            (&self.size, previous.size.to_string(), settings.size.to_string()),
        ] {
            if input.read(cx).value().as_ref() == old.as_str() {
                input.update(cx, |input, cx| input.set_value(new, window, cx));
            }
        }
        cx.notify();
    }

    fn generate(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || self.mode == Mode::Workflow {
            return;
        }
        let backend = cx.global::<Backend>().clone();
        if let Backend::Remote { address } = &backend {
            self.busy = true;
            self.cancellation = CancellationToken::default();
            self.status = "Reading remote credentials…".into();
            let read = crate::credentials::read_token(address, cx);
            cx.spawn_in(window, async move |view, cx| {
                let token = read.await;
                let _ = view.update_in(cx, |view, window, cx| {
                    view.busy = false;
                    if view.cancellation.is_cancelled() {
                        view.status = "Cancelled before connecting.".into();
                    } else {
                        match token {
                            Ok(token) => view.start_generation(backend, token, window, cx),
                            Err(error) => {
                                crate::errors::report("Reading credentials before remote generation", &error, cx);
                                view.status = format!("Could not read remote credentials: {error:#}");
                            },
                        }
                    }
                    cx.notify();
                });
            }).detach();
            cx.notify();
        } else {
            self.start_generation(backend, None, window, cx);
        }
    }

    fn start_generation(&mut self, backend: Backend, token: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(message) = self.models.read(cx).generation_blocker(
            &backend, cx.global::<Settings>().model,
        ) {
            crate::errors::report("Starting generation", &anyhow::anyhow!(message.clone()), cx);
            self.status = message;
            cx.notify();
            return;
        }
        let pinned_identity = match &backend {
            Backend::Remote { address } => match cx.global::<Settings>().server_identities.get(&server_url(address)) {
                Some(pin) => pin.clone(),
                None => {
                    self.status = "Pin this server's identity in Settings → Compute before generating.".into();
                    cx.notify();
                    return;
                }
            },
            Backend::Local => String::new(),
        };
        // Trailing blank lines would otherwise change the prompt's encoding.
        let mut request = Request::new(self.prompt.read(cx).value().trim());
        let size = match parse_size(&self.size.read(cx).value()) {
            Ok(size) => size,
            Err(error) => {
                crate::errors::report("Validating generation settings", &anyhow::anyhow!(error), cx);
                self.status = error.into();
                cx.notify();
                return;
            }
        };
        request.scale = f64::from(size) / 2048.;
        // Keep the Size control's square output dimensions when editing a reference.
        request.ratio = Some("1:1".into());
        request.steps = match parse_steps(&self.steps.read(cx).value()) {
            Ok(steps) => steps,
            Err(error) => {
                crate::errors::report("Validating generation settings", &anyhow::anyhow!(error), cx);
                self.status = error.into();
                cx.notify();
                return;
            }
        };
        if !self.automatic_seed {
            let Ok(seed) = self.seed.read(cx).value().parse::<u64>() else {
                self.status = "Enter a seed from 0 to 18446744073709551615.".into();
                crate::errors::report("Validating generation seed", &anyhow::anyhow!(self.status.clone()), cx);
                cx.notify();
                return;
            };
            request.seed = seed;
        }
        let previews = PreviewMode::from_settings(cx.global::<Settings>());
        if previews.inline() {
            request.preview_every = NonZeroUsize::new(1);
        } else {
            request.preview_control = Some(PreviewControl::default());
        }
        request.pause = Some(PauseControl::default());
        request.attention = if cx.global::<Settings>().bf16_attention {
            AttentionPrecision::BFloat16
        } else {
            AttentionPrecision::Float32
        };
        if let Err(error) = request.dimensions() {
            crate::errors::report("Validating generation dimensions", &error, cx);
            self.status = error.to_string();
            cx.notify();
            return;
        }
        if self.automatic_seed {
            request.seed = self.randomize_seed(window, cx);
        }
        self.busy = true;
        self.previews = previews;
        self.preview_control = request.preview_control.clone();
        self.pause = request.pause.clone();
        self.paused = false;
        self.preview_pending = false;
        self.preview_step = None;
        self.progress = 0.;
        self.status = "Preparing generation…".into();
        self.timing = Some(GenerationTiming::new(request.steps));
        self.ticker = Some(cx.spawn(async move |view, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_secs(1))
                    .await;
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
            let result = {
                request.images = references
                    .into_iter()
                    .map(|reference| (*reference).clone())
                    .collect();
                match &backend {
                    Backend::Local => {
                        let mut generator =
                            generator.lock().unwrap_or_else(|error| error.into_inner());
                        let mut preview_schedule = ManualPreviewSchedule::default();
                        let cancel = cancellation.clone();
                        let preview_control = request.preview_control.clone();
                        let sender = sender.clone();
                        generator.generate(&request, &cancellation, move |event| {
                            let requested = !cancel.is_cancelled()
                                && preview_control.as_ref().is_some_and(|control| {
                                    if previews.automatic {
                                        // Parallel automatic previews decode the latest step
                                        // whenever the decoder is idle, skipping steps otherwise.
                                        if let Event::StepFinished { step, total, .. } = event
                                            && step < total
                                        {
                                            control.request_preview();
                                        }
                                        false
                                    } else {
                                        // Manual checkpoints always pause sampling until decoded.
                                        preview_schedule.on_event(&event, || {
                                            control.request_blocking_preview()
                                        })
                                    }
                                });
                            if sender.send_blocking(Message::Inference(event)).is_err()
                                || (requested
                                    && sender.send_blocking(Message::PreviewRequested).is_err())
                            {
                                cancel.cancel();
                            }
                        })
                    }
                    Backend::Remote { address } => {
                        let cancel = cancellation.clone();
                        let sender = sender.clone();
                        image_forger::remote::run_authenticated(&request, &server_url(address), token.as_deref(), &pinned_identity, &cancellation, move |event| {
                            if sender.send_blocking(Message::Inference(event)).is_err() {
                                cancel.cancel();
                            }
                        })
                    }
                }
            };
            let _ = sender.send_blocking(Message::Complete(result));
        });
        cx.notify();
    }

    fn request_preview(&mut self, cx: &mut Context<Self>) {
        if self.busy
            && !self.cancellation.is_cancelled()
            && !self.preview_pending
            && self.preview_control.as_ref().is_some_and(|control| {
                if self.previews.sequential {
                    control.request_blocking_preview()
                } else {
                    control.request_preview()
                }
            })
        {
            self.preview_pending = true;
            cx.notify();
        }
    }

    fn load_image(&mut self, window: &Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || self.references.len() >= MAX_REFERENCES
        {
            return;
        }
        self.loading_image = true;
        let available = MAX_REFERENCES - self.references.len();
        let answer = rfd::AsyncFileDialog::new()
            .set_parent(window)
            .set_title("Add reference images")
            .add_filter(
                "Images (PNG, JPEG, WebP, HEIC)",
                image_forger::image_input::EXTENSIONS,
            )
            .pick_files();
        cx.spawn_in(window, async move |view, cx| {
            let result = async {
                let Some(files) = answer.await else {
                    return Ok(None);
                };
                let paths: Vec<_> = files.into_iter().map(|file| file.path().to_owned()).collect();
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
                    Err(error) => {
                        crate::errors::report("Loading reference images", &error, cx);
                        view.status = format!("Could not load image: {error:#}");
                    },
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
        self.use_image_as_input(image, "Generated image".into(), window, cx);
    }

    fn use_image_as_input(
        &mut self,
        image: Arc<RgbaImage>,
        name: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Clears the before/after comparison before its reference textures are dropped.
        self.clear_image(window);
        self.stop_painting();
        for reference in self.references.drain(..) {
            let _ = window.drop_image(reference.preview);
        }
        self.references.push(ReferenceImage::new(name, image));
        self.progress = 0.;
        self.status =
            "Generated image is now the reference. Describe the changes you want, then Generate."
                .into();
        cx.notify();
    }

    fn view_reference(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(reference) = self.references.get(index) {
            self.viewing = Some(reference.preview.clone());
            cx.notify();
        }
    }

    fn view_history(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(item) = self.history.get(index) {
            self.viewing = Some(render_image(&item.image));
            cx.notify();
        }
    }

    fn remove_history(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if index >= self.history.len() {
            return;
        }
        let item = self.history.remove(index);
        let _ = window.drop_image(item.thumbnail);
        cx.notify();
    }

    fn use_history_as_input(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy || self.loading_image {
            return;
        }
        if let Some(item) = self.history.get(index) {
            self.use_image_as_input(item.image.clone(), "Generated image".into(), window, cx);
        }
    }

    fn toggle_painting(&mut self, index: usize, cx: &mut Context<Self>) {
        if self.busy || self.loading_image || index >= self.references.len() {
            return;
        }
        if self.painting == Some(index) {
            self.stop_painting();
        } else {
            self.drawing = false;
            self.crop = None;
            self.painting = Some(index);
            self.status = self.tool_status(index);
        }
        cx.notify();
    }

    fn tool_status(&self, index: usize) -> String {
        match self.tool {
            Tool::Draw => format!("Drawing on image {}. Paint over it, then Generate.", index + 1),
            Tool::Crop => format!(
                "Cropping image {}. Drag to select the area to keep, then Apply.",
                index + 1
            ),
        }
    }

    fn set_tool(&mut self, tool: Tool, cx: &mut Context<Self>) {
        self.tool = tool;
        self.drawing = false;
        self.crop = None;
        if let Some(index) = self.painting {
            self.status = self.tool_status(index);
        }
        cx.notify();
    }

    fn stop_painting(&mut self) {
        self.painting = None;
        self.drawing = false;
        self.crop = None;
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

    fn start_crop(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let Some(point) = self.image_point(event.position) else {
            return;
        };
        if !(0. ..=1.).contains(&point.0) || !(0. ..=1.).contains(&point.1) {
            return;
        }
        self.crop = Some(CropSelection {
            anchor: point,
            corner: point,
        });
        self.drawing = true;
        cx.notify();
    }

    fn extend_crop(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
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
        if let Some(crop) = &mut self.crop {
            crop.corner = (x.clamp(0., 1.), y.clamp(0., 1.));
            cx.notify();
        }
    }

    fn apply_crop(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let (Some(index), Some(selection)) = (self.painting, self.crop) else {
            return;
        };
        let reference = &self.references[index];
        let Some(region) = crop_region(selection, reference.image.dimensions(), self.square_crop)
        else {
            self.status = format!("Select at least {MIN_CROP}×{MIN_CROP} pixels to crop.");
            cx.notify();
            return;
        };
        if self
            .before
            .as_ref()
            .is_some_and(|before| Arc::ptr_eq(before, &reference.preview))
        {
            self.clear_comparison();
        }
        self.drawing = false;
        self.crop = None;
        self.references[index].crop(region, window);
        self.status = format!(
            "Cropped image {} to {}×{} pixels.",
            index + 1,
            region[2],
            region[3]
        );
        cx.notify();
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
                        view.receive(message, window, cx);
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

    fn receive(&mut self, message: Message, window: &mut Window, cx: &mut Context<Self>) {
        // Keep the cancellation message visible until the worker has stopped.
        if self.cancellation.is_cancelled()
            && matches!(
                message,
                Message::Inference(_) | Message::PreviewRequested
            )
        {
            return;
        }
        let activity = matches!(
            &message,
            Message::Inference(Event::Preview { .. }) | Message::Complete(_)
        );
        match message {
            Message::PreviewRequested => {
                self.preview_pending = true;
                // Checkpoint previews pause sampling while they decode.
                if !self.status.ends_with(DECODING_PREVIEW) {
                    self.status.push_str(DECODING_PREVIEW);
                }
            }
            Message::Inference(event) => match event {
                Event::Started { steps, .. } => {
                    let mut timing = GenerationTiming::new(steps);
                    timing.set_paused(self.paused);
                    self.timing = Some(timing);
                }
                Event::Progress {
                    stage,
                    completed,
                    total,
                } => {
                    if stage == Stage::Decoding {
                        self.preview_control = None;
                        self.preview_pending = false;
                    }
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
                    if !self.previews.inline()
                        && let Some(timing) = &mut self.timing
                    {
                        timing.complete_step(step);
                    }
                    let decoding = step < total
                        && (self.previews.inline()
                            || (self.preview_pending && self.previews.sequential));
                    self.status = format!(
                        "Step {step}/{total}{}",
                        if decoding { DECODING_PREVIEW } else { "" }
                    );
                }
                Event::Preview { step, total, image } => {
                    self.set_image(image, window);
                    self.preview_pending = false;
                    self.preview_step = Some(step);
                    if self.previews.inline() {
                        if let Some(timing) = &mut self.timing {
                            timing.complete_step(step);
                        }
                        self.status = format!("Step {step}/{total}");
                    } else if let Some(status) = self.status.strip_suffix(DECODING_PREVIEW) {
                        self.status = status.into();
                    }
                }
                _ => {}
            },
            Message::Complete(result) => {
                self.busy = false;
                self.pause = None;
                self.paused = false;
                self.preview_control = None;
                self.preview_pending = false;
                self.timing = None;
                self.ticker = None;
                match result {
                    Ok(generated) => {
                        self.set_image(generated.image.clone(), window);
                        self.history.push(HistoryItem::new(generated.image));
                        // References cannot change while busy, so these are the run's inputs.
                        self.before = self.references.last().map(|r| r.preview.clone());
                        self.completed = true;
                        self.progress = 1.;
                        self.status = format!("Complete — {}", format_duration(generated.elapsed));
                    }
                    Err(error) => {
                        if !error.is::<Cancelled>() { crate::errors::report("Generating image", &error, cx); }
                        self.status = error_status(error);
                    },
                }
            }
        }
        if activity {
            cx.emit(WorkspaceActivity);
        }
    }

    fn clear_image(&mut self, window: &mut Window) {
        self.preview_step = None;
        if let Some(old) = self.rendered.take() {
            let _ = window.drop_image(old);
        }
        self.image = None;
        self.viewing = None;
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
        if let Some(pause) = &self.pause {
            pause.resume();
        }
        self.paused = false;
        self.timing = None;
        self.ticker = None;
        self.status = "Cancelling after the current operation…".into();
        cx.notify();
    }

    fn toggle_pause(&mut self, cx: &mut Context<Self>) {
        let Some(pause) = &self.pause else {
            return;
        };
        if self.cancellation.is_cancelled() {
            return;
        }
        self.paused = !self.paused;
        if self.paused {
            pause.pause();
        } else {
            pause.resume();
        }
        if let Some(timing) = &mut self.timing {
            timing.set_paused(self.paused);
        }
        cx.notify();
    }

    fn save(&mut self, cx: &mut Context<Self>) {
        let Some(image) = self.image.clone() else {
            return;
        };
        self.save_image(image, cx);
    }

    fn save_image(&mut self, image: Arc<RgbaImage>, cx: &mut Context<Self>) {
        let directory = cx.global::<Settings>().save_directory();
        let answer = cx.prompt_for_new_path(&directory, Some("image-forger.png"));
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
                        image.save_with_format(&path, image::ImageFormat::Png).with_context(|| format!("writing PNG to {}", path.display()))?;
                        Ok::<_, anyhow::Error>(Some(format!("Saved {}", path.display())))
                    })
                    .await
            }
            .await;
            let _ = view.update(cx, |view, cx| {
                match result {
                    Ok(Some(status)) => view.status = status,
                    Err(error) => {
                        crate::errors::report("Saving generated image", &error, cx);
                        view.status = format!("Could not save: {error:#}");
                    },
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
        let mode = self.mode;
        div()
            .size_full()
            .flex()
            .bg(palette(cx).background)
            .text_color(palette(cx).text)
            .on_action(cx.listener(|view, _: &Generate, window, cx| view.generate(window, cx)))
            .child(self.sidebar(cx))
            .child(match mode {
                Mode::Simple => self.simple(cx).into_any_element(),
                Mode::Advanced => self.advanced(cx).into_any_element(),
                Mode::Workflow => self.workflow.clone().into_any_element(),
            })
    }
}

impl ImageWindow {
    fn advanced(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        // A local checkpoint with a saved schedule ignores the requested steps.
        let fixed_steps = {
            let settings = cx.global::<Settings>();
            (settings.backend == Backend::Local)
                .then(|| settings.model.sample_sigmas().map(<[f64]>::len))
                .flatten()
        };
        // A running generation keeps the preview settings it started with.
        let previews = if self.busy {
            self.previews
        } else {
            PreviewMode::from_settings(cx.global::<Settings>())
        };
        let manual_previews = !previews.automatic;
        div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_3()
            .p_5()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().flex_1().min_w_0().key_context(PROMPT_CONTEXT)
                        .child(Textarea::new(&self.prompt).disabled(self.busy)))
                    .child(
                        Button::new("generate")
                            .primary()
                            .flex_shrink_0()
                            .label(if self.busy {
                                if self.cancellation.is_cancelled() { "Cancelling…" } else { "Cancel" }
                            } else {
                                "Generate"
                            })
                            .when(!self.busy, |button| button.tooltip("Generate (⌘↩)"))
                            .disabled(self.loading_image || (self.busy && self.cancellation.is_cancelled()))
                            .on_click(cx.listener(|view, _, window, cx| {
                                if view.busy { view.cancel(cx); } else { view.generate(window, cx); }
                            })),
                    )
                    .when(self.busy && self.pause.is_some(), |row| row.child(
                        Button::new("pause")
                            .flex_shrink_0()
                            .icon(Icon::default().path(if self.paused { "icons/play.svg" } else { "icons/pause.svg" }))
                            .selected(self.paused)
                            .tooltip(if self.paused { "Resume generation" } else { "Pause generation after the current layer" })
                            .disabled(self.cancellation.is_cancelled())
                            .on_click(cx.listener(|view, _, _, cx| view.toggle_pause(cx))),
                    ))
            )
            .child(div().flex().flex_wrap().items_center().gap_3()
                .child(div().flex().items_center().gap_2()
                    .child(match fixed_steps {
                        Some(steps) => format!("Steps (fixed at {steps})"),
                        None => "Steps".into(),
                    })
                    .child(Input::new(&self.steps).w(px(72.)).disabled(self.busy || fixed_steps.is_some())))
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
                    .child(Input::new(&self.seed).w(px(200.)).disabled(self.busy || self.automatic_seed)))
                .when(manual_previews, |row| row.child(
                    Button::new("preview-now")
                        .label(if self.preview_pending { "Preparing preview…" } else { "Preview" })
                        .tooltip(if previews.sequential {
                            "Preview the latest completed step, pausing sampling while it decodes"
                        } else { "Preview the latest completed step while generation continues" })
                        .disabled(!self.busy || self.preview_control.is_none() || self.preview_pending || self.cancellation.is_cancelled())
                        .on_click(cx.listener(|view, _, _, cx| view.request_preview(cx))))))
            .child(
                div()
                    .h(px(6.))
                    .w_full()
                    .rounded_md()
                    .overflow_hidden()
                    .bg(palette(cx).border)
                    .child(
                        div()
                            .h_full()
                            .w(relative(self.progress.clamp(0., 1.)))
                            .bg(palette(cx).accent),
                    ),
            )
            .child(div().flex().flex_wrap().items_center().justify_between().gap_2().text_sm()
                .child(if self.paused { format!("Paused · {}", self.status) } else { self.status.clone() })
                .map(|row| match &self.timing {
                    Some(timing) => row.child(div().text_color(palette(cx).muted).child(timing.label())),
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
                let crop_region = self.crop.and_then(|crop| {
                    crop_region(crop, self.references[index].image.dimensions(), self.square_crop)
                });
                view_root.child(div().flex().flex_wrap().items_center().gap_2().text_sm()
                    .child(format!("Image {}", index + 1))
                    .child(Button::new("tool-draw").label("Draw")
                        .selected(self.tool == Tool::Draw)
                        .tooltip("Paint over the image")
                        .on_click(cx.listener(|view, _, _, cx| view.set_tool(Tool::Draw, cx))))
                    .child(Button::new("tool-crop").label("Crop")
                        .selected(self.tool == Tool::Crop)
                        .tooltip("Keep only part of the image")
                        .on_click(cx.listener(|view, _, _, cx| view.set_tool(Tool::Crop, cx))))
                    .child(div().w(px(8.)))
                    .when(self.tool == Tool::Crop, |row| row
                    .child(Button::new("square-crop").label("Square")
                        .selected(self.square_crop)
                        .tooltip("Keep the selection square, like the generated image")
                        .on_click(cx.listener(|view, _, _, cx| {
                            view.square_crop = !view.square_crop;
                            cx.notify();
                        })))
                    .when_some(crop_region, |row, [_, _, width, height]| row.child(
                        div().text_color(palette(cx).muted).child(format!("{width}×{height}"))))
                    .child(Button::new("apply-crop").label("Apply").disabled(crop_region.is_none())
                        .tooltip("Crop the reference to the selection")
                        .on_click(cx.listener(|view, _, window, cx| view.apply_crop(window, cx))))
                    .child(Button::new("clear-crop").label("Clear").disabled(self.crop.is_none())
                        .on_click(cx.listener(|view, _, _, cx| {
                            view.crop = None;
                            view.drawing = false;
                            cx.notify();
                        }))))
                    .when(self.tool == Tool::Draw, |row| row
                    .children(PaintColor::ALL.into_iter().map(|color| {
                        let selected = self.paint_color == color;
                        div().id(color.name()).w(px(24.)).h(px(24.)).rounded_full()
                            .border_2()
                            .border_color(if selected { palette(cx).accent } else { palette(cx).border })
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
                        .on_click(cx.listener(|view, _, _, cx| view.clear_strokes(cx)))))
                    .child(Button::new("done-painting").label("Done").ml_auto()
                        .tooltip("Stop editing. Paint is applied when you Generate.")
                        .on_click(cx.listener(move |view, _, _, cx| view.toggle_painting(index, cx)))))
            })
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .flex()
                    .gap_2()
                    .child(self.references_panel(cx))
                    .child(self.preview_panel(cx))
                    .child(self.history_panel(cx)),
            )
    }

    fn canvas(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex_1()
            .min_h_0()
            .w_full()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .overflow_hidden()
            .bg(palette(cx).card)
            .relative()
            .map(|container| match self.painting.and_then(|index| self.references.get(index)) {
                Some(reference) => {
                    let bounds = self.canvas_bounds.clone();
                    let dimensions = reference.image.dimensions();
                    container
                        .cursor(CursorStyle::Crosshair)
                        .map(|container| match self.tool {
                            Tool::Draw => container
                                .on_mouse_down(MouseButton::Left, cx.listener(|view, event, _, cx| view.start_stroke(event, cx)))
                                .on_mouse_move(cx.listener(|view, event, _, cx| view.extend_stroke(event, cx))),
                            Tool::Crop => container
                                .on_mouse_down(MouseButton::Left, cx.listener(|view, event, _, cx| view.start_crop(event, cx)))
                                .on_mouse_move(cx.listener(|view, event, _, cx| view.extend_crop(event, cx))),
                        })
                        .on_mouse_up(MouseButton::Left, cx.listener(|view, _, _, _| view.drawing = false))
                        .child(img(reference.preview.clone()).size_full().object_fit(ObjectFit::Contain))
                        .child(stroke_overlay(reference, Some(bounds)))
                        .when_some(self.crop.filter(|_| self.tool == Tool::Crop), |container, crop| {
                            container.child(crop_overlay(crop_bounds(crop, dimensions, self.square_crop), dimensions, palette(cx)))
                        })
                }
                None => container.map(|container| match self.viewing.as_ref().or_else(|| self.before.as_ref().filter(|_| self.showing_before)).or(self.rendered.as_ref()).or_else(|| self.references.last().map(|reference| &reference.preview)) {
                Some(image) => container.child(
                    img(image.clone())
                        .size_full()
                        .object_fit(ObjectFit::Contain),
                ),
                None => container.child(
                    div()
                        .text_color(palette(cx).muted)
                        .child("Your image will appear here"),
                ),
            })})
            .when(self.before.is_some() && self.painting.is_none(), |container| container.child(
                div().absolute().top_2().left_2().px_2().py_1().rounded_md().text_xs()
                    .bg(palette(cx).background).text_color(palette(cx).text)
                    .child(if self.showing_before { "Before" } else { "After" })))
            .when(!self.completed && self.painting.is_none() && self.rendered.is_some(), |container| {
                container.when_some(self.preview_step, |container, step| container.child(
                    div().absolute().top_2().left_2().px_2().py_1().rounded_md().text_xs()
                        .bg(palette(cx).background).text_color(palette(cx).text)
                        .child(format!("Preview · step {step}"))))
            })
    }

    fn preview_panel(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_sm().text_color(palette(cx).muted).child("Preview"))
                    .child(
                        Button::new("save")
                            .icon(Icon::default().path("icons/save.svg"))
                            .label("Save")
                            .disabled(self.busy || self.image.is_none())
                            .on_click(cx.listener(|view, _, _, cx| view.save(cx))),
                    ),
            )
            .child(self.canvas(cx))
    }

    fn references_panel(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(220.))
            .flex_shrink_0()
            .flex()
            .flex_col()
            .gap_2()
            .p_2()
            .rounded_md()
            .bg(palette(cx).panel)
            .child(div().text_sm().text_color(palette(cx).muted).child("References"))
            .child(
                Button::new("add-images")
                    .w_full()
                    .flex_shrink_0()
                    .icon(Icon::default().path("icons/image.svg"))
                    .label(if self.references.is_empty() {
                        "Add images"
                    } else {
                        "Add"
                    })
                    .tooltip(if self.loading_image {
                        "Loading images…"
                    } else if self.references.len() >= MAX_REFERENCES {
                        "Maximum of 10 reference images"
                    } else {
                        "Add reference images"
                    })
                    .disabled(
                        self.busy
                            || self.loading_image
                            || self.references.len() >= MAX_REFERENCES,
                    )
                    .on_click(cx.listener(|view, _, window, cx| view.load_image(window, cx))),
            )
            .child(
                div()
                    .id("references-list")
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .overflow_y_scroll()
                    .children(self.references.iter().enumerate().map(|(index, reference)| {
                        let selected = self.painting == Some(index);
                        div()
                            .w_full()
                            .flex_shrink_0()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .id(("reference-view", index))
                                    .relative()
                                    .w_full()
                                    .h(px(110.))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .overflow_hidden()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(if selected {
                                        palette(cx).accent
                                    } else {
                                        palette(cx).border
                                    })
                                    .bg(palette(cx).card)
                                    .when(!self.busy && !self.loading_image, |tile| {
                                        tile.cursor_pointer()
                                    })
                                    .on_click(cx.listener(move |view, _, _, cx| {
                                        view.view_reference(index, cx)
                                    }))
                                    .child(
                                        img(reference.preview.clone())
                                            .size_full()
                                            .object_fit(ObjectFit::Contain),
                                    )
                                    .when(!reference.strokes.is_empty(), |tile| {
                                        tile.child(stroke_overlay(reference, None))
                                    }),
                            )
                            .child(
                                div()
                                    .flex()
                                    .gap_1()
                                    .child(
                                        Button::new(("remove-image", index))
                                            .xsmall()
                                            .icon(Icon::default().path("icons/trash.svg"))
                                            .tooltip(format!("Remove {}", reference.name))
                                            .disabled(self.busy || self.loading_image)
                                            .on_click(cx.listener(move |view, _, window, cx| {
                                                view.remove_reference(index, window, cx);
                                            })),
                                    )
                                    .child(
                                        Button::new(("edit-image", index))
                                            .xsmall()
                                            .flex_1()
                                            .label(if selected { "Done" } else { "Edit" })
                                            .selected(selected)
                                            .tooltip(if selected {
                                                "Finish editing"
                                            } else {
                                                "Paint or crop this image"
                                            })
                                            .disabled(self.busy || self.loading_image)
                                            .on_click(cx.listener(move |view, _, _, cx| {
                                                view.toggle_painting(index, cx)
                                            })),
                                    ),
                            )
                    }))
            )
    }

    fn history_panel(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(220.))
            .flex_shrink_0()
            .flex()
            .flex_col()
            .gap_2()
            .p_2()
            .rounded_md()
            .bg(palette(cx).panel)
            .child(div().text_sm().text_color(palette(cx).muted).child("History"))
            .child(
                div()
                    .id("history-list")
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .overflow_y_scroll()
                    .children(self.history.iter().enumerate().map(|(index, item)| {
                        let thumbnail = item.thumbnail.clone();
                        let image = item.image.clone();
                        div()
                            .w_full()
                            .flex_shrink_0()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .id(("history-view", index))
                                    .w_full()
                                    .h(px(110.))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .overflow_hidden()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(palette(cx).border)
                                    .bg(palette(cx).card)
                                    .cursor_pointer()
                                    .on_click(cx.listener(move |view, _, _, cx| {
                                        view.view_history(index, cx)
                                    }))
                                    .child(
                                        img(thumbnail)
                                            .size_full()
                                            .object_fit(ObjectFit::Contain),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .gap_1()
                                    .child(
                                        Button::new(("remove-history", index))
                                            .xsmall()
                                            .icon(Icon::default().path("icons/trash.svg"))
                                            .tooltip("Remove from history")
                                            .on_click(cx.listener(move |view, _, window, cx| {
                                                view.remove_history(index, window, cx);
                                            })),
                                    )
                                    .child(
                                        Button::new(("save-history", index))
                                            .xsmall()
                                            .flex_1()
                                            .icon(Icon::default().path("icons/save.svg"))
                                            .tooltip("Save PNG")
                                            .on_click(cx.listener(move |view, _, _, cx| {
                                                view.save_image(image.clone(), cx)
                                            })),
                                    )
                                    .child(
                                        Button::new(("use-history", index))
                                            .xsmall()
                                            .flex_1()
                                            .icon(Icon::default().path("icons/image.svg"))
                                            .tooltip("Use as input")
                                            .on_click(cx.listener(move |view, _, window, cx| {
                                                view.use_history_as_input(index, window, cx)
                                            })),
                                    ),
                            )
                    }))
                    .when(self.history.is_empty(), |list| {
                        list.child(
                            div()
                                .text_sm()
                                .text_color(palette(cx).muted)
                                .child("No generations yet."),
                        )
                    }),
            )
    }

    fn sidebar(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let current = self.mode;
        div()
            .flex()
            .flex_col()
            .items_center()
            .gap_1()
            .px_1()
            .py_2()
            .flex_shrink_0()
            .border_r_1()
            .border_color(palette(cx).border)
            .bg(palette(cx).panel)
            .children(Mode::ALL.into_iter().enumerate().map(|(index, mode)| {
                Button::new(("mode", index))
                    .ghost()
                    .large()
                    .w(px(44.))
                    .h(px(44.))
                    .icon(Icon::default().path(mode.icon()))
                    .selected(current == mode)
                    .tooltip(format!("{} mode", mode.label()))
                    .on_click(cx.listener(move |view, _, _, cx| {
                        view.mode = mode;
                        cx.notify();
                    }))
            }))
    }

    fn simple(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_4()
            .p_5()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .key_context(PROMPT_CONTEXT)
                            .child(Textarea::new(&self.prompt).disabled(self.busy)),
                    )
                    .child(
                        Button::new("generate")
                            .primary()
                            .flex_shrink_0()
                            .label(if self.busy {
                                if self.cancellation.is_cancelled() {
                                    "Cancelling…"
                                } else {
                                    "Cancel"
                                }
                            } else {
                                "Generate"
                            })
                            .when(!self.busy, |button| button.tooltip("Generate (⌘↩)"))
                            .on_click(cx.listener(|view, _, window, cx| {
                                if view.busy {
                                    view.cancel(cx);
                                } else {
                                    view.generate(window, cx);
                                }
                            })),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .flex()
                    .gap_2()
                    .child(self.preview_panel(cx))
                    .child(self.history_panel(cx)),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .text_sm()
                    .child(if self.paused {
                        format!("Paused · {}", self.status)
                    } else {
                        self.status.clone()
                    })
                    .when_some(self.timing.as_ref(), |row, timing| {
                        row.child(div().text_color(palette(cx).muted).child(timing.label()))
                    }),
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

/// The selection in image pixels as (left, top, right, bottom). A square
/// selection grows from the anchor without leaving the image.
fn crop_bounds(
    selection: CropSelection,
    (width, height): (u32, u32),
    square: bool,
) -> (f32, f32, f32, f32) {
    let (width, height) = (width as f32, height as f32);
    let (ax, ay) = (selection.anchor.0 * width, selection.anchor.1 * height);
    let (mut dx, mut dy) = (
        selection.corner.0 * width - ax,
        selection.corner.1 * height - ay,
    );
    if square {
        let room_x = if dx < 0. { ax } else { width - ax };
        let room_y = if dy < 0. { ay } else { height - ay };
        let side = dx.abs().max(dy.abs()).min(room_x).min(room_y);
        dx = side.copysign(dx);
        dy = side.copysign(dy);
    }
    (ax.min(ax + dx), ay.min(ay + dy), ax.max(ax + dx), ay.max(ay + dy))
}

/// The selection as a pixel region (x, y, width, height), if it is large enough.
fn crop_region(selection: CropSelection, dimensions: (u32, u32), square: bool) -> Option<[u32; 4]> {
    let (left, top, right, bottom) = crop_bounds(selection, dimensions, square);
    let (x, y) = (left.round() as u32, top.round() as u32);
    let mut width = (right.round() as u32).min(dimensions.0).saturating_sub(x);
    let mut height = (bottom.round() as u32).min(dimensions.1).saturating_sub(y);
    if square {
        width = width.min(height);
        height = width;
    }
    (width >= MIN_CROP && height >= MIN_CROP).then_some([x, y, width, height])
}

/// Maps strokes into a cropped region, dropping those entirely outside it.
fn crop_strokes(
    strokes: &[Stroke],
    [x, y, width, height]: [u32; 4],
    (full_width, full_height): (u32, u32),
) -> Vec<Stroke> {
    // Brush widths are relative to the longer side, which the crop changes.
    let scale = full_width.max(full_height) as f32 / width.max(height) as f32;
    let longer = width.max(height) as f32;
    strokes
        .iter()
        .filter_map(|stroke| {
            let points: Vec<_> = stroke
                .points
                .iter()
                .map(|&(px_, py)| {
                    (
                        (px_ * full_width as f32 - x as f32) / width as f32,
                        (py * full_height as f32 - y as f32) / height as f32,
                    )
                })
                .collect();
            let width_ = stroke.width * scale;
            let (rx, ry) = (
                width_ * longer / 2. / width as f32,
                width_ * longer / 2. / height as f32,
            );
            let (min_x, max_x, min_y, max_y) = points.iter().fold(
                (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
                |(a, b, c, d), &(px_, py)| (a.min(px_), b.max(px_), c.min(py), d.max(py)),
            );
            (max_x >= -rx && min_x <= 1. + rx && max_y >= -ry && min_y <= 1. + ry).then(|| {
                Stroke {
                    color: stroke.color,
                    width: width_,
                    points,
                }
            })
        })
        .collect()
}

// Dims everything outside the crop selection (given in image pixels).
fn crop_overlay(
    (left, top, right, bottom): (f32, f32, f32, f32),
    dimensions: (u32, u32),
    palette: Palette,
) -> impl IntoElement {
    canvas(
        |_, _, _| {},
        move |bounds, (), window, _| {
            let rect = contain(bounds, dimensions);
            let at = |x: f32, y: f32| {
                point(
                    rect.origin.x + rect.size.width * (x / dimensions.0 as f32),
                    rect.origin.y + rect.size.height * (y / dimensions.1 as f32),
                )
            };
            let (width, height) = (dimensions.0 as f32, dimensions.1 as f32);
            let shade = gpui::rgba(0x000000a0);
            for (a, b) in [
                (at(0., 0.), at(width, top)),
                (at(0., bottom), at(width, height)),
                (at(0., top), at(left, bottom)),
                (at(right, top), at(width, bottom)),
            ] {
                window.paint_quad(gpui::fill(Bounds::from_corners(a, b), shade));
            }
            window.paint_quad(gpui::outline(
                Bounds::from_corners(at(left, top), at(right, bottom)),
                palette.accent,
                gpui::BorderStyle::default(),
            ));
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

    fn step_event(step: usize, total: usize) -> Event {
        Event::StepFinished {
            step,
            total,
            duration: Duration::ZERO,
        }
    }

    fn preview_event(step: usize, total: usize) -> Event {
        Event::Preview {
            step,
            total,
            image: Arc::new(RgbaImage::new(1, 1)),
        }
    }

    #[test]
    fn manual_milestones_handle_short_odd_and_overlapping_runs() {
        for (total, expected) in [
            (1, vec![]),
            (2, vec![1]),
            (5, vec![3]),
            (8, vec![4, 5]),
            (9, vec![5]),
            (10, vec![5]),
            (11, vec![5, 6]),
            (20, vec![5, 10]),
        ] {
            let mut schedule = ManualPreviewSchedule::default();
            let mut requested = Vec::new();
            for step in 1..=total {
                if schedule.on_event(&step_event(step, total), || true) {
                    requested.push(step);
                    assert!(
                        !schedule
                            .on_event(&preview_event(step, total), || panic!("duplicate preview"))
                    );
                }
            }
            assert_eq!(requested, expected, "total steps: {total}");
        }
    }

    #[test]
    fn manual_milestones_coalesce_busy_decodes_and_yield_to_final_image() {
        let mut schedule = ManualPreviewSchedule::default();
        assert!(!schedule.on_event(&step_event(5, 20), || false));
        assert!(!schedule.on_event(&step_event(10, 20), || false));
        assert!(schedule.on_event(&preview_event(4, 20), || true));
        assert!(!schedule.on_event(&preview_event(10, 20), || panic!(
            "milestone already delivered"
        )));
        assert!(!schedule.on_event(&step_event(11, 20), || panic!("no milestone due")));

        let mut schedule = ManualPreviewSchedule::default();
        assert!(!schedule.on_event(&step_event(5, 20), || false));
        assert!(!schedule.on_event(&step_event(20, 20), || panic!("final decode is sufficient")));
        assert!(!schedule.on_event(&preview_event(4, 20), || panic!("sampling has finished")));
    }

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

    #[test]
    fn crop_region_orders_corners_and_rejects_tiny_selections() {
        let selection = CropSelection {
            anchor: (0.75, 0.5),
            corner: (0.25, 0.1),
        };
        assert_eq!(crop_region(selection, (200, 100), false), Some([50, 10, 100, 40]));
        let tiny = CropSelection {
            anchor: (0.5, 0.5),
            corner: (0.52, 0.9),
        };
        assert_eq!(crop_region(tiny, (200, 100), false), None);
    }

    #[test]
    fn square_crop_stays_inside_the_image() {
        let selection = CropSelection {
            anchor: (0.5, 0.5),
            corner: (1.0, 0.6),
        };
        // 100px of room to the right, but only 50px below the anchor.
        assert_eq!(crop_region(selection, (200, 100), true), Some([100, 50, 50, 50]));
        let up_left = CropSelection {
            anchor: (0.5, 0.5),
            corner: (0.0, 0.45),
        };
        assert_eq!(crop_region(up_left, (200, 100), true), Some([50, 0, 50, 50]));
    }

    #[test]
    fn cropped_strokes_keep_their_pixels() {
        let stroke = Stroke {
            color: PaintColor::Red,
            width: 0.04,
            points: vec![(0.1, 0.5), (0.9, 0.5)],
        };
        let outside = Stroke {
            color: PaintColor::Blue,
            width: 0.01,
            points: vec![(0.05, 0.05)],
        };
        let region = [20, 30, 50, 40];
        let mut full = RgbaImage::from_pixel(100, 100, image::Rgba([0, 0, 0, 255]));
        rasterize_stroke(&mut full, &stroke);
        let expected = image::imageops::crop_imm(&full, 20, 30, 50, 40).to_image();
        let strokes = crop_strokes(&[stroke, outside], region, (100, 100));
        assert_eq!(strokes.len(), 1);
        let mut cropped = RgbaImage::from_pixel(50, 40, image::Rgba([0, 0, 0, 255]));
        rasterize_stroke(&mut cropped, &strokes[0]);
        let differing = cropped
            .pixels()
            .zip(expected.pixels())
            .filter(|(a, b)| a != b)
            .count();
        assert!(differing <= 4, "{differing} pixels differ");
    }
}
