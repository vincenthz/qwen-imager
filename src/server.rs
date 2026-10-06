//! HTTP transport for the existing blocking generator. GPU work stays on one
//! persistent thread; HTTP handlers only touch small job metadata/CPU images.
use anyhow::Context as _;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{ConnectInfo, DefaultBodyLimit, FromRequest, Multipart, Path, State},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use image_forger::diagnostics::{log, redact};
use image_forger::{
    CancellationToken, Checkpoint, Event, Generator, ModelOptions, PreviewControl, Request,
    RgbaImage, Stage,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    io::Cursor,
    net::SocketAddr,
    num::NonZeroUsize,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore};

const JSON_LIMIT: usize = 64 * 1024;
const UPLOAD_LIMIT: usize = 64 * 1024 * 1024;
const IMAGE_PIXEL_LIMIT: u64 = 16_000_000;
const JOB_PIXEL_LIMIT: u64 = 32_000_000;
const REFERENCE_MEMORY_UNIT: usize = 64 * 1024;
const REFERENCE_MEMORY_LIMIT: usize = 512 * 1024 * 1024;

// Permits follow the actual request buffers through upload, queueing, and
// generation. Dropping a rejected/cancelled/completed request releases them.
struct QueuedRequest {
    request: Request,
    _reference_memory: Vec<OwnedSemaphorePermit>,
}
impl std::ops::Deref for QueuedRequest {
    type Target = Request;
    fn deref(&self) -> &Request {
        &self.request
    }
}

#[derive(Serialize)]
struct ReferenceInfo {
    width: u32,
    height: u32,
}

type ApiResult<T> = Result<T, ApiError>;
#[derive(Debug)]
struct ApiError(StatusCode, String);
#[derive(Clone)]
struct ErrorDetail(String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let detail = ErrorDetail(self.1.clone());
        let mut response = (self.0, Json(serde_json::json!({"error": self.1}))).into_response();
        response.extensions_mut().insert(detail);
        response
    }
}
fn conflict(message: &str) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}

#[derive(Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PreviewMode {
    #[default]
    Manual,
    Auto,
    Off,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct Parameters {
    prompt: String,
    ratio: Option<String>,
    scale: f64,
    steps: usize,
    seed: u64,
    noise_source_size: Option<u32>,
    preview_mode: PreviewMode,
    preview_every: usize,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            ratio: None,
            scale: 1.0,
            steps: 40,
            seed: 42,
            noise_source_size: None,
            preview_mode: PreviewMode::Manual,
            preview_every: 5,
        }
    }
}
impl Parameters {
    fn request(&self, images: Vec<RgbaImage>) -> anyhow::Result<Request> {
        anyhow::ensure!(
            self.prompt.len() <= 16_384,
            "prompt must be at most 16384 UTF-8 bytes"
        );
        anyhow::ensure!(self.steps <= 1000, "steps must be at most 1000");
        anyhow::ensure!(self.preview_every > 0, "preview_every must be positive");
        let mut request = Request::new(&self.prompt);
        request.images = images;
        request.ratio = self.ratio.clone();
        request.scale = self.scale;
        request.steps = self.steps;
        request.seed = self.seed;
        request.noise_source_size = self.noise_source_size;
        match self.preview_mode {
            PreviewMode::Manual => request.preview_control = Some(PreviewControl::default()),
            PreviewMode::Auto => request.preview_every = NonZeroUsize::new(self.preview_every),
            PreviewMode::Off => {}
        }
        request.dimensions()?;
        Ok(request)
    }
}

#[derive(Clone, Copy, Serialize, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
enum Status {
    Queued,
    Running,
    Cancelling,
    Succeeded,
    Failed,
    Cancelled,
}
impl Status {
    fn terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

struct Image {
    pixels: Arc<RgbaImage>,
    step: usize,
    png: OnceCell<Bytes>,
}
impl Image {
    fn new(pixels: Arc<RgbaImage>, step: usize) -> Arc<Self> {
        Arc::new(Self {
            pixels,
            step,
            png: OnceCell::new(),
        })
    }
    async fn response(&self) -> ApiResult<Response> {
        let png = self
            .png
            .get_or_try_init(|| async {
                let pixels = self.pixels.clone();
                tokio::task::spawn_blocking(move || {
                    let mut buffer = Cursor::new(Vec::new());
                    pixels.write_to(&mut buffer, image::ImageFormat::Png)?;
                    Ok::<_, anyhow::Error>(Bytes::from(buffer.into_inner()))
                })
                .await
                .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
            })
            .await?;
        Ok((
            [
                (header::CONTENT_TYPE, "image/png"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            [("x-preview-step", self.step.to_string())],
            png.clone(),
        )
            .into_response())
    }
}

struct Job {
    id: u64,
    parameters: Parameters,
    references: Vec<ReferenceInfo>,
    width: u32,
    height: u32,
    created_ms: u64,
    status: Status,
    stage: &'static str,
    reference_index: Option<usize>,
    stage_completed: usize,
    stage_total: usize,
    completed_steps: usize,
    last_step_s: Option<f64>,
    started: Option<Instant>,
    elapsed_s: Option<f64>,
    error: Option<String>,
    preview: Option<Arc<Image>>,
    preview_pending: bool,
    output: Option<Arc<Image>>,
    control: Option<PreviewControl>,
    cancel: CancellationToken,
}
impl Job {
    fn view(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id.to_string(), "status": self.status, "parameters": self.parameters,
            "references": self.references,
            "width": self.width, "height": self.height, "created_unix_ms": self.created_ms,
            "stage": self.stage, "stage_completed": self.stage_completed, "stage_total": self.stage_total,
            "reference_index": self.reference_index,
            "completed_steps": self.completed_steps, "total_steps": self.parameters.steps,
            "last_step_s": self.last_step_s,
            "elapsed_s": self.elapsed_s.or_else(|| self.started.map(|s| s.elapsed().as_secs_f64())),
            "error": self.error, "preview_pending": self.preview_pending,
            "preview_step": self.preview.as_ref().map(|p| p.step),
            "preview_url": self.preview.as_ref().map(|_| format!("/jobs/{}/preview", self.id)),
            "image_url": self.output.as_ref().map(|_| format!("/jobs/{}/image", self.id)),
            "status_url": format!("/jobs/{}", self.id),
        })
    }
    fn event(&mut self, event: Event) {
        match event {
            Event::Started { .. } => {}
            Event::Progress {
                stage,
                completed,
                total,
            } => {
                self.reference_index = match stage {
                    Stage::ReferenceVision { index } => Some(index),
                    _ => None,
                };
                self.stage = match stage {
                    Stage::Loading => "loading",
                    Stage::ReferenceVision { .. } => "reference_vision",
                    Stage::TextEncoding => "text_encoding",
                    Stage::ReferenceEncoding => "reference_encoding",
                    Stage::DenoiserLoading => "denoiser_loading",
                    Stage::Decoding => "decoding",
                };
                self.stage_completed = completed;
                self.stage_total = total;
            }
            Event::StepFinished {
                step,
                total,
                duration,
            } => {
                self.reference_index = None;
                self.stage = "sampling";
                self.stage_completed = step;
                self.stage_total = total;
                self.completed_steps = step;
                self.last_step_s = Some(duration.as_secs_f64());
            }
            Event::Preview { step, image, .. } => {
                self.preview = Some(Image::new(image, step));
                self.preview_pending = false;
            }
            Event::Finished { .. } => {}
        }
    }
    fn finish(&mut self, result: anyhow::Result<image_forger::Generation>) {
        self.preview_pending = false;
        self.reference_index = None;
        self.elapsed_s = self.started.map(|s| s.elapsed().as_secs_f64());
        match result {
            Ok(result) => {
                self.status = Status::Succeeded;
                self.stage = "finished";
                let image = self
                    .preview
                    .as_ref()
                    .filter(|p| Arc::ptr_eq(&p.pixels, &result.image))
                    .cloned()
                    .unwrap_or_else(|| Image::new(result.image, self.parameters.steps));
                self.preview = Some(image.clone());
                self.output = Some(image);
            }
            Err(error) => {
                if error.is::<image_forger::Cancelled>() {
                    self.status = Status::Cancelled;
                    self.stage = "cancelled";
                } else {
                    self.status = Status::Failed;
                    self.stage = "failed";
                    self.error = Some(format!("{error:#}"));
                }
            }
        }
    }
}

type SharedJob = Arc<Mutex<Job>>;
struct Store {
    jobs: BTreeMap<u64, SharedJob>,
    queue: VecDeque<(u64, QueuedRequest)>,
    next_id: u64,
    stopping: bool,
}
struct Service {
    store: Mutex<Store>,
    wake: Condvar,
    max_jobs: usize,
    authorization: Option<HeaderValue>,
    reference_memory: Arc<Semaphore>,
    uploads: Arc<Semaphore>,
    checkpoint: Checkpoint,
    http_debug: bool,
    next_request: AtomicU64,
}
impl Service {
    #[cfg(test)]
    fn new(max_jobs: usize, authorization: Option<HeaderValue>) -> Arc<Self> {
        Self::with_checkpoint(max_jobs, authorization, Checkpoint::default(), false)
    }
    fn with_checkpoint(
        max_jobs: usize,
        authorization: Option<HeaderValue>,
        checkpoint: Checkpoint,
        http_debug: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            store: Mutex::new(Store {
                jobs: BTreeMap::new(),
                queue: VecDeque::new(),
                next_id: 1,
                stopping: false,
            }),
            wake: Condvar::new(),
            max_jobs,
            authorization,
            reference_memory: Arc::new(Semaphore::new(
                REFERENCE_MEMORY_LIMIT / REFERENCE_MEMORY_UNIT,
            )),
            uploads: Arc::new(Semaphore::new(2)),
            checkpoint,
            http_debug,
            next_request: AtomicU64::new(1),
        })
    }
    fn job(&self, id: u64) -> ApiResult<SharedJob> {
        self.store
            .lock()
            .unwrap()
            .jobs
            .get(&id)
            .cloned()
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "job not found".into()))
    }
    fn submit(&self, parameters: Parameters) -> ApiResult<serde_json::Value> {
        self.submit_images(parameters, Vec::new(), Vec::new())
    }
    fn submit_images(
        &self,
        parameters: Parameters,
        images: Vec<RgbaImage>,
        reference_memory: Vec<OwnedSemaphorePermit>,
    ) -> ApiResult<serde_json::Value> {
        let references = images
            .iter()
            .map(|i| ReferenceInfo {
                width: i.width(),
                height: i.height(),
            })
            .collect();
        let request = parameters
            .request(images)
            .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;
        let (width, height) = request.dimensions().unwrap();
        let mut store = self.store.lock().unwrap();
        if store.stopping {
            return Err(ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "server is stopping".into(),
            ));
        }
        if store.jobs.len() >= self.max_jobs {
            return Err(ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "job capacity reached; DELETE finished jobs to free slots".into(),
            ));
        }
        let id = store.next_id;
        store.next_id += 1;
        let job = Job {
            id,
            parameters,
            references,
            width,
            height,
            created_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            status: Status::Queued,
            stage: "queued",
            reference_index: None,
            stage_completed: 0,
            stage_total: 0,
            completed_steps: 0,
            last_step_s: None,
            started: None,
            elapsed_s: None,
            error: None,
            preview: None,
            preview_pending: false,
            output: None,
            control: request.preview_control.clone(),
            cancel: CancellationToken::default(),
        };
        let view = job.view();
        store.jobs.insert(id, Arc::new(Mutex::new(job)));
        store.queue.push_back((
            id,
            QueuedRequest {
                request,
                _reference_memory: reference_memory,
            },
        ));
        log(
            "INFO",
            "job.queued",
            serde_json::json!({"job_id": id, "width": width, "height": height, "queue_depth": store.queue.len()}),
        );
        self.wake.notify_one();
        Ok(view)
    }
    fn next(&self) -> Option<(SharedJob, QueuedRequest)> {
        let mut store = self.store.lock().unwrap();
        loop {
            if store.stopping {
                return None;
            }
            if let Some((id, request)) = store.queue.pop_front() {
                let job = store.jobs.get(&id).unwrap().clone();
                {
                    let mut data = job.lock().unwrap();
                    data.status = Status::Running;
                    data.stage = "starting";
                    data.started = Some(Instant::now());
                }
                return Some((job, request));
            }
            store = self.wake.wait(store).unwrap();
        }
    }
    fn cancel(&self, id: u64) -> ApiResult<serde_json::Value> {
        let mut store = self.store.lock().unwrap();
        let mut job = store
            .jobs
            .get(&id)
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "job not found".into()))?
            .lock()
            .unwrap();
        if !job.status.terminal() {
            job.cancel.cancel();
            if job.status == Status::Queued {
                job.status = Status::Cancelled;
                job.stage = "cancelled";
            } else {
                job.status = Status::Cancelling;
            }
        }
        let view = job.view();
        log(
            "INFO",
            "job.cancel_requested",
            serde_json::json!({"job_id": id, "status": job.status}),
        );
        drop(job);
        store.queue.retain(|(queued, _)| *queued != id);
        Ok(view)
    }
    fn shutdown(&self) {
        let mut store = self.store.lock().unwrap();
        store.stopping = true;
        store.queue.clear();
        for job in store.jobs.values() {
            let mut job = job.lock().unwrap();
            if !job.status.terminal() {
                job.cancel.cancel();
                if job.status == Status::Queued {
                    job.status = Status::Cancelled;
                    job.stage = "cancelled";
                } else {
                    job.status = Status::Cancelling;
                }
            }
        }
        self.wake.notify_all();
    }
}

fn worker(service: Arc<Service>, options: ModelOptions) {
    let mut generator = Generator::new(options.clone());
    while let Some((job, request)) = service.next() {
        let (id, cancel) = {
            let job = job.lock().unwrap();
            (job.id, job.cancel.clone())
        };
        log(
            "INFO",
            "job.started",
            serde_json::json!({"job_id": id, "model": options.checkpoint.name()}),
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generator.generate(&request, &cancel, |event| {
                let mut job = job.lock().unwrap();
                let previous_stage = job.stage;
                let step_finished = matches!(event, Event::StepFinished { .. });
                job.event(event);
                if service.http_debug && (job.stage != previous_stage || step_finished) {
                    log("DEBUG", "job.progress", serde_json::json!({"job_id": id, "stage": job.stage, "completed": job.stage_completed, "total": job.stage_total}));
                }
            })
        }));
        let result = match result {
            Ok(result) => result,
            Err(_) => {
                generator = Generator::new(options.clone());
                Err(anyhow::anyhow!(
                    "generation worker panicked; model cache reset"
                ))
            }
        };
        let mut job = job.lock().unwrap();
        let last_stage = job.stage;
        job.finish(result);
        let secret = service
            .authorization
            .as_ref()
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if let Some(error) = &mut job.error {
            *error = redact(
                error,
                &[secret, secret.strip_prefix("Bearer ").unwrap_or("")],
            );
        }
        log(
            if job.status == Status::Failed {
                "ERROR"
            } else {
                "INFO"
            },
            "job.finished",
            serde_json::json!({"job_id": id, "status": job.status, "last_stage": last_stage, "completed_steps": job.completed_steps, "elapsed_s": job.elapsed_s, "error": job.error}),
        );
    }
}

async fn guard(
    State(service): State<Arc<Service>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let request_id = service
        .next_request
        .fetch_add(1, Ordering::Relaxed)
        .to_string();
    let started = Instant::now();
    let method = request.method().to_string();
    let secret = service
        .authorization
        .as_ref()
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let path = redact(
        request.uri().path(),
        &[secret, secret.strip_prefix("Bearer ").unwrap_or("")],
    );
    let peer = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0.to_string());
    let auth_present = request.headers().contains_key(header::AUTHORIZATION);
    if service.http_debug {
        log(
            "DEBUG",
            "http.request_started",
            serde_json::json!({"request_id": request_id, "method": method, "path": path, "peer": peer, "auth_present": auth_present}),
        );
    }
    let mut response = if let Some(expected) = &service.authorization
        && request.headers().get(header::AUTHORIZATION) != Some(expected)
    {
        ApiError(
            StatusCode::UNAUTHORIZED,
            "a valid Bearer token is required".into(),
        )
        .into_response()
    } else {
        next.run(request).await
    };
    let secret = service
        .authorization
        .as_ref()
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let reason = response
        .extensions()
        .get::<ErrorDetail>()
        .map(|detail| {
            redact(
                &detail.0,
                &[secret, secret.strip_prefix("Bearer ").unwrap_or("")],
            )
        })
        .or_else(|| {
            response.status().is_client_error().then(|| {
                response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Request rejected")
                    .into()
            })
        });
    log(
        if response.status().is_server_error() {
            "ERROR"
        } else if response.status().is_client_error() {
            "WARN"
        } else {
            "INFO"
        },
        "http.request_finished",
        serde_json::json!({"request_id": request_id, "method": method, "path": path, "peer": peer, "status": response.status().as_u16(), "elapsed_ms": started.elapsed().as_millis(), "error": reason, "location": response.headers().get(header::LOCATION).and_then(|v| v.to_str().ok())}),
    );
    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn health(State(service): State<Arc<Service>>) -> Json<serde_json::Value> {
    let store = service.store.lock().unwrap();
    Json(
        serde_json::json!({"status": if store.stopping { "stopping" } else { "ok" },
        "model": service.checkpoint.repo(), "revision": service.checkpoint.revision(), "retained_jobs": store.jobs.len(), "queued_jobs": store.queue.len(), "max_jobs": service.max_jobs}),
    )
}
fn decode_references(
    files: Vec<Bytes>,
    budget: Arc<Semaphore>,
) -> ApiResult<(Vec<RgbaImage>, Vec<OwnedSemaphorePermit>)> {
    let mut images = Vec::new();
    let mut permits = Vec::new();
    let mut total_pixels = 0;
    for (index, bytes) in files.into_iter().enumerate() {
        let invalid = |error: image::ImageError| {
            let status = if matches!(error, image::ImageError::Limits(_)) {
                StatusCode::PAYLOAD_TOO_LARGE
            } else if matches!(error, image::ImageError::Unsupported(_)) {
                StatusCode::UNSUPPORTED_MEDIA_TYPE
            } else {
                StatusCode::BAD_REQUEST
            };
            ApiError(status, format!("reference image {}: {error}", index + 1))
        };
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(128 * 1024 * 1024);
        let decoder =
            image_forger::image_input::ImageInput::new(&bytes, limits).map_err(invalid)?;
        let (width, height) = decoder.dimensions();
        let pixels = u64::from(width) * u64::from(height);
        total_pixels += pixels;
        if pixels == 0 || pixels > IMAGE_PIXEL_LIMIT || total_pixels > JOB_PIXEL_LIMIT {
            return Err(ApiError(
                StatusCode::PAYLOAD_TOO_LARGE,
                "references exceed the 16-megapixel per-image or 32-megapixel per-job limit".into(),
            ));
        }
        let units = (pixels as usize * 4).div_ceil(REFERENCE_MEMORY_UNIT) as u32;
        let permit = budget.clone().try_acquire_many_owned(units).map_err(|_| {
            ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "reference-image memory capacity reached; retry after queued jobs finish".into(),
            )
        })?;
        // Reserve RGBA memory before decompression, even for a tiny compressed file.
        let decoded = decoder.decode().map_err(invalid)?;
        images.push(decoded);
        permits.push(permit);
    }
    Ok((images, permits))
}

async fn submit(
    State(service): State<Arc<Service>>,
    mut request: axum::extract::Request,
) -> ApiResult<Response> {
    let multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("multipart/form-data"));
    let view = if multipart {
        let upload = service.uploads.clone().try_acquire_owned().map_err(|_| {
            ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "two reference uploads are already in progress; retry shortly".into(),
            )
        })?;
        request
            .extensions_mut()
            .insert(DefaultBodyLimit::max(UPLOAD_LIMIT));
        let mut form = Multipart::from_request(request, &())
            .await
            .map_err(|e| ApiError(e.status(), e.body_text()))?;
        let mut parameters = None;
        let mut files = Vec::new();
        while let Some(mut field) = form
            .next_field()
            .await
            .map_err(|e| ApiError(e.status(), e.body_text()))?
        {
            match field.name() {
                Some("parameters") => {
                    if parameters.is_some() {
                        return Err(ApiError(
                            StatusCode::BAD_REQUEST,
                            "parameters must appear exactly once".into(),
                        ));
                    }
                    let mut data = Vec::new();
                    while let Some(chunk) = field
                        .chunk()
                        .await
                        .map_err(|e| ApiError(e.status(), e.body_text()))?
                    {
                        if data.len() + chunk.len() > JSON_LIMIT {
                            return Err(ApiError(
                                StatusCode::PAYLOAD_TOO_LARGE,
                                "parameters exceed 64 KiB".into(),
                            ));
                        }
                        data.extend_from_slice(&chunk);
                    }
                    parameters =
                        Some(serde_json::from_slice::<Parameters>(&data).map_err(|e| {
                            ApiError(StatusCode::UNPROCESSABLE_ENTITY, e.to_string())
                        })?);
                }
                Some("images") => {
                    if files.len() >= 10 {
                        return Err(ApiError(
                            StatusCode::BAD_REQUEST,
                            "at most 10 reference images are supported".into(),
                        ));
                    }
                    files.push(
                        field
                            .bytes()
                            .await
                            .map_err(|e| ApiError(e.status(), e.body_text()))?,
                    );
                }
                _ => {
                    return Err(ApiError(
                        StatusCode::BAD_REQUEST,
                        "multipart fields must be parameters or images".into(),
                    ));
                }
            }
        }
        let parameters = parameters.ok_or_else(|| {
            ApiError(
                StatusCode::BAD_REQUEST,
                "missing parameters JSON field".into(),
            )
        })?;
        // Parsing/decompression and conversion must not occupy an HTTP runtime thread.
        tokio::task::spawn_blocking(move || {
            let _upload = upload;
            let (images, permits) = decode_references(files, service.reference_memory.clone())?;
            service.submit_images(parameters, images, permits)
        })
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??
    } else {
        let Json(parameters) = Json::<Parameters>::from_request(request, &())
            .await
            .map_err(|e| ApiError(e.status(), e.body_text()))?;
        service.submit(parameters)?
    };
    let location = view["status_url"].as_str().unwrap().to_owned();
    Ok((
        StatusCode::ACCEPTED,
        [(header::LOCATION, location)],
        Json(view),
    )
        .into_response())
}
async fn list(State(service): State<Arc<Service>>) -> Json<serde_json::Value> {
    let store = service.store.lock().unwrap();
    Json(
        serde_json::json!({"jobs": store.jobs.values().map(|j| j.lock().unwrap().view()).collect::<Vec<_>>()}),
    )
}
async fn status(
    State(service): State<Arc<Service>>,
    Path(id): Path<u64>,
) -> ApiResult<Json<serde_json::Value>> {
    Ok(Json(service.job(id)?.lock().unwrap().view()))
}
async fn cancel(
    State(service): State<Arc<Service>>,
    Path(id): Path<u64>,
) -> ApiResult<Json<serde_json::Value>> {
    Ok(Json(service.cancel(id)?))
}
async fn delete(State(service): State<Arc<Service>>, Path(id): Path<u64>) -> ApiResult<StatusCode> {
    let mut store = service.store.lock().unwrap();
    let job = store
        .jobs
        .get(&id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "job not found".into()))?;
    if !job.lock().unwrap().status.terminal() {
        return Err(conflict(
            "cancel the job and wait for a terminal status before deleting",
        ));
    }
    store.jobs.remove(&id);
    Ok(StatusCode::NO_CONTENT)
}
async fn request_preview(
    State(service): State<Arc<Service>>,
    Path(id): Path<u64>,
) -> ApiResult<Response> {
    let job = service.job(id)?;
    let mut job = job.lock().unwrap();
    if job.status != Status::Running {
        return Err(conflict("job must be running to request a preview"));
    }
    let control = job
        .control
        .as_ref()
        .ok_or_else(|| conflict("job is not using manual previews"))?;
    if !control.request_preview() {
        return Err(conflict(
            "preview is already pending or sampling is not active",
        ));
    }
    job.preview_pending = true;
    Ok((StatusCode::ACCEPTED, Json(job.view())).into_response())
}
async fn preview(State(service): State<Arc<Service>>, Path(id): Path<u64>) -> ApiResult<Response> {
    let image = service
        .job(id)?
        .lock()
        .unwrap()
        .preview
        .clone()
        .ok_or_else(|| conflict("no preview available yet"))?;
    image.response().await
}
async fn image(State(service): State<Arc<Service>>, Path(id): Path<u64>) -> ApiResult<Response> {
    let image = service
        .job(id)?
        .lock()
        .unwrap()
        .output
        .clone()
        .ok_or_else(|| conflict("final image is not ready; check job status"))?;
    image.response().await
}
fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/jobs", post(submit).get(list))
        .route("/jobs/{id}", get(status).delete(delete))
        .route("/jobs/{id}/cancel", post(cancel))
        .route("/jobs/{id}/preview", post(request_preview).get(preview))
        .route("/jobs/{id}/image", get(image))
        .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "endpoint not found".into()) })
        .layer(DefaultBodyLimit::max(JSON_LIMIT))
        .layer(middleware::from_fn_with_state(service.clone(), guard))
        .with_state(service)
}

pub fn run(
    address: SocketAddr,
    max_jobs: usize,
    options: ModelOptions,
    http_debug: bool,
) -> anyhow::Result<()> {
    let token = std::env::var("IMAGEFORGER_API_TOKEN").ok();
    anyhow::ensure!(
        token.as_ref().is_none_or(|t| !t.trim().is_empty()),
        "IMAGEFORGER_API_TOKEN must not be empty"
    );
    anyhow::ensure!(
        address.ip().is_loopback() || token.is_some(),
        "set IMAGEFORGER_API_TOKEN to listen beyond localhost"
    );
    let authorization = token
        .map(|t| HeaderValue::from_str(&format!("Bearer {t}")))
        .transpose()
        .context("validating IMAGEFORGER_API_TOKEN as an HTTP header")?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("creating HTTP service runtime")?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(address).await.with_context(|| format!("binding HTTP listener to {address}"))?;
        let service = Service::with_checkpoint(max_jobs, authorization, options.checkpoint, http_debug);
        log("INFO", "http.listening", serde_json::json!({"address": listener.local_addr()?.to_string(), "model": options.checkpoint.name(), "offline": options.offline, "auth_required": service.authorization.is_some(), "max_jobs": max_jobs, "http_debug": http_debug}));
        let worker_service = service.clone();
        let thread = std::thread::Builder::new().name("generation".into()).spawn(move || worker(worker_service, options)).context("starting generation worker")?;
        let shutdown_service = service.clone();
        let result = axum::serve(listener, router(service.clone()).into_make_service_with_connect_info::<SocketAddr>()).with_graceful_shutdown(async move {
            if let Err(error) = tokio::signal::ctrl_c().await { log("ERROR", "http.shutdown_signal_failed", serde_json::json!({"error": error.to_string()})); }
            log("INFO", "http.shutdown_requested", serde_json::json!({}));
            shutdown_service.shutdown();
        }).await;
        service.shutdown();
        tokio::task::spawn_blocking(move || thread.join()).await?
            .map_err(|_| anyhow::anyhow!("generation thread panicked"))?;
        result.context("serving HTTP requests")?;
        log("INFO", "http.stopped", serde_json::json!({}));
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request as HttpRequest,
    };
    use tower::ServiceExt;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }
    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        body: &str,
    ) -> (StatusCode, serde_json::Value) {
        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method(method)
                    .uri(uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (
            status,
            serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null),
        )
    }

    #[test]
    fn api_validates_bounds_and_reclaims_cancelled_queue_slots() {
        runtime().block_on(async {
            let service = Service::new(1, None);
            let app = router(service.clone());
            for body in [
                r#"{"prompt":" "}"#,
                r#"{"prompt":"x","scale":0}"#,
                r#"{"prompt":"x","steps":0}"#,
                r#"{"prompt":"x","steps":1001}"#,
                r#"{"prompt":"x","preview_every":0}"#,
                r#"{"prompt":"x","ratio":"bad"}"#,
                r#"{"prompt":"x","width":512}"#,
                r#"{"prompt":"x","preview_mode":"bad"}"#,
            ] {
                assert!(call(&app, "POST", "/jobs", body).await.0.is_client_error());
            }
            assert_eq!(
                call(
                    &app,
                    "POST",
                    "/jobs",
                    &format!(r#"{{"prompt":"{}"}}"#, "x".repeat(70_000))
                )
                .await
                .0,
                StatusCode::PAYLOAD_TOO_LARGE
            );
            let (code, first) = call(
                &app,
                "POST",
                "/jobs",
                r#"{"prompt":"a teapot","scale":0.25}"#,
            )
            .await;
            assert_eq!(code, StatusCode::ACCEPTED);
            assert_eq!(first["width"], 512);
            assert_eq!(first["status"], "queued");
            assert_eq!(
                call(&app, "POST", "/jobs", r#"{"prompt":"second"}"#)
                    .await
                    .0,
                StatusCode::TOO_MANY_REQUESTS
            );
            assert_eq!(
                call(&app, "GET", "/jobs/1/image", "").await.0,
                StatusCode::CONFLICT
            );
            assert_eq!(
                call(&app, "POST", "/jobs/1/preview", "").await.0,
                StatusCode::CONFLICT
            );
            assert_eq!(
                call(&app, "DELETE", "/jobs/1", "").await.0,
                StatusCode::CONFLICT
            );
            assert_eq!(
                call(&app, "POST", "/jobs/1/cancel", "").await.1["status"],
                "cancelled"
            );
            assert!(service.store.lock().unwrap().queue.is_empty());
            assert_eq!(
                call(&app, "DELETE", "/jobs/1", "").await.0,
                StatusCode::NO_CONTENT
            );
            assert_eq!(
                call(&app, "GET", "/jobs/1", "").await.0,
                StatusCode::NOT_FOUND
            );
            assert_eq!(
                call(&app, "POST", "/jobs", r#"{"prompt":"second"}"#)
                    .await
                    .1["id"],
                "2"
            );
            service.shutdown();
            assert_eq!(
                call(&app, "POST", "/jobs", r#"{"prompt":"third"}"#).await.0,
                StatusCode::SERVICE_UNAVAILABLE
            );
            assert_eq!(
                call(&app, "GET", "/jobs/2", "").await.1["status"],
                "cancelled"
            );
        });
    }

    #[test]
    fn request_ids_and_error_details_cover_auth_validation_and_success() {
        runtime().block_on(async {
            let app = router(Service::new(
                2,
                Some(HeaderValue::from_static("Bearer test-secret")),
            ));
            let mut ids = std::collections::BTreeSet::new();
            for (method, path, token, expected) in [
                ("GET", "/health", "", StatusCode::UNAUTHORIZED),
                ("GET", "/health", "Bearer test-secret", StatusCode::OK),
                (
                    "GET",
                    "/missing",
                    "Bearer test-secret",
                    StatusCode::NOT_FOUND,
                ),
                (
                    "POST",
                    "/jobs",
                    "Bearer test-secret",
                    StatusCode::BAD_REQUEST,
                ),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        HttpRequest::builder()
                            .method(method)
                            .uri(path)
                            .header(header::AUTHORIZATION, token)
                            .header(header::CONTENT_TYPE, "application/json")
                            .body(Body::from("{}"))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), expected);
                assert!(
                    ids.insert(
                        response.headers()["x-request-id"]
                            .to_str()
                            .unwrap()
                            .to_owned()
                    )
                );
                assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
                if expected.is_client_error() {
                    assert!(response.extensions().get::<ErrorDetail>().is_some());
                }
            }
        });
    }

    #[test]
    fn bearer_auth_protects_submission_status_and_images() {
        runtime().block_on(async {
            let app = router(Service::new(
                2,
                Some(HeaderValue::from_static("Bearer test-secret")),
            ));
            assert_eq!(
                multipart(
                    &app,
                    &[("parameters", Bytes::from_static(br#"{"prompt":"test"}"#))]
                )
                .await
                .0,
                StatusCode::UNAUTHORIZED
            );
            for path in [
                "/health",
                "/jobs",
                "/jobs/1",
                "/jobs/1/preview",
                "/jobs/1/image",
            ] {
                assert_eq!(
                    call(&app, "GET", path, "").await.0,
                    StatusCode::UNAUTHORIZED
                );
            }
            for (token, expected) in [
                ("Bearer wrong", StatusCode::UNAUTHORIZED),
                ("Bearer test-secret", StatusCode::OK),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        HttpRequest::builder()
                            .uri("/health")
                            .header(header::AUTHORIZATION, token)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), expected);
            }
        });
    }

    #[test]
    fn progress_and_pngs_survive_late_previews_without_regressing_steps() {
        runtime().block_on(async {
            let service = Service::new(2, None);
            service
                .submit(Parameters {
                    prompt: "test".into(),
                    steps: 3,
                    ..Default::default()
                })
                .unwrap();
            let (job, _) = service.next().unwrap();
            let pixels = Arc::new(RgbaImage::from_pixel(32, 32, image::Rgba([12, 34, 56, 78])));
            {
                let mut job = job.lock().unwrap();
                job.event(Event::StepFinished {
                    step: 2,
                    total: 3,
                    duration: std::time::Duration::from_secs(1),
                });
                job.event(Event::Preview {
                    step: 1,
                    total: 3,
                    image: pixels.clone(),
                });
                assert_eq!(job.completed_steps, 2);
                assert_eq!(job.stage, "sampling");
                assert_eq!(job.view()["preview_step"], 1);
            }
            let app = router(service.clone());
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::builder()
                        .uri("/jobs/1/preview")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.headers()["x-preview-step"], "1");
            assert_eq!(response.headers()[header::CONTENT_TYPE], "image/png");
            let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            assert_eq!(image::load_from_memory(&body).unwrap().to_rgba8(), *pixels);
            {
                let mut job = job.lock().unwrap();
                job.event(Event::StepFinished {
                    step: 3,
                    total: 3,
                    duration: std::time::Duration::from_secs(1),
                });
                job.event(Event::Preview {
                    step: 3,
                    total: 3,
                    image: pixels.clone(),
                });
                job.finish(Ok(image_forger::Generation {
                    image: pixels.clone(),
                    elapsed: std::time::Duration::from_secs(3),
                }));
                assert!(Arc::ptr_eq(
                    job.preview.as_ref().unwrap(),
                    job.output.as_ref().unwrap()
                ));
            }
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::builder()
                        .uri("/jobs/1/image")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["x-preview-step"], "3");
            let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            assert_eq!(image::load_from_memory(&body).unwrap().to_rgba8(), *pixels);
            assert_eq!(
                call(&app, "POST", "/jobs/1/cancel", "").await.1["status"],
                "succeeded"
            );
            assert_eq!(
                call(&app, "DELETE", "/jobs/1", "").await.0,
                StatusCode::NO_CONTENT
            );
        });
    }

    #[test]
    fn fifo_failure_cancellation_and_preview_configuration() {
        let service = Service::new(4, None);
        for mode in [PreviewMode::Manual, PreviewMode::Auto, PreviewMode::Off] {
            service
                .submit(Parameters {
                    prompt: "test".into(),
                    preview_mode: mode,
                    ..Default::default()
                })
                .unwrap();
        }
        let (first, request) = service.next().unwrap();
        assert!(request.preview_control.is_some());
        assert!(request.preview_every.is_none());
        assert_eq!(first.lock().unwrap().id, 1);
        first
            .lock()
            .unwrap()
            .finish(Err(anyhow::anyhow!("test failure")));
        assert_eq!(first.lock().unwrap().view()["error"], "test failure");
        let (second, request) = service.next().unwrap();
        assert_eq!(second.lock().unwrap().id, 2);
        assert_eq!(request.preview_every.unwrap().get(), 5);
        assert!(request.preview_control.is_none());
        assert_eq!(service.cancel(2).unwrap()["status"], "cancelling");
        assert!(second.lock().unwrap().cancel.is_cancelled());
        second
            .lock()
            .unwrap()
            .finish(Err(image_forger::Cancelled.into()));
        assert_eq!(second.lock().unwrap().status, Status::Cancelled);
        let (third, request) = service.next().unwrap();
        assert_eq!(third.lock().unwrap().id, 3);
        assert!(request.preview_control.is_none() && request.preview_every.is_none());
        service.shutdown();
        assert!(third.lock().unwrap().cancel.is_cancelled());
        assert!(service.next().is_none());
    }
    fn encoded(pixels: &RgbaImage, format: image::ImageFormat) -> Bytes {
        let mut out = Cursor::new(Vec::new());
        if format == image::ImageFormat::Jpeg {
            image::DynamicImage::ImageRgba8(pixels.clone())
                .to_rgb8()
                .write_to(&mut out, format)
                .unwrap();
        } else {
            pixels.write_to(&mut out, format).unwrap();
        }
        Bytes::from(out.into_inner())
    }

    async fn multipart(app: &Router, fields: &[(&str, Bytes)]) -> (StatusCode, serde_json::Value) {
        let mut body = Vec::new();
        for (name, data) in fields {
            body.extend_from_slice(
                format!(
                    "--test-boundary\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--test-boundary--\r\n");
        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/jobs")
                    .header(
                        header::CONTENT_TYPE,
                        "multipart/form-data; boundary=test-boundary",
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[test]
    fn multipart_preserves_reference_order_alpha_and_last_image_aspect() {
        runtime().block_on(async {
            let service = Service::new(3, None);
            let app = router(service.clone());
            let landscape = RgbaImage::from_pixel(48, 32, image::Rgba([12, 34, 56, 78]));
            let portrait = RgbaImage::from_pixel(32, 48, image::Rgba([90, 80, 70, 60]));
            let first = encoded(&landscape, image::ImageFormat::Png);
            let second = encoded(&portrait, image::ImageFormat::WebP);
            // Field order is unrestricted; image order has model semantics.
            let fields = [
                ("images", first),
                (
                    "parameters",
                    Bytes::from_static(br#"{"prompt":"edit","scale":0.25}"#),
                ),
                ("images", second),
            ];
            let (code, view) = multipart(&app, &fields).await;
            assert_eq!(code, StatusCode::ACCEPTED, "{view}");
            assert_eq!(
                view["references"],
                serde_json::json!([{"width":48,"height":32},{"width":32,"height":48}])
            );
            assert!(view["height"].as_u64().unwrap() > view["width"].as_u64().unwrap());
            let total = REFERENCE_MEMORY_LIMIT / REFERENCE_MEMORY_UNIT;
            assert_eq!(service.reference_memory.available_permits(), total - 2);
            let (job, request) = service.next().unwrap();
            assert_eq!(request.images, [landscape, portrait]);
            assert_eq!(
                request.dimensions().unwrap(),
                (
                    view["width"].as_u64().unwrap() as u32,
                    view["height"].as_u64().unwrap() as u32
                )
            );
            job.lock().unwrap().event(Event::Progress {
                stage: Stage::ReferenceVision { index: 1 },
                completed: 1,
                total: 27,
            });
            assert_eq!(job.lock().unwrap().view()["reference_index"], 1);
            job.lock().unwrap().event(Event::Progress {
                stage: Stage::TextEncoding,
                completed: 1,
                total: 36,
            });
            assert!(job.lock().unwrap().view()["reference_index"].is_null());
            service.cancel(1).unwrap();
            job.lock()
                .unwrap()
                .finish(Err(image_forger::Cancelled.into()));
            drop(request);
            assert_eq!(service.reference_memory.available_permits(), total);

            let mut fields = fields.clone();
            fields[1].1 = Bytes::from_static(br#"{"prompt":"edit","scale":0.25,"ratio":"1:1"}"#);
            let (code, view) = multipart(&app, &fields).await;
            assert_eq!(code, StatusCode::ACCEPTED);
            assert_eq!(view["width"], 512);
            assert_eq!(view["height"], 512);
            service.cancel(2).unwrap();
            assert_eq!(service.reference_memory.available_permits(), total);
            assert!(service.store.lock().unwrap().queue.is_empty());
        });
    }

    #[test]
    fn multipart_accepts_jpeg_and_heic_and_applies_orientation() {
        runtime().block_on(async {
            let service = Service::new(1, None);
            let app = router(service.clone());
            let fields = [
                (
                    "parameters",
                    Bytes::from_static(br#"{"prompt":"edit","scale":0.25}"#),
                ),
                (
                    "images",
                    Bytes::from_static(include_bytes!("../tests/fixtures/oriented.jpg")),
                ),
                (
                    "images",
                    Bytes::from_static(include_bytes!("../tests/fixtures/oriented.heic")),
                ),
            ];
            let (code, view) = multipart(&app, &fields).await;
            assert_eq!(code, StatusCode::ACCEPTED, "{view}");
            assert_eq!(
                view["references"],
                serde_json::json!([
                    {"width": 32, "height": 64}, {"width": 32, "height": 64}
                ])
            );
            let (_, request) = service.next().unwrap();
            assert_eq!(request.images.len(), 2);
            assert!(request.dimensions().unwrap().1 > request.dimensions().unwrap().0);
        });
        let budget = Arc::new(Semaphore::new(0));
        assert_eq!(
            decode_references(
                vec![Bytes::from_static(include_bytes!(
                    "../tests/fixtures/oriented.heic"
                ))],
                budget,
            )
            .unwrap_err()
            .0,
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn multipart_rejects_bad_forms_files_and_limits_without_queueing() {
        runtime().block_on(async {
            let service = Service::new(1, None);
            let app = router(service.clone());
            let params = ("parameters", Bytes::from_static(br#"{"prompt":"edit"}"#));
            let pixel = encoded(&RgbaImage::new(1, 1), image::ImageFormat::Png);
            for (fields, expected) in [
                (vec![("images", pixel.clone())], StatusCode::BAD_REQUEST),
                (
                    vec![params.clone(), params.clone()],
                    StatusCode::BAD_REQUEST,
                ),
                (
                    vec![params.clone(), ("image", pixel.clone())],
                    StatusCode::BAD_REQUEST,
                ),
                (
                    vec![("parameters", Bytes::from_static(b"invalid"))],
                    StatusCode::UNPROCESSABLE_ENTITY,
                ),
                (
                    vec![params.clone(), ("images", Bytes::from_static(b"GIF89a"))],
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                ),
                (
                    vec![
                        params.clone(),
                        ("images", Bytes::from_static(b"\x89PNG\r\n\x1a\n")),
                    ],
                    StatusCode::BAD_REQUEST,
                ),
                (
                    vec![("parameters", Bytes::from(vec![b' '; JSON_LIMIT + 1]))],
                    StatusCode::PAYLOAD_TOO_LARGE,
                ),
                (
                    std::iter::once(params.clone())
                        .chain((0..11).map(|_| ("images", pixel.clone())))
                        .collect(),
                    StatusCode::BAD_REQUEST,
                ),
                (
                    vec![
                        params.clone(),
                        (
                            "images",
                            encoded(&RgbaImage::new(8193, 1), image::ImageFormat::Png),
                        ),
                    ],
                    StatusCode::PAYLOAD_TOO_LARGE,
                ),
                (
                    vec![
                        params.clone(),
                        ("images", Bytes::from(vec![0; UPLOAD_LIMIT])),
                    ],
                    StatusCode::PAYLOAD_TOO_LARGE,
                ),
            ] {
                let (code, view) = multipart(&app, &fields).await;
                assert_eq!(code, expected, "{view}");
                assert!(view["error"].is_string());
                assert!(service.store.lock().unwrap().queue.is_empty());
                assert_eq!(
                    service.reference_memory.available_permits(),
                    REFERENCE_MEMORY_LIMIT / REFERENCE_MEMORY_UNIT
                );
                assert_eq!(service.uploads.available_permits(), 2);
            }
        });
    }

    #[test]
    fn upload_budget_and_failed_admission_release_reference_buffers() {
        runtime().block_on(async {
            let service = Service::new(1, None);
            let app = router(service.clone());
            let fields = [
                ("parameters", Bytes::from_static(br#"{"prompt":"edit"}"#)),
                (
                    "images",
                    encoded(&RgbaImage::new(32, 32), image::ImageFormat::Jpeg),
                ),
            ];
            let all_memory = service
                .reference_memory
                .clone()
                .try_acquire_many_owned((REFERENCE_MEMORY_LIMIT / REFERENCE_MEMORY_UNIT) as u32)
                .unwrap();
            assert_eq!(
                multipart(&app, &fields).await.0,
                StatusCode::TOO_MANY_REQUESTS
            );
            drop(all_memory);
            let upload_slots = service.uploads.clone().try_acquire_many_owned(2).unwrap();
            assert_eq!(
                multipart(&app, &fields).await.0,
                StatusCode::TOO_MANY_REQUESTS
            );
            drop(upload_slots);
            assert_eq!(multipart(&app, &fields).await.0, StatusCode::ACCEPTED);
            let available = service.reference_memory.available_permits();
            assert_eq!(
                multipart(&app, &fields).await.0,
                StatusCode::TOO_MANY_REQUESTS
            );
            assert_eq!(service.reference_memory.available_permits(), available);
            service.shutdown();
            assert_eq!(
                service.reference_memory.available_permits(),
                REFERENCE_MEMORY_LIMIT / REFERENCE_MEMORY_UNIT
            );
            assert_eq!(
                multipart(&app, &fields).await.0,
                StatusCode::SERVICE_UNAVAILABLE
            );
            assert_eq!(
                service.reference_memory.available_permits(),
                REFERENCE_MEMORY_LIMIT / REFERENCE_MEMORY_UNIT
            );
        });
        let budget = Arc::new(Semaphore::new(1));
        let pixel = encoded(&RgbaImage::new(1, 1), image::ImageFormat::Png);
        assert_eq!(
            decode_references(vec![pixel.clone(), pixel], budget.clone())
                .unwrap_err()
                .0,
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(budget.available_permits(), 1);
    }
}
