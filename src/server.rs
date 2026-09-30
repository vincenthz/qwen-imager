//! HTTP transport for the existing blocking generator. GPU work stays on one
//! persistent thread; HTTP handlers only touch small job metadata/CPU images.
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use qwen_imager::{
    CancellationToken, Event, Generator, ModelOptions, PreviewControl, Request, RgbaImage, Stage,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    io::Cursor,
    net::SocketAddr,
    num::NonZeroUsize,
    sync::{Arc, Condvar, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::OnceCell;

type ApiResult<T> = Result<T, ApiError>;
#[derive(Debug)]
struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({"error": self.1}))).into_response()
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
    fn request(&self) -> anyhow::Result<Request> {
        anyhow::ensure!(
            self.prompt.len() <= 16_384,
            "prompt must be at most 16384 UTF-8 bytes"
        );
        anyhow::ensure!(self.steps <= 1000, "steps must be at most 1000");
        anyhow::ensure!(self.preview_every > 0, "preview_every must be positive");
        let mut request = Request::new(&self.prompt);
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
    width: u32,
    height: u32,
    created_ms: u64,
    status: Status,
    stage: &'static str,
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
            "width": self.width, "height": self.height, "created_unix_ms": self.created_ms,
            "stage": self.stage, "stage_completed": self.stage_completed, "stage_total": self.stage_total,
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
    fn finish(&mut self, result: anyhow::Result<qwen_imager::Generation>) {
        self.preview_pending = false;
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
                if error.is::<qwen_imager::Cancelled>() {
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
    queue: VecDeque<(u64, Request)>,
    next_id: u64,
    stopping: bool,
}
struct Service {
    store: Mutex<Store>,
    wake: Condvar,
    max_jobs: usize,
    authorization: Option<HeaderValue>,
}
impl Service {
    fn new(max_jobs: usize, authorization: Option<HeaderValue>) -> Arc<Self> {
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
        let request = parameters
            .request()
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
            width,
            height,
            created_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            status: Status::Queued,
            stage: "queued",
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
        store.queue.push_back((id, request));
        self.wake.notify_one();
        Ok(view)
    }
    fn next(&self) -> Option<(SharedJob, Request)> {
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
        let cancel = job.lock().unwrap().cancel.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generator.generate(&request, &cancel, |event| job.lock().unwrap().event(event))
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
        job.lock().unwrap().finish(result);
    }
}

async fn guard(
    State(service): State<Arc<Service>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    if let Some(expected) = &service.authorization
        && request.headers().get(header::AUTHORIZATION) != Some(expected)
    {
        return ApiError(
            StatusCode::UNAUTHORIZED,
            "a valid Bearer token is required".into(),
        )
        .into_response();
    }
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}
async fn health(State(service): State<Arc<Service>>) -> Json<serde_json::Value> {
    let store = service.store.lock().unwrap();
    Json(
        serde_json::json!({"status": if store.stopping { "stopping" } else { "ok" },
        "model": qwen_imager::MODEL, "retained_jobs": store.jobs.len(), "queued_jobs": store.queue.len(), "max_jobs": service.max_jobs}),
    )
}
async fn submit(
    State(service): State<Arc<Service>>,
    body: Result<Json<Parameters>, JsonRejection>,
) -> ApiResult<Response> {
    let Json(parameters) = body.map_err(|e| ApiError(e.status(), e.body_text()))?;
    let view = service.submit(parameters)?;
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
        .layer(DefaultBodyLimit::max(64 * 1024))
        .layer(middleware::from_fn_with_state(service.clone(), guard))
        .with_state(service)
}

pub fn run(address: SocketAddr, max_jobs: usize, options: ModelOptions) -> anyhow::Result<()> {
    let token = std::env::var("QWEN_IMAGER_API_TOKEN").ok();
    anyhow::ensure!(
        token.as_ref().is_none_or(|t| !t.trim().is_empty()),
        "QWEN_IMAGER_API_TOKEN must not be empty"
    );
    anyhow::ensure!(
        address.ip().is_loopback() || token.is_some(),
        "set QWEN_IMAGER_API_TOKEN to listen beyond localhost"
    );
    let authorization = token
        .map(|t| HeaderValue::from_str(&format!("Bearer {t}")))
        .transpose()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(address).await?;
        let service = Service::new(max_jobs, authorization);
        let worker_service = service.clone();
        let thread = std::thread::Builder::new().name("generation".into()).spawn(move || worker(worker_service, options))?;
        eprintln!("Qwen HTTP service listening on http://{} (one generation at a time, {max_jobs} retained jobs)", listener.local_addr()?);
        let shutdown_service = service.clone();
        let result = axum::serve(listener, router(service.clone())).with_graceful_shutdown(async move {
            if let Err(error) = tokio::signal::ctrl_c().await { eprintln!("shutdown signal error: {error}"); }
            shutdown_service.shutdown();
        }).await;
        service.shutdown();
        tokio::task::spawn_blocking(move || thread.join()).await?
            .map_err(|_| anyhow::anyhow!("generation thread panicked"))?;
        result?;
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
    fn bearer_auth_protects_submission_status_and_images() {
        runtime().block_on(async {
            let app = router(Service::new(
                2,
                Some(HeaderValue::from_static("Bearer test-secret")),
            ));
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
                job.finish(Ok(qwen_imager::Generation {
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
            .finish(Err(qwen_imager::Cancelled.into()));
        assert_eq!(second.lock().unwrap().status, Status::Cancelled);
        let (third, request) = service.next().unwrap();
        assert_eq!(third.lock().unwrap().id, 3);
        assert!(request.preview_control.is_none() && request.preview_every.is_none());
        service.shutdown();
        assert!(third.lock().unwrap().cancel.is_cancelled());
        assert!(service.next().is_none());
    }
}
