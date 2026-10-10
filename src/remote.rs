//! HTTP client for delegating generation to a remote `image-forger` server.
//!
//! Mirrors the local pipeline's event stream: progress, step completion and
//! previews are polled from the server's `/jobs` API and forwarded as
//! [`Event`]s. Blocking calls run on the caller's worker thread.

use std::{
    io::Read,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, ensure};
use image::RgbaImage;
use std::sync::Arc;

use crate::content_crypto::{self, Identity, Session};
use crate::diagnostics::{endpoint, log, redact};
use crate::{CancellationToken, Event, Generation, Request, Stage};

const BOUNDARY: &str = "----image-forger-boundary";
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// An authenticated connection. Never derive Debug: it holds a secret.
struct Client {
    base: String,
    authorization: Option<String>,
    agent: ureq::Agent,
    crypto: Option<Session>,
}

impl Client {
    fn new(base_url: &str, token: Option<&str>) -> Result<Self> {
        let base = base_url.trim().trim_end_matches('/').to_owned();
        ensure!(
            base.starts_with("http://") || base.starts_with("https://"),
            "Use an HTTP or HTTPS server URL"
        );
        let token = token.map(str::trim).filter(|token| !token.is_empty());
        ensure!(
            token.is_none_or(|token| token.bytes().all(|byte| byte.is_ascii_graphic())),
            "API token must contain only visible ASCII characters, without spaces or line breaks"
        );
        Ok(Self {
            base,
            crypto: None,
            authorization: token.map(|token| format!("Bearer {token}")),
            // Do not forward credentials to a redirect destination.
            agent: ureq::AgentBuilder::new()
                .redirects(0)
                .timeout_connect(Duration::from_secs(10))
                .timeout_read(Duration::from_secs(60))
                .timeout_write(Duration::from_secs(60))
                .build(),
        })
    }

    fn request(&self, method: &str, path: &str) -> ureq::Request {
        let request = self.agent.request(method, &format!("{}{path}", self.base));
        if let Some(authorization) = &self.authorization {
            request.set("Authorization", authorization)
        } else {
            request
        }
    }

    fn safe(&self, message: &str) -> String {
        let authorization = self.authorization.as_deref().unwrap_or("");
        let message = message.replace(&self.base, &endpoint(&self.base));
        redact(
            &message,
            &[
                authorization,
                authorization.strip_prefix("Bearer ").unwrap_or(""),
            ],
        )
    }

    fn response(
        &self,
        method: &str,
        path: &str,
        result: Result<ureq::Response, ureq::Error>,
    ) -> Result<ureq::Response> {
        let operation = format!("{method} {}", endpoint(&format!("{}{path}", self.base)));
        match result {
            Err(ureq::Error::Status(code, response)) => {
                let request_id = response
                    .header("x-request-id")
                    .unwrap_or("unavailable")
                    .to_owned();
                let reason = if code == 401 || code == 403 {
                    "Authentication rejected. Set this host's API token in Settings → Compute to match IMAGEFORGER_API_TOKEN on the server.".into()
                } else {
                    let mut body = String::new();
                    let _ = response.into_reader().take(8192).read_to_string(&mut body);
                    serde_json::from_str::<serde_json::Value>(&body)
                        .ok()
                        .and_then(|body| body["error"].as_str().map(str::to_owned))
                        .unwrap_or_else(|| {
                            if body.trim().is_empty() {
                                "The server returned no error description".into()
                            } else {
                                body
                            }
                        })
                };
                anyhow::bail!(
                    "{}",
                    self.safe(&format!(
                        "{operation} failed: HTTP {code} (request ID {request_id})
{reason}"
                    ))
                )
            }
            Err(ureq::Error::Transport(error)) => {
                let mut causes = format!(
                    "{operation} failed: {:?}: {}",
                    error.kind(),
                    error.message().unwrap_or("HTTP transport error")
                );
                let mut source = std::error::Error::source(&error);
                while let Some(cause) = source {
                    causes.push_str(&format!(
                        "
Caused by: {cause}"
                    ));
                    source = cause.source();
                }
                anyhow::bail!("{}", self.safe(&causes))
            }
            Ok(response) => {
                ensure!(
                    !(300..400).contains(&response.status()),
                    "{operation}: HTTP {} redirect. Configure the final server URL in Settings → Compute.",
                    response.status()
                );
                Ok(response)
            }
        }
    }

    fn get(&self, path: &str) -> Result<ureq::Response> {
        self.response("GET", path, self.request("GET", path).call())
    }

    fn fetch_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let response = self.get(path)?;
        ensure!(
            response.header("Content-Type") == Some(content_crypto::CONTENT_TYPE),
            "Server returned unencrypted image content"
        );
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .with_context(|| {
                format!(
                    "reading image bytes from GET {}",
                    endpoint(&format!("{}{path}", self.base))
                )
            })?;
        self.crypto
            .as_ref()
            .context("Missing content encryption identity")?
            .open_response(&format!("GET {path}"), &bytes)
    }
}

/// Check connectivity and authentication without loading a model or submitting a job.
/// Returns the checkpoint reported by the service.
pub fn test_connection(base_url: &str, token: Option<&str>) -> Result<String> {
    let client = Client::new(base_url, token)?;
    let health: serde_json::Value = client
        .get("/health")?
        .into_json()
        .with_context(|| format!("decoding GET {}/health response", endpoint(base_url)))?;
    ensure!(
        health["status"] == "ok",
        "GET {}/health: server is not ready to accept jobs (status: {})",
        endpoint(base_url),
        health["status"]
    );
    Ok(health["model"]
        .as_str()
        .with_context(|| {
            format!(
                "GET {}/health: response is missing the ImageForger model field",
                endpoint(base_url)
            )
        })?
        .into())
}

/// Verify possession of the pinned private key with an encrypted challenge.
pub fn test_pinned_connection(
    base_url: &str,
    token: Option<&str>,
    pinned_identity: &str,
) -> Result<String> {
    let client = Client::new(base_url, token)?;
    let session = Identity::generate()?.session(content_crypto::public_key(pinned_identity)?)?;
    let body = session.seal_request("POST /identity", b"identity")?;
    let response = client.response(
        "POST",
        "/identity",
        client
            .request("POST", "/identity")
            .set("Content-Type", content_crypto::CONTENT_TYPE)
            .send_bytes(&body),
    )?;
    ensure!(
        response.header("Content-Type") == Some(content_crypto::CONTENT_TYPE),
        "Server did not verify its pinned identity"
    );
    let mut bytes = Vec::new();
    response.into_reader().take(65536).read_to_end(&mut bytes)?;
    let plaintext = session.open_response("POST /identity", &bytes)?;
    let view: serde_json::Value = serde_json::from_slice(&plaintext)?;
    Ok(view["model"]
        .as_str()
        .context("Identity response missing model")?
        .into())
}

/// Run one generation against the remote server at `base_url` (e.g.
/// `http://host:6996`). Emits [`Event::Started`], [`Event::Progress`],
/// [`Event::StepFinished`] and [`Event::Preview`] as they arrive.
pub fn run(
    request: &Request,
    base_url: &str,
    pinned_identity: &str,
    cancellation: &CancellationToken,
    on_event: impl FnMut(Event),
) -> Result<Generation> {
    run_authenticated(
        request,
        base_url,
        None,
        pinned_identity,
        cancellation,
        on_event,
    )
}

/// Like [`run`], using the host's optional Bearer token on every HTTP request.
pub fn run_authenticated(
    request: &Request,
    base_url: &str,
    token: Option<&str>,
    pinned_identity: &str,
    cancellation: &CancellationToken,
    mut on_event: impl FnMut(Event),
) -> Result<Generation> {
    let mut client = Client::new(base_url, token)?;
    client.crypto =
        Some(Identity::generate()?.session(content_crypto::public_key(pinned_identity)?)?);
    let started = Instant::now();
    let image = run_inner(request, &client, cancellation, &mut on_event)?;
    let elapsed = started.elapsed();
    Ok(Generation { image, elapsed })
}

fn run_inner(
    request: &Request,
    client: &Client,
    cancellation: &CancellationToken,
    on_event: &mut impl FnMut(Event),
) -> Result<Arc<RgbaImage>> {
    cancellation.check()?;
    let (preview_mode, preview_every) = match request.preview_every {
        Some(every) => ("auto", every.get()),
        None if request.preview_control.is_some() => ("auto", 5),
        None => ("off", 5),
    };

    let parameters = serde_json::json!({
        "prompt": request.prompt,
        "ratio": request.ratio,
        "scale": request.scale,
        "steps": request.steps,
        "seed": request.seed,
        "noise_source_size": request.noise_source_size,
        "preview_mode": preview_mode,
        "preview_every": preview_every,
    });

    let body = multipart(&parameters, &request.images)?;
    let body = client
        .crypto
        .as_ref()
        .context("Missing pinned server identity")?
        .seal_request("POST /jobs", &body)?;
    let response = client.response(
        "POST",
        "/jobs",
        client
            .request("POST", "/jobs")
            .set("Content-Type", content_crypto::CONTENT_TYPE)
            .send_bytes(&body),
    )?;

    let view: serde_json::Value = response
        .into_json()
        .with_context(|| format!("decoding POST {}/jobs response", endpoint(&client.base)))?;
    let id = view["id"]
        .as_str()
        .context("job response is missing an id")?
        .to_owned();
    let width = view["width"].as_u64().unwrap_or(0) as u32;
    let height = view["height"].as_u64().unwrap_or(0) as u32;
    // The server's checkpoint decides the step count; Turbo ignores the request's.
    let total_steps = view["total_steps"]
        .as_u64()
        .map_or(request.steps, |steps| steps as usize);

    on_event(Event::Started {
        width,
        height,
        steps: total_steps,
        seed: request.seed,
    });

    let mut completed_steps = 0usize;
    let mut preview_step: Option<usize> = None;

    loop {
        cancellation.check()?;
        let status: serde_json::Value = client
            .get(&format!("/jobs/{id}"))?
            .into_json()
            .with_context(|| format!("decoding GET {}/jobs/{id} status", endpoint(&client.base)))?;

        if let (Some(stage), Some(completed), Some(total)) = (
            status["stage"].as_str(),
            status["stage_completed"].as_u64(),
            status["stage_total"].as_u64(),
        ) {
            if let Some(stage) = map_stage(stage, status["reference_index"].as_u64()) {
                on_event(Event::Progress {
                    stage,
                    completed: completed as usize,
                    total: total as usize,
                });
            }
        }

        let steps = status["completed_steps"].as_u64().unwrap_or(0) as usize;
        if steps > completed_steps {
            let duration = Duration::from_secs_f64(status["last_step_s"].as_f64().unwrap_or(0.0));
            for step in completed_steps + 1..=steps {
                on_event(Event::StepFinished {
                    step,
                    total: total_steps,
                    duration,
                });
            }
            completed_steps = steps;
        }

        let current_preview = status["preview_step"].as_u64().map(|s| s as usize);
        if current_preview != preview_step {
            if let Some(step) = current_preview {
                match client
                    .fetch_bytes(&format!("/jobs/{id}/preview"))
                    .and_then(|bytes| decode_png(&bytes))
                {
                    Ok(image) => on_event(Event::Preview {
                        step,
                        total: total_steps,
                        image,
                    }),
                    Err(error) => log(
                        "WARN",
                        "remote.preview_failed",
                        serde_json::json!({"job_id": id, "step": step, "error": client.safe(&format!("{error:#}"))}),
                    ),
                }
            }
            preview_step = current_preview;
        }

        match status["status"].as_str() {
            Some("succeeded") => {
                let bytes = client.fetch_bytes(&format!("/jobs/{id}/image"))?;
                return decode_png(&bytes).with_context(|| {
                    format!(
                        "decoding final image for remote job {id} from {}",
                        endpoint(&client.base)
                    )
                });
            }
            Some("failed") => {
                anyhow::bail!(
                    "{}",
                    status["error"]
                        .as_str()
                        .unwrap_or("remote generation failed")
                );
            }
            Some("cancelled") => return Err(crate::Cancelled.into()),
            _ => {}
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

fn multipart(parameters: &serde_json::Value, images: &[RgbaImage]) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let parameters = serde_json::to_vec(parameters)?;
    body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"parameters\"\r\n");
    body.extend_from_slice(b"Content-Type: application/json\r\n\r\n");
    body.extend_from_slice(&parameters);
    body.extend_from_slice(b"\r\n");
    for (index, image) in images.iter().enumerate() {
        let png = encode_png(image)?;
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"images\"; filename=\"image{index}.png\"\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
        body.extend_from_slice(&png);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    Ok(body)
}

fn encode_png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    image.write_to(&mut cursor, image::ImageFormat::Png)?;
    Ok(cursor.into_inner())
}

fn decode_png(bytes: &[u8]) -> Result<Arc<RgbaImage>> {
    let image = image::load_from_memory(bytes)
        .context("decoding server image")?
        .to_rgba8();
    Ok(Arc::new(image))
}

fn map_stage(stage: &str, reference_index: Option<u64>) -> Option<Stage> {
    match stage {
        "loading" => Some(Stage::Loading),
        "reference_vision" => Some(Stage::ReferenceVision {
            index: reference_index.unwrap_or(0) as usize,
        }),
        "text_encoding" => Some(Stage::TextEncoding),
        "reference_encoding" => Some(Stage::ReferenceEncoding),
        "denoiser_loading" => Some(Stage::DenoiserLoading),
        "decoding" => Some(Stage::Decoding),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::TcpListener, thread};

    struct Reply {
        method: &'static str,
        path: &'static str,
        token: Option<&'static str>,
        status: u16,
        body: Vec<u8>,
    }

    /// A small real HTTP fixture: verifies headers on each request, without a
    /// GPU, model files, or calls to an external service.
    fn server(replies: Vec<Reply>) -> (String, thread::JoinHandle<()>) {
        server_with_identity(replies, None)
    }
    fn server_with_identity(
        replies: Vec<Reply>,
        identity: Option<Identity>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let worker = thread::spawn(move || {
            let mut crypto = None;
            for mut reply in replies {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                Instant::now() < deadline,
                                "client did not send the expected request"
                            );
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("accept: {error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let header_end = loop {
                    let mut buffer = [0u8; 4096];
                    let n = stream.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                    if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
                assert_eq!(
                    headers.lines().next().unwrap(),
                    format!("{} {} HTTP/1.1", reply.method, reply.path)
                );
                let authorization = headers.lines().find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("authorization")
                        .then(|| value.trim().to_owned())
                });
                assert_eq!(
                    authorization,
                    reply.token.map(|token| format!("Bearer {token}"))
                );
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                while bytes.len() < header_end + length {
                    let mut buffer = [0u8; 4096];
                    let n = stream.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                }
                if let Some(identity) = &identity {
                    if reply.method == "POST" {
                        let wire = &bytes[header_end..header_end + length];
                        assert!(
                            !wire
                                .windows(b"private-test-prompt".len())
                                .any(|w| w == b"private-test-prompt")
                        );
                        let (session, plain) = identity.open_request(wire, "POST /jobs").unwrap();
                        assert!(
                            plain
                                .windows(b"private-test-prompt".len())
                                .any(|w| w == b"private-test-prompt")
                        );
                        crypto = Some(session);
                    } else if reply.path.ends_with("/image") || reply.path.ends_with("/preview") {
                        reply.body = crypto
                            .as_ref()
                            .unwrap()
                            .seal_response(&format!("GET {}", reply.path), &reply.body)
                            .unwrap();
                    }
                }
                write!(
                    stream,
                    "HTTP/1.1 {} Test\r\nContent-Type: application/vnd.imageforger.encrypted-v1\r\nContent-Length: {}\r\nX-Request-ID: fixture-42\r\nConnection: close\r\n\r\n",
                    reply.status,
                    reply.body.len()
                )
                .unwrap();
                stream.write_all(&reply.body).unwrap();
            }
        });
        (url, worker)
    }

    #[test]
    fn authenticated_generation_covers_submission_status_preview_and_image() {
        let png = encode_png(&RgbaImage::new(2, 2)).unwrap();
        let replies = vec![
            Reply {
                method: "POST",
                path: "/jobs",
                token: Some("test-secret"),
                status: 202,
                body: br#"{"id":"1","width":2,"height":2}"#.to_vec(),
            },
            Reply {
                method: "GET",
                path: "/jobs/1",
                token: Some("test-secret"),
                status: 200,
                body: br#"{"status":"succeeded","completed_steps":1,"preview_step":1}"#.to_vec(),
            },
            Reply {
                method: "GET",
                path: "/jobs/1/preview",
                token: Some("test-secret"),
                status: 200,
                body: png.clone(),
            },
            Reply {
                method: "GET",
                path: "/jobs/1/image",
                token: Some("test-secret"),
                status: 200,
                body: png,
            },
        ];
        let identity = Identity::generate().unwrap();
        let pin = identity.public_hex();
        let (url, worker) = server_with_identity(replies, Some(identity));
        let mut request = Request::new("private-test-prompt");
        request.steps = 1;
        let mut saw_preview = false;
        let result = run_authenticated(
            &request,
            &url,
            Some("test-secret"),
            &pin,
            &CancellationToken::default(),
            |event| {
                if matches!(event, Event::Preview { .. }) {
                    saw_preview = true;
                }
            },
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(result.image.dimensions(), (2, 2));
        assert!(saw_preview);
    }

    #[test]
    fn connection_check_uses_optional_authentication_without_submitting_jobs() {
        for token in [None, Some("test-secret")] {
            let (url, worker) = server(vec![Reply {
                method: "GET",
                path: "/health",
                token,
                status: 200,
                body: br#"{"status":"ok","model":"test-model"}"#.to_vec(),
            }]);
            assert_eq!(test_connection(&url, token).unwrap(), "test-model");
            worker.join().unwrap();
        }
    }

    #[test]
    fn rejected_credentials_give_actionable_errors_without_exposing_tokens() {
        for (token, code) in [
            (None, 401),
            (Some("wrong-secret"), 401),
            (Some("wrong-secret"), 403),
        ] {
            let (url, worker) = server(vec![Reply {
                method: "GET",
                path: "/health",
                token,
                status: code,
                body: b"do not reflect this response".to_vec(),
            }]);
            let error = test_connection(&url, token).unwrap_err().to_string();
            worker.join().unwrap();
            assert!(error.contains("Settings → Compute"));
            assert!(!error.contains("wrong-secret"));
            assert!(!error.contains("do not reflect"));
        }
    }

    #[test]
    fn server_errors_include_operation_request_id_and_redacted_reason() {
        let (url, worker) = server(vec![Reply {
            method: "GET",
            path: "/health",
            token: Some("test-secret"),
            status: 429,
            body: br#"{"error":"job capacity reached; rejected test-secret"}"#.to_vec(),
        }]);
        let error = test_connection(&url, Some("test-secret"))
            .unwrap_err()
            .to_string();
        worker.join().unwrap();
        assert!(error.contains(&format!("GET {url}/health")));
        assert!(error.contains("HTTP 429"));
        assert!(error.contains("request ID fixture-42"));
        assert!(error.contains("job capacity reached"));
        assert!(!error.contains("test-secret"));
        assert!(error.contains("[REDACTED]"));
    }

    #[test]
    fn connection_failure_retains_the_transport_cause_and_target() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let error = test_connection(&url, None).unwrap_err().to_string();
        assert!(error.contains(&format!("GET {url}/health")));
        assert!(error.contains("Caused by:"), "{error}");
    }

    #[test]
    fn tokens_with_header_control_characters_are_rejected() {
        assert!(Client::new("http://localhost:6996", Some("token\r\nInjected: value")).is_err());
        assert!(Client::new("http://localhost:6996", Some("Bearer token")).is_err());
        let client = Client::new("http://localhost:6996", Some("  test-secret  ")).unwrap();
        assert_eq!(
            client.request("GET", "/health").header("Authorization"),
            Some("Bearer test-secret")
        );
        let client = Client::new("http://localhost:6996", Some("  ")).unwrap();
        assert!(
            client
                .request("GET", "/health")
                .header("Authorization")
                .is_none()
        );
    }
}
