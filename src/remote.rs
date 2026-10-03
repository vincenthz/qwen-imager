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

use crate::{CancellationToken, Event, Generation, Request, Stage};

const BOUNDARY: &str = "----image-forger-boundary";
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Run one generation against the remote server at `base_url` (e.g.
/// `http://host:6996`). Emits [`Event::Started`], [`Event::Progress`],
/// [`Event::StepFinished`] and [`Event::Preview`] as they arrive.
pub fn run(
    request: &Request,
    base_url: &str,
    cancellation: &CancellationToken,
    mut on_event: impl FnMut(Event),
) -> Result<Generation> {
    let started = Instant::now();
    let image = run_inner(request, base_url, cancellation, &mut on_event)?;
    let elapsed = started.elapsed();
    Ok(Generation { image, elapsed })
}

fn run_inner(
    request: &Request,
    base_url: &str,
    cancellation: &CancellationToken,
    on_event: &mut impl FnMut(Event),
) -> Result<Arc<RgbaImage>> {
    let base = base_url.trim_end_matches('/');
    let (preview_mode, preview_every) = match request.preview_every {
        Some(every) => ("auto", every.get()),
        None if request.preview_control.is_some() => ("auto", 5),
        None => ("off", 0),
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
    let response = ureq::post(&format!("{base}/jobs"))
        .set("Content-Type", &format!("multipart/form-data; boundary={BOUNDARY}"))
        .send_bytes(&body)
        .with_context(|| format!("submitting job to {base}"))?;
    ensure!(
        response.status() / 100 == 2,
        "server rejected the job (HTTP {})",
        response.status()
    );

    let view: serde_json::Value = response.into_json().context("reading job response")?;
    let id = view["id"]
        .as_str()
        .context("job response is missing an id")?
        .to_owned();
    let width = view["width"].as_u64().unwrap_or(0) as u32;
    let height = view["height"].as_u64().unwrap_or(0) as u32;

    on_event(Event::Started {
        width,
        height,
        steps: request.steps,
        seed: request.seed,
    });

    let mut completed_steps = 0usize;
    let mut preview_step: Option<usize> = None;

    loop {
        cancellation.check()?;
        let status: serde_json::Value = ureq::get(&format!("{base}/jobs/{id}"))
            .call()
            .with_context(|| format!("polling job {id}"))?
            .into_json()
            .context("reading job status")?;

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
                    total: request.steps,
                    duration,
                });
            }
            completed_steps = steps;
        }

        let current_preview = status["preview_step"].as_u64().map(|s| s as usize);
        if current_preview != preview_step {
            if let Some(step) = current_preview
                && let Ok(bytes) = fetch_bytes(&format!("{base}/jobs/{id}/preview"))
                && let Ok(image) = decode_png(&bytes)
            {
                on_event(Event::Preview {
                    step,
                    total: request.steps,
                    image,
                });
            }
            preview_step = current_preview;
        }

        match status["status"].as_str() {
            Some("succeeded") => {
                let bytes = fetch_bytes(&format!("{base}/jobs/{id}/image"))?;
                return decode_png(&bytes);
            }
            Some("failed") => {
                anyhow::bail!(
                    "{}",
                    status["error"].as_str().unwrap_or("remote generation failed")
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

fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    ensure!(
        response.status() / 100 == 2,
        "GET {url} returned HTTP {}",
        response.status()
    );
    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;
    Ok(bytes)
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
