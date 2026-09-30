use anyhow::{Context, ensure};
use clap::Parser;
use qwen_imager::{CancellationToken, Event, Generator, ModelOptions, Request, Stage};
use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

mod server;

#[derive(Parser)]
#[command(
    version,
    about = "Generate or edit a PNG with Qwen Image 2.1 on Apple Metal"
)]
struct Args {
    /// Text prompt describing the image or edit
    #[arg(required_unless_present = "serve", conflicts_with = "serve")]
    prompt: Option<String>,
    /// Run a persistent HTTP generation service
    #[arg(long, conflicts_with_all = ["output", "images", "ratio", "scale", "steps", "seed", "noise_source_size", "metrics", "preview_dir", "preview_every"])]
    serve: bool,
    /// HTTP listen address; remote access requires QWEN_IMAGER_API_TOKEN
    #[arg(
        long,
        default_value = "127.0.0.1:6996",
        requires = "serve",
        conflicts_with = "prompt"
    )]
    listen: std::net::SocketAddr,
    /// Maximum retained jobs, including queued jobs; delete finished jobs to free slots
    #[arg(long, default_value = "32", value_parser = clap::value_parser!(u32).range(1..=128), requires = "serve", conflicts_with = "prompt")]
    max_jobs: u32,
    /// Output PNG
    #[arg(short, long, default_value = "out.png")]
    output: PathBuf,
    /// Reference image (PNG, JPEG, WebP, or HEIC/HEIF); repeat up to 10 times
    #[arg(short = 'i', long = "image", value_name = "PATH")]
    images: Vec<PathBuf>,
    /// Output aspect ratio; defaults to the last reference's ratio, or 1:1
    #[arg(short, long, value_parser = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"])]
    ratio: Option<String>,
    /// Scale the native 2K size (0 < scale <= 1)
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// Denoising steps
    #[arg(short = 's', long, default_value_t = 40)]
    steps: usize,
    /// Random seed (repeatable in Rust; differs from PyTorch's RNG)
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Experimental: pool noise from this square pixel size (e.g. 2048)
    #[arg(long)]
    noise_source_size: Option<u32>,
    /// Write parameters and precise stage/step timings to a JSON sidecar
    #[arg(long, value_name = "PATH")]
    metrics: Option<PathBuf>,
    /// Existing Qwen Image 2.1 Diffusers snapshot directory
    #[arg(long, value_name = "PATH")]
    model_dir: Option<PathBuf>,
    /// Use cached files only
    #[arg(long)]
    offline: bool,
    /// Save progressive PNG previews to this directory (adds decoding work)
    #[arg(long, value_name = "PATH")]
    preview_dir: Option<PathBuf>,
    /// Preview interval, when --preview-dir is set
    #[arg(long, default_value = "5")]
    preview_every: NonZeroUsize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.serve {
        return server::run(
            args.listen,
            args.max_jobs as usize,
            ModelOptions {
                model_dir: args.model_dir,
                offline: args.offline,
            },
        );
    }
    ensure!(
        args.output
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png")),
        "output must be a .png file to preserve RGBA"
    );
    let mut request = Request::new(args.prompt.expect("clap requires a prompt"));
    request.images = args
        .images
        .iter()
        .map(|p| {
            qwen_imager::image_input::open(p)
                .with_context(|| format!("opening {}", p.display()))
        })
        .collect::<anyhow::Result<_>>()?;
    request.ratio = args.ratio;
    request.scale = args.scale;
    request.steps = args.steps;
    request.seed = args.seed;
    request.noise_source_size = args.noise_source_size;
    request.preview_every = args.preview_dir.as_ref().map(|_| args.preview_every);
    request.dimensions()?;
    if let Some(dir) = &args.preview_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut generator = Generator::new(ModelOptions {
        model_dir: args.model_dir,
        offline: args.offline,
    });
    let cancellation = CancellationToken::default();
    let mut preview_error = None;
    let mut metrics = Vec::new();
    let started = Instant::now();
    let result = generator.generate(&request, &cancellation, |event| {
        if args.metrics.is_some() {
            let mut record = match &event {
                Event::Progress { stage, completed, total } => serde_json::json!({"event": "progress", "stage": format!("{stage:?}"), "completed": completed, "total": total}),
                Event::StepFinished { step, total, duration } => serde_json::json!({"event": "step", "step": step, "total": total, "duration_s": duration.as_secs_f64()}),
                Event::Started { .. } => serde_json::json!({"event": "started"}),
                Event::Preview { step, .. } => serde_json::json!({"event": "preview", "step": step}),
                Event::Finished { .. } => serde_json::json!({"event": "finished"}),
            };
            record["time_s"] = started.elapsed().as_secs_f64().into();
            metrics.push(record);
        }
        match event {
        Event::Started {
            width,
            height,
            steps,
            seed,
        } => eprintln!(
            "{} · Metal · {width}x{height} · {steps} steps · seed {seed}",
            qwen_imager::MODEL
        ),
        Event::Progress {
            stage,
            completed,
            total,
        } => {
            // Keep the terminal concise; a UI can consume every layer event.
            if completed == 0 || completed == total || completed % 8 == 0 {
                let label = match stage {
                    Stage::Loading => "Preparing model".into(),
                    Stage::ReferenceVision { index } => format!("Reference {} vision", index + 1),
                    Stage::TextEncoding => "Text encoder".into(),
                    Stage::ReferenceEncoding => "Reference latents".into(),
                    Stage::DenoiserLoading => "Denoiser prefix".into(),
                    Stage::Decoding => "Decoding RGBA".into(),
                };
                eprintln!("{label} {completed}/{total}");
            }
        }
        Event::StepFinished {
            step,
            total,
            duration,
        } => eprintln!("Step {step}/{total} ({:.1}s)", duration.as_secs_f32()),
        Event::Preview { step, image, .. } => {
            if let Some(dir) = &args.preview_dir {
                let path = dir.join(format!("step-{step:04}.png"));
                if let Err(error) = image
                    .save_with_format(&path, image::ImageFormat::Png)
                    .with_context(|| format!("saving {}", path.display()))
                {
                    preview_error = Some(error);
                    cancellation.cancel();
                }
            }
        }
        Event::Finished { .. } => {}
        }
    });
    if let Some(error) = preview_error {
        return Err(error);
    }
    let generated = result?;
    generated
        .image
        .save_with_format(&args.output, image::ImageFormat::Png)
        .with_context(|| format!("saving {}", args.output.display()))?;
    if let Some(path) = &args.metrics {
        let (width, height) = request.dimensions()?;
        let report = serde_json::json!({
            "prompt": request.prompt, "seed": request.seed, "steps": request.steps,
            "width": width, "height": height, "noise_source_size": request.noise_source_size,
            "model": qwen_imager::MODEL, "revision": qwen_imager::REVISION,
            "generation_s": generated.elapsed.as_secs_f64(),
            "generation_and_save_s": started.elapsed().as_secs_f64(),
            "events": metrics,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing metrics {}", path.display()))?;
    }
    eprintln!(
        "Saved {} ({:.0}s)",
        args.output.display(),
        generated.elapsed.as_secs_f32()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_mode_preserves_generation_cli_and_rejects_ignored_options() {
        let single = Args::try_parse_from(["cli", "a teapot", "--steps", "8"]).unwrap();
        assert!(!single.serve);
        assert_eq!(single.prompt.as_deref(), Some("a teapot"));
        let server = Args::try_parse_from(["cli", "--serve", "--offline"]).unwrap();
        assert!(server.serve);
        assert!(server.prompt.is_none());
        assert!(Args::try_parse_from(["cli"]).is_err());
        assert!(Args::try_parse_from(["cli", "--serve", "--steps", "8"]).is_err());
        assert!(Args::try_parse_from(["cli", "a teapot", "--listen", "127.0.0.1:9000"]).is_err());
    }
}
