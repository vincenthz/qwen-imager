use anyhow::{Context, ensure};
use clap::Parser;
use qwen_imager::{CancellationToken, Event, Generator, ModelOptions, Request, Stage};
use std::{num::NonZeroUsize, path::PathBuf};

#[derive(Parser)]
#[command(
    version,
    about = "Generate or edit a PNG with Qwen Image 2.1 on Apple Metal"
)]
struct Args {
    /// Text prompt describing the image or edit
    prompt: String,
    /// Output PNG
    #[arg(short, long, default_value = "out.png")]
    output: PathBuf,
    /// Reference image; repeat up to 10 times
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
    ensure!(
        args.output
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png")),
        "output must be a .png file to preserve RGBA"
    );
    let mut request = Request::new(args.prompt);
    request.images = args
        .images
        .iter()
        .map(|p| {
            image::open(p)
                .with_context(|| format!("opening {}", p.display()))
                .map(|i| i.to_rgba8())
        })
        .collect::<anyhow::Result<_>>()?;
    request.ratio = args.ratio;
    request.scale = args.scale;
    request.steps = args.steps;
    request.seed = args.seed;
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
    let result = generator.generate(&request, &cancellation, |event| match event {
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
    });
    if let Some(error) = preview_error {
        return Err(error);
    }
    let generated = result?;
    generated
        .image
        .save_with_format(&args.output, image::ImageFormat::Png)
        .with_context(|| format!("saving {}", args.output.display()))?;
    eprintln!(
        "Saved {} ({:.0}s)",
        args.output.display(),
        generated.elapsed.as_secs_f32()
    );
    Ok(())
}
