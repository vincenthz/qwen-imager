# img-gen (Rust / Metal)

Native Rust inference for **Qwen Image 2.1** on Apple Silicon Macs, using
Candle's Metal backend. No Python, PyTorch, server, or C++ inference engine at
runtime. Library-first, with a CLI for text-to-image, up to 10 reference images,
and RGBA PNG output. An optional GPUI desktop window uses the same library.
The library takes and returns in-memory RGBA images.

## Desktop window

From the repository root:

```sh
cargo run --release --no-default-features --features gui --bin QwenImager
```

To build the macOS app bundle, install
[cargo-bundle](https://github.com/burtonageo/cargo-bundle) once, then run:

```sh
cargo install cargo-bundle --version 0.11.0 --locked
cargo bundle-app
open target/release/bundle/osx/QwenImager.app
```

`cargo bundle-app` is an alias for
`cargo bundle --release --format osx --no-default-features --features gui --bin QwenImager`.
The resulting `QwenImager.app` includes the photo-assembly icon and can be copied
to Applications. Model weights stay in the Hugging Face cache and are downloaded
through the app when needed. This produces a local app bundle; distribution
signing and notarization are separate steps.

The icon's transparent master is `assets/app-icon.png`; its generation prompt is
in `assets/app-icon-prompt.txt`. To rebuild the bundled `assets/app-icon.icns`
with all standard and Retina sizes, run `sh scripts/build-app-icon.sh` on macOS,
then `cargo bundle-app` again.

The window checks the Hugging Face cache at startup without network access. If
the model is missing or incomplete, it shows an explanation and a **Download**
button instead of the generation controls. Clicking Download starts fetching
missing files and displays a progress bar with bytes and percentage for the
current file. The generation interface appears when the download finishes;
fully cached models open it directly. Download errors remain visible and you
can retry with Download. The model is approximately 32 GB; `HF_HOME` and
`HF_HUB_CACHE` work as they do in the CLI.

The top bar starts with a **gear** icon that opens **Settings**, followed by
**Workspace 1** and any additional workspaces.
Click **+** to create and select a new workspace. Each workspace retains its own
prompt, settings, reference images and drawings, preview, and finished result.
Switching tabs keeps background generation running. A **bell** appears on a
background tab when a preview arrives or generation ends (including an error or
cancellation); selecting that tab clears the bell. Foreground updates do not
add a bell. The tab strip scrolls when necessary, with **+** always accessible.

Workspaces generate concurrently using independent Metal queues, sampling state,
conditioning caches, previews, and cancellation. They share one pool of immutable
denoiser and VAE weight buffers. Cancelling one workspace does not stop another.
Concurrent runs share GPU time and each needs its own working memory; starting
more runs does not multiply GPU capacity. Workspaces live for the current app session;
closing the app cancels their work. Save images you want to keep before quitting.

Settings are saved in `~/Library/Application Support/QwenImager/settings.json`:

- **Steps** and **Size (px)** for new workspaces (defaults 20 and 512). Idle
  workspaces still showing the previous defaults adopt the new ones.
- **Output directory**, where the Save dialog opens (default `~/Pictures`).
- **Previews**: **Automatic** (the default) or **Manual**, and **Sequential**
  (the default) or **Parallel** decoding. Changes apply from the next generation.
- **Attention precision**: **Float32** (the default, the reference path) or
  **BF16**, which makes each denoising step about 5% faster with slightly
  different pixels. If a BF16 step produces non-finite latents, it is redone in
  Float32 and the rest of that generation stays in Float32. Applies from the
  next generation.

Enter a prompt and click **Generate** to its right. **Steps** and the adjacent
**Size (px)** field (the square image's side length, multiples of 32 from 32 to
2048) start from the settings. Automatic previews decode a preview after every
step. Sequential decoding pauses sampling for each decode; parallel decoding
runs in the background while sampling continues, skipping steps while the
decoder is busy. Manual previews automatically request previews after step 5 and
halfway through sampling (rounded up for odd step counts); these two checkpoints
always pause sampling until decoded, waiting for any background preview first.
Matching milestones produce one preview; the final step uses the normal final
decode. You can also click **Preview** during generation to decode the latest
fully completed step: in parallel mode sampling continues, in sequential mode it
pauses after the current step until the preview is ready. A click during model
loading waits for the first completed step. Only one preview can be pending at a
time; its image shows the source step number. The final image always appears automatically.
A preview still running when sampling finishes is superseded by final decoding.
The decoder has its own Metal queue and reuses its loaded weights; it shares GPU
compute and memory bandwidth with sampling, so previews can still slow generation.

**Auto seed** chooses a new random seed below 2³⁰ for each generation and
displays it in a greyed-out field. **Manual seed** accepts the full unsigned
64-bit range (0–18446744073709551615), so you can edit or reuse any supported seed;
the displayed seed remains available after generation for reproducing an image.
The latest preview replaces the previous one. Generate becomes **Cancel** while
working, stopping at the next safe boundary. The **pause** icon next to it pauses
generation between model layers and becomes **resume**; a paused workspace keeps
its GPU memory, and can still be cancelled. Paused time is excluded from the
elapsed time and estimate. The adjacent save icon (tooltip:
**Save PNG**) opens the native save dialog for the displayed image.
Early previews are estimates and may look rough. Decoding every step adds time.
The status row shows elapsed time and estimated time remaining, updating every
second. With automatic sequential previews, it extrapolates from the average time
per completed step including preview decoding. Otherwise it updates the estimate
at each sampling step, even if no previews are requested. One-time model loading is included
in elapsed time but excluded from the per-step average. The estimate resets for
each generation and clears when the run finishes or is cancelled.

Use the **image +** tile to add references (PNG, JPEG/JPG, WebP, or HEIC/HEIF) through
the native file picker. You can select several files at once, up to 10 images in
total. Each image appears as a numbered thumbnail with a **−** button to remove
it; another add tile stays to the right. The row scrolls horizontally when needed.
Describe the desired changes in the prompt, then Generate. All attached images
are used in their displayed order and stay loaded between runs. When a generation
with references finishes, a **Before** toggle appears at the right of the status row: it
switches the view between the last reference image and the generated result, and
a corner label shows which one is displayed. Save always writes the generated image. After any generation
finishes, **Use as input** (on the same row) replaces all reference images with
the result, so the next Generate edits it. Removing every
thumbnail returns to text-only generation. Output remains square at the selected
Size. Image decoding runs in the background, preserves alpha, and respects photo
orientation. Cancelling the picker, selecting too many files, or choosing an
unreadable file keeps the previous references without adding a partial selection.

Click a reference thumbnail to draw on it. The main view then shows that image,
and a toolbar above it offers red, blue, green and white paint, three brush
sizes (**S**/**M**/**L**, relative to the image's longer side), **Undo**,
**Clear**, and **Done**. Drag on the image to paint freehand. The thumbnail
shows pending paint; click it again or press Done to return to the result
view. When you click Generate, the paint is applied to the full-resolution
reference, so that run and **Before** use the painted image. After that the
paint can no longer be undone; to start over, remove the image and add it again.

Model downloads, inference, and saving run off the UI thread. A bounded event
queue limits retained preview buffers, and old image textures are released when
replaced. Closing the window stops the app. During download, Cancel takes effect
after the current file finishes (the hub client does not support interrupting it).

The GUI supports text-to-image and editing with up to 10 reference images;
advanced options remain available through the library and CLI. The `gui` feature is
optional, so library/CLI builds do not compile GPUI. GPUI compiles its Metal
shaders at runtime; the separate Xcode Metal shader compiler is not required.

## Library

From another Rust project, depend on this directory without the optional CLI:

```toml
[dependencies]
img-gen = { path = "../img-gen/rust", default-features = false }
```

```rust,no_run
use img_gen::{CancellationToken, Event, Generator, ModelOptions, Request};
use std::num::NonZeroUsize;

fn main() -> img_gen::Result<()> {
    let mut generator = Generator::new(ModelOptions::default());
    let mut request = Request::new("a corgi playing guitar in the rain");
    request.scale = 0.25;
    request.steps = 20;
    request.preview_every = NonZeroUsize::new(5); // optional; off by default
    let cancel = CancellationToken::default();
    let generated = generator.generate(&request, &cancel, |event| {
        match event {
            Event::StepFinished { step, total, .. } => println!("{step}/{total}"),
            Event::Preview { image, .. } => {
                // Forward image (Arc<RgbaImage>) to the UI. image.as_raw() is RGBA8.
            }
            _ => {}
        }
    })?;
    generated.image.save("out.png")?;
    Ok(())
}
```

`Generator::generate` blocks. Create/use it on an inference worker and pass owned
`Event` values to the UI. [examples/progress.rs](examples/progress.rs) demonstrates
a worker and bounded channel, without a GUI dependency. The desktop app in
`src/gui/` consumes those events and uploads previews on its UI thread.
The library does not print progress or save images. The caller owns file formats,
UI updates, error display, and output storage. `Request::dimensions()` validates
inputs and determines the output size without GPU access or weight loading.

Events arrive in order: `Started`, stage-local `Progress` (including encoder and
denoiser loading layers), `StepFinished` for each denoising step, optional
`Preview`, and `Finished` after successful final decoding. Background previews can arrive
after later steps have completed; their `step` identifies the snapshot being shown.
No previews arrive after `Finished`. Steps are one-based;
progress counts are local to their stage, not an overall percentage. Errors and
cancellation return from `generate` without `Finished`.

Call `Generator::prepare(&cancel, on_progress)` on a worker before generating
to ensure the entire model is local without allocating GPU resources. Its
`DownloadProgress` callback reports the filename, downloaded bytes (including
resumed bytes), and total bytes for each missing file. The total is `None` while
fetching metadata. Cached files emit no download events. The desktop app uses
this to report downloads before enabling generation. `generate` remains usable
on its own and still downloads any missing files silently.

Preview decoding estimates the clean image as `x_sigma - sigma * velocity`.
Early estimates may look rough. Previews use the full RGBA VAE at output resolution,
so enabling them adds latency and peak memory while the denoiser remains loaded.
The final preview shares the returned image's buffer and needs no extra decode.
Previews never feed back into sampling. For on-demand background previews, set
`request.preview_control = Some(control.clone())` using a fresh
`PreviewControl::default()`, leave `preview_every` unset, and call
`control.request_preview()` from your UI. It returns false when inactive or a
preview is already pending. `control.request_blocking_preview()` instead pauses
sampling after its current step until that preview is decoded, queueing behind a
background decode if one is running. Set `request.attention = AttentionPrecision::BFloat16` for faster
reduced-precision denoiser attention (the default is `Float32`); encoders
always use float32 attention, and a step that overflows in BF16 is recomputed
in float32, which the rest of the generation keeps. Set `request.pause = Some(pause.clone())`
with a `PauseControl` to pause and resume a generation cooperatively between
model layers. Manual mode copies a small clean-latent snapshot to
CPU memory after each completed step (about 2 MiB at 2048×2048); only requested
snapshots are decoded. Callbacks still run on the inference thread. The decoder
uses a separate Metal device/queue and shares one cache across automatic previews,
manual previews, and final decoding. Keep callbacks short, and bound or
coalesce queued previews so the UI doesn't retain every full-resolution buffer.

Clone `CancellationToken` for a Cancel button, and create a fresh token for each
request. Cancellation is cooperative between model layers/stages; it cannot
preempt GPU operations, downloads, or callbacks. A background decode already in
flight may finish after cancellation, but its result is discarded. Dropping or
unloading the generator waits for its decoder worker to exit.
Use `error.is::<img_gen::Cancelled>()` to distinguish it from inference failure.
Use `SharedModel::new(options).generator()` to create independent sessions that
can run on separate threads. Keep one session per workspace to preserve its
prompt/reference cache. Denoiser and VAE weights are loaded once into immutable
Metal buffers shared across all sessions and their preview workers. Loading has
separate queues; each session's computation uses its own queue and allocator.
Text/vision weights are shared while encoders overlap, then released when no
encoder uses them. `unload_models()` on a shared session releases its local
resources; the common weights remain until the shared model and its sessions
are dropped. The desktop tabs use this API. The HTTP service still schedules
one job at a time.

`Generator` accepts sequential requests and keeps its Metal device and loaded
denoiser/VAE weight tensors between requests. VAE previews reuse the same decoder
weights instead of loading them again on every step. Weights load lazily on first
use; text/vision encoder weights are still released after encoding. Denoiser
prefix attention and rotary state are rebuilt for each request, including after
cancellation. `DenoiserLoading` progress therefore also appears on warm runs.
Resident weights use approximately 14.2 GiB for the denoiser and VAE decoder
(about 0.3 GiB more after reference encoding), plus working memory. Drop the
generator or call `Generator::unload_models()` to release its GPU resources.
Unloading preserves the latest request's encoder output, and each reference's
vision features and VAE latents, in CPU memory. Reusing the same `Generator` with an
unchanged prompt and references (a new seed, step count, or size) skips the
text/vision encoders and reference encoding. Changing only the prompt still reuses
the per-reference results. Cached results are bit-identical to recomputing them.
Stages served from the cache emit no `Progress` events.

## HTTP service

Run a persistent generation service with the CLI:

```sh
cargo build --release --bin qwen-imager-cli
./target/release/qwen-imager-cli --serve --offline --listen 127.0.0.1:6996
```

Submit JSON prompts or multipart reference images with JSON parameters to
`POST /jobs`, poll `GET /jobs/{id}`, request a background
preview with `POST /jobs/{id}/preview`, and retrieve preview/final PNGs with
`GET /jobs/{id}/preview` and `GET /jobs/{id}/image`. Jobs run sequentially and
reuse loaded models; HTTP stays responsive during inference. Manual previews
are the default. The bounded queue supports cancellation and explicit cleanup.
See the [HTTP API reference](docs/http-api.md) for parameters, curl examples,
remote access, and lifecycle details.

## CLI

```sh
cd rust
cargo build --release
./target/release/img-gen "a corgi playing guitar in the rain"
./target/release/img-gen "a panoramic mountain landscape" -r 16:9 -o mountains.png
./target/release/img-gen "a cat astronaut" --scale 0.5 --steps 20
./target/release/img-gen "Change the background to a sunset beach" -i ../edit.png -o edited.png
./target/release/img-gen "These characters sit around a campfire" -i a.png -i b.png
./target/release/img-gen "a red teapot" --scale 0.25 --preview-dir previews --preview-every 5
./target/release/img-gen "a red teapot" --scale 0.5 --attention bf16
```

Requires an Apple Silicon Mac with Metal access, Rust, and Xcode command line
tools. This is the full BF16 checkpoint, not a quantized model: weights occupy
about 32 GB on disk and inference needs substantial unified memory. Start with
`--scale 0.25` on a memory-constrained machine. Larger resolutions and multiple
references increase working memory.

The first run downloads the original Hugging Face checkpoint. Existing Python
downloads in `~/.cache/huggingface/hub` are reused. `HF_HOME` and `HF_HUB_CACHE`
are respected. `--offline` forbids downloads; `--model-dir PATH` loads an
existing Diffusers snapshot with `processor/`, `text_encoder/`, `transformer/`,
and `vae/` subdirectories. Keep its files unchanged while the generator exists.
The checkpoint revision is pinned in `src/weights.rs`.

Defaults: native 2K size, 40 steps, seed 42, `out.png`. Without `-r`, editing
follows the last reference's aspect ratio. Dimensions are rounded down to
multiples of 32; reference images are resized to approximately 1 megapixel.
`--scale` accepts values greater than zero and at most one. Reference images may
be PNG, JPEG/JPG, WebP, or HEIC/HEIF; formats are detected from the file contents
and photo orientation is applied. Output is PNG only.

Image loading uses portable Rust decoders (`image` and `heic-rs`), with no
installed HEIC codecs or conversion commands required. HEIC support targets
still photos, including tiled images and HEIF rotation/mirroring. Some HEVC
variants (including lossless files from some encoders) may fail to decode.
The application and inference engine still require Apple Silicon and Metal.
Seeds are reproducible within this implementation, but do not match PyTorch's
random-number generator or guarantee identical results across GPU/library versions.

Experimental shared noise is available in the CLI with `--noise-source-size 2048`.
For a square 512px output (`--scale 0.25`), it generates exactly the same initial
128×128×64 FP32 noise as a native 2048px run with that seed, then pools each 4×4
block per channel, dividing its sum by 4 to preserve unit variance. Only the
initial noise changes; the output-resolution sampling schedule stays unchanged.
The source size must be an integer multiple of the square output size. Omit the
flag for the original noise sequence. This is an experiment in composition
consistency, not a guarantee of matching images across resolutions.

`--metrics PATH.json` writes the prompt, seed, dimensions, checkpoint revision,
generation time, and precise stage/step timestamps. To reproduce the six-image
comparison with macOS RAM and disk-I/O sampling (200ms intervals):

```sh
cargo build --release --bin qwen-imager-cli
python3 scripts/benchmark-noise.py --output output/my-noise-comparison \
  --prompt 'a clown at the circus, with a red teapot, and a dog on a bike' \
  --seeds 42 12345 --steps 8
```

Each image runs in a fresh process, with previews off and locally cached weights.
The OS file cache is not flushed. Compare sampling time separately from setup
and final decoding; two seeds do not constitute a statistical speed benchmark.
The script preserves PNGs, timing JSON, memory samples, process logs, and a
summary CSV/JSON. Peak physical footprint comes from macOS's lifetime peak
counter; RSS is a sampled peak and can include mapped checkpoint pages.

For transparency, use the model's prompt convention:

```sh
./target/release/img-gen 'This is an RGBA image with transparency. A cartoon dragon sticker. The image has alpha channel and the background is transparent.' -o dragon.png
```

The implementation contains only the Qwen3-VL encoder, Qwen 2.1 single-stream
DiT, its RGBA VAE, and the fixed flow-matching Euler schedule. There is no device
selection, CPU/CUDA inference mode, model selection, video, LoRA, training,
batch generation, or prompt rewriting. Encoder layers load as needed; their
weights are released after encoding. Denoiser and VAE weights remain cached
across previews and generations until the generator is dropped or unloaded.
Attention runs in float32 by default for numerical stability; weights
and other activations use BF16. Prefix keys/values are cached. Large VAE convolutions run in
rows with exact overlap to bound temporary buffers.

```sh
cargo test
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
cargo test --features gui
cargo clippy --features gui --all-targets -- -D warnings
# Requires the cached checkpoint and access to the Metal GPU:
cargo test -- --ignored --test-threads=1
```

The Metal tests compare attention (including a captured Qwen regression case)
against a dense CPU reference and the VAE
against fixed Diffusers outputs. An end-to-end Metal test checks ordered progress,
encoder/prefix cancellation, model unloading, and identical final pixels with
previews enabled/disabled. Warm runs are compared with fresh generators after
prompt, reference, seed, and size changes. Manual-preview tests verify snapshot
selection while sampling advances, coalesced clicks, cancelled-session isolation,
and identical pixels with concurrent Metal decoding.
The default tests also check validation and cancellation without GPU access.
Download tests check complete/partial local snapshots and resumed byte counts;
an ignored loopback HTTP test verifies real file transfer progress without
contacting Hugging Face or modifying the user's cache. The GUI tests check
RGBA-to-BGRA preview conversion.
See [fixture provenance](tests/fixtures/README.md).

Architecture and checkpoint reference: [Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1).
Ported from the [Diffusers Qwen 2.1 pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/qwenimage21)
and [Transformers Qwen3-VL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl).
The model weights retain their upstream license. See [NOTICE](NOTICE) for code attribution.
