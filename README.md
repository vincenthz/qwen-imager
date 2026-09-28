# img-gen (Rust / Metal)

Native Rust inference for **Qwen Image 2.1** on Apple Silicon Macs, using
Candle's Metal backend. No Python, PyTorch, server, or C++ inference engine at
runtime. Library-first, with a CLI for text-to-image, up to 10 reference images,
and RGBA PNG output. An optional GPUI desktop window uses the same library.
The library takes and returns in-memory RGBA images.

## Desktop window

```sh
cd rust
cargo run --release --no-default-features --features gui --bin QwenImager
```

The window checks the Hugging Face cache at startup without network access. If
the model is missing or incomplete, it shows an explanation and a **Download**
button instead of the generation controls. Clicking Download starts fetching
missing files and displays a progress bar with bytes and percentage for the
current file. The generation interface appears when the download finishes;
fully cached models open it directly. Download errors remain visible and you
can retry with Download. The model is approximately 32 GB; `HF_HOME` and
`HF_HUB_CACHE` work as they do in the CLI.

Enter a prompt and click **Generate** to its right. **Steps** defaults to 20;
the adjacent **Size (px)** field sets the square image's side length (default
512, multiples of 32 from 32 to 2048). A decoded preview appears after **every
step**. **Auto seed** chooses a new random seed below 2³⁰ for each generation and
displays it in a greyed-out field. **Manual seed** accepts the full unsigned
64-bit range (0–18446744073709551615), so you can edit or reuse any supported seed;
the displayed seed remains available after generation for reproducing an image.
The latest preview replaces the previous one. Generate becomes **Cancel** while
working, stopping at the next safe boundary. The adjacent save icon (tooltip:
**Save PNG**) opens the native save dialog for the displayed image.
Early previews are estimates and may look rough. Decoding every step adds time.
The status row shows elapsed time and estimated time remaining, updating every
second. After the first preview, it extrapolates from the average time per
completed step, including preview decoding. One-time model loading is included
in elapsed time but excluded from the per-step average. The estimate resets for
each generation and clears when the run finishes or is cancelled.

Use the **image +** tile to add references (including PNG, JPEG, or WebP) through
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
`Preview`, and `Finished` after successful final decoding. Steps are one-based;
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
Previews never feed back into sampling. Keep callbacks short, and bound or
coalesce queued previews so the UI doesn't retain every full-resolution buffer.

Clone `CancellationToken` for a Cancel button, and create a fresh token for each
request. Cancellation is cooperative between model layers/stages; it cannot
interrupt a running GPU operation, VAE encode/decode, download, or callback.
Use `error.is::<img_gen::Cancelled>()` to distinguish it from inference failure.
`Generator` accepts sequential requests and releases model tensors between stages
and requests to limit memory; it does not retain a permanently loaded checkpoint.
It does keep the latest request's encoder output, and each reference's vision
features and VAE latents, in CPU memory. Reusing the same `Generator` with an
unchanged prompt and references (a new seed, step count, or size) skips the
text/vision encoders and reference encoding. Changing only the prompt still reuses
the per-reference results. Cached results are bit-identical to recomputing them.
Stages served from the cache emit no `Progress` events.

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
and `vae/` subdirectories. Keep its files unchanged while inference runs.
The checkpoint revision is pinned in `src/weights.rs`.

Defaults: native 2K size, 40 steps, seed 42, `out.png`. Without `-r`, editing
follows the last reference's aspect ratio. Dimensions are rounded down to
multiples of 32; reference images are resized to approximately 1 megapixel.
`--scale` accepts values greater than zero and at most one. Output is PNG only.
Seeds are reproducible within this implementation, but do not match PyTorch's
random-number generator or guarantee identical results across GPU/library versions.

For transparency, use the model's prompt convention:

```sh
./target/release/img-gen 'This is an RGBA image with transparency. A cartoon dragon sticker. The image has alpha channel and the background is transparent.' -o dragon.png
```

The implementation contains only the Qwen3-VL encoder, Qwen 2.1 single-stream
DiT, its RGBA VAE, and the fixed flow-matching Euler schedule. There is no device
selection, CPU/CUDA inference mode, model selection, video, LoRA, training,
batch generation, or prompt rewriting. Encoder layers load as needed; its
weights are released before the denoiser runs, and the denoiser is released
before final decoding. Optional previews decode while the denoiser is resident.
Attention runs in float32 for numerical stability; weights
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
encoder cancellation, and identical final pixels with previews enabled/disabled.
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
