# HTTP generation service

Build and start the same CLI in server mode:

```sh
cargo build --release --bin image-forger-cli
./target/release/image-forger-cli --serve --offline
```

The default address is `127.0.0.1:6996`. `--model`, `--model-dir PATH` and
`--offline` work as in the normal CLI; one service serves one checkpoint. The first job loads the model; later jobs reuse its
loaded weights. One job runs at a time, with other jobs queued in submission
order. HTTP requests remain responsive while inference runs.

For another port, use `--listen 127.0.0.1:9000`. For access from another machine,
set `IMAGEFORGER_API_TOKEN` and use `--listen 0.0.0.0:6996`; include
`Authorization: Bearer YOUR_TOKEN` on every request, including `/health`.
The token is also enforced on localhost when set. The service uses plain HTTP;
use a TLS reverse proxy if you need encrypted transport.

In the desktop GUI, add the service address in **Settings → Compute**, paste the
same `IMAGEFORGER_API_TOKEN` into that host's masked **API token** field (without
the `Bearer` prefix), and click **Save & test**. The token is saved in macOS
Keychain, separately from the settings file. The test calls authenticated
`GET /health` without submitting a job or loading model weights. Select the host
from the title bar's **Local / Remote** menu to generate. The GUI authenticates
submission, polling, preview, and final-image requests; no local model download
is required. **Clear token** removes the credential from Keychain. Removing a
host from the list retains its credential for re-adding the same endpoint.

HTTP redirects are not followed by the GUI client; configure the final service
URL, including its HTTPS scheme or reverse-proxy path, directly in Settings.

## Service diagnostics

The service writes JSON log lines to stderr for startup, completed HTTP requests,
job queueing/start/completion/cancellation, and shutdown. Each completed request
includes its method, path (without query parameters), peer address, HTTP status,
elapsed milliseconds, and any available rejection reason. Responses include an
`X-Request-ID` header matching the request ID in the logs, including authentication
failures. Job lifecycle records include the job ID; submission request logs include
the new job's `Location` header.

Enable additional request-start and generation-stage/step logs with
`--http-debug` (also available as `--debug`):

```sh
./target/release/image-forger-cli --serve --offline --http-debug 2>imageforger-http.log
```

The same flag works with a remote listen address and token authentication. Logs
indicate whether authentication is required/present without recording the
Authorization header. Request bodies, prompts, and image contents are not logged.
HTTP diagnostics redact the configured token and URL user-info/query credentials.
Errors starting the service include context such as the address being bound.

GUI failures appear in persistent root notifications. Click **Details** for the
operation, timestamp, HTTP method/target/status and request ID when available,
and underlying causes (for example, connection refused, DNS/TLS failure, a server
validation error, or a Keychain error). **Copy details** copies this report for
troubleshooting. The same GUI error report is written to stderr as
`gui.operation_failed`. Notifications remain available when changing workspaces
or closing Settings; cancelling an operation does not trigger an error alert.

## Submit and retrieve an image

```sh
curl -sS http://127.0.0.1:6996/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a clown at the circus, with a red teapot, and a dog on a bike","scale":0.25,"steps":8,"seed":42}'
```

This immediately returns `202 Accepted`, a `Location` header, and a job object
with an `id` such as `"1"`. Use that ID in subsequent requests:

```sh
# Poll status (once per second is sufficient for most clients).
curl -sS http://127.0.0.1:6996/jobs/1

# Request a preview from the most recent fully completed step.
curl -sS -X POST http://127.0.0.1:6996/jobs/1/preview

# Once preview_url is non-null, fetch the latest decoded preview.
curl --fail http://127.0.0.1:6996/jobs/1/preview -o preview.png

# Once status is "succeeded", fetch the final RGBA image.
curl --fail http://127.0.0.1:6996/jobs/1/image -o final.png
```

`GET /preview` only retrieves the latest available image; it does not request a
decode. `POST /preview` schedules background decoding while sampling continues.
A request made before the first completed step waits for that step. Repeated
requests while one decode is outstanding return `409`. A preview can lag behind
`completed_steps`; `preview_step` and the PNG's `X-Preview-Step` header identify
its snapshot. Early previews may look rough. Decoding uses additional GPU time
and memory, especially at 2048px.

## Upload reference images for editing

Send a multipart form to the same `POST /jobs` endpoint. Include exactly one
`parameters` field containing the usual JSON, and repeat the `images` field for
each reference image:

```sh
curl --fail-with-body http://127.0.0.1:6996/jobs \
  -F 'parameters={"prompt":"Change the background to a sunset beach","scale":0.25,"steps":8,"seed":42}' \
  -F 'images=@reference.png'

# References keep the order of the images fields.
curl --fail-with-body http://127.0.0.1:6996/jobs \
  -F 'parameters={"prompt":"Put the character from the first image into the scene from the second","scale":0.25,"steps":8}' \
  -F 'images=@character.png' \
  -F 'images=@scene.jpg'
```

Let curl set the multipart Content-Type and boundary. Authenticated workers
require the same Bearer header used for JSON requests. PNG, JPEG, WebP, and HEIC/HEIF are
accepted, detected from the uploaded bytes. Alpha is preserved where the format
supports it, and photo orientation is applied. The filenames are not used as server paths, and uploads are not
written to disk.

The response is the usual `202` job object. Use the existing status, preview,
image, and cancellation endpoints. `references` lists each input's oriented
`width` and `height` in upload order. Without `ratio`, the output follows the
last reference's aspect ratio; an explicit ratio overrides it. References are
resized by the inference pipeline in the same way as local CLI/GUI inputs.

Upload limits:

- Up to **10 images**, with **64 MiB** maximum for the entire multipart body.
- The `parameters` field is limited to **64 KiB**; duplicate `parameters` fields
  and unknown form fields are rejected. Parameters may appear before or after image fields.
- Each image is limited to **16 million pixels** and **8192 pixels per side**;
  the combined inputs are limited to **32 million pixels** per job.
- At most **two uploads** are received/decoded concurrently. Queued and active
  jobs share **512 MiB** of original RGBA reference buffers, accounted in 64 KiB
  units. These are additional to the model's working memory and encoder cache.

Malformed images return `400`, unsupported formats return `415`, and size
limits return `413`. Upload concurrency or reference-memory exhaustion returns
`429`; retry after existing uploads or jobs finish. Rejected requests do not
create partial jobs. Reference buffers are released when a queued job is
cancelled or when active generation exits. Original inputs are not retained
with completed jobs; clients should keep their own copies for resubmission.

## Parameters

The JSON body accepts these fields. Unknown fields are rejected, and the maximum
body size is 64 KiB. These same fields go in the multipart `parameters` field
when uploading reference images. Server file paths are never accepted.

| Field | Default | Meaning |
| --- | --- | --- |
| `prompt` | Required | Nonblank text, at most 16,384 UTF-8 bytes |
| `ratio` | `null` (last reference ratio, or square) | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16` |
| `scale` | `1.0` | Fraction of native 2K size, greater than zero and at most one; `0.25` gives square 512px |
| `steps` | `40` | 1–1000 denoising steps |
| `seed` | `42` | Unsigned 64-bit integer |
| `preview_mode` | `"manual"` | `"manual"`, `"auto"`, or `"off"` |
| `preview_every` | `5` | Positive interval used in `auto` mode |
| `noise_source_size` | `null` | Existing experimental shared-noise option; normally omit |

`manual` supports on-demand background previews. `auto` decodes every
`preview_every` completed steps and pauses sampling for each decode. `off` skips
intermediate snapshots and previews. The final image is available through both
image endpoints in all modes. Preview settings must be chosen at submission.

## Job status

`POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, and a successful
`POST /jobs/{id}/preview` return the same job-object shape:

- `id`, `parameters`, `width`, `height`, `created_unix_ms` identify the request.
- `references` contains the original dimensions of each uploaded image, in order
  (empty for text-only jobs). `reference_index` is the zero-based image currently
  being vision-encoded, or `null` outside that stage.
- `status` is `queued`, `running`, `cancelling`, `succeeded`, `failed`, or `cancelled`.
- `stage`, `stage_completed`, `stage_total` describe the current stage. Stage
  counts reset between loading, reference vision, text encoding, reference
  encoding, denoiser loading, sampling, and decoding; they are not a whole-job percentage.
- `completed_steps` and `total_steps` report sampling progress independently of
  preview progress. `last_step_s` is the latest step's duration.
- `elapsed_s` excludes queue time; it is `null` until the job starts and stops
  increasing on completion.
- `preview_pending`, `preview_step`, and `preview_url` describe preview readiness.
- `image_url` is non-null only on success; `status_url` is always present.
- `error` contains the inference failure message, or `null`.

Image URLs are relative to the server. Fetch a new preview when `preview_step`
changes. The server retains only the latest preview and encodes each snapshot
as PNG lazily, off the inference thread. Repeated fetches reuse the encoded PNG.

## Queue, cancellation, and cleanup

| Method and path | Result |
| --- | --- |
| `GET /health` | Service availability, model ID and pinned revision, queue size, and capacity; does not load/test the model |
| `GET /jobs` | `{"jobs": [...]}` for all retained jobs, oldest first |
| `POST /jobs` | Submit JSON or multipart images + parameters; `202` with job and `Location` |
| `GET /jobs/{id}` | Current job status |
| `POST /jobs/{id}/preview` | Request a manual preview; `202` if accepted |
| `GET /jobs/{id}/preview` | Latest preview PNG, or `409` until available |
| `GET /jobs/{id}/image` | Final PNG, or `409` until successful |
| `POST /jobs/{id}/cancel` | Cancel queued/running job; harmless on terminal jobs |
| `DELETE /jobs/{id}` | Forget a terminal job and release its images; `204` |

The default capacity is **32 retained jobs**, including queued, active, and
finished jobs. Set `--max-jobs 8` to use a smaller limit (allowed range 1–128).
Full capacity returns `429`; explicitly delete finished jobs to free slots.
Running jobs cannot be deleted (`409`): cancel first and wait for `cancelled`.
Cancellation is cooperative between model operations; in-flight GPU work must
finish before cancellation completes. A job that already completed successfully
can still report `succeeded` if cancellation arrived too late.

Requests and images live in process memory and disappear on restart. Download
images before deleting a job or stopping the server. Ctrl-C stops accepting
work, cancels pending/running jobs, and waits for the generation worker to exit.
No files are written by the service itself.

Validation errors return `400` or `422`; oversized bodies return `413`; missing
jobs return `404`; unavailable images or invalid state transitions return `409`.
API errors include an `error` string. If token authentication is enabled,
missing/incorrect authorization returns `401`. Inference failures are reported
in the job's `failed` status so polling clients can distinguish them from HTTP
transport failures.

## Verification

The normal test suite covers request validation, authorization, queue capacity,
FIFO dispatch, cancellation, progress updates, and RGBA PNG responses without
loading the model. For an end-to-end check with cached weights and Metal access:

```sh
python3 scripts/smoke-http-references.py
```

This starts a temporary loopback server, uploads two generated test fixtures,
checks reference progress and manual previews, and compares the final PNG with
the equivalent local CLI edit. It stops the service before running the CLI
comparison. Images, status history, and logs go to `output/http-reference-smoke/`.
Use `--output PATH` to choose another directory.
