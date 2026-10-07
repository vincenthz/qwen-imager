# HTTP generation service

Build and start the same CLI in server mode:

```sh
cargo build --release --bin image-forger-cli
./target/release/image-forger-cli --serve --offline --key-file imageforger.key
```

The default address is `127.0.0.1:6996`. `--model`, `--model-dir PATH` and
`--offline` work as in the normal CLI; one service serves one checkpoint. The first job loads the model; later jobs reuse its
loaded weights. One job runs at a time, with other jobs queued in submission
order. HTTP requests remain responsive while inference runs.

For another port, use `--listen 127.0.0.1:9000`. For access from another machine,
set `IMAGEFORGER_API_TOKEN` and use `--listen 0.0.0.0:6996`; include
`Authorization: Bearer YOUR_TOKEN` on every request, including `/health`.
The token is also enforced on localhost when set. Prompts and images are always
content-encrypted, including over plain HTTP. Bearer tokens and status/control
metadata remain outside this encryption; HTTPS can also protect those fields.
The CLI loads or creates `--key-file PATH` (default `imageforger.key`), prints its
64-character hexadecimal X25519 public identity, and requires Unix key-file
permissions of `0600` or stricter. Back up the private key securely and keep it
across restarts. Invalid files fail closed rather than rotating the identity.

In the desktop GUI, add the service address in **Settings → Compute**, paste the
same `IMAGEFORGER_API_TOKEN` into that host's masked **API token** field (without
the `Bearer` prefix), paste the public identity from the server console into the
identity field, and click **Save & test**. The token is saved in macOS
Keychain, separately from the settings file. The test calls authenticated
`POST /identity` and verifies an encrypted response from the pinned key without
submitting a job or loading model weights. Pins are saved per normalized endpoint
in settings, and are never discovered or silently replaced over HTTP. Select the host
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

## Content encryption v1

This is a breaking change for older HTTP clients: plaintext JSON and multipart
submissions receive `415`; there is no plaintext fallback. Use the GUI or the Rust
`image_forger::remote::run_authenticated(request, url, token, pinned_identity,
cancellation, on_event)` API. It creates a new OS-random X25519 client key for each
generation, encrypts the upload, and decrypts previews and the final image. Only
that request's client context can read its images. Restarting the GUI loses those
in-memory keys; download/save results before closing it.

Cryptoxide supplies X25519, HKDF-SHA256, and ChaCha20-Poly1305. The remote public
identity must be copied through a trusted channel. This static-server exchange
does not provide forward secrecy if the server private key is later compromised.
Content encryption protects transmission, not local image files, saved workspaces,
or the generation server's memory.

All encrypted HTTP bodies use `Content-Type: application/vnd.imageforger.encrypted-v1`.
Byte concatenations below have no separators; integers are big-endian.

1. Each request context has a fresh client identity, 32 random salt bytes, and a
   Unix-seconds timestamp. Its 76-byte header is ASCII `IFC1` + client public key
   (32 bytes) + salt (32 bytes) + timestamp (8 bytes).
2. Compute X25519(client private, pinned server public); reject an all-zero shared
   secret. HKDF-Extract uses the salt and shared secret. HKDF-Expand's info is
   ASCII `ImageForger content v1 X25519 HKDF-SHA256 ChaCha20-Poly1305` + server
   public key + the complete header. Expand to 64 bytes: upload key first,
   download key second, each 32 bytes.
3. A request body is header + nonce (12 bytes) + ciphertext + tag (16 bytes).
   The nonce is four zero bytes + a monotonically increasing 64-bit counter,
   starting at zero in each new context. AEAD additional data is the UTF-8
   method and canonical API path: `POST /jobs` or `POST /identity`.
4. Each response has a new random 32-byte message salt. HKDF-Expand of the download
   key with info ASCII `ImageForger response v1` + message salt produces its
   32-byte message key. This also avoids reusing a key/nonce pair after a server
   restart or replayed identity challenge. The response body is message salt +
   nonce (12 bytes) + ciphertext + tag (16 bytes). Its nonce follows the same
   counter layout. Additional data is `GET /jobs/{id}/preview`,
   `GET /jobs/{id}/image`, or `POST /identity`, as appropriate.
5. Authenticate before parsing or decoding plaintext. Clients reject unencrypted
   image/identity responses, incorrect pins, changed bodies, and substituted
   image paths. Reverse proxy prefixes are not part of the additional data.

`POST /identity` encrypts the literal ASCII `identity`; its encrypted response
contains JSON with the `model` field. It proves possession of the pinned private
key. `GET /health` remains ordinary authenticated JSON and is only a health check.

The timestamp must be within five minutes of the server clock. Accepted upload
contexts cannot be reused: the server retains a bounded replay cache (4096
entries, up to ten minutes). A full cache returns `429`; duplicate uploads return
`409`. This cache is in memory and resets on server restart. New attempts should
create a fresh request context.

## Submit and retrieve an image

The plaintext inside `POST /jobs` is a multipart form with boundary
`----image-forger-boundary`. Include exactly one `parameters` field containing
JSON, for example:

```json
{"prompt":"Change the background to a sunset beach","scale":0.25,"steps":8,"seed":42}
```

Repeat the `images` field for each reference, in input order; omit it for text-only
jobs. The complete multipart body is encrypted before transmission, including the
prompt, images, and form headers. PNG, JPEG, WebP, and HEIC/HEIF are accepted,
detected from the decrypted bytes. Alpha and photo orientation are preserved.
Filenames are not server paths and uploads are not written to disk.

Submission returns ordinary `202` JSON with a job ID and `Location`. Poll
`GET /jobs/{id}` for metadata. `POST /jobs/{id}/preview` requests a manual preview;
`GET /jobs/{id}/preview` and `GET /jobs/{id}/image` return encrypted PNG bytes.
Decrypt using the original request context before opening/saving the PNG.
Fetching a preview does not request decoding. A pending decode returns `409` if
another is requested. `preview_step` and `X-Preview-Step` identify its snapshot.

Status/control calls still work with ordinary HTTP clients:

```sh
curl --fail http://127.0.0.1:6996/jobs/1 -H "Authorization: Bearer $IMAGEFORGER_API_TOKEN"
```

`references` lists each input's oriented dimensions. Without `ratio`, output
follows the last reference's aspect ratio; an explicit ratio overrides it.

Upload limits:

- Up to **10 images**, with **16 MiB** (16,777,216 bytes) maximum for the entire
  multipart body by default, including images, parameters, and form headers.
  An oversized upload returns `413` with the limit in its error message. The GUI
  uploads references as PNG, so the transmitted size can differ from the original
  JPEG or HEIC file size.
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

The encrypted multipart `parameters` field accepts these fields. Unknown fields
are rejected and the maximum parameters size is 64 KiB. Server file paths are
never accepted.

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
  `parameters` omits `prompt` in every status/list/control response.
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
- `error` contains a generic inference failure message, or `null`; underlying
  failures stay in server diagnostics so status cannot echo private content.

Image URLs are relative to the server. Fetch a new preview when `preview_step`
changes. The server retains only the latest preview and encodes each snapshot
as PNG lazily, off the inference thread. Repeated fetches reuse the encoded PNG.

## Queue, cancellation, and cleanup

| Method and path | Result |
| --- | --- |
| `GET /health` | Service availability, model ID and pinned revision, queue size, and capacity; does not load/test the model |
| `GET /jobs` | `{"jobs": [...]}` for all retained jobs, oldest first |
| `POST /identity` | Encrypted proof of the pinned server identity |
| `POST /jobs` | Submit encrypted multipart images + parameters; `202` with job and `Location` |
| `GET /jobs/{id}` | Current job status |
| `POST /jobs/{id}/preview` | Request a manual preview; `202` if accepted |
| `GET /jobs/{id}/preview` | Encrypted latest preview PNG, or `409` until available |
| `GET /jobs/{id}/image` | Encrypted final PNG, or `409` until successful |
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
The identity key file is created when missing; image content is kept in memory.

Validation errors return `400` or `422`; oversized bodies return `413`; missing
jobs return `404`; unavailable images or invalid state transitions return `409`.
API errors include an `error` string. If token authentication is enabled,
missing/incorrect authorization returns `401`. Inference failures are reported
in the job's `failed` status so polling clients can distinguish them from HTTP
transport failures.

## Verification

`cargo test --all-features` covers encryption round trips, tampering, wrong pins,
low-order X25519 inputs, expired/replayed uploads, private key-file reuse,
plaintext rejection, prompt/error redaction, and real loopback client/server
preview and image transfers without loading model weights. Tests that bind
loopback sockets need local network permissions. Existing Metal/model tests
remain opt-in.
