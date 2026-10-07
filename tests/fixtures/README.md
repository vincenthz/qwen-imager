# Reference tensors

`vae.safetensors` contains deterministic input and output tensors, not model
weights. Generated with `AutoencoderKLQwenImage21` from the Diffusers installation
used by the original app, with the Qwen Image 2.1 checkpoint at revision
`790c92633540aa0cb11d9abf19eb46d861714758`, in float32 on CPU, inference mode.

- `input`: `sin(arange(4*32*32) * 0.01)`, shaped `[1,4,32,32]`.
- `latent`: `cos(arange(64*2*2) * 0.02)`, shaped `[1,64,2,2]`.
- `encoded`: VAE posterior **mode**, normalized using the checkpoint's latent
  mean and standard deviation.
- `decoded`: VAE output from `latent * std + mean`, including the reference
  decoder's clamp to `[-1,1]`.

The reference adds a single-frame axis when invoking Diffusers and removes it
when saving. Tests use float32 Metal inference and check maximum absolute
error below 0.002; no Python is needed to run them. Diffusers source revision:
`e0118ade2f60234c41bacf40330a7e2f61108849`.

`attention.safetensors` contains BF16 queries, keys, and values captured from
the checkpoint's first text-encoder layer for the prompt “A studio photograph
of a red ceramic teapot on a white table, soft natural lighting, highly
detailed”. It reproduces non-finite output in Candle 0.9.2's BF16 fused Metal
attention kernel. The test compares the FP32 Metal attention path against
an independent dense CPU softmax implementation, including a finite-value check.

`quantized.safetensors` contains a small synthetic linear layer quantized by
MLX 0.32.3 (`mlx.core.quantize`, group size 64) at 4 and 8 bits, not model
weights. The dense BF16 weight is `sin(arange(96*192) * 0.37) * 0.05`, shaped
`[96,192]`, and the input `x` is `cos(arange(5*192) * 0.11)`, shaped `[5,192]`,
both in BF16. For each `q4`/`q8` prefix it stores the packed `u32` `weight`,
BF16 `scales` and `biases` as MLX writes them, MLX's `dequantized` weight, and
`y = quantized_matmul(x, weight, scales, biases, transpose=True)`. Tests check
the Metal and CPU dequantization against both.

`content-crypto-v1.json` is an independent wire-format vector generated with
Node.js 26's built-in OpenSSL crypto. Test-only X25519 private keys are 32 bytes
of `0x01` (server) and `0x02` (client). The request salt is 32 bytes of `0x03`,
timestamp 1700000000, and nonce zero. The response salt is 32 bytes of `0x04`
and nonce zero. Plaintexts are `private prompt` and `PNG test bytes`, with
additional data `POST /jobs` and `GET /jobs/1/image`, respectively. It verifies
both directions against an implementation independent of cryptoxide; Node is
not needed to run the Rust test. See `docs/http-api.md` for the derivation.
