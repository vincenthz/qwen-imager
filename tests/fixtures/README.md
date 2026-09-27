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
