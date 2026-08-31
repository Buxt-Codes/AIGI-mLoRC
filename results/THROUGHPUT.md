# Run 1 — Inference Throughput

Measured **directly in the main research codebase** (`lorc.model.LoRC`, the
same architecture this repo's `modulated_lorc/model.py` reproduces) — not
derived from this trimmed package — to avoid reporting a number the "real"
implementation can't back up.

**Setup:** RTX 3090 Ti (24GB), 224×224 input, `torch.autocast(bf16)` +
`cudnn.benchmark=True`, pure forward-pass (no dataloader, no backward),
5 warmup iterations then 20 timed iterations per batch size, batch size
swept from 16 to 224.

## fp32-storage backbone + bf16 autocast

| batch size | throughput | peak VRAM |
|---|---|---|
| 16 | 109.0 img/s | 3.46GB |
| 32 | 117.6 img/s | 3.62GB |
| 64 | 127.7 img/s | 3.94GB |
| 96 | 136.3 img/s | 4.27GB |
| 128 | 136.5 img/s | 4.60GB |
| **160 (best)** | **136.6 img/s** | 4.92GB |

## bf16-storage frozen backbone + bf16 autocast

Casting every frozen (non-trainable) parameter to true `bf16` storage
(`cast_frozen_params_to_bf16` in the main codebase's `lorc/utils.py`,
numerically verified elsewhere in that project to be lossless — identical
AUC, 100% decision agreement, max |Δp(fake)| 0.019 vs. fp32 storage on real
checkpoints) both increases throughput and roughly halves peak VRAM, since
it removes the standing fp32 storage entirely rather than just the compute
dtype.

| batch size | throughput | peak VRAM |
|---|---|---|
| 16 | 117.9 img/s | 1.81GB |
| 32 | 124.5 img/s | 1.95GB |
| 64 | 134.0 img/s | 2.24GB |
| 96 | 143.1 img/s | 2.53GB |
| 128 | 143.1 img/s | 2.82GB |
| 160 | 143.0 img/s | 3.10GB |
| **192 (best)** | **143.5 img/s** | 3.39GB |
| 224 | 143.2 img/s | 3.68GB |

## Summary

**Maximum measured throughput: ~143.5 images/sec** on this RTX 3090 Ti
(batch=192, bf16-storage backbone). Both sweeps plateau (further batch
increase buys <1% more throughput while VRAM keeps climbing) well before
the card's 24GB fills — this is compute-bound, not memory-bound, so this is
a real ceiling for this architecture on this GPU, not an artifact of an
under-sized batch or an untested larger batch.
