# mLoRC — Inference Throughput

Measured **directly in the main research codebase** (`lorc.model.LoRC`, the
same architecture this repo's `mlorc/model.py` reproduces) — not
derived from this trimmed package — to avoid reporting a number the "real"
implementation can't back up. Uses the **merged checkpoint** (`mlorc-full.pt`
— LoRA already folded into the backbone weights, no `peft` wrapper at
inference time), the same one this repo ships.

**Setup:** RTX 3090 Ti (24GB), 224×224 input, `torch.autocast(bf16)` +
`cudnn.benchmark=True`, pure forward-pass (no dataloader, no backward),
5 warmup iterations then 20 timed iterations per batch size, batch size
swept from 16 to 224.

## fp32-storage backbone + bf16 autocast

| batch size | throughput | peak VRAM |
|---|---|---|
| 16 | 122.3 img/s | 4.79GB |
| 32 | 131.1 img/s | 3.56GB |
| 64 | 143.5 img/s | 3.89GB |
| 96 | 153.9 img/s | 4.21GB |
| **128 (best)** | **153.9 img/s** | 4.54GB |
| 160 | 153.9 img/s | 4.86GB |

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
| 16 | 133.8 img/s | 1.75GB |
| 32 | 140.3 img/s | 1.89GB |
| 64 | 151.3 img/s | 2.18GB |
| 96 | 162.4 img/s | 2.47GB |
| 128 | 162.7 img/s | 2.76GB |
| **160 (best)** | **163.5 img/s** | 3.04GB |
| 192 | 162.8 img/s | 3.33GB |
| 224 | 162.9 img/s | 3.62GB |

## Summary

**Maximum measured throughput: ~163.5 images/sec** on this RTX 3090 Ti
(batch=160, bf16-storage backbone). Both sweeps plateau (further batch
increase buys <1% more throughput while VRAM keeps climbing) well before
the card's 24GB fills — this is compute-bound, not memory-bound, so this is
a real ceiling for this architecture on this GPU, not an artifact of an
under-sized batch or an untested larger batch.

**~13-14% faster than the pre-merge (LoRA-wrapped) architecture** (previously
136.6/143.5 img/s) — removing the `peft` adapter-dispatch overhead at
inference time was a genuine, measured speedup, not just a loading
convenience. Re-measured after switching to the merged checkpoint rather than
assumed equivalent, since output-equivalence (verified separately, see
`RESULTS.md`) doesn't imply speed-equivalence.
