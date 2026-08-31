# Modulated-LoRC

An AI-generated-image (AIGC) detector: [LoRC](https://arxiv.org/abs/2608.20882)
(Low-Rank Collapse) — a frozen DINOv3 ViT-H+/16 backbone + LoRA adapters, an
orthogonal decomposition of patch tokens against the [CLS] token, and a
Low-Rank Attention Block on the resulting residual subspace — fine-tuned with
a **pair-aware energy augmentation**: each real/fake training pair has its
residual-subspace magnitude randomly rescaled by one *shared* factor, so the
model learns to be invariant to image-composition-linked energy differences
without ever corrupting the within-pair real>fake ordering the whole method
depends on (proven exactly, not just observed empirically — see
[Limitations](#limitations--future-work)).

## Project overview

- **Architecture:** frozen DINOv3 ViT-H+/16 → LoRA (rank=32, α=32, on
  q/k/v/o_proj) → `X_res = X(I − ĉĉᵀ)`, `ĉ = c/‖c‖₂` → Low-Rank Attention
  Block (rank=64) on `X_res` → `Linear([mean-pooled attention output ∥ raw
  CLS token]) → 2 classes` (0 = real, 1 = fake).
- **Training:** one epoch over the DDA-Training-Set (118,287 real/fake
  pairs), Adam lr=1e-4 with cosine warmup, λ_ss=0.1, effective batch 256 —
  plus the pair-aware energy augmentation described above, ramped in
  linearly across the epoch (0% of batches augmented at step 0 → 100% by the
  final step).
- **Weights:** ONE file, `mlorc-full.pt` (~1.6GB, bf16), hosted on a private
  HF hub repo (`buxtcodes/TechJam-Modulated-LoRC`), pulled automatically at
  runtime. LoRA is folded directly into the backbone weights (`peft`'s
  `merge_and_unload()`, verified numerically exact before shipping — max
  output difference 1e-12, pure floating-point noise), so this one file is
  fully self-contained: no separate download of the DINOv3 backbone from
  anywhere else, no LoRA config to keep in sync. No weights are committed to
  this repo itself.

## Setup and installation

```bash
git clone <this repo>
cd TechJam-Modulated-LoRC
bash setup.sh                 # creates venv/, installs requirements.txt
source venv/bin/activate
huggingface-cli login         # or: export HF_TOKEN=hf_...
```

The weights repo is private — you need HF access to
`buxtcodes/TechJam-Modulated-LoRC`. `mlorc-full.pt` is the only *weights*
file downloaded — building the backbone's architecture (before loading those
weights onto it) still fetches DINOv3's small `config.json` (a few KB, not
its weights) from `facebook/dinov3-vith16plus-pretrain-lvd1689m`, which is a
free but gated HF repo — accept its license on the model page once. Nothing
sizeable comes from anywhere but `mlorc-full.pt`.

## Repository structure

```
modulated_lorc/     the model + HF-hub weight loader (see "Directory-only inference" above)
predict.py           image directory -> JSON (image_path, pred)
evaluate/            WildFake evaluator (clean + full/transformed modes) + sample-set downloader
results/             ported WildFake eval results for mLoRC (raw CSVs/JSON + two writeups)
```

## Steps to reproduce results

**Single-image / directory prediction** (what most people want):

```bash
python predict.py --input_dir <path/to/images> --output results.json
```

`results.json` is a list of `{"image_path": ..., "pred": <float 0-1, P(fake)>}`,
one entry per image found (recursively) under `<path/to/images>`.

**Full WildFake benchmark** (what produced every number in this README and
in `results/`) — needs a local copy of the WildFake eval set
(`manifest.csv` + `transform_plan.csv` + the images; not shipped in this repo
directly — too large). Don't have one already? Pull it from the public HF
dataset `buxtcodes/WildFake-Sample` (30,000 images, ~9.5GB) first:

```bash
python evaluate/download_wildfake_sample.py --out_dir wildfake_eval                       # full set, ~9.5GB
python evaluate/download_wildfake_sample.py --out_dir wf_smoke --shards 1 --n_per_group 20 # fast, partial-coverage smoke test (~200MB)
```

`--shards N` bounds download time predictably (fetches only the first N of
10 shard files); `--n_per_group` alone does not make this fast, since the
32 WildFake groups are shard-partitioned rather than interleaved — reaching
every group still means scanning most of the 9.5GB regardless of how few
images per group get kept. Combine both for a bounded-time run, understanding
it won't cover every group.

Then evaluate:

```bash
python evaluate/evaluate_wildfake.py --data_dir <path/to/wildfake_eval> --out_dir results/wildfake_eval --mode both
```

**Verified**: this exact pipeline — fresh `git clone`, fresh venv,
`requirements.txt` only — reproduces the reported numbers. A fresh-environment
clean-mode run against the full 30k set gave BAcc=96.56%/AUC=0.9929
(reported: 96.57%/0.9929 — the 0.01pp difference is ordinary bf16/batch-size
rounding noise, not a discrepancy).

Writes `mLoRC_{clean,full}_per_image.csv`, `_by_group.csv`, `_by_condition.csv`
(full mode only), `_summary.json` — the same files already checked into
`results/wildfake_eval/` from the original run. `results/RESULTS.md` has the
full per-group breakdown (clean, then full, then transform conditions ranked
best-to-worst); `results/THROUGHPUT.md` has the throughput benchmark on its
own. Both were generated from these exact output files.

The error-analysis/robustness-summary figures below were produced by
`analysis/robustness_and_error_report.py` in the main research codebase (not
shipped here — it just reads the same per-image CSVs `evaluate_wildfake.py`
above writes). Inference throughput was likewise benchmarked directly in the
main codebase (`lorc.model.LoRC`, the same architecture this package
reproduces), not re-measured from this trimmed package, to avoid the trimmed
implementation reporting a number the "real" implementation can't back up.

## Robustness Evaluation Summary

Full 30,000-image WildFake eval, clean vs. representative transform
conditions (full per-group breakdown for clean, for full, and every one of
the 15 conditions ranked best-to-worst: `results/RESULTS.md`; raw data:
`results/wildfake_eval/`):

| condition | accuracy | balanced acc | AUC |
|---|---|---|---|
| clean | 95.97% | 96.57% | 0.9929 |
| Blur σ=2.0 | 92.71% | 93.54% | 0.9761 |
| Noise σ=0.10 | 82.38% | 85.88% | 0.9425 |
| JPEG q=30 | 81.89% | 86.11% | 0.9551 |
| Resize 0.25x | 90.08% | 88.15% | 0.9585 |
| **full (all 15 conditions pooled)** | **91.73%** | **92.65%** | **0.9723** |

Heavy JPEG compression (q=30) and aggressive downscaling (0.25x) are the
hardest single conditions — both push accuracy down ~14pp from clean, well
past blur or moderate noise.

## Error Analysis

- **Clean:** false positive rate 0.5%, false negative rate 3.53%.
- **Full (transformed):** false positive rate 1.5%, false negative rate 6.77%
  — both roughly triple under transforms, and false negatives (missed fakes)
  are consistently the larger of the two, meaning the model's dominant
  failure mode is under-flagging rather than over-flagging.
- **Representative false positives** (real images called fake, highest
  confidence): ImageNet/LAION-5B/FFHQ photos, mostly under `Resize 0.25x`,
  `Blur σ=2.0`, or heavy noise in full mode — i.e. transforms that already
  destroy most fine detail push some real photos into the same regime the
  model associates with generator artifacts.
- **Representative false negatives** (fake images called real, highest
  confidence): almost entirely `DDPM`/`ADM` (older diffusion samplers) and,
  under transforms, also `MAGE`/`DDIM` — all under `JPEG q=30/50` or heavy
  blur, where compression/blur artifacts happen to erase the residual-energy
  signature the model relies on.
- **Trade-off:** the clean→full accuracy drop is not uniform across
  generators. The five worst drops are DDPM (-16.7pp), DDIM (-15.6pp),
  **Imagen (-12.8pp)**, styleGAN (-9.9pp), starGAN (-9.2pp) — see
  `results/RESULTS.md`. Imagen (Google) is notable: it's one of
  the strongest clean-mode generators (98.9%) but has the single largest
  per-generator recall drop of any *modern* commercial generator once
  transforms are applied, a real, specific weak point of this checkpoint
  worth flagging rather than averaging away.

## Inference throughput

Full writeup: `results/THROUGHPUT.md`. Measured directly in the main training
codebase (not this trimmed package), RTX 3090 Ti, 224×224 input, `bf16`
autocast + `cudnn.benchmark=True`, batch size swept 16→224:

| config | best throughput |
|---|---|
| fp32-storage backbone | 136.6 img/s (batch=160, 4.92GB) |
| bf16-storage frozen backbone (`cast_frozen_params_to_bf16`) | **143.5 img/s** (batch=192, 3.39GB) |

Both sweeps plateau well before the 3090 Ti's 24GB fills — this is
compute-bound, not memory-bound, so the number above is a real ceiling for
this architecture on this GPU, not an artifact of an under-sized batch.

## Limitations & future work

- **Content-domain blind spot, not energy-related:** SD_LoRA / SD_LyCORIS /
  PersonalizedSD_Dreambooth remain the weakest fake groups (~72-81% BAcc)
  regardless of this augmentation, because their failures are mostly
  non-photographic content (illustration/anime art, ComfyUI workflow
  screenshots) or genuine subject-fidelity personalization — a
  content-domain mismatch the energy-based mechanism was never designed to
  fix. Given more time: a matched content-domain augmentation (or
  additional real training data in these domains) targeting this
  specifically, rather than another energy-space trick.
- **JPEG q=30 and Resize 0.25x are disproportionately hard** (~14pp below
  clean) compared to blur or moderate noise — likely because both destroy
  the fine patch-level texture the residual-subspace signal depends on more
  than the other transforms do. Worth a dedicated compression-aware
  augmentation pass rather than treating JPEG as just one condition among
  many in the existing battery.
- **Imagen (Google) regression under transforms** (98.9%→86.1% recall, the
  single largest per-generator drop in the whole eval) is a known, real
  regression this specific checkpoint introduces relative to a
  no-augmentation baseline — the real-photo gains elsewhere outweigh it in
  aggregate BAcc, but it's a genuine trade-off, not a rounding error. Worth
  investigating whether Imagen's outputs sit in an unusual part of the
  energy distribution that this augmentation happens to perturb badly.
- **Legacy generators remain hardest overall:** DDPM/DDIM/ADM (older
  diffusion samplers) are both the weakest groups in absolute terms and
  have the largest clean→full drops — plausibly because their artifacts are
  already close to the JPEG/blur noise floor, so compression erases the
  little signal that was there. Not obviously fixable by this augmentation
  approach; would need generator-specific analysis.
- **No adversarial or out-of-distribution generator testing:** all results
  are on WildFake's fixed generator roster. An adversarially-aware or
  much-more-recent (2026-era) held-out generator would better test real
  generalization than another pass over the same benchmark.
- **Merged checkpoint is bf16, ~1.6GB** (vs. the earlier partial LoRA+head
  save's ~124MB) — a deliberate trade: one self-contained file instead of
  two separate downloads from two repos. `predict.py`'s first run still
  pays a real download cost either way; this trades a smaller second
  download for a larger single one.

## Team member contributions

Solo project — all components (architecture reuse, the pair-aware energy
augmentation, training, evaluation, this repo) designed and implemented by
a single contributor. No team to attribute sub-components to.
