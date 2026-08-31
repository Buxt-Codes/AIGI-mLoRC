# mLoRC — Detailed WildFake Results

Model: mLoRC (Modulated-LoRC) (DINOv3 ViT-H+/16, LoRA rank=32/α=32, attn_rank=64, pair-aware energy augmentation α∈[0.5,1.5]). Full 30,000-image WildFake eval (10,500 real + 19,500 fake, 32 groups). Raw per-image/per-group/per-condition data: `wildfake_eval/`.

**Overall clean:** BAcc=96.57%  Acc=95.97%  AUC=0.9929  n=30000
**Overall full (all transforms pooled):** BAcc=92.65%  Acc=91.73%  AUC=0.9723  n=30000

## 1. Clean — all groups (ranked best to worst)

Every image gets only the standard q=96 JPEG pass, no other transform.

| rank | group | n | accuracy | balanced acc |
|---|---|---|---|---|
| 1 | DF-GAN | 750 | 100.00% | 100.00% |
| 2 | GALIP | 750 | 100.00% | 100.00% |
| 3 | GigaGAN | 750 | 100.00% | 100.00% |
| 4 | OriginalSD | 750 | 100.00% | 100.00% |
| 5 | PersonalizedSD_Finetune | 750 | 100.00% | 100.00% |
| 6 | VQDM | 750 | 100.00% | 100.00% |
| 7 | VQGAN | 750 | 100.00% | 100.00% |
| 8 | afhq | 1750 | 99.83% | 99.83% |
| 9 | Midjourney_v4 | 750 | 99.60% | 99.60% |
| 10 | SD_ControlNet | 750 | 99.60% | 99.60% |
| 11 | VQVAE | 750 | 99.60% | 99.60% |
| 12 | MAGE | 750 | 99.47% | 99.47% |
| 13 | church | 1750 | 99.37% | 99.37% |
| 14 | imagenet | 1750 | 99.37% | 99.37% |
| 15 | Midjourney_v5 | 750 | 99.33% | 99.33% |
| 16 | BigGAN | 750 | 99.20% | 99.20% |
| 17 | SDXL | 750 | 99.20% | 99.20% |
| 18 | Imagen | 750 | 98.93% | 98.93% |
| 19 | DALLE2 | 750 | 98.80% | 98.80% |
| 20 | DALLE3 | 750 | 98.67% | 98.67% |
| 21 | celebahq | 1750 | 97.89% | 97.89% |
| 22 | ffhq | 1750 | 97.89% | 97.89% |
| 23 | laion5b | 1750 | 97.14% | 97.14% |
| 24 | styleGAN | 750 | 96.53% | 96.53% |
| 25 | MAE | 750 | 94.93% | 94.93% |
| 26 | ADM | 750 | 93.07% | 93.07% |
| 27 | starGAN | 750 | 91.33% | 91.33% |
| 28 | SD_LoRA | 750 | 81.33% | 81.33% |
| 29 | PersonalizedSD_Dreambooth | 750 | 80.80% | 80.80% |
| 30 | SD_LyCORIS | 750 | 79.87% | 79.87% |
| 31 | DDIM | 750 | 77.33% | 77.33% |
| 32 | DDPM | 750 | 71.07% | 71.07% |

## 2. Full (transformed) — all groups (ranked best to worst)

Every image gets its assigned condition from the 15-condition battery (Blur/Noise/JPEG/Resize/CenterCrop/ColorJitter at multiple severities), applied first, then a final q=96 JPEG pass.

| rank | group | n | accuracy | balanced acc |
|---|---|---|---|---|
| 1 | DF-GAN | 750 | 100.00% | 100.00% |
| 2 | VQGAN | 750 | 99.87% | 99.87% |
| 3 | Midjourney_v4 | 750 | 99.73% | 99.73% |
| 4 | OriginalSD | 750 | 99.73% | 99.73% |
| 5 | GigaGAN | 750 | 99.20% | 99.20% |
| 6 | GALIP | 750 | 98.93% | 98.93% |
| 7 | VQVAE | 750 | 98.80% | 98.80% |
| 8 | church | 1750 | 98.69% | 98.69% |
| 9 | afhq | 1750 | 97.89% | 97.89% |
| 10 | DALLE2 | 750 | 97.20% | 97.20% |
| 11 | VQDM | 750 | 96.93% | 96.93% |
| 12 | imagenet | 1750 | 96.80% | 96.80% |
| 13 | PersonalizedSD_Finetune | 750 | 96.67% | 96.67% |
| 14 | MAGE | 750 | 96.13% | 96.13% |
| 15 | DALLE3 | 750 | 95.87% | 95.87% |
| 16 | BigGAN | 750 | 95.73% | 95.73% |
| 17 | SD_ControlNet | 750 | 95.07% | 95.07% |
| 18 | celebahq | 1750 | 94.74% | 94.74% |
| 19 | ffhq | 1750 | 94.40% | 94.40% |
| 20 | Midjourney_v5 | 750 | 94.13% | 94.13% |
| 21 | SDXL | 750 | 93.20% | 93.20% |
| 22 | laion5b | 1750 | 91.83% | 91.83% |
| 23 | MAE | 750 | 91.20% | 91.20% |
| 24 | styleGAN | 750 | 86.67% | 86.67% |
| 25 | Imagen | 750 | 86.13% | 86.13% |
| 26 | ADM | 750 | 85.73% | 85.73% |
| 27 | starGAN | 750 | 82.13% | 82.13% |
| 28 | PersonalizedSD_Dreambooth | 750 | 74.93% | 74.93% |
| 29 | SD_LyCORIS | 750 | 74.93% | 74.93% |
| 30 | SD_LoRA | 750 | 74.00% | 74.00% |
| 31 | DDIM | 750 | 61.73% | 61.73% |
| 32 | DDPM | 750 | 54.40% | 54.40% |

## 3. Transform conditions — ranked best to worst

Same full-mode run, aggregated by which transform condition was applied (pooled across all groups/generators that drew that condition).

| rank | condition | n | accuracy | balanced acc | AUC |
|---|---|---|---|---|---|
| 1 | Blur sigma=0.5 | 2154 | 96.47% | 96.86% | 0.9931 |
| 2 | Blur sigma=1.0 | 2154 | 96.01% | 96.25% | 0.9899 |
| 3 | ColorJitter +-20% | 2128 | 95.72% | 96.12% | 0.9890 |
| 4 | JPEG q=90 | 2154 | 94.99% | 95.91% | 0.9893 |
| 5 | Resize 0.5x | 2154 | 95.13% | 95.11% | 0.9875 |
| 6 | CenterCrop 80% | 2128 | 95.49% | 95.09% | 0.9893 |
| 7 | Blur sigma=2.0 | 2154 | 92.71% | 93.54% | 0.9761 |
| 8 | JPEG q=70 | 2154 | 91.32% | 93.09% | 0.9812 |
| 9 | Noise sigma=0.02 | 2128 | 93.66% | 92.61% | 0.9800 |
| 10 | Noise sigma=0.05 | 2128 | 91.21% | 92.46% | 0.9712 |
| 11 | JPEG q=50 | 2154 | 87.14% | 90.01% | 0.9651 |
| 12 | Resize 0.25x | 2128 | 90.08% | 88.15% | 0.9585 |
| 13 | JPEG q=30 | 2154 | 81.89% | 86.11% | 0.9551 |
| 14 | Noise sigma=0.10 | 2128 | 82.38% | 85.88% | 0.9425 |

## 4. Detailed observations

**Strongest clean-mode groups (100% BAcc):** DF-GAN, GALIP, GigaGAN, OriginalSD, PersonalizedSD_Finetune, VQDM, VQGAN — mostly older/simpler GAN or VAE-style generators with strong, easy-to-detect artifacts.

**Weakest clean-mode groups:** DDPM (71.07%), DDIM (77.33%), SD_LyCORIS (79.87%), PersonalizedSD_Dreambooth (80.80%), SD_LoRA (81.33%) — legacy diffusion samplers (DDPM/DDIM) and the personalization/fine-tuning-based Stable Diffusion variants.

**Transform ranking pattern:** mild transforms (Blur σ=0.5/1.0, ColorJitter, high-quality JPEG q=90, CenterCrop 80%) barely move BAcc from clean (all ≥95%). The four worst conditions are all either heavy compression or heavy resolution loss: Noise σ=0.10 (85.88%), JPEG q=30 (86.11%), Resize 0.25x (88.15%), JPEG q=50 (90.01%) — information-destroying transforms hurt far more than geometric or color transforms of comparable "severity label".

**Clean vs. full, same group:** the ranking is broadly stable at the top and bottom (DDPM/DDIM/SD_LoRA/SD_LyCORIS/PersonalizedSD_Dreambooth are the weakest in both modes) but Imagen and styleGAN drop sharply under transforms despite being strong on clean (Imagen: rank 18→25, 98.93%→86.13%; styleGAN: rank 23→24, 96.53%→86.67%) — see `../ablations/group_tradeoffs.json` for the full clean→full delta ranking and `../ablations/error_analysis.json` for representative failure examples.

