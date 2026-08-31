# Robustness Evaluation Summary

| condition | n | accuracy | balanced acc | AUC |
|---|---|---|---|---|
| clean | 30000 | 95.97% | 96.57% | 0.9929 |
| Blur sigma=0.5 | 2154 | 96.47% | 96.86% | 0.9931 |
| Blur sigma=1.0 | 2154 | 96.01% | 96.25% | 0.9899 |
| Blur sigma=2.0 | 2154 | 92.71% | 93.54% | 0.9761 |
| CenterCrop 80% | 2128 | 95.49% | 95.09% | 0.9893 |
| ColorJitter +-20% | 2128 | 95.72% | 96.12% | 0.989 |
| JPEG q=30 | 2154 | 81.89% | 86.11% | 0.9551 |
| JPEG q=50 | 2154 | 87.14% | 90.01% | 0.9651 |
| JPEG q=70 | 2154 | 91.32% | 93.09% | 0.9812 |
| JPEG q=90 | 2154 | 94.99% | 95.91% | 0.9893 |
| Noise sigma=0.02 | 2128 | 93.66% | 92.61% | 0.98 |
| Noise sigma=0.05 | 2128 | 91.21% | 92.46% | 0.9712 |
| Noise sigma=0.10 | 2128 | 82.38% | 85.88% | 0.9425 |
| Resize 0.25x | 2128 | 90.08% | 88.15% | 0.9585 |
| Resize 0.5x | 2154 | 95.13% | 95.11% | 0.9875 |
| full (all transforms pooled) | 30000 | 91.73% | 92.65% | 0.9723 |

# Error Analysis

## clean mode
False positive rate (real called fake): 0.5%
False negative rate (fake called real): 3.53%

**Most confident false positives (real → predicted fake):**
- `real/imagenet/imagenet_03d26de8c51a.png` (group=imagenet, condition=Clean (q=96), p_fake=1.000)
- `real/laion5b/laion5b_65ed2f54f641.png` (group=laion5b, condition=Clean (q=96), p_fake=0.998)
- `real/laion5b/laion5b_a1c0ccdf9204.png` (group=laion5b, condition=Clean (q=96), p_fake=0.996)
- `real/ffhq/ffhq_68f13b0b58da.png` (group=ffhq, condition=Clean (q=96), p_fake=0.993)
- `real/laion5b/laion5b_3dbf65c940d4.png` (group=laion5b, condition=Clean (q=96), p_fake=0.987)

**Most confident false negatives (fake → predicted real):**
- `fake/DDPM/DDPM_f72274d30c2a.png` (group=DDPM, condition=Clean (q=96), p_fake=0.000)
- `fake/ADM/ADM_e2777c9eef23.png` (group=ADM, condition=Clean (q=96), p_fake=0.000)
- `fake/ADM/ADM_65ac901a0c6c.png` (group=ADM, condition=Clean (q=96), p_fake=0.000)
- `fake/DDPM/DDPM_1bbf2b3f66cb.png` (group=DDPM, condition=Clean (q=96), p_fake=0.000)
- `fake/DDPM/DDPM_78396bb406bd.png` (group=DDPM, condition=Clean (q=96), p_fake=0.000)

## full mode
False positive rate (real called fake): 1.5%
False negative rate (fake called real): 6.77%

**Most confident false positives (real → predicted fake):**
- `real/imagenet/imagenet_1798b60ced9f.png` (group=imagenet, condition=Resize 0.25x, p_fake=1.000)
- `real/celebahq/celebahq_b82f658978ac.png` (group=celebahq, condition=Resize 0.25x, p_fake=1.000)
- `real/laion5b/laion5b_3dbf65c940d4.png` (group=laion5b, condition=Resize 0.25x, p_fake=1.000)
- `real/laion5b/laion5b_2121f2660979.png` (group=laion5b, condition=Blur sigma=2.0, p_fake=1.000)
- `real/laion5b/laion5b_4c7a72b840f3.png` (group=laion5b, condition=Noise sigma=0.02, p_fake=1.000)

**Most confident false negatives (fake → predicted real):**
- `fake/MAGE/MAGE_55590df0f9a0.png` (group=MAGE, condition=JPEG q=30, p_fake=0.000)
- `fake/DDIM/DDIM_fca216fa30fa.png` (group=DDIM, condition=JPEG q=50, p_fake=0.000)
- `fake/DDPM/DDPM_1bbf2b3f66cb.png` (group=DDPM, condition=JPEG q=50, p_fake=0.000)
- `fake/ADM/ADM_65ac901a0c6c.png` (group=ADM, condition=JPEG q=50, p_fake=0.000)
- `fake/ADM/ADM_e2777c9eef23.png` (group=ADM, condition=Blur sigma=1.0, p_fake=0.000)


# Clean → Full Trade-offs (largest accuracy drops)

| group | clean acc | full acc | Δ |
|---|---|---|---|
| DDPM | 71.07% | 54.4% | -16.67pp |
| DDIM | 77.33% | 61.73% | -15.6pp |
| Imagen | 98.93% | 86.13% | -12.8pp |
| styleGAN | 96.53% | 86.67% | -9.87pp |
| starGAN | 91.33% | 82.13% | -9.2pp |
