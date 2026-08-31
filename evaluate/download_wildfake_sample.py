"""Download the WildFake sample eval set (`buxtcodes/WildFake-Sample`,
public, 30,000 images / ~9.5GB across 10 parquet shards) and materialize it
into the manifest.csv + transform_plan.csv + images layout
evaluate_wildfake.py expects.

Two ways to bound the download:
  --shards N     Only fetch the first N of 10 shard files (predictable,
                  fast -- genuinely quick for "does the pipeline work").
                  Does NOT guarantee every one of the 32 groups appears,
                  since groups are shard-partitioned, not interleaved.
  --n_per_group N  Cap images per (split, group) pair once encountered.
                  Combine with --shards for a bounded-time run; used alone
                  it still has to stream past most/all shards to find
                  every group at least once (each row's other columns get
                  deserialized even when skipped), so it is NOT a fast
                  "quick test" by itself -- it only bounds disk usage.

Usage:
    python evaluate/download_wildfake_sample.py --out_dir wildfake_eval                          # full, ~9.5GB
    python evaluate/download_wildfake_sample.py --out_dir wf_smoke --shards 1 --n_per_group 20    # fast, partial-group smoke test

Then:
    python evaluate/evaluate_wildfake.py --data_dir wildfake_eval --out_dir results/wildfake_eval
"""
import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image

REPO_ID = "buxtcodes/WildFake-Sample"
N_SHARDS_TOTAL = 10

REAL_GROUPS = ["afhq", "celebahq", "church", "ffhq", "imagenet", "laion5b"]
FAKE_GROUPS = [
    "ADM", "BigGAN", "DALLE2", "DALLE3", "DDIM", "DDPM", "DF-GAN", "GALIP", "GigaGAN",
    "Imagen", "MAE", "MAGE", "Midjourney_v4", "Midjourney_v5", "OriginalSD",
    "PersonalizedSD_Dreambooth", "PersonalizedSD_Finetune", "SDXL", "SD_ControlNet",
    "SD_LoRA", "SD_LyCORIS", "VQDM", "VQGAN", "VQVAE", "starGAN", "styleGAN",
]
ALL_KEYS = {("real", g) for g in REAL_GROUPS} | {("fake", g) for g in FAKE_GROUPS}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--n_per_group", type=int, default=None,
                    help="Cap images per (split, group) pair -- default: all (~30,000 total)")
    p.add_argument("--shards", type=int, default=None,
                    help=f"Only fetch the first N of {N_SHARDS_TOTAL} shard files -- bounds download "
                         "time predictably, at the cost of not covering every group.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.shards is not None:
        files = [f"data/train-{i:05d}-of-{N_SHARDS_TOTAL:05d}.parquet" for i in range(args.shards)]
        ds = load_dataset(REPO_ID, split="train", streaming=True, data_files={"train": files})
    else:
        ds = load_dataset(REPO_ID, split="train", streaming=True)
    manifest, plan = [], []
    counts, done = defaultdict(int), set()

    for row in ds:
        key = (row["split"], row["group"])
        if key in done:
            continue
        if args.n_per_group is not None and counts[key] >= args.n_per_group:
            done.add(key)
            if done >= ALL_KEYS:
                break  # every group has hit its cap -- no need to keep streaming
            continue
        counts[key] += 1
        rel_path = f"{row['split']}/{row['group']}/{row['split']}_{row['group']}_{counts[key]:05d}.png"
        out_path = args.out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(row["image_bytes"])).convert("RGB").save(out_path)
        manifest.append({"local_path": rel_path, "group": row["group"], "split": row["split"]})
        plan.append({"local_path": rel_path, "condition": row["condition"]})

        n = sum(counts.values())
        if n % 100 == 0:
            print(f"  {n} images written...", flush=True)

    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["local_path", "group", "split"])
        w.writeheader()
        w.writerows(manifest)
    with open(args.out_dir / "transform_plan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["local_path", "condition"])
        w.writeheader()
        w.writerows(plan)

    print(f"\nWrote {len(manifest)} images + manifest.csv + transform_plan.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
