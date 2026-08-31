"""Evaluate Modulated-LoRC against a local copy of the WildFake eval set
(`manifest.csv` + `transform_plan.csv`; not shipped here — point
--data_dir at your own copy).

--mode clean: every image gets only a q=96 JPEG pass.
--mode full:  every image gets its transform_plan.csv condition, then a
              final q=96 JPEG pass (matches training's mandatory-JPEG
              convention).
--mode both (default): both, one manifest load.

Usage:
    python evaluate_wildfake.py --data_dir /path/to/wildfake_eval --out_dir results/wildfake_eval
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlorc import ModulatedLoRC
from transforms import FAMILY_BY_NAME, RAW_BY_NAME, stack_baseline_last, tf_jpeg

LABEL = "mLoRC"
IMAGE_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

_CROP = transforms.CenterCrop(IMAGE_SIZE)
_NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_manifest(data_dir: Path) -> list[dict]:
    with open(data_dir / "manifest.csv") as f:
        return list(csv.DictReader(f))


def load_transform_plan(data_dir: Path) -> dict[str, str]:
    with open(data_dir / "transform_plan.csv") as f:
        return {r["local_path"]: r["condition"] for r in csv.DictReader(f)}


def _to_tensor(img: Image.Image, corrupt_fn) -> torch.Tensor:
    return _NORM(_CROP(corrupt_fn(img)))


@torch.no_grad()
def run_pass(model: ModulatedLoRC, data_dir: Path, rows: list[dict], conditions: list[str],
             batch_size: int) -> list[dict]:
    out = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_conds = conditions[start:start + batch_size]
        tensors = []
        for row, cond in zip(batch_rows, batch_conds):
            img = Image.open(data_dir / row["local_path"]).convert("RGB")
            fn = tf_jpeg(96) if cond == "Clean (q=96)" else stack_baseline_last(RAW_BY_NAME[cond])
            tensors.append(_to_tensor(img, fn))
        batch = torch.stack(tensors).to(model.device)
        with torch.autocast(device_type=model.device, dtype=torch.bfloat16, enabled=(model.device == "cuda")):
            probs, _ = model.model.predict(batch)
        p_fake = probs[:, 1].float().cpu().numpy()

        for row, cond, p in zip(batch_rows, batch_conds, p_fake):
            label = 1 if row["split"] == "fake" else 0
            pred = int(p >= 0.5)
            out.append({
                "local_path": row["local_path"], "group": row["group"], "split": row["split"],
                "condition": cond, "family": FAMILY_BY_NAME.get(cond, "Clean"),
                "p_fake": float(p), "label": label, "pred": pred, "correct": int(pred == label),
            })
        if (start // batch_size) % 20 == 0:
            print(f"  {start + len(batch_rows)}/{len(rows)}", flush=True)
    return out


def summarize(rows: list[dict], key: str) -> list[dict]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key] if key else "ALL"].append(r)
    summary = []
    for k, items in sorted(buckets.items()):
        labels = np.array([it["label"] for it in items])
        probs = np.array([it["p_fake"] for it in items])
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == labels).mean() * 100)
        bacc = float(balanced_accuracy_score(labels, preds) * 100) if len(np.unique(labels)) > 1 else acc
        try:
            auc = float(roc_auc_score(labels, probs))
        except ValueError:
            auc = float("nan")
        summary.append({key or "bucket": k, "n": len(items), "accuracy_pct": acc, "balanced_acc_pct": bacc, "auc": auc})
    return summary


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, type=Path, help="Local WildFake eval dir (manifest.csv + transform_plan.csv)")
    p.add_argument("--out_dir", default=Path("results/wildfake_eval"), type=Path)
    p.add_argument("--mode", default="both", choices=["clean", "full", "both"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hf_token", default=None)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    rows = load_manifest(args.data_dir)
    print(f"{len(rows)} images in manifest")

    print("Loading model...")
    model = ModulatedLoRC.from_pretrained(hf_token=args.hf_token)
    print("Model loaded.\n")

    modes = ["clean", "full"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"=== mode={mode} ===")
        if mode == "clean":
            conditions = ["Clean (q=96)"] * len(rows)
        else:
            plan = load_transform_plan(args.data_dir)
            conditions = [plan[r["local_path"]] for r in rows]

        per_image = run_pass(model, args.data_dir, rows, conditions, args.batch_size)

        write_csv(per_image, args.out_dir / f"{LABEL}_{mode}_per_image.csv")
        overall = summarize(per_image, "")
        by_group = summarize(per_image, "group")
        by_condition = summarize(per_image, "condition") if mode == "full" else []
        with open(args.out_dir / f"{LABEL}_{mode}_summary.json", "w") as f:
            json.dump({"overall": overall, "by_group": by_group, "by_condition": by_condition}, f, indent=2)
        write_csv(by_group, args.out_dir / f"{LABEL}_{mode}_by_group.csv")
        if by_condition:
            write_csv(by_condition, args.out_dir / f"{LABEL}_{mode}_by_condition.csv")

        print(f"  Overall: BAcc={overall[0]['balanced_acc_pct']:.2f}  AUC={overall[0]['auc']:.4f}  n={overall[0]['n']}")
        print(f"  Saved to {args.out_dir}\n")

    print("DONE")


if __name__ == "__main__":
    main()
