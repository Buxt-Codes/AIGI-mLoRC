"""Run Modulated-LoRC over every image in a directory and write a
per-image AIGC-likelihood score to a JSON file.

Usage:
    python predict.py --input_dir path/to/images --output results.json

Output (results.json): a list of
    {"image_path": "<path>", "pred": <float in [0,1], P(image is AI-generated)>}
"""
import argparse
import json
from pathlib import Path

from mlorc import ModulatedLoRC

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, type=Path, help="Directory of images (searched recursively)")
    p.add_argument("--output", default=Path("results.json"), type=Path)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hf_token", default=None, help="HF token for the private weights repo "
                    "(or set the HF_TOKEN env var / run `huggingface-cli login` beforehand)")
    args = p.parse_args()

    images = find_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found under {args.input_dir}")
    print(f"Found {len(images)} images. Loading model...")

    model = ModulatedLoRC.from_pretrained(hf_token=args.hf_token)
    preds = model.predict_images(images, batch_size=args.batch_size)

    results = [{"image_path": str(img), "pred": r["p_fake"]} for img, r in zip(images, preds)]
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
