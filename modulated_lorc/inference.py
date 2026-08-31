"""High-level inference wrapper: build the model, load weights from the
private HF hub repo `buxtcodes/TechJam-Modulated-LoRC` (no manual download
step needed), run predictions on images.

    from modulated_lorc import ModulatedLoRC
    model = ModulatedLoRC.from_pretrained()
    model.predict_image("photo.jpg")   # {"label": "fake", "p_fake": 0.93, "p_real": 0.07}
"""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import LoRC

REPO_ID = "buxtcodes/TechJam-Modulated-LoRC"
WEIGHTS_FILENAME = "modulated-lorc.pt"

IMAGE_SIZE = 224
JPEG_QUALITY = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Must match the config the checkpoint was trained with (run 1: pair-aware
# energy augmentation, DINOv3 ViT-H+/16, LoRA rank=32/α=32, attn_rank=64).
MODEL_KWARGS = dict(attn_rank=64, lora_rank=32, lora_alpha=32)


def _jpeg_pass(img: Image.Image, quality: int = JPEG_QUALITY) -> Image.Image:
    """Standardize the encoding every image is seen through, same as
    training's mandatory-JPEG convention — not a resize, just a re-encode."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ModulatedLoRC:
    def __init__(self, model: LoRC, device: str):
        self.model = model
        self.device = device
        # CENTER CROP, not resize — this is the actual eval/inference
        # transform the checkpoint was trained and evaluated with
        # (`lorc.dataset.make_val_transform`: JPEG q=96 pass, then
        # CenterCrop(224), never a full-image resize). Using Resize here
        # instead would silently feed the model out-of-distribution input
        # and not reproduce the reported WildFake numbers.
        self.tf = transforms.Compose([
            transforms.Lambda(_jpeg_pass),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = REPO_ID,
        filename: str = WEIGHTS_FILENAME,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> "ModulatedLoRC":
        from huggingface_hub import hf_hub_download

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)

        model = LoRC(**MODEL_KWARGS).to(device)
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)  # backbone weights come from the
        # pretrained HF load above; the checkpoint only carries LoRA/head deltas.
        if not any("lora_" in k for k in state):
            raise RuntimeError(f"No LoRA weights found in {weights_path} — refusing "
                                "to serve an unadapted model.")
        model.eval()
        return cls(model, device)

    def _load(self, image) -> torch.Tensor:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        return self.tf(img.convert("RGB"))

    @torch.no_grad()
    def predict_image(self, image: str | Path | Image.Image) -> dict:
        x = self._load(image).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16,
                             enabled=(self.device == "cuda")):
            probs, pred = self.model.predict(x)
        p_real, p_fake = probs[0].float().tolist()
        return {"label": "fake" if pred.item() == 1 else "real", "p_fake": p_fake, "p_real": p_real}

    @torch.no_grad()
    def predict_images(self, images: list, batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self._load(p) for p in images[i:i + batch_size]]).to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16,
                                 enabled=(self.device == "cuda")):
                probs, preds = self.model.predict(batch)
            for p, pred in zip(probs.float().tolist(), preds.tolist()):
                results.append({"label": "fake" if pred == 1 else "real", "p_fake": p[1], "p_real": p[0]})
        return results
