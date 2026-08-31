"""High-level inference wrapper: build the model, load weights from the
private HF hub repo `buxtcodes/TechJam-Modulated-LoRC`, run predictions.

    from mlorc import ModulatedLoRC
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
WEIGHTS_FILENAME = "mlorc-full.pt"  # LoRA merged into the backbone weights -- one file, no separate HF download
CONFIG_FILENAME = "dinov3_config.json"  # architecture metadata only (no weights) -- mirrored here so
# building the backbone's architecture never has to touch DINOv3's gated (manual-review) HF repo at all.
IMAGE_SIZE, JPEG_QUALITY = 224, 96
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# lora_rank=0, pretrained=False: build the bare architecture (config only, no
# weight download) -- mlorc-full.pt's state dict already contains every
# weight (backbone included), so nothing else needs fetching from anywhere.
MODEL_KWARGS = dict(attn_rank=64, lora_rank=0, pretrained=False)


def _jpeg_pass(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ModulatedLoRC:
    def __init__(self, model: LoRC, device: str):
        self.model, self.device = model, device
        # Matches training exactly: JPEG q=96 pass, then CenterCrop — never a resize.
        self.tf = transforms.Compose([
            transforms.Lambda(_jpeg_pass),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @classmethod
    def from_pretrained(cls, repo_id: str = REPO_ID, filename: str = WEIGHTS_FILENAME,
                         device: str | None = None, hf_token: str | None = None) -> "ModulatedLoRC":
        from huggingface_hub import hf_hub_download

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        config_path = hf_hub_download(repo_id=repo_id, filename=CONFIG_FILENAME, token=hf_token)

        model = LoRC(backbone_config_path=config_path, **MODEL_KWARGS).to(device)
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)  # full checkpoint -> exact match required
        model.eval()
        return cls(model, device)

    def _load(self, image) -> torch.Tensor:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        return self.tf(img.convert("RGB"))

    @torch.no_grad()
    def predict_images(self, images: list, batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self._load(p) for p in images[i:i + batch_size]]).to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                probs, _ = self.model.predict(batch)
            for p_real, p_fake in probs.float().tolist():
                results.append({"label": "fake" if p_fake >= 0.5 else "real", "p_fake": p_fake, "p_real": p_real})
        return results

    def predict_image(self, image: str | Path | Image.Image) -> dict:
        return self.predict_images([image])[0]
