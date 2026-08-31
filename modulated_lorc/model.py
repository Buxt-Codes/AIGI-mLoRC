"""Modulated-LoRC — inference-only model definition.

Trimmed, standalone copy of the training repo's `lorc/model.py`, keeping
only what this checkpoint actually needs: the HuggingFace-backbone path
(the checkpoint's backbone is `facebook/dinov3-vith16plus-pretrain-lvd1689m`,
loaded via `transformers.AutoModel`), the Low-Rank Attention Block, and the
LoRC detector head. Every other backbone adapter (timm, TIPSv2, LingBot,
local-checkpoint) and every non-paper "semantic fallback" head from the
training repo have been dropped — this file has no dependency on `timm`.

Architecture (matches arXiv:2608.20882v1, Section 3):
  1. Frozen DINOv3 ViT-H+/16 backbone + LoRA adapters on q/k/v/o_proj.
  2. Orthogonal decomposition:  ĉ = c/‖c‖₂,  X_res = X(I − ĉĉᵀ)
  3. Low-Rank Attention Block (rank r) on X_res.
  4. Classifier: Linear([mean(Y) ∥ c_raw]) → 2 classes (0=real, 1=fake).
"""
import logging
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

logger = logging.getLogger("modulated_lorc")


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace backbone adapter (only backbone path this package supports)
# ═══════════════════════════════════════════════════════════════════════════

class BackboneAdapter(ABC):
    @abstractmethod
    def load(self, name: str, pretrained: bool, **kwargs) -> nn.Module: ...

    @abstractmethod
    def forward_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor: ...

    def num_prefix_tokens(self, backbone, cls_index=0, num_register_tokens=None) -> int:
        if num_register_tokens is not None:
            return 1 + num_register_tokens
        return getattr(backbone, "num_prefix_tokens", 1)

    def output_dim(self, backbone: nn.Module) -> int:
        for attr in ("embed_dim", "num_features", "hidden_size"):
            val = getattr(backbone, attr, None)
            if isinstance(val, int):
                return val
        cfg = getattr(backbone, "config", None)
        if cfg is not None and isinstance(getattr(cfg, "hidden_size", None), int):
            return cfg.hidden_size
        raise ValueError(
            f"Could not infer output dim for backbone {backbone.__class__.__name__}; "
            "pass backbone_dim explicitly."
        )


class _HFBackboneWrapper(nn.Module):
    """Wraps a HuggingFace ViT to match the timm-style forward_features interface.
    Expects `last_hidden_state` [B, T, D] with CLS at index 0.

    Must NOT override parameters()/named_parameters()/state_dict()/
    load_state_dict() — `self.model` is a normal registered submodule, and
    plain nn.Module's default recursive behaviour is what keeps
    named_parameters() and state_dict() key names in agreement once peft
    wraps this again. (A checkpoint-key-mismatch bug from doing this wrong
    once cost a full training run's LoRA weights in the training repo — see
    its lorc/model.py docstring for the full story. Don't reintroduce it.)
    """
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).last_hidden_state  # [B, T, D]


class HFBackboneAdapter(BackboneAdapter):
    def load(self, name: str, pretrained: bool, **kwargs) -> nn.Module:
        from transformers import AutoModel, AutoConfig
        hf_model = (
            AutoModel.from_pretrained(name, **kwargs) if pretrained
            else AutoModel.from_config(AutoConfig.from_pretrained(name, **kwargs))
        )
        return _HFBackboneWrapper(hf_model)

    def forward_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return backbone.forward_features(x)

    def output_dim(self, backbone: nn.Module) -> int:
        cfg = getattr(backbone.model, "config", None)
        if cfg is not None and isinstance(getattr(cfg, "hidden_size", None), int):
            return cfg.hidden_size
        return super().output_dim(backbone)

    def num_prefix_tokens(self, backbone, cls_index=0, num_register_tokens=None) -> int:
        if num_register_tokens is not None:
            return 1 + num_register_tokens
        cfg = getattr(backbone.model, "config", None)
        n_reg = getattr(cfg, "num_register_tokens", 0) if cfg is not None else 0
        return 1 + (n_reg or 0)


# ═══════════════════════════════════════════════════════════════════════════
# Low-Rank Attention Block (Eq. 5-6)
# ═══════════════════════════════════════════════════════════════════════════

class LowRankAttention(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.W_Q = nn.Linear(dim, rank, bias=False)
        self.W_K = nn.Linear(dim, rank, bias=False)
        self.W_V = nn.Linear(dim, rank, bias=False)
        self.W_O = nn.Linear(rank, dim, bias=False)
        self.scale = 1.0 / math.sqrt(rank)

    def forward(self, X_res: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(X_res)
        K = self.W_K(X_res)
        V = self.W_V(X_res)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return self.W_O(torch.bmm(attn, V))


# ═══════════════════════════════════════════════════════════════════════════
# LoRC detector
# ═══════════════════════════════════════════════════════════════════════════

class LoRC(nn.Module):
    """
    Args:
        backbone_name:  HF repo id for the backbone (default: DINOv3 ViT-H+/16,
                         the backbone this checkpoint was trained on).
        backbone_dim:   Output [CLS] dimension. None auto-infers it.
        attn_rank:      Rank of the Low-Rank Attention Block.
        lora_rank:      LoRA rank for backbone fine-tuning.
        lora_alpha:     LoRA α scaling.
        lora_target_modules: Attention sub-module names LoRA is applied to.
        pretrained:     Whether to load pretrained backbone weights (should
                         always be True here — the checkpoint only carries
                         the LoRA/head deltas, not the frozen backbone).
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        backbone_dim: int | None = None,
        attn_rank: int = 64,
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        pretrained: bool = True,
        cls_index: int = 0,
        num_register_tokens: int | None = None,
    ):
        super().__init__()
        self.cls_index = cls_index
        self.num_register_tokens = num_register_tokens

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        adapter = HFBackboneAdapter()
        self._adapter = adapter
        raw_backbone = adapter.load(backbone_name, pretrained)

        if backbone_dim is None:
            backbone_dim = adapter.output_dim(raw_backbone)
        self.backbone_dim = backbone_dim

        # Resolved against the RAW (pre-peft) backbone and cached — see
        # _HFBackboneWrapper's docstring for why this matters.
        self._n_prefix_tokens = adapter.num_prefix_tokens(raw_backbone, cls_index, num_register_tokens)

        for p in raw_backbone.parameters():
            p.requires_grad_(False)

        if lora_rank == 0:
            self.backbone = raw_backbone
        else:
            lora_cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=lora_target_modules, lora_dropout=0.0, bias="none",
            )
            self.backbone = get_peft_model(raw_backbone, lora_cfg)

        self.attn_block = LowRankAttention(dim=backbone_dim, rank=attn_rank)
        self.classifier = nn.Linear(backbone_dim * 2, 2)

    @staticmethod
    def _decompose(c: torch.Tensor, X: torch.Tensor):
        """X_res = X(I − ĉĉᵀ), ĉ = c/‖c‖₂."""
        c_hat = F.normalize(c, dim=-1)
        proj = torch.einsum("bd,bnd->bn", c_hat, X)
        return X - torch.einsum("bn,bd->bnd", proj, c_hat)

    def _encode(self, pixel_values: torch.Tensor):
        hidden = self._adapter.forward_features(self.backbone, pixel_values)  # [B, T, D]
        c = hidden[:, self.cls_index, :]
        X = hidden[:, self._n_prefix_tokens:, :]
        return c, X

    def forward(self, pixel_values: torch.Tensor):
        c, X = self._encode(pixel_values)
        X_res = self._decompose(c, X)
        Y = self.attn_block(X_res)
        pooled = Y.mean(dim=1)
        feat = torch.cat([pooled, c.detach()], dim=-1)
        logits = self.classifier(feat)
        return logits, X_res

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor):
        """Return (probs [B,2], pred_class [B]) — 0=real, 1=fake."""
        self.eval()
        logits, _ = self(pixel_values)
        probs = F.softmax(logits, dim=-1)
        return probs, probs.argmax(dim=-1)
