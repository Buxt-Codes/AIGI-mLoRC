"""Modulated-LoRC — inference-only model definition (arXiv:2608.20882v1, Sec. 3):

  1. Frozen DINOv3 ViT-H+/16 backbone + LoRA on q/k/v/o_proj.
  2. Orthogonal decomposition: ĉ = c/‖c‖₂, X_res = X(I − ĉĉᵀ)
  3. Low-Rank Attention Block (rank r) on X_res.
  4. Classifier: Linear([mean(Y) ∥ c_raw]) → 2 classes (0=real, 1=fake).

The class layout below (BackboneAdapter -> HFBackboneAdapter -> LoRC.backbone)
must stay as-is: it fixes the parameter names the shipped checkpoint was
saved under, so restructuring it breaks loading silently (load_state_dict
runs with strict=False; see `_HFBackboneWrapper` below for why).
"""
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


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
        raise ValueError(f"Could not infer output dim for {backbone.__class__.__name__}")


class _HFBackboneWrapper(nn.Module):
    """`self.model` is a plain submodule on purpose — do not override
    parameters()/state_dict()/load_state_dict() here, or their key names
    stop matching once peft wraps this again (a real bug once dropped a
    trained checkpoint's LoRA weights silently; don't reintroduce it)."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).last_hidden_state  # [B, T, D]


class HFBackboneAdapter(BackboneAdapter):
    def load(self, name: str, pretrained: bool, **kwargs) -> nn.Module:
        from transformers import AutoModel, AutoConfig
        hf_model = (AutoModel.from_pretrained(name, **kwargs) if pretrained
                    else AutoModel.from_config(AutoConfig.from_pretrained(name, **kwargs)))
        return _HFBackboneWrapper(hf_model)

    def forward_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return backbone.forward_features(x)

    def output_dim(self, backbone: nn.Module) -> int:
        cfg = getattr(backbone.model, "config", None)
        return cfg.hidden_size if cfg is not None else super().output_dim(backbone)

    def num_prefix_tokens(self, backbone, cls_index=0, num_register_tokens=None) -> int:
        if num_register_tokens is not None:
            return 1 + num_register_tokens
        cfg = getattr(backbone.model, "config", None)
        return 1 + (getattr(cfg, "num_register_tokens", 0) or 0)


class LowRankAttention(nn.Module):
    """Low-Rank Attention Block (Eq. 5-6)."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.W_Q = nn.Linear(dim, rank, bias=False)
        self.W_K = nn.Linear(dim, rank, bias=False)
        self.W_V = nn.Linear(dim, rank, bias=False)
        self.W_O = nn.Linear(rank, dim, bias=False)
        self.scale = rank ** -0.5

    def forward(self, X_res: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.W_Q(X_res), self.W_K(X_res), self.W_V(X_res)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        return self.W_O(torch.bmm(attn, V))


class LoRC(nn.Module):
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
        adapter = HFBackboneAdapter()
        self._adapter = adapter
        raw_backbone = adapter.load(backbone_name, pretrained)

        backbone_dim = backbone_dim or adapter.output_dim(raw_backbone)
        # Cached against the RAW (pre-peft) backbone — see _HFBackboneWrapper.
        self._n_prefix_tokens = adapter.num_prefix_tokens(raw_backbone, cls_index, num_register_tokens)

        for p in raw_backbone.parameters():
            p.requires_grad_(False)

        if lora_rank == 0:
            # No adapter to add -- either the checkpoint has LoRA already
            # merged into these weights (mlorc-full.pt), or this is a
            # frozen-backbone-only variant. `raw_backbone` IS the model.
            self.backbone = raw_backbone
        else:
            lora_cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.backbone = get_peft_model(raw_backbone, lora_cfg)
        self.attn_block = LowRankAttention(dim=backbone_dim, rank=attn_rank)
        self.classifier = nn.Linear(backbone_dim * 2, 2)

    @staticmethod
    def _decompose(c: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """X_res = X(I − ĉĉᵀ), ĉ = c/‖c‖₂."""
        c_hat = F.normalize(c, dim=-1)
        proj = torch.einsum("bd,bnd->bn", c_hat, X)
        return X - torch.einsum("bn,bd->bnd", proj, c_hat)

    def forward(self, pixel_values: torch.Tensor):
        hidden = self._adapter.forward_features(self.backbone, pixel_values)  # [B, T, D]
        c, X = hidden[:, self.cls_index, :], hidden[:, self._n_prefix_tokens:, :]
        X_res = self._decompose(c, X)
        pooled = self.attn_block(X_res).mean(dim=1)
        logits = self.classifier(torch.cat([pooled, c.detach()], dim=-1))
        return logits, X_res

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor):
        """Return (probs [B,2], pred_class [B]) — 0=real, 1=fake."""
        self.eval()
        logits, _ = self(pixel_values)
        probs = F.softmax(logits, dim=-1)
        return probs, probs.argmax(dim=-1)
