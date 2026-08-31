"""Compat shim: transformers' dependency_versions_check rejects
huggingface_hub>=1.0 even when compatible. Neutralize it before importing
transformers proper. Call once, before any `from transformers import ...`."""
import sys
import types


def patch_transformers_version_check() -> None:
    try:
        import transformers.utils.versions as v
        orig, v.require_version = v.require_version, lambda *a, **k: None
        import transformers.dependency_versions_check as dvc
        dvc.require_version_core = dvc.dep_version_check = lambda *a, **k: None
        v.require_version = orig
    except Exception:
        stub = sys.modules.get("transformers.dependency_versions_check") or types.ModuleType(
            "transformers.dependency_versions_check")
        stub.require_version_core = stub.dep_version_check = lambda *a, **k: None
        sys.modules["transformers.dependency_versions_check"] = stub


patch_transformers_version_check()
