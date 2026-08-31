"""
Compatibility shim for transformers 4.56 + huggingface-hub 1.4.1.

The version check in transformers.dependency_versions_check rejects hub>=1.0.
We patch require_version_core to a no-op so the library loads cleanly.
"""
import sys


def patch_transformers_version_check():
    """Call before any `from transformers import ...`."""
    try:
        # First, import the REAL module so all its other attributes survive
        import importlib.util, types

        # Import the real module directly without triggering the raise
        spec = importlib.util.find_spec("transformers.dependency_versions_check")
        if spec is None:
            return

        # Load via source to avoid going through transformers' __init__
        import importlib._bootstrap_external as _be
        loader = spec.loader
        mod = types.ModuleType(spec.name)
        mod.__spec__ = spec
        mod.__loader__ = loader
        mod.__file__ = getattr(spec, "origin", None)

        # Execute the module source with a patched require_version_core
        orig_code = loader.get_code(spec.name)

        # Build a namespace that no-ops require_version_core
        globs = {
            "__name__": spec.name,
            "__file__": mod.__file__,
            "__loader__": loader,
            "__spec__": spec,
        }

        # Inject a no-op before the module runs
        import builtins
        real_require = None

        # Easier: just import normally but monkey-patch the function
        # after importing the underlying versions module
        from transformers.utils.versions import require_version as _rv
        import transformers.utils.versions as _vmod
        _orig = _vmod.require_version

        def _noop_require(*a, **kw):
            pass

        _vmod.require_version = _noop_require

        # Now import will succeed
        import transformers.dependency_versions_check as _dvc
        _dvc.require_version_core = _noop_require
        _dvc.dep_version_check = _noop_require  # needed by deepspeed integration

        # Restore
        _vmod.require_version = _orig

    except Exception:
        # If the whole thing fails, try a simpler approach: create a stub
        # that provides both names deepspeed integration needs
        import types
        stub = sys.modules.get("transformers.dependency_versions_check")
        if stub is None:
            stub = types.ModuleType("transformers.dependency_versions_check")
        stub.require_version_core = lambda *a, **kw: None
        stub.dep_version_check = lambda *a, **kw: None
        sys.modules["transformers.dependency_versions_check"] = stub


# Apply immediately
patch_transformers_version_check()
