from __future__ import annotations

"""
AlignScope — Framework Auto-Patching System

Applies monkey-patches to supported MARL frameworks so users
get alignment observability with zero code changes.

Usage:
    alignscope patch rllib       # CLI
    alignscope.patch("rllib")    # Python
"""

_PATCHES = {
    "rllib": "alignscope.patches.rllib",
    "pettingzoo": "alignscope.patches.pettingzoo",
    "pymarl": "alignscope.patches.pymarl",
}


def apply_patch(framework: str) -> bool:
    """
    Apply an auto-patch to a MARL framework.

    Returns True if successful, False if framework not found.
    """
    if framework not in _PATCHES:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Supported: {', '.join(_PATCHES.keys())}"
        )

    module_path = _PATCHES[framework]

    try:
        import importlib
        mod = importlib.import_module(module_path)
        mod.apply()
        return True
    except ImportError as e:
        print(f"[AlignScope] Could not patch {framework}: {e}")
        return False
    except Exception as e:
        print(f"[AlignScope] Error patching {framework}: {e}")
        return False


def available_patches() -> list[str]:
    """Return list of supported framework names."""
    return list(_PATCHES.keys())
