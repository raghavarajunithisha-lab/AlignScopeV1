from __future__ import annotations

"""
AlignScope — Integration Bridges

Auto-detects installed ML observability tools (W&B, MLflow) and
forwards alignment metrics to them. This ensures zero vendor lock-in —
AlignScope adds to your existing stack, never replaces it.
"""


def detect_integrations() -> dict[str, bool]:
    """Check which ML tools are available."""
    available = {}

    try:
        import wandb
        available["wandb"] = True
    except ImportError:
        available["wandb"] = False

    try:
        import mlflow
        available["mlflow"] = True
    except ImportError:
        available["mlflow"] = False

    return available
