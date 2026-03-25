"""
AlignScope — Weights & Biases Bridge

If the user has W&B installed and an active run, AlignScope
automatically forwards alignment metrics as W&B custom metrics.
This means they see AlignScope data right alongside their reward
curves in the W&B dashboard.

Zero vendor lock-in: AlignScope never replaces W&B — it adds to it.
"""


class WandbBridge:
    """Forwards AlignScope metrics to an active W&B run."""

    def __init__(self):
        self._wandb = None
        self._active = False

        try:
            import wandb
            self._wandb = wandb

            # Check if there's an active run
            if wandb.run is not None:
                self._active = True
                print("[AlignScope] ✓ Detected active W&B run — forwarding metrics")
            else:
                print("[AlignScope] W&B installed but no active run. "
                      "Call wandb.init() first to enable forwarding.")
        except ImportError:
            pass

    def log(self, step: int, metrics: dict, events: list):
        """Forward alignment metrics to W&B."""
        if not self._active or not self._wandb:
            return

        # Re-check for active run (may have been initialized after AlignScope)
        if self._wandb.run is None:
            return

        try:
            # Log overall alignment score
            log_data = {
                "alignscope/overall_alignment": metrics.get("overall_alignment_score", 0),
            }

            # Log per-team metrics
            team_metrics = metrics.get("team_metrics", {})
            for tid, tm in team_metrics.items():
                prefix = f"alignscope/team_{tid}"
                log_data[f"{prefix}/role_stability"] = tm.get("avg_role_stability", 0)
                log_data[f"{prefix}/coalitions"] = tm.get("active_coalitions", 0)
                log_data[f"{prefix}/defectors"] = tm.get("defector_count", 0)

            # Log event counts
            if events:
                by_type = {}
                for e in events:
                    t = e.get("type", "unknown")
                    by_type[t] = by_type.get(t, 0) + 1

                for etype, count in by_type.items():
                    log_data[f"alignscope/events/{etype}"] = count

            self._wandb.log(log_data, step=step)

        except Exception:
            pass  # Never crash

    def finish(self, summary: dict):
        """Log final summary to W&B."""
        if not self._active or not self._wandb or self._wandb.run is None:
            return

        try:
            for key, value in summary.items():
                self._wandb.run.summary[f"alignscope/{key}"] = value
        except Exception:
            pass
