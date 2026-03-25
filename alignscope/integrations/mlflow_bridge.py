"""
AlignScope — MLflow Bridge

If the user has MLflow installed and an active run, AlignScope
automatically logs alignment metrics as MLflow metrics.

Zero vendor lock-in: AlignScope enriches MLflow — never replaces it.
"""


class MlflowBridge:
    """Forwards AlignScope metrics to an active MLflow run."""

    def __init__(self):
        self._mlflow = None
        self._active = False

        try:
            import mlflow
            self._mlflow = mlflow

            # Check for active run
            if mlflow.active_run() is not None:
                self._active = True
                print("[AlignScope] ✓ Detected active MLflow run — forwarding metrics")
            else:
                print("[AlignScope] MLflow installed but no active run. "
                      "Call mlflow.start_run() first to enable forwarding.")
        except ImportError:
            pass

    def log(self, step: int, metrics: dict, events: list):
        """Forward alignment metrics to MLflow."""
        if not self._active or not self._mlflow:
            return

        # Re-check for active run
        if self._mlflow.active_run() is None:
            return

        try:
            log_data = {
                "alignscope.overall_alignment": metrics.get("overall_alignment_score", 0),
            }

            team_metrics = metrics.get("team_metrics", {})
            for tid, tm in team_metrics.items():
                prefix = f"alignscope.team_{tid}"
                log_data[f"{prefix}.role_stability"] = tm.get("avg_role_stability", 0)
                log_data[f"{prefix}.coalitions"] = tm.get("active_coalitions", 0)
                log_data[f"{prefix}.defectors"] = tm.get("defector_count", 0)

            self._mlflow.log_metrics(log_data, step=step)

        except Exception:
            pass

    def finish(self, summary: dict):
        """Log final summary params to MLflow."""
        if not self._active or not self._mlflow:
            return

        try:
            params = {
                f"alignscope_{k}": str(v)
                for k, v in summary.items()
            }
            self._mlflow.log_params(params)
        except Exception:
            pass
