from __future__ import annotations

"""
AlignScope — Defection & Anomaly Detector

Monitors alignment metrics over time and flags moments when
an agent breaks from its coalition. These are the scientifically
interesting events in alignment research — the exact tick where
cooperation fails.

Detection methods:
1. Direct defection events from the data source
2. Metric-based anomalies: sudden drops in reciprocity or role stability
3. Coalition fragmentation: when coalition sizes shrink unexpectedly

Environment-agnostic: works with any data source that provides
the standard AlignScope tick/metrics format.
"""


class DefectionDetector:
    """Accumulates metric history and detects alignment anomalies."""

    def __init__(
        self,
        reciprocity_drop_threshold: float = 0.3,
        stability_drop_threshold: float = 0.2,
        lookback_window: int = 10,
    ):
        self.reciprocity_drop_threshold = reciprocity_drop_threshold
        self.stability_drop_threshold = stability_drop_threshold
        self.lookback_window = lookback_window

        self.metric_history: list[dict] = []
        self.all_events: list[dict] = []

    def analyze(self, tick_data: dict, metrics: dict) -> list[dict]:
        """
        Analyze one tick for defection and anomaly events.

        Args:
            tick_data: Raw game state from any compatible data source
            metrics: Computed alignment metrics from AlignmentMetrics

        Returns:
            List of detected events (may be empty)
        """
        events = []
        tick = tick_data["tick"]

        # 1. Forward direct defection events from data source
        for defection in tick_data.get("defection_events", []):
            event = {
                "tick": tick,
                "type": "defection",
                "agent_id": defection["agent_id"],
                "team": defection["team"],
                "severity": defection["severity"],
                "description": (
                    f"Agent {defection['agent_id']} (team {defection.get('team', '?')}, "
                    f"{defection.get('previous_role', 'unknown')}) defected. "
                    f"Reason: {defection.get('reason', 'no reason provided')}"
                ),
                "details": defection,
            }
            events.append(event)

        # 2. Check for reciprocity drops (metric-based anomalies)
        if len(self.metric_history) >= self.lookback_window:
            reciprocity_events = self._detect_reciprocity_anomalies(
                tick, metrics
            )
            events.extend(reciprocity_events)

        # 3. Check for role stability drops
        if len(self.metric_history) >= self.lookback_window:
            stability_events = self._detect_stability_anomalies(
                tick, metrics
            )
            events.extend(stability_events)

        # 4. Check coalition fragmentation
        coalition_events = self._detect_coalition_changes(tick, metrics)
        events.extend(coalition_events)

        # Store for history
        self.metric_history.append(metrics)
        self.all_events.extend(events)

        return events

    def _detect_reciprocity_anomalies(
        self, tick: int, current: dict
    ) -> list[dict]:
        """Detect sudden drops in pair reciprocity."""
        events = []
        current_pairs = {
            (p["agent_a"], p["agent_b"]): p["reciprocity"]
            for p in current.get("pair_metrics", [])
        }

        for pair, current_val in current_pairs.items():
            historical = []
            for past in self.metric_history[-self.lookback_window:]:
                for p in past.get("pair_metrics", []):
                    if (p["agent_a"], p["agent_b"]) == pair:
                        historical.append(p["reciprocity"])

            if historical:
                avg_historical = sum(historical) / len(historical)
                drop = avg_historical - current_val

                if drop > self.reciprocity_drop_threshold:
                    events.append({
                        "tick": tick,
                        "type": "reciprocity_drop",
                        "agent_id": pair[0],
                        "partner_id": pair[1],
                        "severity": round(min(1.0, drop / 0.5), 3),
                        "description": (
                            f"Reciprocity between agents {pair[0]} and {pair[1]} "
                            f"dropped from {avg_historical:.2f} to {current_val:.2f}"
                        ),
                        "team": None,
                    })

        return events

    def _detect_stability_anomalies(
        self, tick: int, current: dict
    ) -> list[dict]:
        """Detect sudden role stability drops for individual agents."""
        events = []
        current_agents = current.get("agent_metrics", {})

        for aid, metrics in current_agents.items():
            current_stability = metrics["role_stability"]

            historical = []
            for past in self.metric_history[-self.lookback_window:]:
                past_agents = past.get("agent_metrics", {})
                if aid in past_agents:
                    historical.append(past_agents[aid]["role_stability"])

            if historical:
                avg_historical = sum(historical) / len(historical)
                drop = avg_historical - current_stability

                if drop > self.stability_drop_threshold:
                    events.append({
                        "tick": tick,
                        "type": "stability_drop",
                        "agent_id": aid,
                        "severity": round(min(1.0, drop / 0.4), 3),
                        "description": (
                            f"Agent {aid} role stability dropped from "
                            f"{avg_historical:.2f} to {current_stability:.2f} — "
                            f"possible role confusion or strategic shift"
                        ),
                        "team": metrics.get("team"),
                    })

        return events

    def _detect_coalition_changes(
        self, tick: int, current: dict
    ) -> list[dict]:
        """
        Detect coalition fragmentation.

        FIX: Removed the 'prev_coalitions > 0' guard that was preventing
        events from firing after coalitions first dropped to 0.

        Before:
            curr_coalitions < prev_coalitions AND prev_coalitions > 0
            → Once coalitions hit 0 they could never drop further,
              so no events fired after the first fragmentation.

        After:
            curr_coalitions < prev_coalitions
            → Fires every time the count drops, regardless of floor value.
        """
        events = []

        if not self.metric_history:
            return events

        prev = self.metric_history[-1]
        prev_teams = prev.get("team_metrics", {})
        curr_teams = current.get("team_metrics", {})

        for tid in curr_teams:
            if tid in prev_teams:
                prev_coalitions = prev_teams[tid].get("active_coalitions", 0)
                curr_coalitions = curr_teams[tid].get("active_coalitions", 0)

                # FIX: removed "and prev_coalitions > 0"
                if (isinstance(prev_coalitions, int)
                        and isinstance(curr_coalitions, int)
                        and curr_coalitions < prev_coalitions):
                    lost = prev_coalitions - curr_coalitions
                    events.append({
                        "tick": tick,
                        "type": "coalition_fragmentation",
                        "agent_id": None,
                        "team": tid,
                        "severity": round(min(1.0, lost / 2.0), 3),
                        "description": (
                            f"Team {tid} lost {lost} coalition(s): "
                            f"{prev_coalitions} → {curr_coalitions}"
                        ),
                    })

        return events

    def get_summary(self) -> dict:
        """Return aggregate defection statistics."""
        if not self.all_events:
            return {
                "total_events": 0,
                "defections": 0,
                "reciprocity_drops": 0,
                "stability_drops": 0,
                "coalition_fragmentations": 0,
                "avg_severity": 0,
            }

        by_type = {}
        for e in self.all_events:
            t = e["type"]
            by_type[t] = by_type.get(t, 0) + 1

        avg_sev = sum(e.get("severity", 0) for e in self.all_events) / len(self.all_events)

        return {
            "total_events": len(self.all_events),
            "defections": by_type.get("defection", 0),
            "reciprocity_drops": by_type.get("reciprocity_drop", 0),
            "stability_drops": by_type.get("stability_drop", 0),
            "coalition_fragmentations": by_type.get("coalition_fragmentation", 0),
            "avg_severity": round(avg_sev, 3),
        }