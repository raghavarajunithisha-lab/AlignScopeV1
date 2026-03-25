"""
AlignScope — Core SDK Tracker

The AlignScopeTracker is the heart of the SDK. It:
1. Accepts raw MARL data via .log() calls
2. Normalizes data from any framework format into the standard schema
3. Runs the metrics engine and anomaly detector in real-time
4. Sends results to the dashboard server via WebSocket
5. Optionally forwards metrics to W&B and/or MLflow
"""

import json
import time
import random
import asyncio
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

from alignscope.metrics import AlignmentMetrics
from alignscope.detector import DefectionDetector


class AlignScopeTracker:
    """
    Core tracker that bridges training code → dashboard.

    Handles data normalization, metric computation, and WebSocket streaming.
    Thread-safe — can be called from training threads.
    """

    def __init__(
        self,
        project: str = "default",
        server_url: str = "ws://localhost:8000/ws/sdk",
        preset: Optional[str] = None,
        paradigm: Optional[dict] = None,
        metrics: Optional[list] = None,
        events: Optional[list] = None,
        topology: Optional[dict] = None,
        config: Optional[dict] = None,
        forward_wandb: bool = True,
        forward_mlflow: bool = True,
    ):
        self.project = project
        self.server_url = server_url
        
        self.preset = preset
        self.paradigm = paradigm or {"environment": preset if preset else "cooperative", "learning": "decentralized"}
        self.metrics_config = metrics or []
        self.events_config = events or []

        if preset == "zero-sum":
            if not self.metrics_config:
                self.metrics_config = [
                    {"id": "win_rate", "label": "Win Rate", "type": "scalar"},
                    {"id": "exploitability", "label": "Exploitability", "type": "scalar"},
                    {"id": "nash_gap", "label": "Nash Gap", "type": "scalar"},
                ]
            if not self.events_config:
                self.events_config = [
                    {"id": "policy_shift", "label": "Policy Shift", "color": "#dc4a4a"},
                    {"id": "role_reversal", "label": "Role Reversal", "color": "#e8925a"},
                ]
        elif preset == "mean-field":
            if not self.metrics_config:
                self.metrics_config = [
                    {"id": "population_distribution", "label": "Pop. Dist.", "type": "scalar"},
                    {"id": "mean_reward", "label": "Mean Reward", "type": "scalar"},
                    {"id": "field_entropy", "label": "Field Entropy", "type": "scalar"},
                ]
        elif preset == "cooperative" or not preset:
            if not self.metrics_config:
                self.metrics_config = [
                    {"id": "role_stability", "label": "Role Stability", "type": "scalar"},
                    {"id": "coalitions", "label": "Coalitions", "type": "scalar"},
                    {"id": "defectors", "label": "Defectors", "type": "scalar"},
                    {"id": "score", "label": "Score", "type": "scalar"},
                ]
            if not self.events_config:
                self.events_config = [
                    {"id": "defection", "label": "Defection", "color": "#dc4a4a"},
                    {"id": "reciprocity_drop", "label": "Reciprocity ↓", "color": "#c97b3a"},
                    {"id": "stability_drop", "label": "Stability ↓", "color": "#b06ec7"},
                    {"id": "coalition_fragmentation", "label": "Coalition ↓", "color": "#d4a843"},
                ]

        self.topology_config = topology or {
            "nodeLabel": "agent",
            "edgeTypes": [
                {"id": "comm", "label": "Communication", "style": "dashed"},
                {"id": "collab", "label": "Collaboration", "style": "solid"}
            ],
            "groupBy": "team"
        }

        self.config = config or {}

        # Metric engines
        self._metrics = AlignmentMetrics()
        self._detector = DefectionDetector()

        # State
        self._step = 0
        self._agents_cache: Dict[Union[int, str], dict] = {}
        self._help_tracker: Dict[tuple, int] = {}
        self._role_history: Dict[Union[int, str], list] = {}
        self._started = False
        self._lock = threading.Lock()

        # WebSocket connection — single persistent daemon thread
        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_thread_started = False
        self._send_queue: deque = deque()
        self._ws_stop_event = threading.Event()

        # Integration bridges
        self._wandb_bridge = None
        self._mlflow_bridge = None

        if forward_wandb:
            self._init_wandb_bridge()
        if forward_mlflow:
            self._init_mlflow_bridge()

        self._print_banner()

    def _print_banner(self):
        """Print a nice startup message like W&B does."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel.fit(
                f"[bold cyan]AlignScope[/] v0.1.0\n"
                f"Project: [bold]{self.project}[/]\n"
                f"Dashboard: [link]http://localhost:8000[/link]",
                title="AlignScope",
                border_style="cyan",
            ))
        except (ImportError, Exception):
            print(f"AlignScope v0.1.0 | Project: {self.project}")
            print(f"   Dashboard: http://localhost:8000")

    def log(
        self,
        step: int,
        agents: Optional[Union[list, dict]] = None,
        obs: Any = None,
        actions: Any = None,
        rewards: Any = None,
        **kwargs,
    ) -> None:
        """
        Log one step of multi-agent data.

        Accepts data in multiple formats:
        - Raw dicts (standard schema)
        - NumPy arrays (auto-converted)
        - Framework-specific formats (auto-detected and normalized)
        """
        with self._lock:
            self._step = step
            normalized = self._normalize_data(step, agents, obs, actions, rewards, **kwargs)

            # Compute metrics
            tick_metrics = self._metrics.update(normalized)

            # Detect anomalies
            events = self._detector.analyze(normalized, tick_metrics)

            # Build payload
            payload = {
                "type": "tick",
                "data": {
                    "tick": step,
                    "agents": normalized["agents"],
                    "objectives": normalized.get("objectives", []),
                    "team_scores": normalized.get("team_scores", {}),
                    "metrics": {
                        "agent_metrics": tick_metrics["agent_metrics"],
                        "pair_metrics": tick_metrics["pair_metrics"],
                        "team_metrics": {
                            str(k): v for k, v in tick_metrics["team_metrics"].items()
                        },
                        "overall_alignment_score": tick_metrics["overall_alignment_score"],
                    },
                    "relationships": self._build_relationships(),
                    "events": [
                        {
                            "tick": e["tick"],
                            "type": e["type"],
                            "agent_id": e.get("agent_id"),
                            "team": e.get("team"),
                            "severity": e.get("severity", 0.5),
                            "description": e["description"],
                        }
                        for e in events
                    ],
                },
            }

            # Queue for WebSocket send
            self._send_queue.append(payload)
            self._flush_queue()

            # Forward to integrations
            self._forward_metrics(step, tick_metrics, events)

    def report(self, tick: int, agent: Union[str, int], metrics: dict) -> None:
        """Dynamically report custom metrics for an agent."""
        with self._lock:
            payload = {
                "type": "tick",
                "data": {
                    "tick": tick,
                    "agents": [],
                    "metrics": {
                        "agent_metrics": {str(agent): metrics},
                        "team_metrics": {},
                    },
                    "team_scores": {},
                    "relationships": [],
                    "events": []
                }
            }
            self._send_queue.append(payload)
            self._flush_queue()

    def event(self, tick: int, type: str, agent: Union[str, int], detail: str, severity: float = 0.5) -> None:
        """Dynamically report custom events."""
        with self._lock:
            team = self._agents_cache.get(agent, {}).get("team")
            payload = {
                "type": "tick",
                "data": {
                    "tick": tick,
                    "agents": [],
                    "metrics": None,
                    "team_scores": {},
                    "relationships": [],
                    "events": [
                        {
                            "tick": tick,
                            "type": type,
                            "agent_id": agent,
                            "team": team,
                            "severity": severity,
                            "description": detail
                        }
                    ],
                }
            }
            self._send_queue.append(payload)
            self._flush_queue()

    def _normalize_data(
        self,
        step: int,
        agents: Any,
        obs: Any,
        actions: Any,
        rewards: Any,
        **kwargs,
    ) -> dict:
        """
        Normalize data from any format into AlignScope standard schema.

        Handles:
        - List of dicts (standard schema) → pass through
        - Dict of {agent_id: data} → convert to list
        - NumPy arrays → convert to lists
        - None values → infer from cache
        """
        normalized_agents = []
        normalized_actions = []

        # --- Normalize agents ---
        if agents is None:
            # Use cached state
            normalized_agents = list(self._agents_cache.values())
        elif isinstance(agents, list):
            if agents and isinstance(agents[0], dict):
                # Already in standard format
                normalized_agents = agents
            else:
                # List of raw values — create minimal agent entries
                for i, a in enumerate(agents):
                    entry = self._make_agent_entry(i, a)
                    normalized_agents.append(entry)
        elif isinstance(agents, dict):
            # Dict of {agent_id: data}
            for aid, data in agents.items():
                if isinstance(data, dict):
                    data.setdefault("agent_id", aid)
                    normalized_agents.append(data)
                else:
                    normalized_agents.append(self._make_agent_entry(aid, data))
        else:
            # Try numpy conversion
            try:
                import numpy as np
                if isinstance(agents, np.ndarray):
                    for i in range(agents.shape[0]):
                        normalized_agents.append(self._make_agent_entry(i, agents[i]))
            except (ImportError, Exception):
                pass

        # Fill in defaults for any agent missing fields
        import math
        total_agents = len(normalized_agents)
        for i, agent in enumerate(normalized_agents):
            agent.setdefault("agent_id", str(i))
            agent.setdefault("team", 0)
            agent.setdefault("role", "agent")
            
            # Deterministic circular layout fallback
            radius = 100
            default_x = math.cos(2 * math.pi * i / max(1, total_agents)) * radius
            default_y = math.sin(2 * math.pi * i / max(1, total_agents)) * radius
            
            agent.setdefault("x", round(default_x, 2))
            agent.setdefault("y", round(default_y, 2))
            agent.setdefault("resources", 0)
            agent.setdefault("hearts", 0)
            agent.setdefault("energy", 0)
            agent.setdefault("is_defector", False)
            agent.setdefault("coalition_id", agent.get("team", 0))

            # Track roles for stability metrics
            aid = agent["agent_id"]
            if aid not in self._role_history:
                self._role_history[aid] = []
            self._role_history[aid].append(agent["role"])

            # Update cache
            self._agents_cache[aid] = agent

        # --- Normalize actions ---
        if actions is not None:
            if isinstance(actions, list) and actions and isinstance(actions[0], dict):
                normalized_actions = actions
            elif isinstance(actions, dict):
                # {agent_id: action} format
                for aid, act in actions.items():
                    action_str = str(act) if not isinstance(act, str) else act
                    entry = {
                        "tick": step,
                        "agent_id": aid,
                        "action": action_str,
                        "target_id": None,
                        "detail": "",
                    }
                    normalized_actions.append(entry)
            elif isinstance(actions, list):
                # List of raw action values
                for i, act in enumerate(actions):
                    agent_id = normalized_agents[i]["agent_id"] if i < len(normalized_agents) else i
                    normalized_actions.append({
                        "tick": step,
                        "agent_id": agent_id,
                        "action": str(act),
                        "target_id": None,
                        "detail": "",
                    })
            else:
                try:
                    import numpy as np
                    if isinstance(actions, np.ndarray):
                        for i in range(actions.shape[0]):
                            agent_id = normalized_agents[i]["agent_id"] if i < len(normalized_agents) else i
                            normalized_actions.append({
                                "tick": step,
                                "agent_id": agent_id,
                                "action": str(actions[i]),
                                "target_id": None,
                                "detail": "",
                            })
                except (ImportError, Exception):
                    pass

        # --- Normalize rewards and attach to agents ---
        if rewards is not None:
            if isinstance(rewards, dict):
                for aid, r in rewards.items():
                    for agent in normalized_agents:
                        if agent["agent_id"] == aid:
                            agent["energy"] = float(r) if not isinstance(r, (int, float)) else r
            elif isinstance(rewards, (list, tuple)):
                for i, r in enumerate(rewards):
                    if i < len(normalized_agents):
                        normalized_agents[i]["energy"] = float(r)
            else:
                try:
                    import numpy as np
                    if isinstance(rewards, np.ndarray):
                        for i in range(min(rewards.shape[0], len(normalized_agents))):
                            normalized_agents[i]["energy"] = float(rewards[i])
                except (ImportError, Exception):
                    pass

        return {
            "tick": step,
            "step": step,
            "agents": normalized_agents,
            "actions": normalized_actions,
            "defection_events": kwargs.get("defection_events", []),
            "objectives": kwargs.get("objectives", []),
            "team_scores": kwargs.get("team_scores", {}),
        }

    def _make_agent_entry(self, agent_id: Any, data: Any) -> dict:
        """Create a minimal agent dict from raw data."""
        if isinstance(data, dict):
            entry = dict(data)
            entry.setdefault("agent_id", agent_id)
            return entry
        return {
            "agent_id": agent_id,
            "team": 0,
            "role": "agent",
            "x": 0,
            "y": 0,
            "resources": 0,
            "hearts": 0,
            "energy": 0,
            "is_defector": False,
            "coalition_id": 0,
        }

    def _build_relationships(self) -> List[dict]:
        """Build relationship edges from accumulated help tracking."""
        edges = []
        for (a, b), count in self._metrics.help_matrix.items():
            if count > 0:
                reverse = self._metrics.help_matrix.get((b, a), 0)
                total = count + reverse
                reciprocity = (min(count, reverse) / max(count, reverse)) if max(count, reverse) > 0 else 0
                # Determine same_team from agents cache
                team_a = self._agents_cache.get(a, {}).get("team")
                team_b = self._agents_cache.get(b, {}).get("team")
                edges.append({
                    "source": a,
                    "target": b,
                    "weight": total,
                    "reciprocity": round(reciprocity, 3),
                    "same_team": team_a == team_b if (team_a is not None and team_b is not None) else False,
                })
        return edges

    def _flush_queue(self):
        """Ensure the persistent WebSocket daemon thread is running."""
        if not self._ws_thread_started:
            self._ws_thread_started = True
            self._ws_stop_event.clear()
            self._ws_thread = threading.Thread(
                target=self._ws_sender_loop,
                daemon=True,
                name="alignscope-ws",
            )
            self._ws_thread.start()

    def _ws_sender_loop(self):
        """
        Single persistent background thread. Handles connection,
        reconnection with exponential backoff, and queue draining.
        Never spawns additional threads.
        """
        backoff = 0.5
        max_backoff = 30.0

        while not self._ws_stop_event.is_set():
            try:
                asyncio.run(self._ws_send_async())
            except Exception as e:
                print(f"[AlignScope] WS disconnected: {type(e).__name__}: {e}")

            if self._ws_stop_event.is_set():
                break

            # Exponential backoff before reconnect
            print(f"[AlignScope] Reconnecting in {backoff:.1f}s...")
            self._ws_stop_event.wait(timeout=backoff)
            backoff = min(backoff * 2, max_backoff)

        print("[AlignScope] WS sender thread exiting.")

    def _build_auto_config(self) -> dict:
        """Auto-derive config from seen agents when none is explicitly provided."""
        teams = {}
        roles = set()
        for agent in self._agents_cache.values():
            tid = agent.get("team", 0)
            if tid not in teams:
                teams[tid] = {"id": tid, "name": f"Team {tid}", "size": 0, "color": ""}
            teams[tid]["size"] += 1
            roles.add(agent.get("role", "agent"))
        
        team_colors = ['#6d9eeb', '#e8925a', '#4abe7d', '#b06ec7']
        team_list = []
        for tid in sorted(teams.keys()):
            t = teams[tid]
            t["color"] = team_colors[tid % len(team_colors)]
            team_list.append(t)
        
        return {
            "num_agents": len(self._agents_cache),
            "teams": team_list,
            "roles": list(roles),
            "num_objectives": 0,
            "max_ticks": 500,
            "preset": self.preset,
            "paradigm": self.paradigm,
            "metrics": self.metrics_config,
            "events": self.events_config,
            "topology": self.topology_config,
        }

    async def _ws_send_async(self):
        """
        Async WebSocket sender. Connects, sends config, then drains queue.
        Raises on disconnect so the caller loop can reconnect.
        """
        import websockets
        print(f"[AlignScope] Connecting to {self.server_url}...")
        async with websockets.connect(self.server_url) as ws:
            self._ws = ws
            print(f"[AlignScope] Connected to {self.server_url}")

            # Send config on every (re)connect
            config_to_send = self.config
            if not config_to_send:
                config_to_send = self._build_auto_config()
            if config_to_send:
                await ws.send(json.dumps({"type": "config", "data": config_to_send}))

            sent_count = 0
            while not self._ws_stop_event.is_set():
                if self._send_queue:
                    payload = self._send_queue.popleft()
                    await ws.send(json.dumps(payload))
                    sent_count += 1
                    if sent_count % 50 == 0:
                        print(f"[AlignScope] Sent {sent_count} ticks ({len(self._send_queue)} queued)")
                else:
                    await asyncio.sleep(0.01)

    def _init_wandb_bridge(self):
        """Initialize W&B forwarding if wandb is installed."""
        try:
            from alignscope.integrations.wandb_bridge import WandbBridge
            self._wandb_bridge = WandbBridge()
        except (ImportError, Exception):
            pass

    def _init_mlflow_bridge(self):
        """Initialize MLflow forwarding if mlflow is installed."""
        try:
            from alignscope.integrations.mlflow_bridge import MlflowBridge
            self._mlflow_bridge = MlflowBridge()
        except (ImportError, Exception):
            pass

    def _forward_metrics(self, step: int, metrics: dict, events: list):
        """Forward alignment metrics to all active integrations."""
        if self._wandb_bridge:
            try:
                self._wandb_bridge.log(step, metrics, events)
            except Exception:
                pass

        if self._mlflow_bridge:
            try:
                self._mlflow_bridge.log(step, metrics, events)
            except Exception:
                pass

    def reset(self):
        """Reset metrics state between episodes. Call from env.reset()."""
        with self._lock:
            self._metrics.reset()
            self._detector = DefectionDetector()
            self._step = 0
            self._role_history.clear()
            # Keep _agents_cache and _send_queue intact for continuity

    def finish(self):
        """Finalize the tracking session."""
        summary = self._detector.get_summary()
        if self._wandb_bridge:
            try:
                self._wandb_bridge.finish(summary)
            except Exception:
                pass
        if self._mlflow_bridge:
            try:
                self._mlflow_bridge.finish(summary)
            except Exception:
                pass
        # Signal the persistent WS thread to stop
        self._ws_stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        self._ws_thread_started = False
        self._send_queue.clear()
