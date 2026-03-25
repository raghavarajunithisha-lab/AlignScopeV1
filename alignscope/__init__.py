"""
AlignScope — Real-time alignment observability for multi-agent RL.

Usage:
    import alignscope

    # Initialize a tracking run
    alignscope.init(project="my-experiment")

    # In your training loop, add one line:
    alignscope.log(step, agents, obs, actions, rewards)

    # Or start the dashboard server:
    alignscope.start(port=8000, demo=True)

    # Framework-specific integrations:
    alignscope.patch("rllib")           # Auto-patch RLlib
    env = alignscope.wrap(env)          # Wrap PettingZoo env
"""

from typing import Optional, Union

__version__ = "0.1.0"

from alignscope.sdk import AlignScopeTracker

# Module-level singleton tracker
_tracker: Optional[AlignScopeTracker] = None


def init(
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
) -> AlignScopeTracker:
    """
    Initialize an AlignScope tracking session.

    Args:
        project: Name for this experiment run
        server_url: WebSocket URL of the AlignScope dashboard server
        config: Optional environment config (teams, roles, etc.)
        forward_wandb: If True and wandb is installed, forward metrics to W&B
        forward_mlflow: If True and mlflow is installed, forward metrics to MLflow

    Returns:
        AlignScopeTracker instance
    """
    global _tracker
    _tracker = AlignScopeTracker(
        project=project,
        server_url=server_url,
        preset=preset,
        paradigm=paradigm,
        metrics=metrics,
        events=events,
        topology=topology,
        config=config,
        forward_wandb=forward_wandb,
        forward_mlflow=forward_mlflow,
    )
    return _tracker


def log(
    step: int,
    agents = None,
    obs: object = None,
    actions: object = None,
    rewards: object = None,
    **kwargs,
) -> None:
    """
    Log one step of multi-agent data. This is the Tier 2 one-line API.

    Args:
        step: Current timestep / tick number
        agents: Agent states — list of dicts or framework-specific format
        obs: Observations (auto-normalized from framework format)
        actions: Actions taken (auto-normalized)
        rewards: Rewards received (auto-normalized)
        **kwargs: Additional data to attach to this step
    """
    global _tracker
    if _tracker is None:
        # Auto-init with defaults if not explicitly initialized
        init()
    _tracker.log(step, agents=agents, obs=obs, actions=actions, rewards=rewards, **kwargs)


def report(tick: int, agent: Union[str, int], metrics: dict) -> None:
    """Dynamically report custom metrics for an agent."""
    global _tracker
    if _tracker is None:
        init()
    _tracker.report(tick, agent, metrics)


def event(tick: int, type: str, agent: Union[str, int], detail: str, severity: float = 0.5) -> None:
    """Dynamically report custom events."""
    global _tracker
    if _tracker is None:
        init()
    _tracker.event(tick, type, agent, detail, severity)


def start(port: int = 8000, host: str = "0.0.0.0", demo: bool = False) -> None:
    """
    Start the AlignScope dashboard server.

    Args:
        port: Port to serve on (default 8000)
        host: Host to bind to (default 0.0.0.0)
        demo: If True, run the built-in demo simulator
    """
    from alignscope.server import run_server
    run_server(host=host, port=port, demo=demo)


def patch(framework: str) -> None:
    """
    Auto-patch a MARL framework for zero-code integration (Tier 1).

    Args:
        framework: One of "rllib", "pettingzoo", "pymarl"
    """
    from alignscope.patches import apply_patch
    apply_patch(framework)


def wrap(env, **kwargs):
    """
    Wrap a PettingZoo environment for automatic logging (Tier 3).

    Args:
        env: A PettingZoo environment instance
        **kwargs: Additional config passed to the wrapper

    Returns:
        Wrapped environment that auto-logs to AlignScope
    """
    from alignscope.patches.pettingzoo import AlignScopeWrapper
    return AlignScopeWrapper(env, **kwargs)
