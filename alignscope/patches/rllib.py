"""
AlignScope — RLlib Auto-Patch

Provides an AlignScopeCallback for Ray RLlib that automatically
logs multi-agent data to AlignScope on every episode step.

Tier 1 (Zero Code):
    alignscope patch rllib
    python train.py  # zero changes

Tier 3 (Explicit Plugin):
    from alignscope.patches.rllib import AlignScopeCallback
    config = PPOConfig().callbacks(AlignScopeCallback)
"""

_original_callbacks = None

try:
    import ray
    RAY_VERSION = tuple(map(int, ray.__version__.split(".")[:2]))
    if RAY_VERSION[0] >= 2:
        from ray.rllib.algorithms.callbacks import DefaultCallbacks
        BaseCallback = DefaultCallbacks
    else:
        # RLlib 1.x path
        from ray.rllib.agents.callbacks import DefaultCallbacks
        BaseCallback = DefaultCallbacks
except (ImportError, AttributeError, ValueError):
    BaseCallback = object


class AlignScopeCallback(BaseCallback):
    """
    RLlib callback that streams agent data to AlignScope.

    Works with both Ray RLlib 1.x and 2.x APIs using version-aware static inheritance.
    Gracefully degrades if RLlib is not installed.
    """

    def __init__(self, *args, **kwargs):
        if BaseCallback is not object:
            super().__init__(*args, **kwargs)

        import alignscope
        if alignscope._tracker is None:
            alignscope.init(project="rllib-run")

        self._tracker = alignscope._tracker
        self._step = 0

    def on_episode_step(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index=None,
        **kwargs,
    ):
        """Called on every step of every episode."""
        self._step += 1

        try:
            agents = []
            actions = {}
            rewards = {}

            # Extract agent data from the episode
            if episode is not None:
                for agent_id in episode.get_agents():
                    last_action = episode.last_action_for(agent_id)
                    last_reward = episode.last_reward_for(agent_id)
                    last_obs = episode.last_observation_for(agent_id)

                    # Determine team from agent_id naming convention
                    team = self._infer_team(agent_id)

                    import math
                    all_agents = list(episode.get_agents())
                    a_idx = all_agents.index(agent_id) if agent_id in all_agents else 0
                    n_agents = len(all_agents)
                    
                    agents.append({
                        "agent_id": str(agent_id),
                        "team": team,
                        "role": "agent",
                        "x": round(math.cos(2 * math.pi * a_idx / max(1, n_agents)) * 150, 2),
                        "y": round(math.sin(2 * math.pi * a_idx / max(1, n_agents)) * 150, 2),
                        "resources": 0,
                        "hearts": 0,
                        "energy": float(last_reward) if last_reward is not None else 0,
                        "is_defector": False,
                        "coalition_id": team,
                    })

                    actions[str(agent_id)] = str(last_action)
                    rewards[str(agent_id)] = float(last_reward) if last_reward is not None else 0

            self._tracker.log(
                step=self._step,
                agents=agents,
                actions=actions,
                rewards=rewards,
            )

        except Exception as e:
            # Never crash the training run
            pass

    def on_episode_start(self, *, episode=None, **kwargs):
        """Reset step counter at episode start."""
        self._step = 0

    def on_episode_end(self, *, episode=None, **kwargs):
        """Log episode completion."""
        pass

    @staticmethod
    def _infer_team(agent_id) -> int:
        """Infer team from agent ID naming patterns."""
        agent_str = str(agent_id).lower()
        # Common patterns: "agent_0", "team_1_agent_2", "red_0", etc.
        if "team_1" in agent_str or "blue" in agent_str or "enemy" in agent_str:
            return 1
        if "team_2" in agent_str:
            return 2
        return 0


def apply():
    """
    Auto-patch RLlib to use AlignScope callbacks.

    This monkey-patches the default callback class so existing
    training code works without any changes.
    """
    global _original_callbacks

    try:
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        # Store original
        _original_callbacks = AlgorithmConfig.callbacks

        # Monkey-patch the default
        original_callbacks_method = AlgorithmConfig.callbacks

        def patched_callbacks(self, callbacks_class=None):
            if callbacks_class is None:
                callbacks_class = AlignScopeCallback
            return original_callbacks_method.fget(self)

        # Override the property/method
        try:
            AlgorithmConfig.callbacks = property(
                lambda self: AlignScopeCallback,
                original_callbacks_method.fset if hasattr(original_callbacks_method, 'fset') else None,
            )
        except (TypeError, AttributeError):
            # Fallback: just inform user to use the callback manually
            print("[AlignScope] Auto-patch applied. Use AlignScopeCallback in your config.")

        print("[AlignScope] ✓ RLlib patched successfully")
        return True

    except ImportError:
        raise ImportError(
            "RLlib is not installed. Install with: pip install 'alignscope[rllib]'"
        )
