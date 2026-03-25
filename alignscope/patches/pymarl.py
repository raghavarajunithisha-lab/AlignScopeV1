"""
AlignScope — PyMARL / EPyMARL Integration

Hooks into PyMARL's EpisodeRunner to extract TRUE per-agent data
(individual rewards, actions, alive masks) before they get averaged
into aggregate stats like return_mean.

Usage:
    Tier 3 (Config-only):
        # In config.yaml:
        logger: alignscope

    Tier 2 (Programmatic):
        import alignscope
        alignscope.patch("pymarl")
"""


class AlignScopeLogger:
    """
    PyMARL-compatible logger that forwards multi-agent data to AlignScope.

    PyMARL loggers implement log_stat(key, value, step) and can
    optionally implement setup(log_dir, args) and console_logger.
    """

    def __init__(self):
        self._step = 0
        self._buffer = {}
        self._tracker = None
        self._num_agents = 5  # Default, updated from env_info
        self._map_name = "unknown"

    def setup(self, log_dir=None, args=None):
        """Called by PyMARL when the logger is initialized."""
        import alignscope

        project = "pymarl-run"
        if args and hasattr(args, 'env_args'):
            self._map_name = args.env_args.get('map_name', 'unknown')
            project = f"pymarl-{self._map_name}"

        if alignscope._tracker is None:
            alignscope.init(project=project)
        self._tracker = alignscope._tracker

    def log_stat(self, key: str, value, step: int):
        """
        Called by PyMARL for each logged statistic.

        We buffer everything and flush on 'end of episode' marker keys.
        """
        self._buffer[key] = value
        self._step = step

        # Flush on common "end of step" keys
        if key in ("return_mean", "ep_length_mean", "test_return_mean"):
            self._flush(step)

    def ingest_episode(self, episode_batch, step: int):
        """
        Direct hook into EpisodeRunner's episode data.
        This receives the FULL episode batch with per-agent, per-timestep data
        before PyMARL averages it away.

        episode_batch is a PyMARL EpisodeBatch with shape:
            rewards:      (batch, timesteps, n_agents)
            actions:      (batch, timesteps, n_agents, 1)
            avail_actions:(batch, timesteps, n_agents, n_actions)
            terminated:   (batch, timesteps, 1)
        """
        if not self._tracker:
            return

        try:
            rewards = episode_batch["reward"]       # (batch, T, n_agents or 1)
            actions = episode_batch["actions"]      # (batch, T, n_agents, 1)
            terminated = episode_batch["terminated"] # (batch, T, 1)

            batch_size = rewards.shape[0]
            timesteps = rewards.shape[1]
            
            # Determine n_agents from actions shape
            n_agents = actions.shape[2] if len(actions.shape) > 2 else self._num_agents

            for b in range(min(batch_size, 1)):  # Process first episode in batch
                for t in range(timesteps):
                    if terminated[b, t, 0]:
                        break

                    agents = []
                    agent_rewards = {}
                    agent_actions = {}

                    for a in range(n_agents):
                        # Per-agent reward (may be shared in cooperative)
                        if len(rewards.shape) == 3 and rewards.shape[2] > 1:
                            r = float(rewards[b, t, a])
                        else:
                            # Shared reward — distribute equally
                            r = float(rewards[b, t, 0]) / n_agents

                        act = int(actions[b, t, a, 0]) if len(actions.shape) == 4 else int(actions[b, t, a])

                        import math
                        agent_id = f"agent_{a}"
                        agents.append({
                            "agent_id": agent_id,
                            "team": 0,  # PyMARL is cooperative (same team)
                            "role": "agent",
                            "x": round(math.cos(2 * math.pi * a / max(1, n_agents)) * 150, 2),
                            "y": round(math.sin(2 * math.pi * a / max(1, n_agents)) * 150, 2),
                            "resources": 0,
                            "hearts": 0,
                            "energy": r,
                            "is_defector": False,
                            "coalition_id": 0,
                        })
                        agent_rewards[agent_id] = r
                        agent_actions[agent_id] = str(act)

                    self._tracker.log(
                        step=step + t,
                        agents=agents,
                        actions=agent_actions,
                        rewards=agent_rewards,
                    )

        except Exception:
            pass  # Never crash training

    def _flush(self, step: int):
        """
        Fallback: send buffered aggregate data to AlignScope when
        per-agent episode data is not available (e.g. older PyMARL forks).
        """
        if not self._tracker:
            return

        try:
            agents = []
            rewards = {}

            # Try to reconstruct from per-agent keys first
            agent_ids = set()
            for key in self._buffer:
                if key.startswith("agent_"):
                    parts = key.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        agent_ids.add(int(parts[1]))

            if not agent_ids:
                num_agents = self._buffer.get("n_agents", self._num_agents)
                agent_ids = set(range(num_agents))

            # Distribute shared reward across agents
            shared_reward = self._buffer.get("return_mean", 0)
            per_agent_reward = float(shared_reward) / max(len(agent_ids), 1)

            battle_won = self._buffer.get("battle_won_mean", None)

            for aid in sorted(agent_ids):
                # Use per-agent reward if available, else distribute shared reward
                r = self._buffer.get(f"agent_{aid}_reward", per_agent_reward)
                agents.append({
                    "agent_id": f"agent_{aid}",
                    "team": 0,
                    "role": "agent",
                    "x": float(aid * 60),
                    "y": 0.0,
                    "resources": 0,
                    "hearts": 0,
                    "energy": float(r),
                    "is_defector": False,
                    "coalition_id": 0,
                })
                rewards[f"agent_{aid}"] = float(r)

            # Generate defection events if battle was lost
            defection_events = []
            if battle_won is not None and float(battle_won) < 0.5:
                defection_events.append({
                    "agent_id": "team",
                    "team": 0,
                    "previous_role": "agent",
                    "severity": 1.0 - float(battle_won),
                    "reason": f"Battle lost (win_rate={battle_won:.2f})",
                })

            self._tracker.log(
                step=step,
                agents=agents,
                rewards=rewards,
                defection_events=defection_events,
            )

        except Exception:
            pass  # Never crash training

        self._buffer = {}

    def print_recent_stats(self):
        """Called by PyMARL to print recent stats to console."""
        pass


def apply():
    """
    Auto-patch PyMARL to use AlignScope.

    Hooks into TWO locations:
    1. Logger.log_stat — captures aggregate stats (fallback)
    2. EpisodeRunner.run — captures TRUE per-agent episode data (primary)
    """
    try:
        # --- Hook 1: Logger (aggregate stats fallback) ---
        try:
            import src.utils.logging as logging_module
        except ImportError:
            import utils.logging as logging_module

        if not hasattr(logging_module, "Logger"):
            print("[AlignScope] Native Logger class not found.")
            return False

        _original_init = logging_module.Logger.__init__
        _original_log_stat = logging_module.Logger.log_stat

        def patched_init(self, console_logger):
            _original_init(self, console_logger)
            self._alignscope_logger = AlignScopeLogger()
            self._alignscope_logger.setup()

        def patched_log_stat(self, key, value, t, to_sacred=True):
            _original_log_stat(self, key, value, t, to_sacred)
            if hasattr(self, '_alignscope_logger'):
                self._alignscope_logger.log_stat(key, value, t)

        logging_module.Logger.__init__ = patched_init
        logging_module.Logger.log_stat = patched_log_stat

        print("[AlignScope] [OK] EPyMARL Logger patched")

        # --- Hook 2: EpisodeRunner (per-agent data) ---
        try:
            try:
                from src.runners.episode_runner import EpisodeRunner
            except ImportError:
                from runners.episode_runner import EpisodeRunner

            _original_run = EpisodeRunner.run

            def patched_run(self, test_mode=False):
                episode_batch = _original_run(self, test_mode=test_mode)

                # Forward the raw episode batch to AlignScope
                try:
                    if hasattr(self, 'logger') and hasattr(self.logger, '_alignscope_logger'):
                        self.logger._alignscope_logger.ingest_episode(
                            episode_batch,
                            step=self.t_env,
                        )
                except Exception:
                    pass  # Never crash training

                return episode_batch

            EpisodeRunner.run = patched_run
            print("[AlignScope] [OK] EpisodeRunner.run patched (per-agent extraction)")
        except ImportError:
            print("[AlignScope] EpisodeRunner not found — using aggregate logger only")

        return True

    except ImportError:
        print("[AlignScope] PyMARL not found in standard locations.")
        return False
