"""
AlignScope — PettingZoo Integration (Universal)

Provides a wrapper that auto-logs every step of ANY PettingZoo
environment to AlignScope — no environment-specific code.

Usage:
    import alignscope
    env = alignscope.wrap(your_pettingzoo_env)
    # done — every step auto-logged

How it works:
    1. Roles inferred from agent name  (knight_0 → "knight", player_0 → "player")
    2. Teams inferred from name keywords or index-based fallback
    3. Positions extracted from env state if available, else grid layout
    4. Deaths detected by comparing alive sets between rounds
    5. AEC vs Parallel API auto-detected
    6. Works with KAZ, MPE, Classic games, Atari, SISL, etc.
"""


class AlignScopeWrapper:
    """
    Wraps any PettingZoo environment to auto-log agent interactions.
    Compatible with both AEC and Parallel API environments.
    """

    def __init__(self, env, project: str = "pettingzoo-run", **kwargs):
        self.env = env
        self._step = 0

        self._round_actions: dict = {}
        self._round_rewards: dict = {}
        # Fixed at init — possible_agents never changes mid-episode
        self._round_size = len(env.possible_agents)

        # Track alive agents to detect deaths each round
        self._prev_alive: set = set(env.possible_agents)

        import alignscope
        if alignscope._tracker is None:
            alignscope.init(project=project)
        self._tracker = alignscope._tracker

        # Only AEC envs have agent_iter
        self._is_parallel = not hasattr(env, 'agent_iter')

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------------ #
    # Parallel API                                                         #
    # ------------------------------------------------------------------ #

    def step(self, actions):
        if self._is_parallel:
            return self._parallel_step(actions)
        else:
            return self._aec_step(actions)

    def _parallel_step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._step += 1

        currently_alive = set(self.env.agents)
        defection_events = self._detect_deaths(currently_alive)
        self._prev_alive = currently_alive

        agents = []
        for i, agent_id in enumerate(self.env.possible_agents):
            alive = agent_id in currently_alive
            reward = rewards.get(agent_id, 0) if isinstance(rewards, dict) else 0
            role, x, y = self._extract_state(agent_id, i)
            agents.append({
                "agent_id": str(agent_id),
                "team": self._infer_team(agent_id, i),
                "role": role,
                "x": x,
                "y": y,
                "resources": 0,
                "hearts": 0,
                "energy": float(reward),
                "is_defector": not alive,
                "coalition_id": self._infer_team(agent_id, i) if alive else -1,
            })

        self._tracker.log(
            step=self._step,
            agents=agents,
            actions={str(k): str(v) for k, v in actions.items()},
            rewards={str(k): float(v) for k, v in rewards.items()},
            defection_events=defection_events,
        )

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # AEC API                                                              #
    # ------------------------------------------------------------------ #

    def _aec_step(self, action):
        """
        Accumulate per-round actions/rewards keyed by possible_agents
        count. Flush once all agents have acted. Robust to agent deaths.
        """
        agent_id = self.env.agent_selection
        self.env.step(action)

        self._round_actions[str(agent_id)] = str(action)

        for aid in self.env.possible_agents:
            self._round_rewards[str(aid)] = float(
                self.env.rewards.get(aid, 0)
            )

        # Flush when all agents have acted, or if only dead agents remain
        if len(self._round_actions) >= len(self.env.agents) or len(self._round_actions) >= self._round_size:
            self._flush_round()
            self._round_actions = {}
            self._round_rewards = {}

    def _flush_round(self):
        """Build agent list and log one AlignScope tick."""
        self._step += 1
        agents = []
        currently_alive = set(self.env.agents)

        # Emit defection_events for newly dead agents so the
        # timeline shows a red Defection marker for each death
        defection_events = self._detect_deaths(currently_alive)

        for i, aid in enumerate(self.env.possible_agents):
            alive = aid in currently_alive
            reward = self._round_rewards.get(str(aid), 0.0)
            role, x, y = self._extract_state(aid, i)

            agents.append({
                "agent_id": str(aid),
                "team": self._infer_team(aid, i),
                "role": role,
                "x": x,
                "y": y,
                "resources": 0,
                "hearts": 0,
                "energy": reward,
                "is_defector": not alive,
                "coalition_id": self._infer_team(aid, i) if alive else -1,
            })

        # Update AFTER building defection events — compare this round to last
        self._prev_alive = currently_alive

        self._tracker.log(
            step=self._step,
            agents=agents,
            actions=dict(self._round_actions),
            rewards=dict(self._round_rewards),
            defection_events=defection_events,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _detect_deaths(self, currently_alive: set) -> list:
        """
        Returns one defection_event dict per agent that died this round
        (was in _prev_alive but not in currently_alive).
        detector.py forwards these to the timeline as red markers.
        """
        events = []
        newly_dead = self._prev_alive - currently_alive
        possible = list(self.env.possible_agents)
        total_agents = len(possible)
        remaining = len(currently_alive)

        for aid in newly_dead:
            i = possible.index(aid) if aid in possible else 0
            # Dynamic severity: losing agents when few remain is more severe
            severity = float(f"{min(1.0, 1.0 - (remaining / max(total_agents, 1))):.2f}")
            events.append({
                "agent_id": str(aid),
                "team": self._infer_team(aid, i),
                "previous_role": self._infer_role(aid),
                "severity": severity,
                "reason": f"{aid} was eliminated (terminated by environment)",
            })
        return events

    def _extract_state(self, agent_id: str, index: int):
        """
        Uses adapters.py for advanced physics engine extraction (MPE/KAZ).
        Falls back to basic static heuristics otherwise.
        """
        try:
            from alignscope.adapters import try_extract_env_state
            state = try_extract_env_state(self.env, agent_id)
            
            # 1. Physics coords
            if state["x"] != 0.0 or state["y"] != 0.0:
                # Scale MPE physics engine so it graphs nicely in pixels
                x = float(state["x"] * 200 + 400)
                y = float(state["y"] * 200 + 300)
            else:
                x, y = self._get_fallback_position(agent_id, index)
                
            # 2. Roles
            role = state.get("role")
            if not role or role == "agent":
                role = self._infer_role(agent_id)
                
            return role, x, y
        except Exception:
            # Absolute fallback
            return self._infer_role(agent_id), float(index * 60), float(self._infer_team(agent_id, index) * 100)

    def _get_fallback_position(self, agent_id: str, index: int):
        try:
            unwrapped = self.env.unwrapped

            # Strategy: Pygame rect-based position (KAZ, Pistonball, etc.)
            if hasattr(unwrapped, 'agent_name_mapping') and hasattr(unwrapped, 'agent_list'):
                idx = unwrapped.agent_name_mapping.get(agent_id)
                if idx is not None:
                    agent_obj = unwrapped.agent_list[idx]
                    if hasattr(agent_obj, 'rect'):
                        return float(agent_obj.rect.x), float(agent_obj.rect.y)

            # Strategy 2: Generic .position attribute
            if hasattr(unwrapped, 'agents_dict'):
                agent_obj = unwrapped.agents_dict.get(agent_id)
                if agent_obj and hasattr(agent_obj, 'position'):
                    pos = agent_obj.position
                    return float(pos[0]), float(pos[1])
        except Exception:
            pass
            
        # Fallback: grid layout based on index and team
        return float(index * 60), float(self._infer_team(agent_id, index) * 100)

    @staticmethod
    def _infer_role(agent_id) -> str:
        """
        Extract role from agent name using PettingZoo's universal
        naming convention: {type}_{id}.
            knight_0  → "knight"
            archer_1  → "archer"
            player_0  → "player"
            adversary_0 → "adversary"
        """
        name = str(agent_id).lower()
        if "_" in name:
            return name.rsplit("_", 1)[0]  # handles multi-word like "evader_agent_0"
        return "agent"

    @staticmethod
    def _infer_team(agent_id, index: int) -> int:
        """
        Infer team from agent name using keyword heuristics.
        Works across PettingZoo envs without any env-specific logic.

        Priority:
            1. Explicit team keywords ("team_0", "enemy", "ally")
            2. Adversarial keywords ("adversary" → team 1)
            3. Fallback: all agents on team 0 (cooperative default)
        """
        agent_str = str(agent_id).lower()

        # Explicit team keywords
        for keyword, team in [
            ("second", 1), ("blue", 1), ("enemy", 1), ("team_1", 1),
            ("first", 0), ("red", 0), ("ally", 0), ("team_0", 0),
        ]:
            if keyword in agent_str:
                return team

        # Adversarial agents go to team 1 (MPE adversary, predator, etc.)
        for adversarial in ["adversary", "predator", "evader"]:
            if adversarial in agent_str:
                return 1

        # Default: team 0 (most PettingZoo envs are cooperative)
        return 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None, **kwargs):
        """Clear all round buffers, reset alive tracking, and isolate metrics between episodes."""
        self._step = 0
        self._round_actions = {}
        self._round_rewards = {}
        self._prev_alive = set(self.env.possible_agents)
        # Propagate reset to the tracker's metrics engine for inter-episode isolation
        if hasattr(self._tracker, 'reset'):
            self._tracker.reset()
        return self.env.reset(seed=seed, options=options, **kwargs)

    def last(self):
        return self.env.last()

    def agent_iter(self):
        return self.env.agent_iter()

    def close(self):
        return self.env.close()


def apply():
    """Auto-patch PettingZoo's make() to auto-wrap environments."""
    try:
        import pettingzoo

        if hasattr(pettingzoo, 'make'):
            original_make = pettingzoo.make

            def patched_make(*args, **kwargs):
                env = original_make(*args, **kwargs)
                return AlignScopeWrapper(env)

            pettingzoo.make = patched_make

        print("[AlignScope] ✓ PettingZoo patched successfully")
        print("[AlignScope]   New environments will be auto-wrapped.")
        print("[AlignScope]   Or use: env = alignscope.wrap(your_env)")
        return True

    except ImportError:
        raise ImportError(
            "PettingZoo is not installed. Install with: pip install 'alignscope[pettingzoo]'"
        )