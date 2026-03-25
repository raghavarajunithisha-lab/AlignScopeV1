from __future__ import annotations

"""
AlignScope — Multi-Agent RL Episode Simulator

Generates realistic multi-agent RL episode traces for demonstration.
Models key dynamics common to cooperative multi-agent environments:
  - Multiple teams of agents with configurable specialized roles
  - Resource gathering, objective capture, territory defense
  - Coalition formation and occasional defection events

This is a DEMO simulator. Replace this module with a real data source
(e.g., episode log parser, live environment bridge) for production use.
See the README for the generic data schema.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TeamConfig:
    """Configuration for a single team."""
    name: str
    size: int
    color: str = ""  # optional color hint for frontend


@dataclass
class SimulatorConfig:
    """
    Environment-agnostic configuration for the MARL simulator.
    Pass this to MARLSimulator to define teams, roles, objectives, etc.
    """
    teams: list[TeamConfig] = field(default_factory=lambda: [
        TeamConfig(name="Alpha", size=5, color="#6d9eeb"),
        TeamConfig(name="Beta", size=5, color="#e8925a"),
    ])
    roles: list[str] = field(default_factory=lambda: [
        "gatherer", "coordinator", "disruptor", "scout"
    ])
    num_objectives: int = 8
    map_width: int = 20
    map_height: int = 20
    max_ticks: int = 500
    defection_probability: float = 0.003
    seed: Optional[int] = None


# Default action types common across MARL environments
DEFAULT_ACTIONS = [
    "gather", "deposit", "capture", "defend",
    "disrupt", "scout_area", "switch_role",
    "move", "help_ally", "noop",
]


@dataclass
class Agent:
    agent_id: int
    team: int
    role: str
    x: float
    y: float
    resources: int = 0
    hearts: int = 0
    role_history: list = field(default_factory=list)
    help_given: dict = field(default_factory=dict)   # {target_id: count}
    help_received: dict = field(default_factory=dict) # {source_id: count}
    action_trajectory: list = field(default_factory=list)
    is_defector: bool = False
    defection_tick: Optional[int] = None
    coalition_id: Optional[int] = None

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "team": self.team,
            "role": self.role,
            "x": int(self.x),
            "y": int(self.y),
            "resources": self.resources,
            "hearts": self.hearts,
            "energy": self.resources + self.hearts,
            "is_defector": self.is_defector,
            "coalition_id": self.coalition_id,
        }


@dataclass
class Objective:
    """A shared objective/resource point on the map (territory, flag, resource node, etc.)."""
    objective_id: int
    x: float
    y: float
    owner: Optional[int] = None  # team id or None
    control_strength: float = 0.0

    def to_dict(self):
        return {
            "objective_id": self.objective_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "owner": self.owner,
            "control_strength": round(self.control_strength, 2),
        }


@dataclass
class GameAction:
    tick: int
    agent_id: int
    action: str
    target_id: Optional[int] = None
    detail: str = ""

    def to_dict(self):
        return {
            "tick": self.tick,
            "agent_id": self.agent_id,
            "action": self.action,
            "target_id": self.target_id,
            "detail": self.detail,
        }


class MARLSimulator:
    """
    Generic Multi-Agent RL episode simulator.

    Simulates N teams competing in a shared environment with:
    - Role-based specialization with interdependencies
    - Emergent coalitions within teams
    - Occasional defection events where an agent stops cooperating

    Configurable via SimulatorConfig for any MARL environment.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        if config is None:
            config = SimulatorConfig()

        self.config = config
        self.teams = config.teams
        self.roles = config.roles
        self.num_objectives = config.num_objectives
        self.map_width = config.map_width
        self.map_height = config.map_height
        self.max_ticks = config.max_ticks
        self.defection_probability = config.defection_probability
        self.rng = random.Random(config.seed)

        self.agents: list[Agent] = []
        self.objectives: list[Objective] = []
        self.tick = 0
        self.actions_log: list[GameAction] = []

        self._initialize_game()

    def _initialize_game(self):
        """Set up agents and objectives based on config."""
        agent_id_counter = 0
        num_teams = len(self.teams)

        for team_idx, team_cfg in enumerate(self.teams):
            # Distribute teams across the map horizontally
            x_start = (team_idx / num_teams) * self.map_width * 0.3 + 1
            x_end = x_start + self.map_width * 0.3

            for i in range(team_cfg.size):
                role = self.roles[i % len(self.roles)]
                agent = Agent(
                    agent_id=agent_id_counter,
                    team=team_idx,
                    role=role,
                    x=self.rng.uniform(x_start, x_end),
                    y=self.rng.uniform(1, self.map_height - 1),
                    coalition_id=team_idx * 100 + i // 3,
                )
                agent.role_history.append(role)
                self.agents.append(agent)
                agent_id_counter += 1

        # Create objectives distributed across the map
        for j in range(self.num_objectives):
            self.objectives.append(Objective(
                objective_id=j,
                x=self.rng.uniform(2, self.map_width - 2),
                y=self.rng.uniform(2, self.map_height - 2),
            ))

    def get_config_payload(self) -> dict:
        """
        Returns the full environment configuration for the frontend.
        Sent once at connection start so the UI can dynamically build itself.
        """
        total_agents = sum(t.size for t in self.teams)
        return {
            "num_agents": total_agents,
            "teams": [
                {
                    "id": i,
                    "name": t.name,
                    "size": t.size,
                    "color": t.color,
                }
                for i, t in enumerate(self.teams)
            ],
            "roles": self.roles,
            "num_objectives": self.num_objectives,
            "map_size": {"width": self.map_width, "height": self.map_height},
            "max_ticks": self.max_ticks,
        }

    def step(self) -> dict:
        """
        Advance the simulation by one tick.
        Returns the full game state for this tick.
        """
        self.tick += 1
        tick_actions = []

        for agent in self.agents:
            action = self._decide_action(agent)
            self._execute_action(agent, action)
            tick_actions.append(action)

        # Update objective control
        self._update_objectives()

        # Check for defection events
        defection_events = self._check_defections()

        # Compute team scores
        team_scores = self._compute_scores()

        return {
            "step": self.tick,
            "tick": self.tick,
            "agents": [a.to_dict() for a in self.agents],
            "objectives": [o.to_dict() for o in self.objectives],
            "actions": [a.to_dict() for a in tick_actions],
            "defection_events": defection_events,
            "team_scores": team_scores,
            "map_size": {"width": self.map_width, "height": self.map_height},
        }

    def _execute_action(self, agent: Agent, action: GameAction):
        """Track action in agent's trajectory for goal convergence metrics."""
        agent.action_trajectory.append(action.action)
        if len(agent.action_trajectory) > 100:
            agent.action_trajectory = agent.action_trajectory[-60:]
        self.actions_log.append(action)

    def _decide_action(self, agent: Agent) -> GameAction:
        """Determine an agent's action based on role, state, and relationships."""
        if agent.is_defector:
            return self._defector_action(agent)

        # Role-based behavior — map roles to behavior profiles
        role_idx = self.roles.index(agent.role) if agent.role in self.roles else 0
        behavior = role_idx % 4  # cycle through 4 behavior profiles

        if behavior == 0:
            return self._gatherer_action(agent)
        elif behavior == 1:
            return self._coordinator_action(agent)
        elif behavior == 2:
            return self._disruptor_action(agent)
        else:
            return self._scout_action(agent)

    def _gatherer_action(self, agent: Agent) -> GameAction:
        """Gatherers collect resources and deliver them to coordinators."""
        r = self.rng.random()

        if r < 0.4:
            agent.resources += self.rng.randint(1, 3)
            return GameAction(self.tick, agent.agent_id, "gather",
                              detail=f"resources={agent.resources}")
        elif r < 0.7:
            # Help a coordinator on the team
            coordinators = [a for a in self.agents
                            if a.team == agent.team
                            and self.roles.index(a.role) % 4 == 1
                            and not a.is_defector]
            if coordinators:
                target = self.rng.choice(coordinators)
                transfer = min(agent.resources, self.rng.randint(1, 2))
                agent.resources -= transfer
                target.hearts += transfer
                self._record_help(agent, target)
                return GameAction(self.tick, agent.agent_id, "help_ally",
                                  target_id=target.agent_id,
                                  detail=f"deposited {transfer} resources")
            return GameAction(self.tick, agent.agent_id, "gather",
                              detail="no coordinators nearby")
        else:
            self._move_agent(agent)
            return GameAction(self.tick, agent.agent_id, "move")

    def _coordinator_action(self, agent: Agent) -> GameAction:
        """Coordinators capture neutral objectives using accumulated resources."""
        r = self.rng.random()

        if r < 0.45 and agent.hearts > 0:
            target_o = self._nearest_capturable_objective(agent)
            if target_o:
                agent.hearts -= 1
                target_o.owner = agent.team
                target_o.control_strength = min(1.0, target_o.control_strength + 0.3)
                return GameAction(self.tick, agent.agent_id, "capture",
                                  target_id=target_o.objective_id,
                                  detail=f"captured objective {target_o.objective_id}")
        elif r < 0.65:
            friendly = [o for o in self.objectives if o.owner == agent.team]
            if friendly:
                o = self.rng.choice(friendly)
                o.control_strength = min(1.0, o.control_strength + 0.1)
                return GameAction(self.tick, agent.agent_id, "defend",
                                  target_id=o.objective_id)

        self._move_agent(agent)
        return GameAction(self.tick, agent.agent_id, "move")

    def _disruptor_action(self, agent: Agent) -> GameAction:
        """Disruptors weaken enemy-controlled objectives."""
        r = self.rng.random()
        if r < 0.5 and agent.hearts > 0:
            enemy = [o for o in self.objectives
                     if o.owner is not None and o.owner != agent.team]
            if enemy:
                target_o = self.rng.choice(enemy)
                agent.hearts -= 1
                target_o.control_strength = max(0, target_o.control_strength - 0.4)
                if target_o.control_strength <= 0:
                    target_o.owner = None
                return GameAction(self.tick, agent.agent_id, "disrupt",
                                  target_id=target_o.objective_id,
                                  detail="disrupted enemy objective")
        self._move_agent(agent)
        return GameAction(self.tick, agent.agent_id, "move")

    def _scout_action(self, agent: Agent) -> GameAction:
        """Scouts explore and sometimes help nearby allies."""
        r = self.rng.random()
        if r < 0.3:
            allies = [a for a in self.agents
                      if a.team == agent.team and a.agent_id != agent.agent_id
                      and not a.is_defector]
            if allies:
                target = self.rng.choice(allies)
                self._record_help(agent, target)
                return GameAction(self.tick, agent.agent_id, "help_ally",
                                  target_id=target.agent_id, detail="scouted for ally")
        elif r < 0.5:
            return GameAction(self.tick, agent.agent_id, "scout_area",
                              detail="surveying territory")

        self._move_agent(agent)
        return GameAction(self.tick, agent.agent_id, "move")

    def _defector_action(self, agent: Agent) -> GameAction:
        """A defecting agent acts selfishly — hoards, doesn't help."""
        r = self.rng.random()
        if r < 0.6:
            agent.resources += self.rng.randint(1, 2)
            return GameAction(self.tick, agent.agent_id, "gather",
                              detail="hoarding (defector)")
        self._move_agent(agent)
        return GameAction(self.tick, agent.agent_id, "move")

    def _check_defections(self) -> list[dict]:
        """Check if any agent defects this tick."""
        events = []
        for agent in self.agents:
            if agent.is_defector:
                continue
            tick_factor = self.tick / self.max_ticks
            help_factor = 1.0 / (1.0 + sum(agent.help_received.values()))
            p = self.defection_probability * tick_factor * help_factor

            if self.rng.random() < p:
                agent.is_defector = True
                agent.defection_tick = self.tick
                old_coalition = agent.coalition_id
                agent.coalition_id = -1

                connections = len(agent.help_given) + len(agent.help_received)
                severity = min(1.0, connections / 6.0)

                events.append({
                    "tick": self.tick,
                    "agent_id": agent.agent_id,
                    "team": agent.team,
                    "previous_role": agent.role,
                    "previous_coalition": old_coalition,
                    "severity": round(severity, 3),
                    "reason": self._defection_reason(agent),
                })
        return events

    def _defection_reason(self, agent: Agent) -> str:
        """Generate a plausible reason for defection."""
        reasons = [
            "low reciprocity from coalition members",
            "resource imbalance — contributing more than receiving",
            "role instability — switched too often, lost team position",
            "isolated from team — too far from nearest allies",
            "opportunity: saw undefended objective and went solo",
        ]
        return self.rng.choice(reasons)

    def _record_help(self, helper: Agent, recipient: Agent):
        """Track help interactions for reciprocity calculation."""
        helper.help_given[recipient.agent_id] = \
            helper.help_given.get(recipient.agent_id, 0) + 1
        recipient.help_received[helper.agent_id] = \
            recipient.help_received.get(helper.agent_id, 0) + 1

    def _move_agent(self, agent: Agent):
        """Move agent with some directional bias toward objectives."""
        dx = self.rng.uniform(-1.2, 1.2)
        dy = self.rng.uniform(-1.2, 1.2)

        center_x = self.map_width / 2
        center_y = self.map_height / 2
        dx += (center_x - agent.x) * 0.03
        dy += (center_y - agent.y) * 0.03

        agent.x = max(0.5, min(self.map_width - 0.5, agent.x + dx))
        agent.y = max(0.5, min(self.map_height - 0.5, agent.y + dy))

    def _nearest_capturable_objective(self, agent: Agent) -> Optional[Objective]:
        """Find the closest objective not owned by this agent's team."""
        candidates = [o for o in self.objectives if o.owner != agent.team]
        if not candidates:
            return None
        return min(candidates,
                   key=lambda o: math.hypot(o.x - agent.x, o.y - agent.y))

    def _update_objectives(self):
        """Decay undefended objective control over time."""
        for o in self.objectives:
            if o.owner is not None:
                o.control_strength = max(0, o.control_strength - 0.01)
                if o.control_strength <= 0:
                    o.owner = None

    def _compute_scores(self) -> dict:
        """Compute team scores based on objectives controlled."""
        scores = {i: 0.0 for i in range(len(self.teams))}
        for o in self.objectives:
            if o.owner is not None:
                scores[o.owner] += o.control_strength
        return {k: round(v, 2) for k, v in scores.items()}

    def get_agent_relationships(self) -> list[dict]:
        """
        Extract pairwise relationship data for the topology graph.
        Returns edges with weight = mutual help count.
        """
        edges = []
        seen = set()
        for agent in self.agents:
            for target_id, count in agent.help_given.items():
                pair = tuple(sorted([agent.agent_id, target_id]))
                if pair in seen:
                    continue
                seen.add(pair)

                target = self.agents[target_id] if target_id < len(self.agents) else None
                if target is None:
                    continue

                reverse = target.help_given.get(agent.agent_id, 0)
                total = count + reverse
                reciprocity = min(count, reverse) / max(count, reverse) if max(count, reverse) > 0 else 0

                edges.append({
                    "source": pair[0],
                    "target": pair[1],
                    "weight": total,
                    "reciprocity": round(reciprocity, 3),
                    "same_team": agent.team == target.team,
                })
        return edges

    @property
    def is_finished(self) -> bool:
        return self.tick >= self.max_ticks
