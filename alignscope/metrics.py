from __future__ import annotations

"""
AlignScope — Alignment Signal Extractor

Computes three core alignment metrics per agent-pair per tick:

1. Role Stability Score
   Measures how consistently an agent maintains its specialized role.
   Low entropy in role-switch history = strong specialization = alignment signal.
   Applicable to any MARL environment where agents have assignable or emergent roles.

2. Reciprocity Index
   Measures mutual help between agent pairs. When Agent A helps Agent B
   and B reciprocates, this is the seed of coalition formation — individuals
   learning to form cooperative wholes.

3. Goal Convergence Delta
   Measures whether two agents' action trajectories are converging or diverging.
   Uses cosine similarity of recent action-type frequency vectors.
   Convergence = agents learning to complement each other.

These metrics are environment-agnostic and work with any MARL data source
that provides agent roles, actions, and help/interaction events.
"""

import math
from collections import Counter, deque
from typing import Any


class AlignmentMetrics:
    """Stateful metric computer that accumulates agent data across ticks."""
    
    role_histories: dict[int, deque[str]]
    help_matrix: dict[tuple, int]
    action_histories: dict[int, list[str]]
    tick_metrics: list[dict]

    def __init__(self):
        self._init_state()

    def _init_state(self):
        self.role_histories = {}
        self.help_matrix = {}
        self.action_histories = {}
        self.tick_metrics = []
        
    def reset(self):
        """Reset internal metrics state (e.g., between episodes)."""
        self._init_state()

    def update(self, tick_data: dict) -> dict:
        """
        Process one tick of game data and return computed metrics.

        Args:
            tick_data: Output from MARLSimulator.step() or any compatible data source.

        Returns:
            Dictionary of alignment metrics for this tick.
        """
        tick = tick_data["tick"]
        agents = tick_data["agents"]
        actions = tick_data["actions"]

        # Update role histories
        for agent in agents:
            aid = agent["agent_id"]
            role = agent["role"]
            if aid not in self.role_histories:
                self.role_histories[aid] = deque(maxlen=100)
            self.role_histories[aid].append(role)

        # Update action histories and infer reciprocity based on shared fate or spatial proximity
        interaction_groups = {}
        for agent in agents:
            team = agent["team"]
            coal = agent.get("coalition_id", -1)
            
            # Group by explicit coalition or strictly positive/negative shared reward 
            # (indicating shared fate like both taking damage or both succeeding)
            energy = round(agent.get("energy", 0), 1)
            group_key = f"t{team}_c{coal}_e{energy}" if energy != 0 else f"t{team}_c{coal}"
            
            if group_key not in interaction_groups:
                interaction_groups[group_key] = []
            interaction_groups[group_key].append(agent)
                
        # If multiple agents share fate or are physically very close, infer reciprocity
        for group, a_list in interaction_groups.items():
            if len(a_list) < 2:
                continue
            for i, a1 in enumerate(a_list):
                for a2 in a_list[i+1:]:
                    # Check spatial proximity if coordinates exist
                    dist = 1000.0
                    if "x" in a1 and "y" in a1 and "x" in a2 and "y" in a2:
                        dist = math.sqrt((a1["x"] - a2["x"])**2 + (a1["y"] - a2["y"])**2)
                    
                    # Add to help matrix if they share non-zero reward, or are very close
                    # or explicitly share a coalition.
                    energy_val = a1.get("energy", 0)
                    if energy_val != 0 or dist < 25.0 or a1.get("coalition_id", -1) != -1:
                        pair = (a1["agent_id"], a2["agent_id"])
                        rev_pair = (a2["agent_id"], a1["agent_id"])
                        weight = 1 if energy_val > 0 else 0.5 # weaker signal for shared negative
                        self.help_matrix[pair] = self.help_matrix.get(pair, 0) + weight
                        self.help_matrix[rev_pair] = self.help_matrix.get(rev_pair, 0) + weight

        for action in actions:
            aid = action["agent_id"]
            atype = action["action"]
            if aid not in self.action_histories:
                self.action_histories[aid] = []
            self.action_histories[aid].append(atype)

            # Support explicit 'help_ally' if the environment explicitly provides it
            if atype == "help_ally" and action.get("target_id") is not None:
                pair = (aid, action["target_id"])
                self.help_matrix[pair] = self.help_matrix.get(pair, 0) + 2 # Stronger explicit signal

        # Compute per-agent metrics
        agent_metrics = {}
        for agent in agents:
            aid = agent["agent_id"]
            agent_metrics[aid] = {
                "role_stability": self._role_stability(aid),
                "team": agent["team"],
                "role": agent["role"],
                "is_defector": agent["is_defector"],
                "coalition_id": agent["coalition_id"],
            }

        # Compute pairwise metrics
        pair_metrics = self._compute_pair_metrics(agents)

        # Compute team-level aggregates
        team_metrics = self._compute_team_metrics(agent_metrics, pair_metrics, agents)

        result = {
            "tick": tick,
            "agent_metrics": agent_metrics,
            "pair_metrics": pair_metrics,
            "team_metrics": team_metrics,
            "overall_alignment_score": self._overall_alignment(team_metrics),
        }

        self.tick_metrics.append(result)
        return result

    def _role_stability(self, agent_id: int) -> float:
        """
        Compute role stability using normalized entropy.

        Returns a value in [0, 1] where:
          1.0 = agent has never switched roles (perfect specialization)
          0.0 = agent switches roles uniformly at random (no specialization)
        """
        history = self.role_histories.get(agent_id, [])
        if len(history) <= 1:
            return 1.0

        counts = Counter(history)
        total = len(history)
        num_roles = len(counts)

        if num_roles == 1:
            return 1.0

        # Shannon entropy
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values()
        )

        # Normalize by max possible entropy
        max_entropy = math.log2(num_roles)
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        # Invert: low entropy = high stability
        val = 1.0 - normalized
        return round(float(val), 4)

    def _compute_pair_metrics(self, agents: list[dict]) -> list[dict]:
        """Compute reciprocity and goal convergence for agent pairs."""
        pairs = []
        agent_ids = [a["agent_id"] for a in agents]

        for i, aid_a in enumerate(agent_ids):
            for aid_b in agent_ids[i + 1:]:
                reciprocity = self._reciprocity_index(aid_a, aid_b)
                convergence = self._goal_convergence(aid_a, aid_b)

                # Only include pairs with some interaction
                if reciprocity > 0 or convergence > 0.1:
                    # Determine if same team
                    team_a = next(a["team"] for a in agents if a["agent_id"] == aid_a)
                    team_b = next(a["team"] for a in agents if a["agent_id"] == aid_b)

                    pairs.append({
                        "agent_a": aid_a,
                        "agent_b": aid_b,
                        "reciprocity": reciprocity,
                        "goal_convergence": convergence,
                        "same_team": team_a == team_b,
                        "relationship_strength": round(
                            (reciprocity * 0.6 + convergence * 0.4), 4
                        ),
                    })

        return pairs

    def _reciprocity_index(self, agent_a: int, agent_b: int) -> float:
        """
        Compute mutual help ratio between two agents.

        Returns value in [0, 1]:
          1.0 = perfectly reciprocal (equal help given/received)
          0.0 = no interaction or completely one-sided
        """
        a_to_b = self.help_matrix.get((agent_a, agent_b), 0)
        b_to_a = self.help_matrix.get((agent_b, agent_a), 0)

        total = a_to_b + b_to_a
        if total == 0:
            return 0.0

        # Reciprocity = 1 - |imbalance| / total
        imbalance = abs(a_to_b - b_to_a)
        val = 1.0 - (imbalance / total)
        return round(float(val), 4)

    def _goal_convergence(self, agent_a: int, agent_b: int) -> float:
        """
        Compute cosine similarity of recent action-type frequency vectors.

        Uses last 50 actions to focus on recent behavior.
        High similarity = agents pursuing complementary/aligned goals.
        """
        window = 50
        ha = self.action_histories.get(agent_a, [])
        hb = self.action_histories.get(agent_b, [])
        hist_a = ha[-window:] if ha else []
        hist_b = hb[-window:] if hb else []

        if not hist_a or not hist_b:
            return 0.0

        # Build frequency vectors over all action types
        all_actions = set(hist_a) | set(hist_b)
        vec_a = Counter(hist_a)
        vec_b = Counter(hist_b)

        # Cosine similarity
        dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_actions)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        val = dot / (mag_a * mag_b)
        return round(float(val), 4)

    def _compute_team_metrics(
        self, agent_metrics: dict, pair_metrics: list[dict], agents: list[dict]
    ) -> dict:
        """Aggregate metrics per team including stability, convergence, and reciprocity."""
        teams: dict[int, dict[str, Any]] = {}
        for agent in agents:
            tid = int(agent["team"])
            if tid not in teams:
                teams[tid] = {
                    "avg_role_stability": 0.0,
                    "avg_convergence": 0.0,
                    "avg_reciprocity": 0.0,
                    "agent_count": 0,
                    "pair_count": 0,
                    "defector_count": 0,
                    "active_coalitions": set(),
                }
            aid = agent["agent_id"]
            am = agent_metrics[aid]
            teams[tid]["avg_role_stability"] += am["role_stability"]
            teams[tid]["agent_count"] += 1
            if am["is_defector"]:
                teams[tid]["defector_count"] += 1
            if am["coalition_id"] is not None and am["coalition_id"] >= 0:
                teams[tid]["active_coalitions"].add(am["coalition_id"])
                
        # Distribute pair metrics to teams
        for p in pair_metrics:
            if p["same_team"]:
                agent_a = next(a for a in agents if a["agent_id"] == p["agent_a"])
                tid = agent_a["team"]
                teams[tid]["avg_convergence"] += p["goal_convergence"]
                teams[tid]["avg_reciprocity"] += p["reciprocity"]
                teams[tid]["pair_count"] += 1

        # Initialize trackers for new metrics
        out_teams = {}
        total_global_energy = sum(agent.get("energy", 0) for agent in agents)
        total_active_agents = sum(1 for agent in agents if not agent.get("is_defector", False))

        for tid, tdata in teams.items():
            n = float(tdata["agent_count"])
            pcount = float(tdata["pair_count"])
            team_energy = sum(a.get("energy", 0) for a in agents if a["team"] == tid)
            team_active = sum(1 for a in agents if a["team"] == tid and not a.get("is_defector", False))
            
            # Action distribution for field entropy
            team_actions = [a.get("action", "none") for a in agents if a["team"] == tid]
            action_counts = Counter(team_actions)
            field_entropy = 0.0
            if len(team_actions) > 0:
                field_entropy = -sum((c/len(team_actions)) * math.log2(c/len(team_actions)) for c in action_counts.values())

            # Pseudo-metrics for Zero-Sum and Mean-Field
            mean_reward = team_energy / n if n > 0 else 0
            population_distribution = team_active / total_active_agents if total_active_agents > 0 else 0
            
            # Win rate proxy: logistic function over team's recent energy share
            win_rate_proxy = 0.5
            if total_global_energy != 0:
                share = team_energy / abs(total_global_energy)
                win_rate_proxy = 1.0 / (1.0 + math.exp(-share * 5)) # compress to 0-1
                
            # Exploitability proxy: 1.0 minus role stability (highly unstable teams are exploitable)
            avg_stability = float(tdata["avg_role_stability"] / n if n > 0 else 0)
            exploitability = max(0.0, 1.0 - avg_stability)
            
            # Nash gap proxy: difference between current field entropy and max possible
            max_ent = math.log2(len(action_counts)) if len(action_counts) > 0 else 1.0
            nash_gap = max(0.0, max_ent - field_entropy)

            out_teams[tid] = {
                "avg_role_stability": round(avg_stability, 4),
                "avg_convergence": round(float(tdata["avg_convergence"] / pcount if pcount > 0 else 0), 4),
                "avg_reciprocity": round(float(tdata["avg_reciprocity"] / pcount if pcount > 0 else 0), 4),
                "agent_count": int(tdata["agent_count"]),
                "defector_count": int(tdata["defector_count"]),
                "active_coalitions": len(tdata["active_coalitions"]),
                
                # Zero-Sum & Mean-Field Preset Support
                "mean_reward": round(mean_reward, 3),
                "population_distribution": round(population_distribution, 3),
                "field_entropy": round(field_entropy, 3),
                "win_rate": round(win_rate_proxy, 3),
                "exploitability": round(exploitability, 3),
                "nash_gap": round(nash_gap, 3)
            }

        return out_teams

    def _overall_alignment(self, team_metrics: dict) -> float:
        """
        True overall alignment score: mathematical aggregation of the
        three core pillars (stability, convergence, reciprocity) penalized by defections.
        """
        if not team_metrics:
            return 0.0

        total_stability = sum(t["avg_role_stability"] for t in team_metrics.values())
        total_convergence = sum(t["avg_convergence"] for t in team_metrics.values())
        total_reciprocity = sum(t["avg_reciprocity"] for t in team_metrics.values())
        
        total_defectors = sum(t["defector_count"] for t in team_metrics.values())
        total_agents = sum(t["agent_count"] for t in team_metrics.values())
        
        num_teams = len(team_metrics)
        avg_stability = total_stability / num_teams
        avg_convergence = total_convergence / num_teams
        avg_reciprocity = total_reciprocity / num_teams
        
        # Base alignment is a blend of the three pillars
        base_alignment = (avg_stability * 0.4) + (avg_convergence * 0.3) + (avg_reciprocity * 0.3)

        defection_penalty = total_defectors / total_agents if total_agents > 0 else 0

        # Floor at 0.0, ceiling at 1.0
        val = base_alignment * (1 - defection_penalty * 0.8)
        return float(max(0.0, min(1.0, round(float(val), 4))))
