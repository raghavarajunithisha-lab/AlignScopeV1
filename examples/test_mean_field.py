"""
AlignScope Example — Mean-Field: Large-Scale Battle Simulation

Two armies of 32 agents each battle on a grid.
Each agent's policy depends on the mean action of nearby agents
(mean-field approximation), not individual agent states.

Demonstrates: Mean-Field MARL with population-level dynamics
Dependencies: None (self-contained environment)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Mean-Field Battle Grid
# ──────────────────────────────────────────────────

class MeanFieldBattle:
    """Grid-based battle game inspired by mlii/mfrl Battle scenario."""

    GRID_SIZE = 20
    ACTIONS = ["attack", "defend", "advance", "retreat", "hold"]

    def __init__(self, army_size=32):
        self.army_size = army_size
        self.reset()

    def reset(self):
        self.agents = {}
        # Army 0: left side
        for i in range(self.army_size):
            row = i % 8
            col = i // 8
            self.agents[f"red_{i}"] = {
                "team": 0, "x": col + 1, "y": row + 6,
                "hp": 10, "attack_power": random.uniform(1, 3),
                "kills": 0,
            }
        # Army 1: right side
        for i in range(self.army_size):
            row = i % 8
            col = i // 8
            self.agents[f"blue_{i}"] = {
                "team": 1, "x": self.GRID_SIZE - col - 2, "y": row + 6,
                "hp": 10, "attack_power": random.uniform(1, 3),
                "kills": 0,
            }
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return dict(self.agents)

    def _get_mean_action(self, team: int, actions: dict) -> dict:
        """Compute mean action distribution for a team (mean-field approximation)."""
        counts = {a: 0 for a in self.ACTIONS}
        total = 0
        for aid, action in actions.items():
            if self.agents[aid]["team"] == team and self.agents[aid]["hp"] > 0:
                counts[action] += 1
                total += 1
        if total == 0:
            return {a: 1.0 / len(self.ACTIONS) for a in self.ACTIONS}
        return {a: c / total for a, c in counts.items()}

    def step(self, actions: dict):
        self.step_count += 1
        rewards = {}
        events = []

        # Compute mean-field for each team
        mf = {0: self._get_mean_action(0, actions), 1: self._get_mean_action(1, actions)}

        for aid, action in actions.items():
            a = self.agents[aid]
            if a["hp"] <= 0:
                rewards[aid] = -0.5
                continue

            team = a["team"]
            enemy_team = 1 - team
            my_mf = mf[team]
            enemy_mf = mf[enemy_team]

            # Action effects depend on mean-field
            if action == "advance":
                direction = 1 if team == 0 else -1
                a["x"] = max(0, min(self.GRID_SIZE - 1, a["x"] + direction))
                rewards[aid] = 0.05

            elif action == "retreat":
                direction = -1 if team == 0 else 1
                a["x"] = max(0, min(self.GRID_SIZE - 1, a["x"] + direction))
                rewards[aid] = -0.02

            elif action == "attack":
                # Damage effectiveness scales with team coordination (mean-field)
                coordination = my_mf.get("attack", 0) + my_mf.get("advance", 0)
                damage = a["attack_power"] * (0.5 + coordination)
                # Find nearest enemy
                enemies = [(eid, e) for eid, e in self.agents.items()
                          if e["team"] == enemy_team and e["hp"] > 0]
                if enemies:
                    nearest = min(enemies, key=lambda e: abs(e[1]["x"]-a["x"]) + abs(e[1]["y"]-a["y"]))
                    dist = abs(nearest[1]["x"] - a["x"]) + abs(nearest[1]["y"] - a["y"])
                    if dist <= 3:
                        nearest[1]["hp"] -= damage
                        rewards[aid] = 0.3
                        if nearest[1]["hp"] <= 0:
                            a["kills"] += 1
                            rewards[aid] = 1.0
                            events.append({
                                "agent_id": nearest[0], "team": enemy_team,
                                "previous_role": "soldier", "severity": 0.8,
                                "reason": f"{nearest[0]} eliminated by {aid}",
                            })
                    else:
                        rewards[aid] = -0.01
                else:
                    rewards[aid] = 0.0

            elif action == "defend":
                # Defending reduces incoming damage — reflected in mean field
                rewards[aid] = 0.02

            else:  # hold
                rewards[aid] = 0.0

        done = self.step_count >= 150
        # Also end if one army is eliminated
        alive = {0: 0, 1: 0}
        for a in self.agents.values():
            if a["hp"] > 0:
                alive[a["team"]] += 1
        if alive[0] == 0 or alive[1] == 0:
            done = True

        return self._get_obs(), rewards, events, done, alive


# ──────────────────────────────────────────────────
# Mean-Field Policy: action depends on team mean action
# ──────────────────────────────────────────────────

def mean_field_policy(agent, team_mf, enemy_mf, step):
    """Agent picks action influenced by what teammates are doing (mean field)."""
    if agent["hp"] <= 0:
        return "hold"

    # If team is mostly attacking, join the attack (herd behavior)
    if team_mf.get("attack", 0) > 0.4:
        return random.choices(["attack", "advance"], weights=[0.7, 0.3])[0]

    # If team is mostly defending, hold or defend
    if team_mf.get("defend", 0) > 0.4:
        return random.choices(["defend", "hold"], weights=[0.6, 0.4])[0]

    # Early game: advance
    if step < 30:
        return random.choices(["advance", "attack"], weights=[0.6, 0.4])[0]

    # Default: mixed strategy
    return random.choices(
        MeanFieldBattle.ACTIONS,
        weights=[0.35, 0.15, 0.25, 0.05, 0.20]
    )[0]


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("🏟️  Mean-Field Battle — 64 Agents, 2 Armies")
    print("    Each agent acts based on team mean-field statistics")
    print("=" * 55)

    tracker = alignscope.init(project="mean-field-battle", preset="mean-field")
    env = MeanFieldBattle(army_size=32)

    for ep in range(2):
        obs = env.reset()
        tracker.reset()
        done = False
        prev_mf = {0: {a: 0.2 for a in env.ACTIONS}, 1: {a: 0.2 for a in env.ACTIONS}}
        print(f"\n⚔️ Battle {ep + 1}")

        while not done:
            # Each agent picks action based on mean-field
            actions = {}
            for aid, a in env.agents.items():
                team = a["team"]
                actions[aid] = mean_field_policy(a, prev_mf[team], prev_mf[1-team], env.step_count)

            obs, rewards, events, done, alive = env.step(actions)

            # Update mean-field for next step
            for team in [0, 1]:
                prev_mf[team] = env._get_mean_action(team, actions)

            # Log to AlignScope
            scale = 20
            tracker.log(
                step=env.step_count + ep * 150,
                agents=[
                    {
                        "agent_id": aid,
                        "team": a["team"],
                        "role": "soldier",
                        "x": float(a["x"] * scale),
                        "y": float(a["y"] * scale),
                        "resources": a["kills"],
                        "hearts": max(0, a["hp"]),
                        "energy": rewards.get(aid, 0),
                        "is_defector": a["hp"] <= 0,
                        "coalition_id": a["team"] if a["hp"] > 0 else -1,
                    }
                    for aid, a in env.agents.items()
                ],
                actions=actions,
                rewards=rewards,
                defection_events=events,
            )
            time.sleep(0.06)

        print(f"   Survivors — Red: {alive[0]}/32, Blue: {alive[1]}/32")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see army dynamics.")


if __name__ == "__main__":
    main()
