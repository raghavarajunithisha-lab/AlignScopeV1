"""
AlignScope Example — Competitive Predator-Prey Arena

Two teams on a continuous 2D field: 3 predators chase 2 prey.
Predators are rewarded for catching, prey are rewarded for escaping.
Demonstrates zero-sum competitive MARL integration.

Dependencies: None (self-contained environment)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Predator-Prey Arena
# ──────────────────────────────────────────────────

class PredatorPreyArena:
    """Continuous 2D arena: predators chase prey, prey evade."""

    ARENA_SIZE = 400.0
    PREDATOR_SPEED = 3.0
    PREY_SPEED = 4.0  # Prey slightly faster
    CATCH_RADIUS = 15.0

    def __init__(self, n_predators=3, n_prey=2):
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.reset()

    def reset(self):
        self.predators = {
            f"predator_{i}": {"x": random.uniform(0, 100), "y": random.uniform(0, self.ARENA_SIZE), "catches": 0}
            for i in range(self.n_predators)
        }
        self.prey = {
            f"prey_{i}": {
                "x": random.uniform(300, self.ARENA_SIZE), "y": random.uniform(0, self.ARENA_SIZE),
                "alive": True, "escapes": 0,
            }
            for i in range(self.n_prey)
        }
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "predators": {pid: (p["x"], p["y"]) for pid, p in self.predators.items()},
            "prey": {pid: (p["x"], p["y"]) for pid, p in self.prey.items() if p["alive"]},
        }

    def step(self, pred_actions, prey_actions):
        """pred_actions: {id: (dx, dy)}, prey_actions: {id: (dx, dy)}"""
        rewards = {}
        events = []
        self.step_count += 1

        # Move predators
        for pid, (dx, dy) in pred_actions.items():
            p = self.predators[pid]
            mag = math.sqrt(dx*dx + dy*dy) or 1
            p["x"] = max(0, min(self.ARENA_SIZE, p["x"] + dx / mag * self.PREDATOR_SPEED))
            p["y"] = max(0, min(self.ARENA_SIZE, p["y"] + dy / mag * self.PREDATOR_SPEED))
            rewards[pid] = -0.01  # Small penalty for not catching

        # Move prey
        for pid, (dx, dy) in prey_actions.items():
            p = self.prey[pid]
            if not p["alive"]:
                continue
            mag = math.sqrt(dx*dx + dy*dy) or 1
            p["x"] = max(0, min(self.ARENA_SIZE, p["x"] + dx / mag * self.PREY_SPEED))
            p["y"] = max(0, min(self.ARENA_SIZE, p["y"] + dy / mag * self.PREY_SPEED))
            rewards[pid] = 0.01  # Small reward for surviving
            p["escapes"] += 1

        # Check catches
        for pred_id, pred in self.predators.items():
            for prey_id, prey in self.prey.items():
                if not prey["alive"]:
                    continue
                dist = math.sqrt((pred["x"] - prey["x"])**2 + (pred["y"] - prey["y"])**2)
                if dist < self.CATCH_RADIUS:
                    prey["alive"] = False
                    pred["catches"] += 1
                    rewards[pred_id] = 1.0  # Predator rewarded
                    rewards[prey_id] = -1.0  # Prey penalized
                    events.append({
                        "agent_id": prey_id, "team": 1, "previous_role": "prey",
                        "severity": 0.9, "reason": f"{prey_id} caught by {pred_id}",
                    })

        # Respawn dead prey after 20 steps
        for pid, p in self.prey.items():
            if not p["alive"] and self.step_count % 20 == 0:
                p["alive"] = True
                p["x"] = random.uniform(250, self.ARENA_SIZE)
                p["y"] = random.uniform(0, self.ARENA_SIZE)

        done = self.step_count >= 300
        return self._get_obs(), rewards, events, done


# ──────────────────────────────────────────────────
# Policies
# ──────────────────────────────────────────────────

def predator_policy(obs, agent_id):
    """Chase nearest prey."""
    px, py = obs["predators"][agent_id]
    if not obs["prey"]:
        return (random.uniform(-1, 1), random.uniform(-1, 1))
    nearest = min(obs["prey"].values(), key=lambda p: (p[0]-px)**2 + (p[1]-py)**2)
    return (nearest[0] - px, nearest[1] - py)


def prey_policy(obs, agent_id):
    """Flee from nearest predator."""
    px, py = obs["prey"][agent_id]
    if not obs["predators"]:
        return (0, 0)
    nearest = min(obs["predators"].values(), key=lambda p: (p[0]-px)**2 + (p[1]-py)**2)
    return (px - nearest[0], py - nearest[1])  # Move away


# ──────────────────────────────────────────────────
# Training Loop with AlignScope
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("⚔️  Predator-Prey Arena — Competitive MARL")
    print("    3 predators vs 2 prey on a continuous field")
    print("=" * 55)

    tracker = alignscope.init(project="predator-prey-arena", preset="zero-sum")
    env = PredatorPreyArena(n_predators=3, n_prey=2)

    for ep in range(2):
        obs = env.reset()
        tracker.reset()
        done = False
        total_catches = 0
        print(f"\n🏟️ Episode {ep + 1}")

        while not done:
            # Compute actions
            pred_actions = {pid: predator_policy(obs, pid) for pid in env.predators}
            prey_actions = {pid: prey_policy(obs, pid) for pid in env.prey if env.prey[pid]["alive"]}

            obs, rewards, events, done = env.step(pred_actions, prey_actions)
            total_catches += len(events)

            # Build agent list for AlignScope
            agent_list = []
            for pid, p in env.predators.items():
                agent_list.append({
                    "agent_id": pid, "team": 0, "role": "predator",
                    "x": p["x"], "y": p["y"],
                    "resources": p["catches"], "hearts": 3,
                    "energy": rewards.get(pid, 0), "is_defector": False, "coalition_id": 0,
                })
            for pid, p in env.prey.items():
                agent_list.append({
                    "agent_id": pid, "team": 1, "role": "prey",
                    "x": p["x"], "y": p["y"],
                    "resources": p["escapes"], "hearts": 1 if p["alive"] else 0,
                    "energy": rewards.get(pid, 0), "is_defector": not p["alive"], "coalition_id": 1,
                })

            tracker.log(
                step=env.step_count + ep * 300,
                agents=agent_list,
                rewards=rewards,
                defection_events=events,
            )
            time.sleep(0.04)

        print(f"   Catches: {total_catches}")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see predator-prey dynamics.")


if __name__ == "__main__":
    main()
