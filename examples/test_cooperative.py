"""
AlignScope Example — Cooperative Multi-Agent Gridworld

A team of 4 agents on a 10×10 grid must collect resources and bring
them to a shared depot. Demonstrates how a researcher building a
custom cooperative environment would integrate AlignScope.

Dependencies: None (self-contained environment)
"""
import time
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Resource Gathering Gridworld
# ──────────────────────────────────────────────────

class ResourceGrid:
    """10×10 grid where agents gather resources to a central depot."""

    def __init__(self, n_agents=4, grid_size=10, n_resources=8):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.depot = (grid_size // 2, grid_size // 2)
        self.reset()

    def reset(self):
        self.agents = {
            f"agent_{i}": {
                "x": random.randint(0, self.grid_size - 1),
                "y": random.randint(0, self.grid_size - 1),
                "carrying": False,
                "delivered": 0,
            }
            for i in range(self.n_agents)
        }
        self.resources = [
            (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            for _ in range(self.n_resources)
        ]
        self.team_score = 0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            aid: {"pos": (a["x"], a["y"]), "carrying": a["carrying"], "resources": self.resources[:]}
            for aid, a in self.agents.items()
        }

    def step(self, actions: dict):
        """actions: {agent_id: 'up'|'down'|'left'|'right'|'pickup'|'drop'}"""
        rewards = {}
        self.step_count += 1

        for aid, action in actions.items():
            agent = self.agents[aid]
            # Movement
            if action == "up" and agent["y"] > 0:
                agent["y"] -= 1
            elif action == "down" and agent["y"] < self.grid_size - 1:
                agent["y"] += 1
            elif action == "left" and agent["x"] > 0:
                agent["x"] -= 1
            elif action == "right" and agent["x"] < self.grid_size - 1:
                agent["x"] += 1
            elif action == "pickup" and not agent["carrying"]:
                pos = (agent["x"], agent["y"])
                if pos in self.resources:
                    self.resources.remove(pos)
                    agent["carrying"] = True
            elif action == "drop" and agent["carrying"]:
                if (agent["x"], agent["y"]) == self.depot:
                    agent["carrying"] = False
                    agent["delivered"] += 1
                    self.team_score += 1

            # Reward: +1 for delivery, small penalty for each step
            rewards[aid] = -0.01
            if action == "drop" and (agent["x"], agent["y"]) == self.depot and not agent["carrying"]:
                rewards[aid] = 1.0

        # Respawn resources
        while len(self.resources) < self.n_resources:
            self.resources.append(
                (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            )

        done = self.step_count >= 200
        return self._get_obs(), rewards, done


# ──────────────────────────────────────────────────
# Simple Policy: move toward nearest resource or depot
# ──────────────────────────────────────────────────

def greedy_policy(obs, agent_id):
    """Simple hand-coded policy — move toward resource if empty, toward depot if carrying."""
    pos = obs[agent_id]["pos"]
    carrying = obs[agent_id]["carrying"]

    if carrying:
        target = (5, 5)  # depot
        if pos == target:
            return "drop"
    else:
        resources = obs[agent_id]["resources"]
        if not resources:
            return random.choice(["up", "down", "left", "right"])
        # Find nearest resource
        target = min(resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))
        if pos == target:
            return "pickup"

    # Move toward target
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


# ──────────────────────────────────────────────────
# Training Loop with AlignScope Integration
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("🏗️  Cooperative Gridworld — Resource Gathering")
    print("    4 agents collect resources → deliver to depot")
    print("=" * 55)

    # 1. Initialize AlignScope
    tracker = alignscope.init(project="cooperative-gridworld")

    # 2. Create environment
    env = ResourceGrid(n_agents=4, grid_size=10, n_resources=8)

    n_episodes = 3
    for ep in range(n_episodes):
        obs = env.reset()
        tracker.reset()  # Isolate metrics between episodes
        done = False
        ep_reward = 0
        print(f"\n📦 Episode {ep + 1}/{n_episodes}")

        while not done:
            # Each agent picks an action using the policy
            actions = {aid: greedy_policy(obs, aid) for aid in env.agents}

            # Step the environment
            obs, rewards, done = env.step(actions)
            ep_reward += sum(rewards.values())

            # 3. Log to AlignScope — this is the integration point
            scale = 40  # Scale grid coords for dashboard visibility
            tracker.log(
                step=env.step_count + ep * 200,
                agents=[
                    {
                        "agent_id": aid,
                        "team": 0,
                        "role": "carrier" if a["carrying"] else "gatherer",
                        "x": float(a["x"] * scale),
                        "y": float(a["y"] * scale),
                        "resources": a["delivered"],
                        "hearts": 3,
                        "energy": rewards[aid],
                        "is_defector": False,
                        "coalition_id": 0,
                    }
                    for aid, a in env.agents.items()
                ],
                actions=actions,
                rewards=rewards,
            )
            time.sleep(0.05)  # Slow down for dashboard visualization

        print(f"   Score: {env.team_score} deliveries | Total reward: {ep_reward:.2f}")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see agent cooperation.")


if __name__ == "__main__":
    main()
