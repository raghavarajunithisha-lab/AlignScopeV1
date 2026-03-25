"""
AlignScope Example — EPyMARL: QMIX on Multi-Robot Warehouse

Simulates a warehouse with 4 robots that must pick items and deliver
them to loading zones. Uses QMIX-style value decomposition:
  Q_total = mix(Q_1, Q_2, ..., Q_n)

Demonstrates: How an EPyMARL researcher would integrate AlignScope
Dependencies: None (self-contained environment)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Multi-Robot Warehouse (RWARE-inspired)
# ──────────────────────────────────────────────────

class Warehouse:
    """Grid warehouse: robots pick shelves and deliver to loading zone."""

    SIZE = 12

    def __init__(self, n_robots=4, n_shelves=6):
        self.n_robots = n_robots
        self.n_shelves = n_shelves
        self.loading_zone = (self.SIZE - 1, self.SIZE // 2)
        self.reset()

    def reset(self):
        self.robots = {
            f"robot_{i}": {
                "x": random.randint(0, 3), "y": random.randint(0, self.SIZE - 1),
                "carrying": None, "deliveries": 0,
            }
            for i in range(self.n_robots)
        }
        self.shelves = {
            f"shelf_{j}": {
                "x": random.randint(4, self.SIZE - 3),
                "y": random.randint(1, self.SIZE - 2),
                "picked": False,
            }
            for j in range(self.n_shelves)
        }
        self.step_count = 0
        self.total_deliveries = 0
        return self._obs()

    def _obs(self):
        return {
            rid: {
                "pos": (r["x"], r["y"]),
                "carrying": r["carrying"],
                "shelves": [(sid, s["x"], s["y"]) for sid, s in self.shelves.items() if not s["picked"]],
                "loading_zone": self.loading_zone,
            }
            for rid, r in self.robots.items()
        }

    def step(self, actions):
        """actions: {robot_id: 'up'|'down'|'left'|'right'|'pick'|'deliver'}"""
        self.step_count += 1
        rewards = {}
        events = []

        for rid, action in actions.items():
            r = self.robots[rid]
            rewards[rid] = -0.01  # Step cost

            if action == "up" and r["y"] > 0:
                r["y"] -= 1
            elif action == "down" and r["y"] < self.SIZE - 1:
                r["y"] += 1
            elif action == "left" and r["x"] > 0:
                r["x"] -= 1
            elif action == "right" and r["x"] < self.SIZE - 1:
                r["x"] += 1
            elif action == "pick" and r["carrying"] is None:
                for sid, s in self.shelves.items():
                    if not s["picked"] and s["x"] == r["x"] and s["y"] == r["y"]:
                        s["picked"] = True
                        r["carrying"] = sid
                        rewards[rid] = 0.2
                        break
            elif action == "deliver" and r["carrying"] is not None:
                if (r["x"], r["y"]) == self.loading_zone:
                    r["carrying"] = None
                    r["deliveries"] += 1
                    self.total_deliveries += 1
                    rewards[rid] = 1.0

            # Collision penalty
            for other_id, other in self.robots.items():
                if other_id != rid and other["x"] == r["x"] and other["y"] == r["y"]:
                    rewards[rid] -= 0.1

        # Respawn delivered shelves
        for sid, s in self.shelves.items():
            if s["picked"] and not any(r["carrying"] == sid for r in self.robots.values()):
                s["x"] = random.randint(4, self.SIZE - 3)
                s["y"] = random.randint(1, self.SIZE - 2)
                s["picked"] = False

        done = self.step_count >= 200
        return self._obs(), rewards, events, done


# ──────────────────────────────────────────────────
# QMIX-style Training: Individual Q + Monotonic Mixing
# ──────────────────────────────────────────────────

class SimpleQTable:
    """Epsilon-greedy Q-table for a single robot."""

    ACTIONS = ["up", "down", "left", "right", "pick", "deliver"]

    def __init__(self, epsilon=0.3, alpha=0.1, gamma=0.95):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def _state_key(self, obs):
        return (obs["pos"], obs["carrying"] is not None,
                len(obs["shelves"]) > 0,
                obs["pos"] == obs["loading_zone"])

    def act(self, obs):
        state = self._state_key(obs)
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.ACTIONS}
        return max(self.q[state], key=self.q[state].get)

    def update(self, obs, action, reward, next_obs):
        state = self._state_key(obs)
        next_state = self._state_key(next_obs)
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.ACTIONS}
        if next_state not in self.q:
            self.q[next_state] = {a: 0.0 for a in self.ACTIONS}

        max_next = max(self.q[next_state].values())
        self.q[state][action] += self.alpha * (
            reward + self.gamma * max_next - self.q[state][action]
        )


def qmix_total(individual_qs: list) -> float:
    """Monotonic mixing (simplified QMIX): Q_total = sum(abs_weights * Q_i)."""
    # In real QMIX, weights come from a hypernetwork; here we use fixed positive weights.
    weights = [0.3, 0.25, 0.25, 0.2]
    return sum(w * q for w, q in zip(weights, individual_qs))


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("🏭 Multi-Robot Warehouse — EPyMARL QMIX")
    print("    4 robots pick shelves → deliver to loading zone")
    print("    Training: Q-table per robot + monotonic mixing")
    print("=" * 55)

    tracker = alignscope.init(project="warehouse-qmix")
    env = Warehouse(n_robots=4, n_shelves=6)

    # One Q-table per robot (individual value functions)
    q_tables = {f"robot_{i}": SimpleQTable(epsilon=0.3) for i in range(4)}

    for ep in range(5):
        obs = env.reset()
        tracker.reset()
        done = False
        ep_reward = 0
        print(f"\n📦 Episode {ep + 1}/5  (ε={q_tables['robot_0'].epsilon:.2f})")

        prev_obs = dict(obs)

        while not done:
            # Each robot selects action from its Q-table
            actions = {rid: q_tables[rid].act(obs[rid]) for rid in env.robots}

            next_obs, rewards, events, done = env.step(actions)
            ep_reward += sum(rewards.values())

            # Update individual Q-tables
            for rid in env.robots:
                q_tables[rid].update(prev_obs[rid], actions[rid], rewards[rid], next_obs[rid])

            prev_obs = dict(next_obs)
            obs = next_obs

            # Log to AlignScope
            scale = 35
            tracker.log(
                step=env.step_count + ep * 200,
                agents=[
                    {
                        "agent_id": rid,
                        "team": 0,
                        "role": "carrier" if r["carrying"] else "picker",
                        "x": float(r["x"] * scale),
                        "y": float(r["y"] * scale),
                        "resources": r["deliveries"],
                        "hearts": 3,
                        "energy": rewards[rid],
                        "is_defector": False,
                        "coalition_id": 0,
                    }
                    for rid, r in env.robots.items()
                ],
                actions=actions,
                rewards=rewards,
            )
            time.sleep(0.04)

        # Decay exploration
        for qt in q_tables.values():
            qt.epsilon = max(0.05, qt.epsilon * 0.8)

        print(f"   Deliveries: {env.total_deliveries} | Reward: {ep_reward:.2f}")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see warehouse coordination.")


if __name__ == "__main__":
    main()
