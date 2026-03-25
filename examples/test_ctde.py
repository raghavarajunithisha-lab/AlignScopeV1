"""
AlignScope Example — CTDE: Drone Swarm Coordination

4 drones controlled by a central planner must cover observation zones.
During training, the planner sees all drones (centralized).
During execution, each drone navigates with local sensors (decentralized).

Demonstrates: Centralized Training, Decentralized Execution (CTDE)
Dependencies: None (self-contained environment)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Drone Swarm Coverage
# ──────────────────────────────────────────────────

class DroneSwarm:
    """4 drones must cover 4 observation zones. Central planner assigns targets."""

    FIELD_SIZE = 400.0
    DRONE_SPEED = 5.0
    ZONE_RADIUS = 40.0

    def __init__(self, n_drones=4):
        self.n_drones = n_drones
        self.zones = [
            (100, 100), (300, 100), (100, 300), (300, 300)
        ]
        self.reset()

    def reset(self):
        self.drones = {
            f"drone_{i}": {
                "x": self.FIELD_SIZE / 2 + random.uniform(-20, 20),
                "y": self.FIELD_SIZE / 2 + random.uniform(-20, 20),
                "battery": 100.0,
                "zone_time": 0,
                "assigned_zone": -1,
            }
            for i in range(self.n_drones)
        }
        self.coverage = {i: 0.0 for i in range(len(self.zones))}
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        """Centralized obs: all drone positions (for training critic)."""
        return {
            "drones": {did: (d["x"], d["y"], d["battery"]) for did, d in self.drones.items()},
            "zones": self.zones[:],
            "coverage": dict(self.coverage),
        }

    def _local_obs(self, drone_id):
        """Decentralized obs: only own position + nearest zone."""
        d = self.drones[drone_id]
        nearest_zone = min(range(len(self.zones)),
                          key=lambda z: (self.zones[z][0]-d["x"])**2 + (self.zones[z][1]-d["y"])**2)
        return {"pos": (d["x"], d["y"]), "battery": d["battery"], "nearest_zone": nearest_zone}

    def assign_zones(self):
        """Central planner: assign each drone to the nearest uncovered zone."""
        available_zones = list(range(len(self.zones)))
        assignments = {}
        drones_sorted = sorted(self.drones.keys())

        for did in drones_sorted:
            if not available_zones:
                assignments[did] = 0
                continue
            d = self.drones[did]
            best_zone = min(available_zones,
                           key=lambda z: (self.zones[z][0]-d["x"])**2 + (self.zones[z][1]-d["y"])**2)
            assignments[did] = best_zone
            available_zones.remove(best_zone)
            self.drones[did]["assigned_zone"] = best_zone

        return assignments

    def step(self, actions: dict):
        """actions: {drone_id: (target_x, target_y)} — decentralized navigation."""
        self.step_count += 1
        rewards = {}

        for did, (tx, ty) in actions.items():
            d = self.drones[did]
            # Move toward target
            dx, dy = tx - d["x"], ty - d["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 1:
                d["x"] += dx / dist * min(self.DRONE_SPEED, dist)
                d["y"] += dy / dist * min(self.DRONE_SPEED, dist)
            d["battery"] -= 0.1  # Drain battery

            # Check zone coverage
            reward = -0.01
            for zi, (zx, zy) in enumerate(self.zones):
                if math.sqrt((d["x"]-zx)**2 + (d["y"]-zy)**2) < self.ZONE_RADIUS:
                    self.coverage[zi] += 0.1
                    d["zone_time"] += 1
                    reward = 0.1  # Rewarded for being in zone
            rewards[did] = reward

        done = self.step_count >= 250 or all(c >= 10.0 for c in self.coverage.values())
        return self._get_obs(), rewards, done


# ──────────────────────────────────────────────────
# Training Loop: CTDE Pattern
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("🛸 Drone Swarm — CTDE (Centralized Train, Decentral Exec)")
    print("    4 drones cover 4 zones via central planner")
    print("=" * 55)

    tracker = alignscope.init(project="drone-swarm-ctde")
    env = DroneSwarm(n_drones=4)

    for ep in range(3):
        obs = env.reset()
        tracker.reset()
        done = False
        print(f"\n🛸 Episode {ep + 1}")

        while not done:
            # CENTRALIZED: planner assigns zones using global state
            assignments = env.assign_zones()

            # DECENTRALIZED: each drone navigates to its assigned zone
            actions = {}
            for did in env.drones:
                zone_idx = assignments[did]
                target = env.zones[zone_idx]
                # Add noise to simulate imperfect local execution
                actions[did] = (
                    target[0] + random.uniform(-5, 5),
                    target[1] + random.uniform(-5, 5),
                )

            obs, rewards, done = env.step(actions)

            # Log to AlignScope
            roles = ["scout", "observer", "relay", "guard"]
            tracker.log(
                step=env.step_count + ep * 250,
                agents=[
                    {
                        "agent_id": did,
                        "team": 0,
                        "role": roles[i % len(roles)],
                        "x": d["x"],
                        "y": d["y"],
                        "resources": d["zone_time"],
                        "hearts": int(d["battery"] / 25),
                        "energy": rewards[did],
                        "is_defector": d["battery"] < 10,
                        "coalition_id": d["assigned_zone"],
                    }
                    for i, (did, d) in enumerate(env.drones.items())
                ],
                rewards=rewards,
            )
            time.sleep(0.04)

        total_coverage = sum(env.coverage.values())
        print(f"   Coverage: {total_coverage:.1f} | Zones fully covered: "
              f"{sum(1 for c in env.coverage.values() if c >= 10)}/4")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see drone coordination.")


if __name__ == "__main__":
    main()
