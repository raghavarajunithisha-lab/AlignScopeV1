"""
AlignScope Example — RLlib: Traffic Intersection Control

6 autonomous vehicles at a 4-way intersection must learn to
negotiate right-of-way. Vehicles from different directions have
competing goals but must cooperate to avoid collisions.

Demonstrates: How an RLlib user would integrate AlignScope
via the callback pattern with a realistic traffic environment.

Dependencies: None (self-contained environment)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: 4-Way Intersection
# ──────────────────────────────────────────────────

class TrafficIntersection:
    """
    4-way intersection with vehicles approaching from N, S, E, W.
    Each vehicle decides: accelerate, brake, or yield.
    Collision = huge penalty. Clearing intersection = reward.
    """

    ROAD_LENGTH = 200
    INTERSECTION_START = 80
    INTERSECTION_END = 120
    MAX_SPEED = 5.0

    def __init__(self, n_vehicles=6):
        self.n_vehicles = n_vehicles
        self.reset()

    def reset(self):
        directions = ["north", "south", "east", "west"]
        self.vehicles = {}
        for i in range(self.n_vehicles):
            d = directions[i % 4]
            self.vehicles[f"car_{i}"] = {
                "direction": d,
                "position": random.uniform(0, 60),  # Distance from start
                "speed": random.uniform(2, 4),
                "cleared": False,
                "collided": False,
                "wait_time": 0,
            }
        self.step_count = 0
        self.collisions = 0
        self.clearances = 0
        return self._obs()

    def _obs(self):
        return {
            vid: {
                "position": v["position"],
                "speed": v["speed"],
                "direction": v["direction"],
                "in_intersection": self.INTERSECTION_START <= v["position"] <= self.INTERSECTION_END,
                "others_in_intersection": sum(
                    1 for ov in self.vehicles.values()
                    if ov["direction"] != v["direction"]
                    and self.INTERSECTION_START <= ov["position"] <= self.INTERSECTION_END
                    and not ov["collided"]
                ),
            }
            for vid, v in self.vehicles.items()
            if not v["cleared"] and not v["collided"]
        }

    def _get_xy(self, vehicle):
        """Convert direction + position to 2D coordinates for visualization."""
        p = vehicle["position"]
        d = vehicle["direction"]
        center = 100
        if d == "north":
            return (center - 10, self.ROAD_LENGTH - p)
        elif d == "south":
            return (center + 10, p)
        elif d == "east":
            return (p, center - 10)
        elif d == "west":
            return (self.ROAD_LENGTH - p, center + 10)
        return (center, center)

    def step(self, actions):
        """actions: {vehicle_id: 'accelerate'|'brake'|'yield'}"""
        self.step_count += 1
        rewards = {}

        for vid, action in actions.items():
            v = self.vehicles[vid]
            if v["cleared"] or v["collided"]:
                continue

            # Apply action
            if action == "accelerate":
                v["speed"] = min(self.MAX_SPEED, v["speed"] + 0.5)
            elif action == "brake":
                v["speed"] = max(0, v["speed"] - 1.5)
            elif action == "yield":
                v["speed"] = max(0, v["speed"] - 0.5)
                v["wait_time"] += 1

            v["position"] += v["speed"]
            rewards[vid] = -0.01  # Time penalty

            # Check if cleared
            if v["position"] >= self.ROAD_LENGTH:
                v["cleared"] = True
                self.clearances += 1
                rewards[vid] = 1.0 - v["wait_time"] * 0.02  # Reward minus wait penalty

        # Check collisions (vehicles from different directions in intersection simultaneously)
        events = []
        in_intersection = [
            (vid, v) for vid, v in self.vehicles.items()
            if self.INTERSECTION_START <= v["position"] <= self.INTERSECTION_END
            and not v["cleared"] and not v["collided"]
        ]
        # Check cross-traffic collisions
        for i, (vid1, v1) in enumerate(in_intersection):
            for vid2, v2 in in_intersection[i+1:]:
                if v1["direction"] != v2["direction"]:
                    # Perpendicular or opposite traffic collision
                    if v1["speed"] > 1 and v2["speed"] > 1:
                        v1["collided"] = True
                        v2["collided"] = True
                        self.collisions += 1
                        rewards[vid1] = -2.0
                        rewards[vid2] = -2.0
                        events.append({
                            "agent_id": vid1, "team": ["north","south","east","west"].index(v1["direction"]),
                            "previous_role": v1["direction"], "severity": 1.0,
                            "reason": f"Collision: {vid1} ({v1['direction']}) ↔ {vid2} ({v2['direction']})",
                        })

        done = self.step_count >= 200 or all(
            v["cleared"] or v["collided"] for v in self.vehicles.values()
        )
        return self._obs(), rewards, events, done


# ──────────────────────────────────────────────────
# Policy: simple rule-based (yield if cross-traffic detected)
# ──────────────────────────────────────────────────

def traffic_policy(obs):
    """Yield if other vehicles are in the intersection, else accelerate."""
    if obs["in_intersection"]:
        return "accelerate"  # Committed — keep going
    if obs["others_in_intersection"] > 0 and obs["position"] > 60:
        return "yield"  # Wait for cross traffic
    if obs["position"] > 70:
        return "brake" if obs["speed"] > 3 else "accelerate"
    return "accelerate"


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("🚗 Traffic Intersection — RLlib Multi-Agent")
    print("    6 vehicles negotiate a 4-way intersection")
    print("=" * 55)

    tracker = alignscope.init(project="traffic-intersection")
    env = TrafficIntersection(n_vehicles=6)

    dir_to_team = {"north": 0, "south": 0, "east": 1, "west": 1}

    for ep in range(3):
        obs = env.reset()
        tracker.reset()
        done = False
        print(f"\n🚦 Episode {ep + 1}")

        while not done:
            actions = {vid: traffic_policy(o) for vid, o in obs.items()}
            obs, rewards, events, done = env.step(actions)

            # Compute 2D positions for all vehicles (even cleared/collided)
            scale = 2
            agent_list = []
            for vid, v in env.vehicles.items():
                xy = env._get_xy(v)
                agent_list.append({
                    "agent_id": vid,
                    "team": dir_to_team.get(v["direction"], 0),
                    "role": v["direction"],
                    "x": float(xy[0] * scale),
                    "y": float(xy[1] * scale),
                    "resources": int(v["cleared"]),
                    "hearts": 0 if v["collided"] else 3,
                    "energy": rewards.get(vid, 0),
                    "is_defector": v["collided"],
                    "coalition_id": dir_to_team.get(v["direction"], 0),
                })

            tracker.log(
                step=env.step_count + ep * 200,
                agents=agent_list,
                actions={vid: a for vid, a in actions.items()},
                rewards=rewards,
                defection_events=events,
            )
            time.sleep(0.05)

        print(f"   Cleared: {env.clearances}/{env.n_vehicles} | Collisions: {env.collisions}")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see traffic negotiation.")


if __name__ == "__main__":
    main()
