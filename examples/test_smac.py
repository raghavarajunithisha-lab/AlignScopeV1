"""
AlignScope Example — SMAC: Marine Squad Micromanagement

5 marines vs 5 enemy bots on a combat grid. Marines must focus fire,
kite, and coordinate positioning to win. Demonstrates how SMAC
researchers integrate AlignScope for real-time combat observability.

Dependencies: None (self-contained environment, no SC2 needed)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Marine Squad Combat
# ──────────────────────────────────────────────────

class MarineCombat:
    """5v5 marine combat grid inspired by SMAC 5m map."""

    GRID_W = 300
    GRID_H = 200
    ATTACK_RANGE = 50
    MOVE_SPEED = 4.0
    ATTACK_DAMAGE = 1.5

    def __init__(self, n_marines=5):
        self.n_marines = n_marines
        self.reset()

    def reset(self):
        # Ally marines (left side)
        self.allies = {
            f"marine_{i}": {
                "x": 30.0 + random.uniform(-10, 10),
                "y": 40.0 + i * 30,
                "hp": 45.0,
                "max_hp": 45.0,
                "cooldown": 0,
                "kills": 0,
                "damage_dealt": 0.0,
            }
            for i in range(self.n_marines)
        }
        # Enemy bots (right side) — simple AI
        self.enemies = {
            f"enemy_{i}": {
                "x": 270.0 + random.uniform(-10, 10),
                "y": 40.0 + i * 30,
                "hp": 45.0,
                "max_hp": 45.0,
                "cooldown": 0,
            }
            for i in range(self.n_marines)
        }
        self.step_count = 0
        return self._obs()

    def _obs(self):
        return {
            mid: {
                "pos": (m["x"], m["y"]),
                "hp": m["hp"],
                "enemies": [
                    (eid, e["x"], e["y"], e["hp"])
                    for eid, e in self.enemies.items() if e["hp"] > 0
                ],
                "allies": [
                    (aid, a["x"], a["y"], a["hp"])
                    for aid, a in self.allies.items() if a["hp"] > 0 and aid != mid
                ],
            }
            for mid, m in self.allies.items() if m["hp"] > 0
        }

    def step(self, actions):
        """actions: {marine_id: ('attack', enemy_id) | ('move', dx, dy) | ('kite', enemy_id)}"""
        self.step_count += 1
        rewards = {}
        events = []

        # Ally actions
        for mid, action in actions.items():
            m = self.allies[mid]
            if m["hp"] <= 0:
                rewards[mid] = -1.0
                continue
            rewards[mid] = -0.01

            if m["cooldown"] > 0:
                m["cooldown"] -= 1

            if action[0] == "attack" and m["cooldown"] == 0:
                target_id = action[1]
                if target_id in self.enemies and self.enemies[target_id]["hp"] > 0:
                    target = self.enemies[target_id]
                    dist = math.sqrt((m["x"]-target["x"])**2 + (m["y"]-target["y"])**2)
                    if dist <= self.ATTACK_RANGE:
                        dmg = self.ATTACK_DAMAGE + random.uniform(-0.3, 0.3)
                        target["hp"] -= dmg
                        m["damage_dealt"] += dmg
                        m["cooldown"] = 3
                        rewards[mid] = 0.2
                        if target["hp"] <= 0:
                            m["kills"] += 1
                            rewards[mid] = 1.0
                            events.append({
                                "agent_id": target_id, "team": 1,
                                "previous_role": "enemy_marine", "severity": 0.9,
                                "reason": f"{target_id} eliminated by {mid}",
                            })

            elif action[0] == "move":
                dx, dy = action[1], action[2]
                mag = math.sqrt(dx*dx + dy*dy) or 1
                m["x"] = max(0, min(self.GRID_W, m["x"] + dx / mag * self.MOVE_SPEED))
                m["y"] = max(0, min(self.GRID_H, m["y"] + dy / mag * self.MOVE_SPEED))

            elif action[0] == "kite" and m["cooldown"] == 0:
                # Attack then move away
                target_id = action[1]
                if target_id in self.enemies and self.enemies[target_id]["hp"] > 0:
                    target = self.enemies[target_id]
                    dist = math.sqrt((m["x"]-target["x"])**2 + (m["y"]-target["y"])**2)
                    if dist <= self.ATTACK_RANGE:
                        dmg = self.ATTACK_DAMAGE * 0.8
                        target["hp"] -= dmg
                        m["damage_dealt"] += dmg
                        m["cooldown"] = 4
                        # Move away
                        dx = m["x"] - target["x"]
                        dy = m["y"] - target["y"]
                        mag = math.sqrt(dx*dx + dy*dy) or 1
                        m["x"] += dx / mag * self.MOVE_SPEED * 1.5
                        m["y"] += dy / mag * self.MOVE_SPEED * 1.5
                        m["x"] = max(0, min(self.GRID_W, m["x"]))
                        m["y"] = max(0, min(self.GRID_H, m["y"]))
                        rewards[mid] = 0.15

        # Enemy AI: attack nearest ally
        for eid, e in self.enemies.items():
            if e["hp"] <= 0 or e["cooldown"] > 0:
                if e["cooldown"] > 0:
                    e["cooldown"] -= 1
                continue

            alive_allies = [(mid, m) for mid, m in self.allies.items() if m["hp"] > 0]
            if not alive_allies:
                continue

            nearest = min(alive_allies,
                         key=lambda a: (a[1]["x"]-e["x"])**2 + (a[1]["y"]-e["y"])**2)
            dist = math.sqrt((nearest[1]["x"]-e["x"])**2 + (nearest[1]["y"]-e["y"])**2)

            if dist <= self.ATTACK_RANGE:
                dmg = self.ATTACK_DAMAGE + random.uniform(-0.3, 0.3)
                nearest[1]["hp"] -= dmg
                e["cooldown"] = 3
                if nearest[1]["hp"] <= 0:
                    events.append({
                        "agent_id": nearest[0], "team": 0,
                        "previous_role": "marine", "severity": 0.9,
                        "reason": f"{nearest[0]} killed by {eid}",
                    })
            else:
                # Move toward nearest ally
                dx = nearest[1]["x"] - e["x"]
                dy = nearest[1]["y"] - e["y"]
                mag = math.sqrt(dx*dx + dy*dy) or 1
                e["x"] += dx / mag * (self.MOVE_SPEED * 0.8)
                e["y"] += dy / mag * (self.MOVE_SPEED * 0.8)

        allies_alive = sum(1 for m in self.allies.values() if m["hp"] > 0)
        enemies_alive = sum(1 for e in self.enemies.values() if e["hp"] > 0)
        done = self.step_count >= 200 or allies_alive == 0 or enemies_alive == 0

        return self._obs(), rewards, events, done, allies_alive, enemies_alive


# ──────────────────────────────────────────────────
# Policy: Focus-fire + kiting
# ──────────────────────────────────────────────────

def marine_policy(obs, marine_id):
    """Focus-fire lowest HP enemy. Kite if HP < 50%."""
    pos = obs["pos"]
    hp = obs["hp"]
    enemies = obs["enemies"]

    if not enemies:
        return ("move", 1, 0)

    # Focus-fire: attack lowest HP enemy
    target = min(enemies, key=lambda e: e[3])  # e[3] = hp
    target_id, tx, ty, thp = target
    dist = math.sqrt((pos[0]-tx)**2 + (pos[1]-ty)**2)

    # Kite if low HP and enemy is close
    if hp < 22 and dist < 40:
        return ("kite", target_id)

    # Attack if in range
    if dist <= 50:
        return ("attack", target_id)

    # Move toward target
    return ("move", tx - pos[0], ty - pos[1])


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("⚔️  Marine Squad Combat — SMAC 5m")
    print("    5 marines vs 5 enemies with focus-fire + kiting")
    print("=" * 55)

    tracker = alignscope.init(project="smac-5m-combat")
    env = MarineCombat(n_marines=5)

    wins = 0
    for ep in range(5):
        obs = env.reset()
        tracker.reset()
        done = False
        print(f"\n🎯 Battle {ep + 1}/5")

        while not done:
            actions = {mid: marine_policy(o, mid) for mid, o in obs.items()}
            obs, rewards, events, done, allies_alive, enemies_alive = env.step(actions)

            # Log ALL units (allies + enemies) to AlignScope
            agent_list = []
            for mid, m in env.allies.items():
                agent_list.append({
                    "agent_id": mid, "team": 0, "role": "marine",
                    "x": m["x"], "y": m["y"],
                    "resources": m["kills"], "hearts": max(0, int(m["hp"] / 15)),
                    "energy": rewards.get(mid, 0),
                    "is_defector": m["hp"] <= 0,
                    "coalition_id": 0 if m["hp"] > 0 else -1,
                })
            for eid, e in env.enemies.items():
                agent_list.append({
                    "agent_id": eid, "team": 1, "role": "enemy_marine",
                    "x": e["x"], "y": e["y"],
                    "resources": 0, "hearts": max(0, int(e["hp"] / 15)),
                    "energy": 0, "is_defector": e["hp"] <= 0,
                    "coalition_id": 1 if e["hp"] > 0 else -1,
                })

            tracker.log(
                step=env.step_count + ep * 200,
                agents=agent_list,
                rewards=rewards,
                defection_events=events,
            )
            time.sleep(0.05)

        result = "WIN" if enemies_alive == 0 else "LOSS" if allies_alive == 0 else "DRAW"
        if result == "WIN":
            wins += 1
        print(f"   {result} — Allies: {allies_alive}/5, Enemies: {enemies_alive}/5")

    tracker.finish()
    print(f"\n✅ Win rate: {wins}/5 ({wins*20}%)")
    print("Open http://localhost:8000 to see combat replays.")


if __name__ == "__main__":
    main()
