"""
AlignScope Example — Mixed-Interest: Multi-Agent Trading Market

6 traders across 3 firms compete for market share while
cooperating within their firm. Demonstrates mixed cooperative-
competitive dynamics with real supply/demand mechanics.

Dependencies: None (self-contained environment)
"""
import time
import math
import random
import alignscope


# ──────────────────────────────────────────────────
# Custom Environment: Trading Market
# ──────────────────────────────────────────────────

class TradingMarket:
    """Simulated commodity market with intra-firm cooperation and inter-firm competition."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Market state
        self.price = 100.0
        self.volatility = 0.05
        self.step_count = 0

        # 3 firms, 2 traders each
        self.traders = {}
        firms = [(0, "alpha"), (1, "beta"), (2, "gamma")]
        for firm_id, firm_name in firms:
            for j in range(2):
                role = "analyst" if j == 0 else "executor"
                tid = f"{firm_name}_{role}"
                self.traders[tid] = {
                    "firm": firm_id,
                    "role": role,
                    "cash": 10000.0,
                    "inventory": 0,
                    "pnl": 0.0,  # Profit and loss
                    "trades": 0,
                }
        return self._get_obs()

    def _get_obs(self):
        return {
            "price": self.price,
            "traders": {tid: dict(t) for tid, t in self.traders.items()},
        }

    def step(self, actions: dict):
        """actions: {trader_id: ('buy'|'sell'|'hold', quantity)}"""
        self.step_count += 1
        rewards = {}
        events = []

        # Price follows geometric Brownian motion with order pressure
        buy_pressure = sum(q for _, (a, q) in actions.items() if a == "buy")
        sell_pressure = sum(q for _, (a, q) in actions.items() if a == "sell")
        net_pressure = (buy_pressure - sell_pressure) * 0.02
        noise = random.gauss(0, self.volatility * self.price)
        trend = math.sin(self.step_count * 0.03) * 0.5  # Slow trend
        self.price = max(10, self.price + noise + net_pressure + trend)

        for tid, (action, quantity) in actions.items():
            t = self.traders[tid]
            old_value = t["cash"] + t["inventory"] * self.price

            if action == "buy" and t["cash"] >= self.price * quantity:
                t["cash"] -= self.price * quantity
                t["inventory"] += quantity
                t["trades"] += 1
            elif action == "sell" and t["inventory"] >= quantity:
                t["cash"] += self.price * quantity
                t["inventory"] -= quantity
                t["trades"] += 1

            new_value = t["cash"] + t["inventory"] * self.price
            t["pnl"] = new_value - 10000.0  # P&L from starting cash
            rewards[tid] = new_value - old_value

            # Detect "defection" — trader acting against firm interest
            firm_members = [t2 for t2id, t2 in self.traders.items()
                           if t2["firm"] == t["firm"] and t2id != tid]
            if firm_members:
                partner = firm_members[0]
                if (action == "buy" and partner["inventory"] > 5) or \
                   (action == "sell" and partner["inventory"] < -5):
                    if random.random() < 0.3:  # Not always obvious
                        events.append({
                            "agent_id": tid, "team": t["firm"],
                            "previous_role": t["role"], "severity": 0.6,
                            "reason": f"{tid} traded against firm position",
                        })

        done = self.step_count >= 200
        return self._get_obs(), rewards, events, done


# ──────────────────────────────────────────────────
# Trading Policies
# ──────────────────────────────────────────────────

def analyst_policy(obs, agent_id):
    """Analysts are trend-followers — buy when price is rising."""
    price = obs["price"]
    inventory = obs["traders"][agent_id]["inventory"]
    cash = obs["traders"][agent_id]["cash"]

    if price < 95 and cash > price * 3:
        return ("buy", 3)
    elif price > 105 and inventory > 2:
        return ("sell", 2)
    elif inventory > 8:
        return ("sell", 3)
    return ("hold", 0)


def executor_policy(obs, agent_id):
    """Executors are contrarian — buy dips, sell rallies."""
    price = obs["price"]
    inventory = obs["traders"][agent_id]["inventory"]
    cash = obs["traders"][agent_id]["cash"]

    if price < 90 and cash > price * 5:
        return ("buy", 5)
    elif price > 110 and inventory > 0:
        return ("sell", min(inventory, 4))
    elif random.random() < 0.2 and cash > price:
        return ("buy", 1)
    return ("hold", 0)


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("💹 Trading Market — Mixed-Interest MARL")
    print("    3 firms × 2 traders: cooperate within, compete across")
    print("=" * 55)

    tracker = alignscope.init(project="trading-market")
    env = TradingMarket()

    for ep in range(2):
        obs = env.reset()
        tracker.reset()
        done = False
        print(f"\n📈 Episode {ep + 1}")

        while not done:
            actions = {}
            for tid, t in env.traders.items():
                if t["role"] == "analyst":
                    actions[tid] = analyst_policy(obs, tid)
                else:
                    actions[tid] = executor_policy(obs, tid)

            obs, rewards, events, done = env.step(actions)

            # Log to AlignScope
            firm_colors_x = {0: 80, 1: 200, 2: 320}
            tracker.log(
                step=env.step_count + ep * 200,
                agents=[
                    {
                        "agent_id": tid,
                        "team": t["firm"],
                        "role": t["role"],
                        "x": float(firm_colors_x[t["firm"]] + (30 if t["role"] == "executor" else 0)),
                        "y": float(obs["price"] * 2),  # Y tracks price
                        "resources": t["trades"],
                        "hearts": 3,
                        "energy": rewards[tid],
                        "is_defector": t["pnl"] < -500,
                        "coalition_id": t["firm"],
                    }
                    for tid, t in env.traders.items()
                ],
                actions={tid: a for tid, (a, _) in actions.items()},
                rewards=rewards,
                defection_events=events,
            )
            time.sleep(0.05)

        # Print P&L summary
        for tid, t in env.traders.items():
            print(f"   {tid:20s} P&L: ${t['pnl']:+.0f}  ({t['trades']} trades)")

    tracker.finish()
    print("\n✅ Done! Open http://localhost:8000 to see firm dynamics.")


if __name__ == "__main__":
    main()
