"""
AlignScope — Knights Archers Zombies (KAZ) Real Data Test

Tests AlignScope with real PettingZoo KAZ data.

Install:
    pip install 'pettingzoo[butterfly]' pygame alignscope

Run:
    1.  alignscope start          (in one terminal)
    2.  python test_kaz.py        (in another terminal)
    3.  Open http://localhost:8000
"""

import time
import alignscope
from pettingzoo.butterfly import knights_archers_zombies_v10

# ------------------------------------------------------------------
# What you should see on the dashboard:
#
#  Agent Topology:
#    - 4 nodes: archer_0, archer_1, knight_0, knight_1
#    - All team-0 color (blue) — they are all cooperative
#    - Diamonds for archers, circles for knights (different roles)
#    - Nodes gradually pull together as the episode progresses
#      (no help_ally edges yet — see note below)
#
#  Alignment Metrics:
#    - Role stability climbs toward 1.0 quickly (roles never change)
#    - Coalition count = 1 (all same team, same coalition)
#    - Defectors count goes UP as agents are killed by zombies
#      (dead agents are flagged is_defector=True so they detach visually)
#
#  Defection Timeline:
#    - Red markers appear when agents die (killed = "defected" from team)
#    - Timing matches when zombies reach agents in the game
#
#  Note on edges (topology links):
#    KAZ has no "help_ally" action — agents only attack zombies.
#    So the help_matrix stays empty and no topology edges appear.
#    This is correct behavior. To add edges, you could instrument
#    the "attack" (action=5) action as a proxy for coordination —
#    see the commented block at the bottom of this file.
# ------------------------------------------------------------------

NUM_EPISODES = 3

alignscope.init(project="kaz-real-data")

for episode in range(NUM_EPISODES):
    print(f"\n⚔️  Episode {episode + 1}/{NUM_EPISODES}")

    env = knights_archers_zombies_v10.env(
        num_archers=2,
        num_knights=2,
        max_zombies=10,
        killable_knights=True,
        killable_archers=True,
        max_cycles=300,
        vector_state=True,
        render_mode=None,      # set "human" to watch the game live
    )

    # Wrap with AlignScope — one line
    env = alignscope.wrap(env)

    env.reset()

    step = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Random policy — replace with your trained policy here
            action = env.action_space(agent).sample()

        env.step(action)
        step += 1

        # Pace the simulation so the dashboard can render it smoothly
        time.sleep(0.02)

    env.close()
    print(f"   Episode {episode + 1} done — {step} steps")

print("\n✅ KAZ test complete. Check http://localhost:8000")
time.sleep(2) # Allow websocket to flush remaining payloads


# ------------------------------------------------------------------
# OPTIONAL: Instrument attack coordination as "help_ally" edges
#
# KAZ action 5 = "attack". Knights and archers attack zombies that
# are threatening the whole team. You can treat synchronized attacks
# (two agents attacking in the same cycle) as a coordination signal
# and manually log it as help_ally. Uncomment and modify _flush_round
# in pettingzoo.py to call this:
#
# def _detect_coordination(self, round_actions: dict) -> list:
#     """Flag agents who both attacked this round as mutually helping."""
#     attackers = [aid for aid, act in round_actions.items() if act == "5"]
#     help_events = []
#     for i, a in enumerate(attackers):
#         for b in attackers[i+1:]:
#             help_events.append({
#                 "tick": self._step,
#                 "agent_id": a,
#                 "action": "help_ally",
#                 "target_id": b,
#                 "detail": "coordinated attack",
#             })
#             help_events.append({
#                 "tick": self._step,
#                 "agent_id": b,
#                 "action": "help_ally",
#                 "target_id": a,
#                 "detail": "coordinated attack",
#             })
#     return help_events
# ------------------------------------------------------------------
