"""
AlignScope — PettingZoo Integration Example (Tier 3: Framework Plugin)

Shows how to add AlignScope to any PettingZoo environment with
ONE line of code.

Prerequisites:
    pip install alignscope[pettingzoo]

Usage:
    1. Start the dashboard:  alignscope start
    2. Run this script:      python examples/pettingzoo_example.py
"""

try:
    import time
    import alignscope
    from pettingzoo.classic import rps_v2  # Rock-Paper-Scissors

    # Create your PettingZoo environment as normal
    env = rps_v2.env()

    # Wrap it with AlignScope — ONE LINE:
    env = alignscope.wrap(env)  # ← THIS IS THE ONLY CHANGE

    NUM_EPISODES = 5  # Run many short episodes so you can watch the dashboard

    print(f"🎮 Running {NUM_EPISODES} episodes of Rock-Paper-Scissors...")
    print(f"   Open http://localhost:8000 to watch live\n")

    for episode in range(NUM_EPISODES):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()

            env.step(action)
            time.sleep(0.15)  # Slow down for visualization

        if (episode + 1) % 10 == 0:
            print(f"   Episode {episode + 1}/{NUM_EPISODES} done")

    env.close()
    print("\n✅ All episodes complete! Check the dashboard for alignment metrics.")

except ImportError:
    print("This example requires PettingZoo.")
    print("Install with: pip install 'alignscope[pettingzoo]'")
    print()
    print("Showing what the code looks like instead:")
    print()
    print("  import alignscope")
    print("  env = alignscope.wrap(your_pettingzoo_env)")
    print("  # That's it — every step auto-logged!")
