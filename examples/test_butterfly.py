"""
AlignScope Example — PettingZoo Butterfly Environments (Cooperative)
Tests: Pistonball, Cooperative Pong

Install: pip install pettingzoo[butterfly]
"""
import time
import alignscope


def test_pistonball():
    """Pistonball — cooperative physics: pistons must coordinate to move a ball."""
    from pettingzoo.butterfly import pistonball_v6
    
    print("🦋 Testing Pistonball (20 pistons cooperating)...")
    env = pistonball_v6.env()
    env = alignscope.wrap(env)
    env.reset()
    
    steps = 0
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        steps += 1
        if steps > 500:  # Limit for testing
            break
    env.close()
    print(f"   ✅ Pistonball complete ({steps} steps)")


def test_cooperative_pong():
    """Cooperative Pong — two paddles cooperate to keep ball in play."""
    from pettingzoo.butterfly import cooperative_pong_v5
    
    print("🏓 Testing Cooperative Pong...")
    env = cooperative_pong_v5.env()
    env = alignscope.wrap(env)
    env.reset()
    
    steps = 0
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        steps += 1
        if steps > 500:
            break
    env.close()
    print(f"   ✅ Cooperative Pong complete ({steps} steps)")


def main():
    alignscope.init(project="pettingzoo-butterfly-cooperative")
    print("=" * 50)
    print("AlignScope — Butterfly Environments Test")
    print("Paradigm: Cooperative (agents must coordinate)")
    print("=" * 50)

    try:
        test_pistonball()
    except ImportError as e:
        print(f"   ⚠️ Skipped Pistonball: {e}")

    try:
        test_cooperative_pong()
    except ImportError as e:
        print(f"   ⚠️ Skipped Cooperative Pong: {e}")

    print("\n✅ All Butterfly environments tested!")
    print("Open http://localhost:8000 to visualize agent cooperation.")


if __name__ == "__main__":
    main()
