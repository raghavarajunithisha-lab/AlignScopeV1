"""
AlignScope Example — PettingZoo SISL Environments (Cooperative)
Tests: Waterworld, Pursuit

Install: pip install pettingzoo[sisl]
"""
import time
import alignscope


def test_waterworld():
    """Waterworld — agents chase food and avoid poison in continuous space."""
    from pettingzoo.sisl import waterworld_v4
    
    print("🌊 Testing Waterworld (5 pursuers cooperating)...")
    env = waterworld_v4.env(n_pursuers=5, n_evaders=5, n_poisons=10)
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
    print(f"   ✅ Waterworld complete ({steps} steps)")


def test_pursuit():
    """Pursuit — cooperative predators must surround prey on a grid."""
    from pettingzoo.sisl import pursuit_v4
    
    print("🎯 Testing Pursuit (8 pursuers chasing 30 evaders)...")
    env = pursuit_v4.env(n_pursuers=8, n_evaders=30)
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
    print(f"   ✅ Pursuit complete ({steps} steps)")


def main():
    alignscope.init(project="pettingzoo-sisl-cooperative")
    print("=" * 50)
    print("AlignScope — SISL Environments Test")
    print("Paradigm: Cooperative (continuous / grid)")
    print("=" * 50)

    try:
        test_waterworld()
    except ImportError as e:
        print(f"   ⚠️ Skipped Waterworld: {e}")

    try:
        test_pursuit()
    except ImportError as e:
        print(f"   ⚠️ Skipped Pursuit: {e}")

    print("\n✅ All SISL environments tested!")
    print("Open http://localhost:8000 to visualize.")


if __name__ == "__main__":
    main()
