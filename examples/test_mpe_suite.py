"""
AlignScope Example — MPE Full Suite (CTDE / Cooperative / Competitive / Mixed)
Tests: simple_spread, simple_tag, simple_adversary, simple_push, simple_crypto

Install: pip install pettingzoo[mpe]
"""
import time
import alignscope


def run_mpe_env(env_fn, name, description, max_steps=300):
    """Generic runner for any MPE environment."""
    print(f"\n🌐 {name}: {description}")
    env = env_fn()
    env = alignscope.wrap(env)
    env.reset()
    
    steps = 0
    try:
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                action = env.action_space(agent).sample()
            env.step(action)
            steps += 1
            time.sleep(0.005)
            if steps > max_steps:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print(f"   ✅ {name} complete ({steps} steps)")


def main():
    alignscope.init(project="mpe-full-suite")
    print("=" * 60)
    print("AlignScope — MPE Full Suite Test")
    print("Tests continuous physics positions via MPE Adapter")
    print("=" * 60)

    # 1. Cooperative Navigation (simple_spread)
    try:
        from pettingzoo.mpe import simple_spread_v3
        run_mpe_env(
            lambda: simple_spread_v3.env(N=3, continuous_actions=True),
            "simple_spread",
            "Cooperative — 3 agents spread to landmarks"
        )
    except ImportError as e:
        print(f"   ⚠️ Skipped simple_spread: {e}")

    # 2. Predator-Prey (simple_tag)
    try:
        from pettingzoo.mpe import simple_tag_v3
        run_mpe_env(
            lambda: simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, continuous_actions=True),
            "simple_tag",
            "Mixed — 3 predators chase 1 prey"
        )
    except ImportError as e:
        print(f"   ⚠️ Skipped simple_tag: {e}")

    # 3. Keep-Away (simple_adversary)
    try:
        from pettingzoo.mpe import simple_adversary_v3
        run_mpe_env(
            lambda: simple_adversary_v3.env(N=2, continuous_actions=True),
            "simple_adversary",
            "Competitive — adversary vs cooperating agents"
        )
    except ImportError as e:
        print(f"   ⚠️ Skipped simple_adversary: {e}")

    # 4. Push (simple_push)    
    try:
        from pettingzoo.mpe import simple_push_v3
        run_mpe_env(
            lambda: simple_push_v3.env(continuous_actions=True),
            "simple_push",
            "Competitive — agent pushes adversary from landmark"
        )
    except ImportError as e:
        print(f"   ⚠️ Skipped simple_push: {e}")

    # 5. Crypto (simple_crypto)
    try:
        from pettingzoo.mpe import simple_crypto_v3
        run_mpe_env(
            lambda: simple_crypto_v3.env(continuous_actions=True),
            "simple_crypto",
            "Competitive — communication with adversary eavesdropping"
        )
    except ImportError as e:
        print(f"   ⚠️ Skipped simple_crypto: {e}")

    print("\n" + "=" * 60)
    print("✅ MPE suite complete!")
    print("Dashboard shows real (x,y) positions from physics engine.")
    print("Open http://localhost:8000 to visualize.")


if __name__ == "__main__":
    main()
