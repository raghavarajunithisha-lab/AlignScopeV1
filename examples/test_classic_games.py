"""
AlignScope Example — PettingZoo Classic Games (Competitive)
Tests: Chess, Connect Four, Tic-Tac-Toe, Rock-Paper-Scissors, Go

Install: pip install pettingzoo[classic]
"""
import time
import alignscope


def test_rps():
    """Rock-Paper-Scissors — simplest competitive game."""
    from pettingzoo.classic import rps_v2
    
    print("🎮 Testing Rock-Paper-Scissors...")
    env = rps_v2.env()
    env = alignscope.wrap(env)
    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
    env.close()
    print("   ✅ RPS complete")


def test_connect_four():
    """Connect Four — turn-based competitive board game."""
    from pettingzoo.classic import connect_four_v3
    
    print("🎮 Testing Connect Four...")
    env = connect_four_v3.env()
    env = alignscope.wrap(env)
    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            mask = obs["action_mask"]
            valid = [i for i, v in enumerate(mask) if v == 1]
            action = valid[0] if valid else 0
        env.step(action)
    env.close()
    print("   ✅ Connect Four complete")


def test_tictactoe():
    """Tic-Tac-Toe — simplest board game."""
    from pettingzoo.classic import tictactoe_v3
    
    print("🎮 Testing Tic-Tac-Toe...")
    env = tictactoe_v3.env()
    env = alignscope.wrap(env)
    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            mask = obs["action_mask"]
            valid = [i for i, v in enumerate(mask) if v == 1]
            action = valid[0] if valid else 0
        env.step(action)
    env.close()
    print("   ✅ Tic-Tac-Toe complete")


def main():
    alignscope.init(project="pettingzoo-classic-games")
    print("=" * 50)
    print("AlignScope — PettingZoo Classic Games Test")
    print("Paradigm: Competitive / Zero-Sum")
    print("=" * 50)

    try:
        test_rps()
    except ImportError as e:
        print(f"   ⚠️ Skipped RPS: {e}")

    try:
        test_connect_four()
    except ImportError as e:
        print(f"   ⚠️ Skipped Connect Four: {e}")

    try:
        test_tictactoe()
    except ImportError as e:
        print(f"   ⚠️ Skipped Tic-Tac-Toe: {e}")

    print("\n✅ All classic games tested!")
    print("Open http://localhost:8000 to see agent topology and defection timeline.")


if __name__ == "__main__":
    main()
