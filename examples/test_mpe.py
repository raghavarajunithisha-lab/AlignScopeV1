import time
import alignscope
from pettingzoo.mpe import simple_tag_v3

def main():
    print("Initializing MPE (Simple Tag)")
    print("This requires pettingzoo[mpe] to be installed.")
    
    # 3 adversaries, 1 good agent (prey)
    env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, continuous_actions=True)
    
    # Wrap it with AlignScope to auto-log real physical continuous positions
    alignscope.init(project="mpe-simple-tag")
    env = alignscope.wrap(env)
    
    env.reset()
    
    # Run the equivalent of a simple random policy
    try:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                # Random continuous action
                action = env.action_space(agent).sample()
                
            env.step(action)
            
            # Simple delay to visualize live 
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()