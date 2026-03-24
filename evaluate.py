import gymnasium as gym
import torch
import numpy as np
from agent import Agent

def evaluate(env_name="LunarLander-v3", checkpoint_path="checkpoint.pth", num_episodes=5):
    """
    Evaluates the trained agent in the specified environment.
    """
    print(f"Initializing {env_name} environment for evaluation...")
    # Initialize the environment with render_mode="human" to watch the agent
    try:
        env = gym.make(env_name, render_mode="human")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("If you are using a custom 3D Lunar Lander, make sure it is properly registered or imported.")
        return
    
    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    # Load the trained weights
    try:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        print(f"Successfully loaded weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Could not find {checkpoint_path}.")
        print("Please ensure you have trained the model and saved the weights to this path before evaluating.")
        env.close()
        return
    except Exception as e:
        print(f"Error loading weights: {e}")
        env.close()
        return

    print(f"Evaluating agent for {num_episodes} episodes...")
    for i_episode in range(1, num_episodes + 1):
        state, info = env.reset()
        score = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # The agent chooses an action greedily based on learned policy (eps=0.0)
            action = agent.act(state, eps=0.0)
            
            # Take the action in the environment
            next_state, reward, done, truncated, info = env.step(action)
            
            score += reward
            state = next_state
            
        print(f"Episode {i_episode}\tScore: {score:.2f}")
        
    env.close()
    print("Evaluation completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent.")
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Name of the environment")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to the trained model weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    
    evaluate(env_name=args.env, checkpoint_path=args.checkpoint, num_episodes=args.episodes)
