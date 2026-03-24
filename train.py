import gymnasium as gym
import torch
import numpy as np
from collections import deque
import argparse

from agent import Agent
from utils import plot_learning_curve

def dqn(env_name, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.997, save_path='checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        env_name (str): environment name
        agent (Agent): the RL agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        save_path (str): path to save the model
    """
    # Create training environment
    env = gym.make(env_name)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_avg_score = -np.inf            # Track the best average score seen so far
    
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        prev_action = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, base_reward, done, truncated, info = env.step(action)
            
            # --- Custom Reward Shaping ---
            reward = base_reward
            
            # 1. Fuel optimization (Extra penalties for engine use)
            if action in [1, 3]:
                reward -= 0.5   # Side thrusters
            elif action == 2:
                reward -= 0.5   # Main engine
                
            # 2. Prevent left/right thruster cancelling (Jittering)
            if (prev_action == 1 and action == 3) or (prev_action == 3 and action == 1):
                reward -= 5.0   # Massive penalty for reversing thrust immediately
                
            # 3. Structural stability (Constant acceleration / Low Jerk)
            # state[3] is y_velocity. We calculate magnitude of vertical acceleration:
            accel_y = abs(next_state[3] - state[3]) 
            # We penalize high sudden accelerations (jolts)
            reward -= accel_y * 10.0
            
            # 4. Encourage faster descent (Penalize moving too slowly vertically)
            # state[3] is negative when moving down. If velocity is between -0.1 and 0 (too slow), penalize it.
            if -0.1 < state[3] < 0:
                reward -= 1.0 # Penalty for hovering/descending too slowly
                
            # 5. Optimal Pathing (Penalize horizontal deviation from the center line)
            # state[0] is the horizontal coordinate. The landing pad is exactly at x=0.
            # We penalize whenever the agent is not directly above the landing pad.
            # This forces the agent to align itself horizontally *first* and then drop straight down.
            reward -= abs(state[0]) * 5.0
            
            # 6. Time Penalty (Penalize each frame it stays in the air)
            # This combats it "scoring well" despite floating sideways slowly
            reward -= 0.5 
            
            # 7. Guaranteed Perfect Landing Bonus
            # If the episode is 'done' and the final base_reward is 100 (which is given for a safe landing, not a crash)
            # We add a massive positive reward to force it to prioritize this above everything else.
            if done and base_reward == 100:
                reward += 500.0
            
            prev_action = action
            # -----------------------------
            
            agent.step(state, action, reward, next_state, done or truncated)
            state = next_state
            score += base_reward # Record the base score for accurate evaluation
            if done or truncated:
                break 
                
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        current_avg = np.mean(scores_window)
        
        # Save the best weights we've seen so far!
        if i_episode >= 100 and current_avg > max_avg_score:
            max_avg_score = current_avg
            torch.save(agent.qnetwork_local.state_dict(), save_path)
        
        print('\rEpisode {}\tAverage Score: {:.2f} (Best: {:.2f})'.format(i_episode, current_avg, max_avg_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} (Best: {:.2f})'.format(i_episode, current_avg, max_avg_score))
        
        # Wait until it hits a consistent, near-perfect upper bound (280+) before stopping completely
        if current_avg >= 280.0:
            print('\nEnvironment mastered in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, current_avg))
            break
            
    print(f'\nFinished training. Best model weights saved to {save_path} with average score of {max_avg_score:.2f}')
    
    env.close()
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Name of the environment")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train")
    parser.add_argument("--out", type=str, default="checkpoint.pth", help="Output file for the model weights")
    parser.add_argument("--plot", type=str, default="scores.png", help="Output file for the learning curve plot")
    
    args = parser.parse_args()
    
    print(f"Initializing {args.env} environment for training...")
    try:
        # Create a dummy environment just to extract the shapes
        env_eval = gym.make(args.env)
        state_size = env_eval.observation_space.shape[0]
        action_size = env_eval.action_space.n
        env_eval.close()
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit(1)
        
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    scores = dqn(env_name=args.env, agent=agent, n_episodes=args.episodes, save_path=args.out)
    
    # Plot the scores
    plot_learning_curve(scores, filename=args.plot)
    print(f"Learning curve saved to {args.plot}")
    
    # --- Watch the trained agent play ---
    print("\n--- Training Complete! Loading best weights to watch the agent... ---")
    try:
        agent.qnetwork_local.load_state_dict(torch.load(args.out, map_location=torch.device('cpu')))
        env_watch = gym.make(args.env, render_mode="human")
        
        for i_episode in range(1, 4): # Watch 3 episodes
            state, info = env_watch.reset()
            score = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Act greedily (no epsilon exploration)
                action = agent.act(state, eps=0.0)
                state, reward, done, truncated, info = env_watch.step(action)
                score += reward
                
            print(f"Evaluation Episode {i_episode}\tScore: {score:.2f}")
            
        env_watch.close()
    except Exception as e:
        print(f"Could not render the environment: {e}")
