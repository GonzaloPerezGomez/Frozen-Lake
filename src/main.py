import gymnasium as gym
import numpy as np
from q_learn import train_agent_q
from sarsa import train_agent_sarsa
from src.test import test_agent

# step() returns {Position, Reward, Termination, Truncation}

def init(saved: bool):
    
    if saved:
        q_table = np.load("q_table.npy")
    else:
        q_table = np.zeros(shape=(16, 4))
    
    return q_table

def print_info(n_games, reward, n_steps):
    print(f"Game: {n_games} | Reward: {reward} | Steps: {n_steps}")


if __name__ == "__main__":
    
    # Initialize the q_table to 0s
    q_table = init(False)
    
    # Create the env
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    
    # Train with sarsa or Q_learn
    q_table = train_agent_sarsa(env, q_table)
    q_table = train_agent_q(env, q_table)
    
    # Test with epislon = 0s
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
    test_agent(env, q_table, episodes=10)
    
    
    
    