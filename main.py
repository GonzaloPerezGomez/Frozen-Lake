import gymnasium as gym
import numpy as np
from q_learn import train_agent_q
from sarsa import train_agent_sarsa
# step() returns {Position, Reward, Termination, Truncation}

def init(saved: bool):
    
    if saved:
        q_table = np.load("q_table.npy")
    else:
        q_table = np.zeros(shape=(16, 4))
    
    return q_table

def print_info(n_games, reward, n_steps):
    print(f"Game: {n_games} | Reward: {reward} | Steps: {n_steps}")
    
def test_agent(env:gym.Env, q_table, episodes=100):
    total_rewards = 0

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Política completamente greedy (epsilon = 0)
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward

    avg_reward = total_rewards / episodes
    print(f"Recompensa media en {episodes} episodios (ε=0): {avg_reward:.2f}")
    return avg_reward



    
if __name__ == "__main__":
    
    # Inicializamos la q_table con 0s
    q_table = init(False)
    
    #Creamos el entorno
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    q_table = train_agent_sarsa(env, q_table)
    test_agent(env, q_table)
    
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
    test_agent(env, q_table, episodes=10)
    
    
    
    