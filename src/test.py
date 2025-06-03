import numpy as np


def test_agent(env, q_table, episodes=100):
    total_rewards = 0
    n_games = 0
    
    while n_games < episodes:
        state, _ = env.reset()
        done = False

        while not done:
            # Política completamente greedy (epsilon = 0)
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward
            
        n_games += 1

    avg_reward = total_rewards / episodes
    print(f"Recompensa media en {episodes} episodios (ε=0): {avg_reward:.2f}")
    return avg_reward