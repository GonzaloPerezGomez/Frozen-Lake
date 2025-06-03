import gymnasium as gym
import numpy as np
import random

EPSILON = 0.99
GAMES_LIMIT = 20000
LR = 0.01
GAMMA = 0.9

def train_agent_q(env:gym.Env, q_table):

    n_games = 1
    
    #Por cada serie/partida
    while n_games <= GAMES_LIMIT:
        
        termination, truncation = False, False
        done = termination + truncation
        
        state, _ = env.reset() 
        n_steps = 1
        
        while not done:
            
            # Update the epsilon value
            epsilon = max(0.1, EPSILON * (0.995 ** n_games))
            
            # Use epsilon-greedy
            random_float = random.random()
            if random_float < epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Act greedy (explotation)
                action = np.argmax(q_table[state])
                
            # We perform the neww action
            new_state, reward, termination, truncation, _ = env.step(action)
            
            #Adjust (optional)
            if termination == True and reward == 0:
                reward = -1
            
            # We calculate the new q value with Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            new_q_value = q_table[state][action] + LR * (reward + GAMMA * q_table[new_state][np.argmax(q_table[new_state])] - q_table[state][action])
            
            # Update the value
            q_table[state][action] = new_q_value
            
            # Update the variables
            state = new_state
            done = termination + truncation
            n_steps += 1
            
        if n_games % 20 == 0:    
            print(f"Game: {n_games} | Reward: {reward} | Steps: {n_steps}")
        
        # Increment the game´s counter
        n_games += 1
            
    np.save("q_table.npy", q_table)
    return q_table