#pip install gym[atari]
#pip install gym[toy_text]
#pip install stable-baselines3
import numpy as np
import random
from IPython.display import clear_output # Public APIfor display tools in IPython
import gym
import time

environment = gym.make('Taxi-v3').env
environment.reset()               # For handeling the posibility to face with Taxi-v3 error we reset the environment before rendering it
environment.render()


alpha = 0.1 #learning rate
gamma = 0.6
epsilon = 0.1 #random value


q_table = np.zeros([environment.observation_space.n, environment.action_space.n])

num_of_episodes = 100000
for episode in range(0 ,num_of_episodes):
    # reset the environment
    state = environment.reset()
    
    # Initialized the variables
    reward = 0
    terminated = False
    # if we want to not terminale the loop
    while not terminated:
        # Take learned path or explore new actions based on the epsilon
        if random.uniform(0,1)<epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            # Take action
            next_state, reward, terminated, info = environment.step(action)
            
            
            # Recalculate
            q_value = q_table[state , action]
            max_value = np.max(q_table[next_state])
            new_q_value = (1-alpha)* q_value + alpha * (reward + gamma * max_value)
           
        
            # update q_table
            q_table[state , action] = new_q_value
            state = next_state
    if (episode + 1)%100 == 0:
                clear_output(wait = True)
                print('Episode: {}'.format(episode + 1))
                environment.render()

              
                
                
                
print('*****************************************')
print('Training is done!\n')
print('*****************************************')



total_epochs, total_penalties = 0, 0
episodes = 5

for _ in range(episodes):
    state = environment.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = environment.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        environment.render()
        time.sleep(1)

    total_penalties += penalties
    total_epochs += epochs


print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
