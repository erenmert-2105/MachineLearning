#%% Lib
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from IPython.display import clear_output
import tensorflow as tf

import time
import dill
import psutil
import pickle
import pyglet
import gc


#%%   DeepRf      
class DeepRf:
    
    def __init__(self,env,gamma,learning_rate,epsilon):
       # parameter / hyperparameter
       self.state_size = env.observation_space.shape[0]
       self.action_size = env.action_space.n

       """
       self.gama: This is the discount factor (gamma), which determines the
       importance of future rewards in the Q-Learning algorithm. It is a value 
       between 0 and 1, with a higher value indicating that future rewards are 
       more important.
       self.leraning_rate: This is the learning rate, which determines the 
       size of the update step taken at each iteration of the Q-Learning 
       algorithm. A smaller learning rate may result in slower convergence, 
       while a larger learning rate may cause overshooting or instability.
       
       self.epsilon: This is the exploration rate or epsilon,
       which determines the probability of the agent taking a random 
       action instead of following the action with the highest Q-value.
       A higher value of epsilon means that the agent is more likely 
       to explore its environment, while a lower value means it is more 
       likely to exploit what it already knows.
       
       self.epsilon_decay: This is the rate at which the value of
       epsilon is decreased over time. This allows the agent to gradually 
       reduce its exploration and increase its exploitation as it gains more
       information about the environment.
       self.epsilon_min: This is the minimum value that epsilon can take.
       This sets a lower bound on the exploration rate and ensures that the
       agent will never fully stop exploring its environment.
       """
       self.gamma = gamma
       self.learning_rate = learning_rate
       
       self.epsilon = epsilon  # explore
       self.epsilon_decay = 0.995
       self.epsilon_min = 0.01
       
       self.memory = deque(maxlen = 1000)

       self.model = self.build_model()

        
      
    def build_model(self):



        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "tanh"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model

        
            
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward 
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target, verbose = 0,use_multiprocessing=True)
        gc.collect()
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
#%%   Load value         
load=0

def get_ram_usage():
    return psutil.virtual_memory().percent

if __name__ == "__main__":
    
    # initialize gym env and agent
    env = gym.make("CartPole-v1")
#%%    Run
    gamma = 0.95 
    # reward importance
    learning_rate = 0.001
    # high learning rate results in quick but unstable convergence, while a low learning rate may produce slow but stable convergence
    epsilon = 1 
    # explore rate do not change that its changing over time in algo
    batch_size = 10
    # it is how many steps between backward and forward 
    # it effect result dramaticly low batch_size makes more accure and high makes more experiance so learns faster but too high or low makes learning harder
    
    episodes = 2
    # how many times play after done 

    
    
    agent = DeepRf(env,gamma,learning_rate,epsilon)
    
    if load ==1:
        model = tf.keras.models.load_model("DRFmodel.h5")
        memory = dill.load(open('buffer.dill', 'rb'))
        agent.model=model
        agent.memory=memory
        
    scores=np.array([])    
    for e in range(episodes+1):
        
        # initialize environment
        state = env.reset()
        
        state = np.reshape(state,[1,4])
        
        time = 0
        if get_ram_usage() >= 95:
            tf.keras.models.save_model(agent.model, "DRFmodel.h5")
            dill.dump(agent.memory, open('buffer.dill', 'wb'))
        
            break
        else: 
            while True:
                
                # act
                action = agent.act(state) # select an action
                
                # step
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state,[1,4])
                
                # remember / storage
                agent.remember(state, action, reward, next_state, done)
                
                # update state
                state = next_state
                
                # replay
                agent.replay(batch_size)
                
                # adjust epsilon
                agent.adaptiveEGreedy()
                
                time += 1
                if done:
                    print("Episode: {}, time: {}".format(e,time))
                    scores=np.append(scores,time)
                    break
                
                 
                
# %% test
"""
   
   trained_model = agent
   trained_model.epsilon=0
   state = env.reset()
   state = np.reshape(state, [1,4])
   step = 0
   t_reward=0
   while True:
       action = trained_model.act(state)
       next_state, reward, done, _ = env.step(action)
       t_reward=t_reward+reward
       next_state = np.reshape(next_state, [1,4])
       state = next_state
       step += 1
       
       print("Step: {},Total Reward: {}".format(step,t_reward))
       #time.sleep(0.4)
       if done:
           break
   print("Done")
"""
#%%  save      
"""        
tf.keras.models.save_model(agent.model, "DRFmodel.h5")
dill.dump(agent.memory, open('buffer.dill', 'wb'))


"""
#%%load                        
"""
model = tf.keras.models.load_model("DRFmodel.h5")
memory = dill.load(open('buffer.dill', 'rb'))
"""                
