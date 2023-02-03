import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from IPython.display import clear_output



        
class DQLAgent:
    def __init__(self, env):
        # parameter / hyperparameter
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        #%%
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
        #%%
        self.gamma = 0.95
        self.learning_rate = 0.001 
        
        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
        
        
    def build_model(self):
        #%%
        """
        Sigmoid: This activation function maps any input to a value 
        between 0 and 1, making it useful for binary classification problems.
        However, the sigmoid activation function can cause the vanishing 
        gradient problem, where the gradients become very small, making
        it difficult to train deep networks.

        Tanh: The tanh activation function maps any input to a value
        between -1 and 1. It is similar to the sigmoid function, but 
        the outputs are centered around 0, which can help the model
        converge faster.
        
        ReLU (Rectified Linear Unit): This activation function is widely 
        used in deep learning because of its simplicity and efficiency. 
        The ReLU activation function returns the input if it is positive,
        and 0 otherwise. This allows the network to learn sparse 
        representations, as only a subset of neurons will be activated
        in each layer.
        
        Leaky ReLU: This is a variant of the ReLU activation function 
        that returns a small negative value (e.g. 0.01) for negative 
        inputs instead of 0. This helps prevent the "dying ReLU" problem,
        where some neurons never activate and become stuck at 0.
        
        Softmax: This activation function is commonly used in the output 
        layer of a deep learning model for multiclass classification problems. 
        It maps the inputs to a probability distribution over the classes, 
        allowing the model to output a predicted class.
        """
        #%%
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
            self.model.fit(state,train_target, verbose = 0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    
    # initialize gym env and agent
    env = gym.make("CartPole-v1")

    agent = DQLAgent(env)
    
    batch_size = 16
    episodes = 2
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        
        state = np.reshape(state,[1,4])
        
        time = 0
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
                break
            
            
# %% test
import time
trained_model = agent
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
            
            


    

            
            
            
            
            
            
            
            
            
            
            
            
    
