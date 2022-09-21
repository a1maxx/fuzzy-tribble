import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random




class CustomEnv(Env):

    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.float32(np.array([0])), high=np.float32(np.array([100])), dtype=np.float32)
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        #  Since the actions are 0 ,1 ,2 corresponding to down, stay, up

        self.state += action - 1
        self.shower_length -= 1

        # Calculate the reward
        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1

        # Checking if the shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Setting the placeholder info

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return self.state


