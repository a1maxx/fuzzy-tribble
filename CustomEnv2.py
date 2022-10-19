import numpy as np
from numpy import random
import gym
from gym import Env, spaces
from gym.spaces import Box, Discrete
import random
from scipy import stats
from scipy.optimize import minimize


def gaussian(params, sample_data):
    mean = params[0]
    sd = params[1]
    # Calculate negative log likelihood
    nll = -np.sum(stats.norm.logpdf(sample_data, loc=mean, scale=sd))

    return nll


class CustomEnv2(Env):

    def __init__(self):
        self.numberOfPeriods = 2
        self.numberOfSensors = 3
        self.distributions = ['normal', 'normal', 'normal']
        self.mean = [2, 5, 10]
        self.eMean = [1, 4, 9]
        self.eStandardDeviation = [0.6, 0.2, 0.2]
        self.standardDeviation = [1, 0.2, 0.1]
        self.driftProbability = [0.1, 0.2, 0.5]
        self.driftRateOfMean = [1, 2, 3]
        self.state = [10, 15, 20]  # Resource Allocated
        self.resourceCapacity = 55

        self.action_space = gym.spaces.Box(low=np.zeros(self.numberOfSensors),
                                           high=np.ones(self.numberOfSensors) * 5, dtype=np.int16)

        self.observation_space = spaces.Box(low=np.array([0,0,0]),
                                            high=np.array([100, 100, 100]),
                                            dtype=np.int16)

        self.action_space

    def reset(self):
        self.numberOfPeriods = 2
        self.numberOfSensors = 3
        self.distributions = ['normal', 'normal', 'normal']
        self.mean = [2, 5, 10]
        self.eMean = [1, 4, 9]
        self.standardDeviation = [1, 0.2, 0.1]
        self.driftProbability = [0.1, 0.2, 0.5]
        self.driftRateOfMean = [1, 2, 3]
        self.state = [10, 15, 20]
        # self.resourceCapacity = 55

        return self.state

    def step(self, action):
        self.state += action
        self.periods -= 1

        for i in range(self.numberOfSensors):
            if random.uniform(0, 1, 1)[0] < self.driftProbability:
                if random.uniform(0, 1, 1)[0] < 0.5:
                    self.mean[i] -= self.driftRateOfMean
                else:
                    self.mean[i] += self.driftRateOfMean
            measurements = self.createMeasurements(i)
            results = minimize(gaussian, [1, 1], measurements, method='Nelder-Mead')
            self.eMean = results.x[0]

        reward = np.sum(-abs(np.array(self.eMean) - np.array(self.mean))) * np.sum(np.array(self.state))

        if self.periods == 0:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def createMeasurements(self, sensorNo):
        return random.normal(self.mean[sensorNo], self.standardDeviation[sensorNo], self.state[sensorNo] * 10)
