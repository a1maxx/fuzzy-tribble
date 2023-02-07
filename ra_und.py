import math
from math import log
import numpy as np
import gym
from gym import spaces

TOTAL_BANDWIDTH = 10 # Mhz
NUMBER_OF_RBS = 50 # RB - resource block
NUMBER_OF_SENB = np.arange(6, 16, 2)
SMALL_CELL_INTER_SITE_DISTANCE = 50 # m
POWER_OF_MGNB = 46 # p_M of macro gnodeB # dBm
POWER_OF_SGNB = 30 # p_n of small gnodeB # dBm
CIRCUIT_POWER_OF_SENB = 6.8 # watt
R0_OF_SUE = 1 # Mbps

SHADOWING_FADE_VARIANCE_OF_SGNB = 10 # dB
EFFECTIVE_THERMAL_NOISE_POWER_SIGMA = -174 # dBm/Hz
MAXIMUM_NUMBER_OF_UE_SMALL_CELL = 30


SIZE_OF_REPLAY_MEMORY = 10000
LEARNING_RATE = 0.1




def path_loss_of_sgnb(d):
    return 140.7+37.6*log(d)

def channel_gain(d):
    pass

def sinr(x_n_m:np.array,n: int):
    N = NUMBER_OF_SENB[n]
    den = sum(x_n_m[i]*POWER_OF_SGNB for i in range(0,N)) + EFFECTIVE_THERMAL_NOISE_POWER_SIGMA**2






class RaUnd(gym.env):
    def __init__(self):
        super(RaUnd, self).__init__()
