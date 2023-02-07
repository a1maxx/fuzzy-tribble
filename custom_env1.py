import numpy as np
import gym
from gym import spaces
from abstract_NOMA_model import solve_noma
from stable_baselines.common.env_checker import check_env
from helper_functions import setRandomState, transition_state


class NomaEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, nSensors):
        super(NomaEnv, self).__init__()
        self.M = 2
        self.frame_length = 10
        self.nSensors = nSensors
        n_actions = 3
        self.state = setRandomState(nSensors=self.nSensors, cluster_size=self.M)
        self.action_space = spaces.Discrete(n_actions)
        _l = np.hstack((np.array([0]).repeat(nSensors),np.array([1])))
        _h = np.hstack((np.array([2]).repeat(nSensors), np.array([4])))
        self.observation_space = spaces.Box(low=_l, high=_h,
                                            shape=(self.nSensors + 1,), dtype=np.int16)

    def reset(self):
        self.M = 2
        self.frame_length = 10
        self.state = setRandomState(nSensors=self.nSensors, cluster_size=self.M)
        return self.state

    def step(self, action):
        if action == 0:
            self.M = 2
        elif action == 1:
            self.M = 3
        elif action == 2:
            self.M = 4
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.state = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = bool(self.frame_length == 0)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = solve_noma(self.M)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        transitioned_state = update_pris(self.state, self.M)

        self.frame_length -= 1
        return transitioned_state, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass


# %%
env = NomaEnv(4)
check_env(env, warn=True)

states = ["state_1", "state_2", "state_3"]

# Possible sequences of events
transitionName = [["s_11", "s_12", "s_13"], ["s_21", "s_22", "s_23"], ["s_31", "s_32", "s_33"]]

# Probabilities matrix (transition matrix)
transitionMatrix = [[0.2, 0.6, 0.2], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]
dict_pris = {1: 0.1, 2: 0.3, 3: 0.6}
nSensors = 4
dict_sensors = {i: [0, 0] for i in range(0, nSensors)}


def setRandomState(nSensors, cluster_size):
    dict_sensors = {i: [0, 0] for i in range(0, nSensors)}
    for i in range(0, nSensors):
        dict_sensors[i][0] = np.random.randint(low=1, high=3)
        dict_sensors[i][1] = dict_pris[dict_sensors[i][0]]

    return np.hstack((np.array([dict_sensors[i][1] for i in dict_sensors]), np.array([cluster_size])))


def update_pris():
    for i in range(0, nSensors):
        change = np.random.choice(transitionName[dict_sensors[i][0]], replace=True,
                                  p=transitionMatrix[dict_sensors[i][0]])
        dict_sensors[i][0] = int(change[-1])
        dict_sensors[i][1] = dict_pris[dict_sensors[i][0]]

        return dict_sensors
