import numpy as np
import gym
# from charset_normalizer import detect
import pyomo.common.errors
from gym import spaces
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.env_checker import check_env
from helper_functions import setRandomState, transition_state, create_param_set, solve_noma
from helper_functions import solve_noma2
# from stable_baselines3 import PPO, A2C  # DQN coming soon
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_util import make_vec_env
import logging


class NomaEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, nSensors):
        super(NomaEnv, self).__init__()
        self.M = 2
        self.frame_length = 10
        self.nSensors = nSensors
        n_actions = 3
        self.state = setRandomState(nSensors=self.nSensors, cluster_size=self.M)
        self.action_space = spaces.Discrete(n_actions)
        _l = np.hstack((np.array([0]).repeat(nSensors), np.array([1])))
        _h = np.hstack((np.array([2]).repeat(nSensors), np.array([4])))
        self.observation_space = spaces.Box(low=_l, high=_h,
                                            shape=(self.nSensors + 1,), dtype=np.int16)
        self.cum_rew = 0

    def reset(self):
        self.M = 2
        self.frame_length = 10
        self.state = setRandomState(nSensors=self.nSensors, cluster_size=self.M)
        self.cum_rew = 0
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
        self.state = np.hstack(
            (np.clip(self.state[0:len(self.state) - 1], 0, 2), np.clip(self.state[len(self.state) - 1], 2, 4)))

        # Are we at the end of planning horizon
        done = bool(self.frame_length == 0)

        # Reward as the system serves

        param_set = create_param_set(self.state)
        try:
            reward = solve_noma2(cluster_size=self.M, param_set=param_set)
        except pyomo.common.errors.ApplicationError:
            print(param_set)
            reward = -1

        self.cum_rew += reward

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        transitioned_state = transition_state(self.state, self.M)

        self.frame_length -= 1
        return transitioned_state, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
            # agent is represented as a cross, rest as a dot
        print("State: {0}".format(self.state))

    def close(self):
        pass


# %% DQN
from stable_baselines3 import DQN

# from stable_baselines3.common.evaluation import evaluate_policy

env = NomaEnv(12)
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(3e3))

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)

model.save("models/m1_dqn_3e3")
# %% A2C

from stable_baselines3 import A2C

env = NomaEnv(12)
model = A2C("MlpPolicy", env, verbose=1)
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
model.learn(total_timesteps=int(3e3))
model.save("models/m1_a2c_3e3")

# %%
from stable_baselines3 import PPO

env = NomaEnv(12)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(3e4))
model.save("models/m2_ppo_3e4")

# %%


# %%
import tensorflow as tf

print(tf.__version__)

obs = env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

model_path = "models/dqn_5e4.zip"
model = DQN.load(model_path, env=env)

n = 1
env.reset()
flag = True
while flag and n <= 1000:
    obs = env.observation_space.sample()
    action, _ = model.predict(obs, deterministic=True)
    print("Action: ", action)
    n += 1
    print("Currently at iteration: {0}".format(n))
    if action != 0 or obs[len(obs) - 1] not in [3, 4]:
        print("State: {0}, at iteration {1}"
              " \ncluster size was {2}"
              "\tand the action {3} was taken".format(obs, n, obs[len(obs) - 1], action))
        flag = False

# check_env(env, warn=True)
initial_random_state = setRandomState(4, 2)
model.predict(initial_random_state, deterministic=True)
param_set = create_param_set(initial_random_state)
solve_noma(2, param_set)

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


param_set = {'gains': np.array([4.01527253, 3.68951722, 0.80962714, 0.72630315, 0.7072012,
                    0.58994122, 0.38433832, 0.3090835, 0.22363532, 0.21936055,
                    0.19948185, 0.126238]),
 'scores': [0.3, 0.3, 0.6, 0.3, 0.3, 0.1, 0.3, 0.3, 0.3, 0.3, 0.6, 0.3],
 'bits': {(1, 1): 98.12515292147677, (2, 1): 136.87307897834182, (3, 1): 140.0274026288061, (4, 1): 115.30971314760019,
          (5, 1): 108.872381895524, (6, 1): 125.79438807569262, (7, 1): 142.4566046943769, (8, 1): 103.6554618628379,
          (9, 1): 107.62415922144285, (10, 1): 92.59885158264831,
          (11, 1): 148.85044861559692, (12, 1): 92.03018379898484}}
