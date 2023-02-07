import gym
from gym import spaces, Env
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
import time
from stable_baselines.common.env_checker import check_env


class OurCustomEnv(gym.Env):
    def __init__(self, sales_function, obs_range, action_range):
        super(OurCustomEnv, self).__init__()

        self.budget = 1000  # fixing the budget
        self.sales_function = sales_function  # calculating sales based on spends

        # we create an observation space with predefined range
        self.observation_space = spaces.Box(low=obs_range[0], high=obs_range[1],
                                            dtype=np.float32)
        self.state = [0, 0]
        # similar to observation, we define action space
        self.action_space = spaces.Box(low=action_range[0], high=action_range[1],
                                       dtype=np.float32)

    def step(self, action):
        self.budget = 1000

        self.state *= 1000

        self.state[0] = int(np.min([1000, self.state[0] + action[0] * 500]))
        self.state[0] = int(np.max([self.state[0], 0]))

        self.state[1] = int(np.min([1000, self.state[1] + action[1] * 500]))
        self.state[1] = int(np.max([self.state[1], 0]))

        if np.sum(self.state) > 0:
            self.budget -= np.sum(self.state)
        else:
            self.budget = np.sum(self.state)

        if self.budget < 0:
            reward = -1
            done = True  # Condition for completion of episode
        else:
            reward = self.sales_function(self.state)
            done = True

        info = {}

        return self.state / 1000, reward, done, info

    def reset(self):
        self.budget = 1000
        self.state = np.random.uniform(low=0, high=1, size=2)
        return self.state


#%%

# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import ACKTR
#
# obs_range = np.array([[0, 0], [1, 1]])
# act_range = np.array([[-1, -1], [1, 1]])
#
# def sales_function(a):
#     return np.random.normal(10, 0.01) * a[0] + np.random.normal(7, 0.01) * a[1]
#
#
# env = OurCustomEnv(sales_function, obs_range, act_range)
# check_env(env,warn=True)
#
# env = make_vec_env(lambda: env, n_envs=1)
# model = ACKTR('MlpPolicy', env, verbose=1).learn(5000)
#
# filename = 'CustomENv' +  time.ctime()[14:16]
# figure_file = 'C:\\Users\\Administrator\\Desktop\\' + filename + '.png'
#
# obs = env.reset()
# n_steps = 1000
# score_history = []
# for step in range(n_steps):
#   score = 0
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   score += reward
#   score_history.append(score)
#
# x = [i + 1 for i in range(n_steps)]
# plot_learning_curve(x, score_history, figure_file)
#
#
#
# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())


# %%

# agent.load_models()
np.random.seed(0)
if __name__ == '__main__':
    # env = gym.make('LunarLanderContinuous-v2')
    obs_range = np.array([[0, 0], [1, 1]])
    act_range = np.array([[-1, -1], [1, 1]])


    def sales_function(a):
        return np.random.normal(10, 0) * a[0] + np.random.normal(7, 0) * a[1]


    env = OurCustomEnv(sales_function, obs_range, act_range)


    # agent = Agent(alpha=0.0001, beta=0.001,
    #               input_dims=env.observation_space.shape, tau=0.001,
    #               batch_size=64, fc1_dims=400, fc2_dims=300,
    #               n_actions=env.action_space.shape[0])

    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=600, fc2_dims=400,
                  n_actions=env.action_space.shape[0])

    n_games = 500
    filename = 'CustomENv' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + '_games' + time.ctime()[14:16]
    figure_file = 'C:\\Users\\Administrator\\Desktop\\' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)


observation = env.reset()
agent.actor.eval()
state = T.tensor([observation], dtype=T.float).to(agent.actor.device)
mu = agent.actor.forward(state).to(agent.actor.device)
