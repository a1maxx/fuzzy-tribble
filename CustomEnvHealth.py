state = {
    'alertness': 0,
    'hypertension': 0,
    'intoxication': 0,
    'time_since_slept': 10,
    'time_elapsed': 0,
    'work_done': 0
}

import numpy as np


def wakeup(state):
    state['alertness'] = np.random.uniform(0.7, 1.3)
    state['time_since_slept'] = 0


def drink_coffee(state):
    state['alertness'] += np.random.uniform(0, 1)
    state['hypertension'] += np.random.uniform(0, 0.3)


def drink_beer(state):
    state['alertness'] -= np.random.uniform(0, 0.5)
    state['hypertension'] += np.random.uniform(0, 0.3)
    # source https://attorneydwi.com/b-a-c-per-drink/
    state['intoxication'] += np.random.uniform(0.01, 0.03)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heart_attack_risk(hypertension, heart_attack_proclivity=0.5):
    return heart_attack_proclivity * sigmoid(hypertension - 6)


# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))
# xspace = np.linspace(0, 10, 100)
# plt.xlabel('hypertension')
# plt.ylabel('heart attack risk')
# plt.plot(xspace, heart_attack_risk(xspace))
# plt.show()


def heart_attack_occured(state, heart_attack_proclivity=0.5):
    return np.random.uniform(0, 1) < heart_attack_risk(state['hypertension'], heart_attack_proclivity)


heart_attack_occured(state)


def alertness_decay(time_since_slept):
    return sigmoid((time_since_slept - 40) / 10)


# plt.figure(figsize=(12,8))
# xspace = np.linspace(0, 100, 100)
# plt.xlabel('halfhours since slept')
# plt.ylabel('alertness decay')
# plt.plot(xspace, alertness_decay(xspace))
#
# plt.show()

decay_rate = 0.97
half_life = decay_rate ** 24
half_life


def half_hour_passed(state):
    state['alertness'] -= alertness_decay(state['time_since_slept'])
    state['hypertension'] = decay_rate * state['hypertension']
    state['intoxication'] = decay_rate * state['intoxication']
    state['time_since_slept'] += 1
    state['time_elapsed'] += 1


half_hour_passed(state)
state


def crippling_anxiety(alertness):
    return sigmoid(alertness - 3)


# plt.figure(figsize=(12, 8))
# xspace = np.linspace(0, 10, 100)
# plt.xlabel('alertness')
# plt.ylabel('crippling anxiety')
# plt.plot(xspace, crippling_anxiety(xspace))
# plt.show()


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def ballmer_function(intoxication):
    return sigmoid((0.05 - intoxication) * 50) + 2 * gaussian(intoxication, 0.135, 0.005)


# plt.figure(figsize=(12,8))
# xspace = np.linspace(0, 0.26, 1000)
# plt.xlabel('blood alcohol content')
# plt.ylabel('programming skill')
# plt.plot(xspace, ballmer_function(xspace))
# plt.show()

def productivity(state):
    p = 1
    p *= state['alertness']
    p *= 1 - crippling_anxiety(state['alertness'])
    p *= ballmer_function(state['intoxication'])
    return p


def work(state):
    state['work_done'] += productivity(state)
    half_hour_passed(state)


def do_nothing(state):
    pass


def sleep(state):
    """Have 16 half-hours of healthy sleep"""
    for hh in range(16):
        half_hour_passed(state)
    wakeup(state)


import gym

actions = [do_nothing, drink_coffee, drink_beer, sleep]
heartpole_action_space = gym.spaces.Discrete(len(actions))

observations = ['alertness', 'hypertension', 'intoxication',
                'time_since_slept', 'time_elapsed', 'work_done']


def make_heartpole_obs_space():
    lower_obs_bound = {
        'alertness': - np.inf,
        'hypertension': 0,
        'intoxication': 0,
        'time_since_slept': 0,
        'time_elapsed': 0,
        'work_done': - np.inf
    }
    higher_obs_bound = {
        'alertness': np.inf,
        'hypertension': np.inf,
        'intoxication': np.inf,
        'time_since_slept': np.inf,
        'time_elapsed': np.inf,
        'work_done': np.inf
    }

    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    return gym.spaces.Box(low, high, shape)


class HeartPole(gym.Env):
    def __init__(self, heart_attack_proclivity=0.5, max_steps=1000):
        self.actions = actions
        self.observations = observations
        self.action_space = heartpole_action_space
        self.observation_space = make_heartpole_obs_space()
        self.heart_attack_proclivity = heart_attack_proclivity
        self.log = ''
        self.max_steps = max_steps

    def observation(self):
        return np.array([self.state[o] for o in self.observations])

    def reset(self):
        self.state = {
            'alertness': 0,
            'hypertension': 0,
            'intoxication': 0,
            'time_since_slept': 0,
            'time_elapsed': 0,
            'work_done': 0
        }
        self.steps_left = self.max_steps

        wakeup(self.state)
        return self.observation()

    def step(self, action):
        if self.state['time_elapsed'] == 0:
            old_score = 0
        else:
            old_score = self.state['work_done'] / self.state['time_elapsed']

        # Do selected action
        self.actions[action](self.state)
        self.log += f'Chosen action: {self.actions[action].__name__}\n'

        # Do work
        work(self.state)

        new_score = self.state['work_done'] / self.state['time_elapsed']

        reward = new_score - old_score

        if heart_attack_occured(self.state, self.heart_attack_proclivity):
            self.log += f'HEART ATTACK\n'

            # We would like to avoid this
            reward -= 100

            # A heart attack is like purgatory - painful, but cleansing
            # You can tell I am not a doctor
            self.state['hypertension'] = 0

        self.log += str(self.state) + '\n'

        self.steps_left -= 1
        done = (self.steps_left <= 0)

        return self.observation(), reward, done, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''


heartpole = HeartPole()
observation = heartpole.reset()
cups = 0

while 'HEART ATTACK' not in heartpole.log:
    observation, reward, done, _ = heartpole.step(1)
    cups += 1

print(cups)


from stable_baselines3 import DQN

model = DQN("MlpPolicy", heartpole, verbose=1, policy_kwargs={'net_arch': [16,16]})

model.learn(total_timesteps=500000, log_interval=10)

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, heartpole, n_eval_episodes=5)
print(f'Reward per episode {mean_reward}±{std_reward}')


obs = heartpole.reset()
total_reward = 0
for step in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = heartpole.step(action)
    heartpole.render()