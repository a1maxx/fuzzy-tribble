import pandas as pd
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from gym import spaces
import gym
from helper_functions import setRandomState, transition_state, create_param_set, solve_noma
from pyomo.common.timing import TicTocTimer

ACTION_SET = {0: 2, 1: 3, 2: 4}


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
        reward = solve_noma(cluster_size=self.M, param_set=param_set)
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


# %%

# timer = TicTocTimer()
# timer.tic("starting timer")
# timer.toc("Elapsed Time")


env = NomaEnv(12)
model_path = "models/dqn_3e3.zip"
model = DQN.load(model_path, env=env)

# df = pd.DataFrame(columns=['NSD_Static', 'NSD_Dynamic', 'OBJ_Static',
#                            'OBJ_Dynamic', 'CT_Static'
#                                           'CT_Dynamic'])
df = pd.DataFrame(columns=['NSD_Static', 'NSD_Dynamic'])

for i in range(100):
    S = env.reset()
    temp = np.zeros(shape=(2,))
    for j in range(10):
        action, _ = model.predict(S, deterministic=True)
        S, _, _, _ = env.step(action)
        param_set = create_param_set(S)
        temp[0] += solve_noma(2, param_set)
        temp[1] += solve_noma(ACTION_SET[int(action)], param_set)

    df.loc[i] = temp

df.to_pickle("pickles/dqn.pkl")

# %%
env = NomaEnv(12)
model_DQN_path = "models/dqn_3e3.zip"
model_DQN = DQN.load(model_DQN_path, env=env)

model_A2C_path = "models/a2c_3e3.zip"
model_A2C = A2C.load(model_A2C_path, env=env)

model_PPO_path = "models/ppo_3e3.zip"
model_PPO = PPO.load(model_A2C_path, env=env)

df = pd.DataFrame(columns=['Constant', 'DQN', 'A2C', 'PPO'])

for i in range(5):
    S = env.reset()
    temp = np.zeros(shape=(df.shape[1],))
    for j in range(10):
        action_DQN, _ = model_DQN.predict(S, deterministic=True)
        action_A2C, _ = model_A2C.predict(S, deterministic=True)
        action_PPO, _ = model_PPO.predict(S, deterministic=True)

        S, _, _, _ = env.step(action_DQN)
        param_set = create_param_set(S)
        temp[0] += solve_noma(2, param_set)
        temp[1] += solve_noma(ACTION_SET[int(action_DQN)], param_set)
        temp[2] += solve_noma(ACTION_SET[int(action_A2C)], param_set)
        temp[3] += solve_noma(ACTION_SET[int(action_PPO)], param_set)

    df.loc[i] = temp

df.to_pickle("pickles/comparison_1.pkl")

# %%

import pandas as pd
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from helper_functions import setRandomState, transition_state, \
    create_param_set, solve_noma2

env = NomaEnv(12)
model_DQN_path = "models/m1_dqn_3e3.zip"
model_DQN = DQN.load(model_DQN_path, env=env)

model_A2C_path = "models/m1_a2c_3e3.zip"
model_A2C = A2C.load(model_A2C_path, env=env)

model_PPO_path = "models/m1_ppo_3e3.zip"
model_PPO = PPO.load(model_A2C_path, env=env)

df = pd.DataFrame(columns=['Constant(2)', 'Constant(3)', 'Constant(4)',
                           'DQN', 'A2C', 'PPO'])
ACTION_SET = {0: 2, 1: 3, 2: 4}

for i in range(50):
    S = env.reset()
    temp = np.zeros(shape=(df.shape[1],))
    for j in range(10):
        action_DQN, _ = model_DQN.predict(S, deterministic=True)
        action_A2C, _ = model_A2C.predict(S, deterministic=True)
        action_PPO, _ = model_PPO.predict(S, deterministic=True)

        S, _, _, _ = env.step(action_DQN)
        param_set = create_param_set(S)
        temp[0] += solve_noma2(2, param_set)
        temp[1] += solve_noma2(3, param_set)
        temp[2] += solve_noma2(4, param_set)
        temp[3] += solve_noma2(ACTION_SET[int(action_DQN)], param_set)
        temp[4] += solve_noma2(ACTION_SET[int(action_A2C)], param_set)
        temp[5] += solve_noma2(ACTION_SET[int(action_PPO)], param_set)

    df.loc[i] = temp

df.to_pickle("pickles/comparison_4.pkl")

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams["figure.dpi"] = 300

df = pd.read_pickle("pickles/comparison_4.pkl")

df2 = df.sub(df.mean(axis=1), axis=0)
df3 = pd.DataFrame(df2['Constant(2)'].sort_values()).reset_index(drop=True)
for i in df2.columns:
    if i != 'Constant(2)':
        df3 = df3.join(df2[i].sort_values().reset_index(drop=True))

# df3 = df3.drop('Constant(2)',axis=1)

melted_df = pd.melt(df3.iloc[:, [2, 3, 4, 5]])
df4 = df3.iloc[:, [2, 3, 4, 5]]
melted_df['Index'] = np.tile(np.arange(1, len(df4) + 1), len(df4.columns))

g = sns.relplot(melted_df, kind='line', x='Index', y="value", hue="variable")
g._legend.remove()
plt.legend(title='', fontsize='10', title_fontsize='14')
plt.xlabel("Rank")
plt.ylabel("Objective Function")
plt.show()

g = sns.lmplot(melted_df, x='Index', y="value",
               hue="variable", markers=["o", "x"])
g._legend.set_title(" ")
new_labels = ['DQN', 'Constant']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

plt.xlabel(r'$\theta$ ' + "Time Step")
plt.ylabel("Objective Function")
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

df = pd.read_pickle("pickles/comparison_2.pkl")
melted_df = pd.melt(df, value_vars=['NSD_Static', 'NSD_Dynamic'])
melted_df['Index'] = np.tile(np.arange(100), 2)
g = sns.relplot(melted_df, kind='line', x='Index', y="value", hue="variable")
g._legend.set_title(" ")
plt.xlabel("Time Step")
plt.ylabel("Objective Function")
plt.show()

g = sns.lmplot(melted_df, x='Index', y="value",
               hue="variable", markers=["o", "x"])
g._legend.set_title(" ")
new_labels = ['DQN', 'Constant']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

plt.xlabel(r'$\theta$ ' + "Time Step")
plt.ylabel("Objective Function")
plt.show()
