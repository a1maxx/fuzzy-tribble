import gym
from gym import spaces, Env
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OurCustomEnv(gym.Env):
    def __init__(self, sales_function, obs_range, action_range):
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
        self.budget -= np.sum(action)  # remaining budget will be old budget-sum of both spends

        self.state[0] += action[0]
        self.state[1] += action[1]

         # gives total sales based on spends
        reward = 0

        if self.budget <= 0:
            reward = self.budget * 1e3
            done = True  # Condition for completion of episode
        else:
            reward = self.sales_function(self.state)
            done = True

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.budget = 1000
        self.state = np.random.randint(low=0,high=1000, size = 2)
        return self.state


class ReplayBuffer():

    def __init__(self, max_size, input_shape=2, n_actions=2):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta):
        super(CriticNetwork, self).__init__()
        self.input_dims = 2  # fb, insta
        self.fc1_dims = 256  # hidden layers
        self.fc2_dims = 256  # hidden layers
        self.n_actions = 2  # fb, insta
        self.fc1 = nn.Linear(2 + 2, self.fc1_dims)  # state + action
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        q1 = self.q1(q1_action_value)
        return q1


class ActorNetwork(nn.Module):
    def __init__(self, alpha):
        super(ActorNetwork, self).__init__()
        self.input_dims = 2
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.n_actions = 2
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        # fixing each agent between 0 and 1 and transforming each action in env
        # mu = T.sigmoid(self.mu(prob))
        return mu


class Agent(object):
    def __init__(self, env, batch_size, alpha=0.01, beta=0.01, input_dims=2, tau=0.001, gamma=0.99,
                 n_actions=2, max_size=1000000):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size
        self.actor = ActorNetwork(alpha)
        self.critic = CriticNetwork(beta)
        self.target_actor = ActorNetwork(alpha)
        self.target_critic = CriticNetwork(beta)

        self.scale = 1.0
        self.noise = np.random.normal(scale=self.scale, size=(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise,
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)


obs_range = np.array([[0, 0], [1000, 1000]])
act_range = np.array([[0, 0], [1000, 1000]])


def sales_function(a):
    return np.random.normal(10, 1) * a[0] + np.random.normal(7, 6) * a[1]


env = OurCustomEnv(sales_function, obs_range, act_range)

# %%
agent = Agent(alpha=0.000025, beta=0.00025, tau=0.001, env=env,
              batch_size=64, n_actions=2)

score_history = []
for i in range(10000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)


