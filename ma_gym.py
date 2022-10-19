import gym
import time

env = gym.make('ma_gym:Checkers-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    time.sleep(5)
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)


env = gym.make("MountainCar-v0")
state = env.reset()

done = False
while not done:
    action = 2  # always go right!
    env.step(action)
    env.render(mode="human")

env.close()