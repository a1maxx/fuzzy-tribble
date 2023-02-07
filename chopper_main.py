from ChopperScape import ChopperScape
from ddpg_torch import Agent
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from IPython import display
#
# env = ChopperScape()
# obs = env.reset()
#
# while True:
#     # Take a random action
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#
#     # Render the game
#     env.render()
#     if done == True:
#         break
#
# env.close()


env = ChopperScape()
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[32400], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=6)

np.random.seed(0)

score_history = []
for i in range(1000):
    obs = env.reset()
    obs = obs.flatten()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))