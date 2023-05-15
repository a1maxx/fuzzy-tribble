# import custom_env1
# from CustomEnv import CustomEnv
# from Helpers import build_model, build_agent
# from tensorflow import keras
# from keras.optimizers import Adam
# import numpy as np
# import numpy as np
# import gym
# from gym import spaces
# from stable_baselines.common.env_checker import check_env
from helper_functions import setRandomState, transition_state, create_param_set, solve_noma
import CustomEnv

def main():
    # env = CustomEnv()
    # episodes = 20  # 20 shower episodes
    # states = env.observation_space.shape
    # actions = env.action_space.n
    # model = build_model(states, actions)
    # model.summary()
    # dqn = build_agent(model, actions)
    # dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    # dqn.fit(env, nb_steps=60000, visualize=False, verbose=1)
    #
    # results = dqn.test(env, nb_episodes=150, visualize=False)
    # print(np.mean(results.history['episode_reward']))

    # env = custom_env1.NomaEnv(4)
    # check_env(env, warn=True)
    initial_random_state = setRandomState(12, 2)
    param_set = create_param_set(initial_random_state)
    print(solve_noma(2, param_set))

    # for episode in range(1, episodes + 1):
    #     state = env.reset()
    #     done = False
    #     score = 0
    #
    #     while not done:
    #         action = env.action_space.sample()
    #         n_state, reward, done, info = env.step(action)
    #         score += reward
    #
    #     print('Episode:{} Score:{}'.format(episode, score))
    #
    # state = env.observation_space.shape


if __name__ == "__main__":
    main()
