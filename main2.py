from CustomEnv2 import CustomEnv2
from Helpers import build_model2, build_agent
from tensorflow import keras
from keras.optimizers import Adam
import numpy as np


def main():
    env = CustomEnv2()
    episodes = 20  # 20 shower episodes/
    states = env.observation_space.shape
    actions = env.action_space.shape
    # model = build_model2(states, actions)
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, activation='relu', input_shape=(1, 3)))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(actions, activation='linear'))
    dqn = build_agent(model, actions)
    model.summary()
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=10, visualize=False, verbose=1)

    a = []
    for i in range(100):
        a.append(dqn.forward([np.random.randint(0,100) for _ in range(3)]))

    results = dqn.test(env, nb_episodes=150, visualize=False)
    print(np.mean(results.history['episode_reward']))


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

    # state = env.observation_space.shape


if __name__ == "__main__":
    main()
