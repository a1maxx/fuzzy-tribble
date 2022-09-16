import CustomEnv as Ce



def main():
    env = Ce.CustomEnv()
    episodes = 20  # 20 shower episodes

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward

        print('Episode:{} Score:{}'.format(episode, score))

    state = env.observation_space.shape



if __name__ == "__main__":
    main()
