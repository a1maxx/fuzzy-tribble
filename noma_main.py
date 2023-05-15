from stable_baselines3.common.env_checker import check_env
import custom_env1


env = custom_env1.NomaEnv(4)
check_env(env, warn=True)