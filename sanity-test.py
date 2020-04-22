import gym
import pyrat_env
from stable_baselines.common.env_checker import check_env
import random
from time import sleep

if __name__ == '__main__':
    env = gym.make("PyRatEnv-v0")
    check_env(env)
    # obs = env.reset()
    # env.render()
    # print(obs)
    # print(env.maze)
    #
    # while True:
    #     obs, _, _, _ = env.step((random.randint(0, 3), random.randint(0, 3)))
    #     env.render()
    #     print(obs)
    #     print(env.maze)
    #     sleep(0.3)

    print(env.observation_space)
