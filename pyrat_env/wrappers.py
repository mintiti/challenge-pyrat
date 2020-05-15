from gym.core import ObservationWrapper, RewardWrapper
import gym.spaces as spaces
import numpy as np
from gym.spaces import Box, Dict, Discrete, Tuple, MultiBinary, MultiDiscrete


def flatdim2(space):
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return int(sum([flatdim2(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim2(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten2(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float16).flatten()
    elif isinstance(space, Discrete):
        return [x]
    elif isinstance(space, Tuple):
        return np.concatenate([flatten2(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        return np.concatenate([flatten2(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, MultiBinary):
        return np.asarray(x).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).flatten()
    else:
        raise NotImplementedError


class FlattenObservationMatrices(ObservationWrapper):
    def __init__(self, env):
        super(FlattenObservationMatrices, self).__init__(env)

        flatdim = flatdim2(env.observation_space)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(flatdim,), dtype=np.float16)

    def observation(self, observation):
        return flatten2(self.env.observation_space, observation)


class MatricizePositions(ObservationWrapper):
    def __init__(self, env):
        super(MatricizePositions, self).__init__(env)

        obs_dim = (9, 21, 15)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=obs_dim, dtype=np.float16)

    def observation(self, observation):
        ret = []
        ret.append(observation['Maze_left'])
        ret.append(observation['Maze_up'])
        ret.append(observation['Maze_right'])
        ret.append(observation['Maze_down'])
        ret.append(observation['pieces_of_cheese'])
        player1_score_matrix = np.full_like(observation['pieces_of_cheese'], observation['player1_score'], dtype= np.float16)
        player2_score_matrix = np.full_like(observation['pieces_of_cheese'], observation['player2_score'], dtype= np.float16)
        ret.append(player1_score_matrix)
        ret.append(player2_score_matrix)

        # player location matrices
        player1 = np.zeros(ret[0].shape, dtype= np.float16)
        player2 = np.zeros(ret[0].shape, dtype= np.float16)

        player1[observation['player1_location']] = 1
        player2[observation['player2_location']] = 1

        ret.append(player1)
        ret.append(player2)

        return np.array(ret, dtype= np.float16)


class FinalReward(RewardWrapper):
    def __init__(self, env):
        super(FinalReward, self).__init__(env)
        self.cumulated_reward = 0

    def reward(self, reward):
        self.cumulated_reward += reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.reward(reward)
        if done:
            reward = self.cumulated_reward / abs(self.cumulated_reward)
        else:
            reward = 0

        return observation, reward, done, info


def AlphaZero(env):
    return FinalReward(MatricizePositions(env))
