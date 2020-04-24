from agents.alphazero.alphazerogeneral.pyrat.PyratGame import PyratGame
from agents.alphazero.alphazerogeneral import MCTS2
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
import numpy as np
import random

if __name__ == '__main__':
    env = AlphaZero(PyratEnv())
    pyratgame = PyratGame(env)
    obs = pyratgame.getInitBoard()
    player = 1

    for i in range(1500):
        p1_action = random.randint(0,3)
        p2_action = random.randint(0,3)
        obs, player = pyratgame.getNextState(obs, player, p1_action, p2_action)

    print("number of turns",obs[10])




