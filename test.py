from agents.alphazero.alphazerogeneral.pyrat.PyratGame import PyratGame, Symmetries
from agents.alphazero.alphazerogeneral import MCTS2
from agents.alphazero.neural_net import ResidualNet
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
import numpy as np
import random

if __name__ == '__main__':
    env = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(env)
    obs = pyratgame.getInitBoard()
    player = -1

    nn= ResidualNet(64,3)
    nn.load_checkpoint(folder="temp/3x64", filename='best.pth.tar')




