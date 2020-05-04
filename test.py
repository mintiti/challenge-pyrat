from agents.alphazero.alphazerogeneral.pyrat.PyratGame import PyratGame, Symmetries
from agents.alphazero.alphazerogeneral import MCTS2
from agents.alphazero.neural_net import ResidualNet
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
import numpy as np
import random
import time
from agents.alphazero.parallel.mcts import RootParentNode, Node, DEFAULT_MCTS_PARAMS, MCTS
class NeuralNetWrapper :
    def __init__(self, nb_filters, nb_residual_blocks):
        self.model = ResidualNet(nb_filters, nb_residual_blocks)

    def compute_priors_and_value(self,obs):
        return self.model.predict(obs)

if __name__ == '__main__':
    game = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(game)
    obs = pyratgame.getInitBoard()
    # player = -1
    #
    model = NeuralNetWrapper(64, 3)
    model.model.load_checkpoint(folder="temp/3x64", filename='best.pth.tar')
    mcts = MCTS(model, DEFAULT_MCTS_PARAMS)
    root_parent  = RootParentNode(pyratgame)

    DEFAULT_MCTS_PARAMS["num_simulations"] = 10

    root_node = Node(action= None, obs = obs[:10], done= False, reward=0, state= obs, player=1, mcts= mcts, parent= root_parent)

    mcts.compute_action(root_node)







