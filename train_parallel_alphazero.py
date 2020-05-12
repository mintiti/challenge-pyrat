from agents.alphazero.alphazerogeneral.pyrat.PyratGame import PyratGame, Symmetries
from agents.alphazero.alphazerogeneral import MCTS2
from agents.alphazero.neural_net import ResidualNet
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
import numpy as np
import random
import time
from agents.alphazero.parallel.mcts import RootParentNode, Node, DEFAULT_MCTS_PARAMS, MCTS
from agents.alphazero.parallel.coach import Coach, NeuralNetWrapper
from agents.alphazero.parallel.buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import ray
args = {
    'numIters': 1000,
    'numEps': 20,  # Number of complete self-play games to simulate during a new iteration.
    'updateThreshold': 0.5790,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'buffer_size': 500000,  # Number of game examples to train the neural networks.
    'arenaCompare': 20,  # Number of games to play during arena play to determine if new net will be accepted.
    'min_buffer_size': 100000,
    'self_play_mcts_params': {
        "temperature": 1,
        "add_dirichlet_noise": True,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 2.5,
        "num_simulations": 600,
        "exploit": False,
        "puct_coefficient": 2,
        "argmax_tree_policy": False,
        'temp_threshold': 20
    },

    'eval_mcts_params': {
        "temperature": 1,
        "add_dirichlet_noise": False,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 2.5,
        "num_simulations": 600,  # number of mcts games to simulate
        "exploit": True,
        "puct_coefficient": 2,
        "argmax_tree_policy": True
    },

    'checkpoint': './temp/3x64/',
    'load_model': False,
    'load_folder_file': ('/dev/models/3x64/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    # NN config
    'residual_blocks': 3,
    'filters': 64,
}



def train():
    nmodel = NeuralNetWrapper(args['filters'], args['residual_blocks'])
    nmodel.load_checkpoint(folder=args['checkpoint'] + 'models/', filename='best.pth.tar')
    env = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(env)
    buffer = ReplayBuffer(args['checkpoint'] + 'examples/', maxlen=args['buffer_size'])
    logger = SummaryWriter()

    c = Coach(pyratgame, nmodel, args, buffer, logger)
    c.fill_buffer()

    c.learn()