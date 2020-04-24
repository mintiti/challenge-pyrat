from agents.alphazero.alphazerogeneral.Coach2 import  Coach2
from agents.alphazero.alphazerogeneral.pyrat.PyratGame import PyratGame
from agents.alphazero.neural_net import ResidualNet
from agents.alphazero.alphazerogeneral.utils import *
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero

args = dotdict({
    'numIters': 1000,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 200,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/2x128/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    # NN config
    'residual_blocks' : 6,
    'filters' : 128,


})

if __name__ == '__main__':
    env = PyratEnv(symmetry= False, mud_density=0,start_random= True)
    env = AlphaZero(env)
    game = PyratGame(env)
    nn = ResidualNet(args.filters,args.residual_blocks)

    c= Coach2(game, nn,args)

    c.learn()