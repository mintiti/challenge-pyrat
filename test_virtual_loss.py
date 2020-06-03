from agents.alphazero.pyrat.PyratGame import PyratGame
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
import time

from agents.alphazero.sequential.coach import Coach
from agents.alphazero.ray_training.ray_coach import InferenceActor, LearningActor, SelfPlayActor, NeuralNetWrapper
from agents.alphazero.buffer import ReplayBuffer
from agents.alphazero.sequential.arena import Arena
from torch.utils.tensorboard import SummaryWriter
import ray
from ray.util import ActorPool
from agents.alphazero.virtual_loss.mcts import MCTS, RootParentNode, Node
from train_parallel_alphazero import create_best_net
args = {
    'run_name': "t3-3x64",
    'numIters': 1000, # number of self-play iterations to play
    'numEps': 60,  # Number of complete self-play games to simulate during a new iteration.
    'updateThreshold': 0.5790,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'buffer_size': 500000,  # Number of game examples to train the neural networks.
    'arenaCompare': 30,  # Number of games to play during arena play to determine if new net will be accepted.
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
        'temp_threshold': 20,
        'virtual_loss_batch_size' : 16
    },

    'eval_mcts_params': {
        "temperature": 1,
        "add_dirichlet_noise": False,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 2.5,
        "num_simulations": 600,  # number of mcts games to simulate
        "exploit": True,
        "puct_coefficient": 2,
        "argmax_tree_policy": True,
        'virtual_loss_batch_size': 8
    },

    'checkpoint': './temp/t3-3x64/',
    'load_model': True,
    'load_folder_file': ('/dev/models/3x64/', 'best.pth.tar'),

    # NN config
    'residual_blocks': 3,
    'filters': 64,

    # Worker config
    'num_models': 3,
    'num_workers_model': 1,

}

if __name__ == '__main__':
    pmodel = create_best_net()
    env = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(env)
    times = []
    mcts = MCTS(pmodel, args['self_play_mcts_params'])
    board = pyratgame.getInitBoard()

    root_parent = RootParentNode(pyratgame)  # dummy node
    current_node = Node(action=None, obs=board[:9], done=False, reward=0, state=board, player=1, mcts=mcts,
                        parent=root_parent)
    for i in range(100):

        start = time.time()
        probs, action, next_node = mcts.tree_search(current_node)
        current_node = mcts.make_move(current_node, action)
        end = time.time()
