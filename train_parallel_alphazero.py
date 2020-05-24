from agents.alphazero.alphazerogeneral.pyrat.PyratGame import PyratGame
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
import time
from agents.alphazero.parallel.mcts import  MCTS
from agents.alphazero.parallel.coach import Coach, NeuralNetWrapper, SelfPlayActor, InferenceActor, LearningActor
from agents.alphazero.parallel.buffer import ReplayBuffer
from agents.alphazero.parallel.arena import Arena
from torch.utils.tensorboard import SummaryWriter
import ray
from ray.util import ActorPool

args = {
    'numIters': 1000,
    'numEps': 60,  # Number of complete self-play games to simulate during a new iteration.
    'updateThreshold': 0.5790,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'buffer_size': 500000,  # Number of game examples to train the neural networks.
    'arenaCompare': 30,  # Number of games to play during arena play to determine if new net will be accepted.
    'min_buffer_size': 100000,
    'self_play_mcts_params': {
        "temperature": 1,
        "add_dirichlet_noise": True,
        "dirichlet_epsilon": 0,
        "dirichlet_noise": 2.5,
        "num_simulations": 600,
        "exploit": False,
        "puct_coefficient": 3,
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
        "puct_coefficient": 3,
        "argmax_tree_policy": True
    },

    'checkpoint': './temp/t1-3x64/',
    'load_model': True,
    'load_folder_file': ('/dev/models/3x64/', 'best.pth.tar'),

    # NN config
    'residual_blocks': 3,
    'filters': 64,

    # Worker config
    'num_models': 3,
    'num_workers_model': 1,

}


def get_training_samples(model_creator, game, args):
    # create the actor pool
    start = time.time()
    nb_models = args['num_models']
    nb_workers_per_model = args['num_workers_model']
    models = [model_creator() for _ in range(nb_models)]
    actors = []
    for model in models:
        for _ in range(nb_workers_per_model):
            actors.append(SelfPlayActor.remote(model, args['self_play_mcts_params'], game))
    actor_pool = ActorPool(actors)

    # Launch the games
    nb_games = args['numEps']
    m = actor_pool.map_unordered(lambda a, v: a.play.remote(v)[0], [i + 1 for i in range(nb_games)])

    results = [res for res in m]
    print(f"played {nb_games} in {time.time() - start}s")
    return results


def train():
    nmodel = create_latest_net()
    env = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(env)
    buffer = ReplayBuffer(args['checkpoint'] + 'examples/', maxlen=args['buffer_size'])
    buffer.load()
    print(len(buffer))
    logger = SummaryWriter(log_dir="./runs/t1-3x64--Deepmind")
    c = Coach(pyratgame, nmodel, args, buffer, logger)
    #c.fill_buffer()

    c.learn()

def learn_on_buffer(buffer):
    learning_model = create_learning_model()
    infos = learning_model.train.remote(buffer.storage)
    infos = ray.get(infos)
    ray.get(learning_model.save_checkpoint.remote(folder=args['checkpoint'] + 'models/', filename='temp.pth.tar'))
    return infos

def create_best_net_inference():
    pmodel = InferenceActor.remote(args['filters'], args['residual_blocks'])
    ray.get(pmodel.load_checkpoint.remote(folder=args['checkpoint'] + 'models/', filename='best.pth.tar'))
    return pmodel

def create_best_net():
    nmodel = NeuralNetWrapper(args['filters'], args['residual_blocks'])
    nmodel.load_checkpoint(folder=args['checkpoint'] + 'models/', filename='best.pth.tar')
    return nmodel

def create_latest_net():
    nmodel = NeuralNetWrapper(args['filters'], args['residual_blocks'])
    nmodel.load_checkpoint(folder=args['checkpoint'] + 'models/', filename='temp.pth.tar')
    return nmodel

def create_learning_model():
    learner = LearningActor.remote(args['filters'], args['residual_blocks'])
    ray.get(learner.load_checkpoint.remote(folder=args['checkpoint'] + 'models/', filename='temp.pth.tar'))
    return learner

def evaluate(pyratgame,buffer):
    pnet = create_best_net()
    nnet = create_latest_net()
    eval_params = args['eval_mcts_params']
    pmcts = MCTS(pnet, eval_params)
    nmcts = MCTS(nnet, eval_params)
    arena = Arena(pmcts, nmcts, pyratgame)

    pWon, nWon, draws = arena.playGames(args['arenaCompare'])
    del arena
    del pmcts
    del nmcts
    print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nWon, pWon, draws))
    if pWon + nWon == 0 or float(nWon) / (pWon + nWon) < args['updateThreshold']:
        print('REJECTING NEW MODEL')
    else:
        print('ACCEPTING NEW MODEL')
        nnet.save_checkpoint(folder=args['checkpoint'] + 'models/',
                             filename='checkpoint_' + buffer.get_n_iters() + '.pth.tar')
        nnet.save_checkpoint(folder=args['checkpoint'] + 'models/', filename='best.pth.tar')
        nnet.clear_cache()
    del pnet
    del nnet



def train_parallel():
    # Preparation
    # Load the buffer
    logger = SummaryWriter(log_dir="./runs/t1-3x64--Deepmind")
    buffer = ReplayBuffer(args['checkpoint'] + 'examples/', maxlen=args['buffer_size'])
    buffer.load()
    print(f"Buffer loaded. Buffer size {len(buffer)}")
    # Create the game
    env = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(env)
    for i in range(args['numIters']):
        print('------ITER ' + str(i +1) + '------')
        # get the data
        print(f"Starting {args['numEps']} self-play games...")
        train_examples = get_training_samples(create_best_net_inference, pyratgame, args)
        # store the data
        for ep_samples in train_examples:
            buffer.store(ep_samples)

        buffer.save()

        # train the net on the data
        infos = learn_on_buffer(buffer)

        # Book keeping
        global_step = buffer.get_n_iters()
        logger.add_scalars("Training losses", {"Value loss": infos['value_loss'],
                                               "Policy loss": infos['policy_loss'],
                                               "Total_loss": infos['value_loss'] + infos['policy_loss']}, global_step)


        # Run evaluation games
        evaluate(pyratgame,buffer)
        
        time.sleep(1)


if __name__ == '__main__':
    train()
    #
    # ray.init(num_gpus=1)
    # train_parallel()
