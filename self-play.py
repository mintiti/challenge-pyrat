import time

from torch.utils.tensorboard import SummaryWriter

from agents.alphazero.pyrat.PyratGame import PyratGame
from agents.alphazero.buffer import ReplayBuffer
from agents.alphazero.sequential.coach import Coach
from agents.alphazero.ray_training.ray_coach import NeuralNetWrapper
from pyrat_env.envs import PyratEnv
from pyrat_env.wrappers import AlphaZero
from train_parallel_alphazero import args


def self_play():
    env = AlphaZero(PyratEnv(symmetry=False, mud_density=0, start_random=True, target_density=0))
    pyratgame = PyratGame(env)
    buffer = ReplayBuffer(args['checkpoint'] + 'examples/', maxlen=args['buffer_size'])

    logger = SummaryWriter(log_dir="./runs/t1-3x64--Deepmind")
    i = 1
    while True:
        print(f"Starting self play game {i}")
        start = time.time()

        nmodel = NeuralNetWrapper(args['filters'], args['residual_blocks'])
        nmodel.load_checkpoint(folder=args['checkpoint'] + 'models/', filename='best.pth.tar')
        c = Coach(pyratgame, nmodel, args, buffer, logger)

        # play the game
        game_samples = c.execute_episode()
        print(f"finished game {i} in {time.time() - start}s")
        # load the latest buffer
        buffer.load()
        # Store the game samples
        buffer.store(game_samples)
        # Save the buffer to disk to replace the file on disk
        buffer._save()  # inplace saving the games to the last iteration examples

        i += 1

if __name__ == '__main__':
    self_play()
