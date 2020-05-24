from .mcts import MCTS, RootParentNode, Node, MCTSActor
from collections import deque
from tqdm.auto import trange
from ..neural_net import ResidualNet
from .arena import Arena, ParallelArena
import ray
import time

class Coach:
    # TODO : code the buffer
    def __init__(self, game, nnet, args, buffer, logger):
        self.game = game
        self.nnet = nnet  # neural nets need to be wrapped inside of NeuralNetWrapper
        self.pnet = self.nnet.__class__(args['filters'], args['residual_blocks'])  # the competitor network
        self.args = args
        self.replay_buffer = buffer
        self.logger = logger

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
            terminal_node : the last node reached
        """
        episode_examples = []
        board = self.game.getInitBoard()

        # Make the player's mcts
        self.pnet.clear_cache()
        self_play_params = self.args['self_play_mcts_params']
        mcts = MCTS(self.pnet, self_play_params)
        mcts.self_play()

        # create the root node
        root_parent = RootParentNode(self.game)  # dummy node
        current_node = Node(action=None, obs=board[:9], done=False, reward=0, state=board, player=1, mcts=mcts,
                            parent=root_parent)

        # run the game
        episode_step = 0
        while not current_node.done:
            episode_step += 1
            exploit = episode_step > self_play_params['temp_threshold']
            mcts.exploit = exploit
            # get the tree policy, the action chosen and the next node
            tree, action, next_node = mcts.compute_action(current_node)

            symmetries = self.game.getSymmetries(current_node.obs, tree)

            for obs, pi in symmetries:
                episode_examples.append([obs, current_node.current_player, pi])

            current_node = next_node

        return [(x[0], x[2], current_node.reward * ((-1) ** (x[1] != current_node.current_player))) for x in
                episode_examples], current_node

    def fill_buffer(self):
        # Load the best net
        self.pnet.load_checkpoint(folder=self.args['checkpoint'] + 'models/', filename='best.pth.tar')
        i = 0
        while len(self.replay_buffer) < self.args['min_buffer_size']:
            print(f"Filling buffer ...\nBuffer size {len(self.replay_buffer)}\nStarting game {i + 1}")
            episode_examples, end_node = self.execute_episode()
            self.replay_buffer.store(episode_examples)
            if i%20 == 0:
                print(f"game {i + 1 }, saving to disk...")
                self.replay_buffer.temp_save()
            i+=1

        print(f"Buffer size {len(self.replay_buffer)}\nSaving buffer ... ")
        self.replay_buffer.save()

    def get_n_iters(self):
        return self.replay_buffer.get_n_iters()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        if not self.args['load_model']:
            self.pnet.save_checkpoint(folder=self.args['checkpoint'] + 'models/', filename='temp.pth.tar')

        for i in range(1, self.args['numIters']):
            print('------ITER ' + str(i) + '------')
            # load the best net
            self.pnet.load_checkpoint(folder=self.args['checkpoint'] + 'models/', filename='best.pth.tar')
            start = time.time()
            # Run self play episodes

            #Book keeping
            iter_cheese_average = 0
            iter_turn_average = 0
            for eps in trange(self.args['numEps'], desc='Running self-play episodes', unit='episode'):
                print(f"\nstarting episode {eps + 1}")
                episode_examples, end_node = self.execute_episode()
                self.replay_buffer.store(episode_examples)

                # Book keeping
                iter_turn_average += end_node.state[9][0][0]
                iter_cheese_average += end_node.state[5][0][0] + end_node.state[6][0][0]
            self.logger.add_scalar('Number of turns', iter_turn_average / self.args['numEps'], self.get_n_iters())
            self.logger.add_scalar('Number of cheese captures',iter_cheese_average / self.args['numEps'] ,
                                   self.get_n_iters())

            infos = self.nnet.train(self.replay_buffer.storage)
            self.nnet.clear_cache()
            self.nnet.save_checkpoint(folder=self.args['checkpoint'] + 'models/', filename='temp.pth.tar')

            # Book keeping
            global_step = self.get_n_iters()
            self.logger.add_scalars("Training losses", {"Value loss": infos['value_loss'],
                                                   "Policy loss": infos['policy_loss'],
                                                   "Total_loss": infos['value_loss'] + infos['policy_loss']},
                               global_step)

            self.replay_buffer.save()
            # Run the eval games
            eval_params = self.args['eval_mcts_params']
            pmcts = MCTS(self.pnet, eval_params)
            nmcts = MCTS(self.nnet, eval_params)
            arena = Arena(pmcts, nmcts, self.game)

            pWon, nWon, draws = arena.playGames(self.args['arenaCompare'])

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nWon, pWon, draws))
            if pWon + nWon == 0 or float(nWon) / (pWon + nWon) < self.args['updateThreshold']:
                print('REJECTING NEW MODEL')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args['checkpoint'] + 'models/',
                                          filename=self.getCheckpointFile())
                self.nnet.save_checkpoint(folder=self.args['checkpoint'] + 'models/', filename='best.pth.tar')
                self.nnet.clear_cache()

            # Book keeping
            self.logger.add_scalars("Winrate vs previous version", {"Threshold" : self.args['updateThreshold'],
                                                                    "Winrate" : float(nWon) / (pWon + nWon),
                                                                    "Draws" : float(draws / (draws + nWon + pWon))},
                                    self.get_n_iters())

    def getCheckpointFile(self):
        return 'checkpoint_' + str(self.replay_buffer.get_n_iters()) + '.pth.tar'


class NeuralNetWrapper:
    def __init__(self, nb_filters, nb_residual_blocks):
        self.model = ResidualNet(nb_filters, nb_residual_blocks)

    def compute_priors_and_value(self, obs):
        child_priors, value = self.model.predict(obs)
        return child_priors, float(value)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.model.load_checkpoint(folder=folder, filename=filename)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.model.save_checkpoint(folder=folder, filename=filename)

    def clear_cache(self):
        self.model.clear_cache()

    def train(self, examples):
        return self.model.train(examples)

@ray.remote(num_gpus=1/3)
class InferenceActor:
    def __init__(self, nb_filters, nb_residual_blocks):
        self.model = ResidualNet(nb_filters, nb_residual_blocks)
    @ray.method(num_return_vals=2)
    def compute_priors_and_value(self, obs):
        child_priors, value = self.model.predict(obs)
        return child_priors, float(value)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.model.load_checkpoint(folder=folder, filename=filename)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.model.save_checkpoint(folder=folder, filename=filename)

    def clear_cache(self):
        self.model.clear_cache()

    def train(self, examples):
        return self.model.train(examples)

@ray.remote(num_gpus=1)
class LearningActor:
    def __init__(self, nb_filters, nb_residual_blocks):
        self.model = ResidualNet(nb_filters, nb_residual_blocks)

    @ray.method(num_return_vals=2)
    def compute_priors_and_value(self, obs):
        child_priors, value = self.model.predict(obs)
        return child_priors, float(value)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.model.load_checkpoint(folder=folder, filename=filename)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.model.save_checkpoint(folder=folder, filename=filename)

    def clear_cache(self):
        self.model.clear_cache()

    def train(self, examples):
        return self.model.train(examples)

@ray.remote(num_cpus=1)
class SelfPlayActor:
    def __init__(self, model, mcts_config, game):
        self.model = model
        self.mcts  = MCTSActor(self.model, mcts_config)
        self.game  = game
        self.episode_examples = []
        self.args = mcts_config

    @ray.method(num_return_vals= 2)
    def play(self,numgame):
        print(f"Starting game {numgame}")
        start = time.time()
        self.episode_examples = []
        board = self.game.getInitBoard()

        # Make the player's mcts
        ray.get(self.model.clear_cache.remote())


        self.mcts.self_play()
        self_play_params = self.args
        # create the root node
        root_parent = RootParentNode(self.game)  # dummy node
        current_node = Node(action=None, obs=board[:9], done=False, reward=0, state=board, player=1, mcts=self.mcts,
                            parent=root_parent)

        # run the game
        episode_step = 0
        while not current_node.done:
            episode_step += 1
            over_threshold = episode_step > self_play_params['temp_threshold']
            self.mcts.set_exploit(over_threshold)
            # get the tree policy, the action chosen and the next node
            tree, action, next_node = self.mcts.compute_action(current_node)

            symmetries = self.game.getSymmetries(current_node.obs, tree)

            for obs, pi in symmetries:
                self.episode_examples.append([obs, current_node.current_player, pi])

            current_node = next_node

        end = time.time()
        print(f"Game {numgame} ended in {end-start}")

        return [(x[0], x[2], current_node.reward * ((-1) ** (x[1] != current_node.current_player))) for x in
                self.episode_examples], current_node