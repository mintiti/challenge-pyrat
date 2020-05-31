import time

import ray

from agents.alphazero.neural_net import ResidualNet
from agents.alphazero.sequential.mcts import RootParentNode, Node
from agents.alphazero.ray_training.ray_mcts import MCTSActor


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