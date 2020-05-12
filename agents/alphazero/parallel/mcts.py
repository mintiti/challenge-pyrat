"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import numpy as np

DEFAULT_MCTS_PARAMS = {
    "temperature" : 1,
    "add_dirichlet_noise": True,
    "dirichlet_epsilon" : 0.25,
    "dirichlet_noise" : 2.5,
    "num_simulations" : 600,
    "exploit" : False,
    "puct_coefficient" : 2,
    "argmax_tree_policy" : False
}
class Node:
    def __init__(self, action, obs, done, reward, state, mcts, player, parent=None):
        self.game = parent.game
        self.action = action  # Action used to go to this state
        self.current_player = player

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.game.getActionSize()
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32)  # Q
        self.child_priors = np.zeros(
            [self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)  # N
        self.valid_actions = [1]*4

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs

        self.mcts = mcts

    @property
    def number_visits(self):
        """Returns the number of thimes the current node has been visited"""
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        """Return the total value of the whole subtree below current node"""
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):

        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        """Returns an array containing the Q values of the children of the current node"""
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        """Calculates the U value for each child of the current node.
        :return: Array of size (action space,) """
        return math.sqrt(self.number_visits) * self.child_priors / (
            1 + self.child_number_visits)

    def best_action(self):
        """Chooses the action with the highest PUCT value.
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        return np.argmax(masked_child_score)

    def select(self):
        """Goes down the tree while the node has been expanded
        :return: the first node that is not expanded, i.e. the first that has neither been evaluated """
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        """expands the current node after child priors evaluation
        :arg child_priors: array of size (action_space,) containing the children visit priorities from the neural net"""
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        """Gets the child of the current node when action a is done"""
        if action not in self.children:
            #self.game.set_state(self.state)
            #obs, reward, done, _ = self.game.step(action)
            next_state, next_player = self.game.getNextState(self.state,self.current_player,action, previous_move = self.action)
            reward = 0
            game_ended = self.game.getGameEnded(next_state, -1)
            done = (game_ended != 0)
            if self.done == True:
                done = True
            if game_ended == 1 or game_ended == -1:
                reward = game_ended

            obs  = next_state[:9]
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts,
                player= next_player)
        return self.children[action]

    def backup(self, value):
        """Backs up the value v through the tree.
            :arg v: the value as evaluated at a leaf node (game ended) or by the neural network"""
        current = self
        while current.parent is not None:
            value *= -1
            current.number_visits += 1
            current.total_value += value
            current = current.parent


class RootParentNode:
    def __init__(self, game):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.game = game


class MCTS:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]

    def eval(self):
        self.exploit = True
        self.add_dirichlet_noise = False

    def self_play(self):
        self.exploit = False
        self.add_dirichlet_noise = True

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.compute_priors_and_value(
                    leaf.obs)
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size)

                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(
            tree_policy)  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(
                np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]