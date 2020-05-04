from ray.rllib.env import MultiAgentEnv
import gym
from gym import spaces
from ..pyrat import move
import numpy as np
from ..config import cfg
from pickle import load, dump
from ..imports.maze import *
from ..imports.display import *
from ..imports.parameters import *
from ray.tune.utils.util import merge_dicts
from gym.wrappers.flatten_observation import FlattenObservation

# CONSTANTS
DECISION_FROM_ACTION_DICT = {
    0: 'L',
    1: 'U',
    2: "R",
    3: 'D'
}

PYRAT_MULTIAGENT_DEFAULT_CONFIG = {
    "width": 21,
    "height": 15,
    "nb_pieces_of_cheese": 41,
    "max_turns": 1000,
    "target_density": 0.7,
    "connected": True,
    "symmetry": True,
    "mud_density": 0,
    "mud_range": 10,
    "maze_file": "",
    "start_random": False,
    "flatten" : False}


class PyratMultiAgent(MultiAgentEnv):
    # TODO : add mud
    # TODO : code a replay system and a replayer
    """
    Description:
        2 agents compete in a maze for rewards randomly dispersed in the maze. The goal is to collect the most.
    Observation:
                - array of size (10,21,15) with each layer containing :
                            0) Maze_left
                            1) Maze_up
                            2) Maze_right
                            3) Maze_down
                            4) Pieces of cheese location
                            5) Player 1 score
                            6) Player 2 score
                            7) Player 1 location
                            8) Player 2 location
                            9) Player perspective plane

    Actions:
        Type: Tuple( Discrete(4), Discrete(4))
        For each agent :
        Num	Action
        0	Agent to the left
        1	Agent up
        2   Agent to the right
        3   Agent down

    Reward:
        Reward is 1 if player 1 eats a cheese and player 2 doesn't, 0 if both take a cheese and -1 if player 2 takes a cheese and player1 doesn't
    Starting State:
        A random (connected) maze with random cheese disposition
        For now no mud is included
        Each player start respectively on the lower left and higher right corner
    Episode Termination:
        Max number of turns is reached
        One player takes more than half of the cheeses available
    """
    # Gym API methods
    metadata = {'render.modes': ['human', 'none']}
    reward_range = (-1, 1)

    RAT = 1
    PYTHON = -1

    def __init__(self, env_config ):
        dict = merge_dicts(PYRAT_MULTIAGENT_DEFAULT_CONFIG, env_config)

        width = dict["width"]
        height = dict["height"]
        nb_pieces_of_cheese = dict["nb_pieces_of_cheese"]
        max_turns = dict["max_turns"]
        target_density = dict["target_density"]
        connected = dict["connected"]
        symmetry = dict["symmetry"]
        mud_density = dict["mud_density"]
        mud_range = dict["mud_range"]
        maze_file = dict["maze_file"]
        start_random = dict["start_random"]
        self.flatten = dict["flatten"]



        self.rat = "rat"
        self.python = "python"
        # Precompute
        self.RAT_matrix = np.full((width, height), self.RAT)
        self.PYTHON_matrix = np.full((width, height), self.PYTHON)

        self.max_turns = max_turns
        self.target_density = target_density
        self.connected = connected
        self.symmetry = symmetry
        self.mud_density = mud_density
        self.mud_range = mud_range
        self.maze_file = maze_file
        self.start_random = start_random
        self.turn = 0
        self.maze = None
        self.maze_dimension = (width, height)
        self.pieces_of_cheese = []
        self.nb_pieces_of_cheese = nb_pieces_of_cheese
        self.player1_location = None
        self.player2_location = None
        self.player1_score = 0
        self.player2_score = 0
        self.player1_misses = 0
        self.player2_misses = 0
        self.player1_moves = 0
        self.player2_moves = 0
        self.random_seed = random.randint(0, sys.maxsize)
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0],
                                                                                  self.maze_dimension[1],
                                                                                  self.target_density,
                                                                                  self.connected, self.symmetry,
                                                                                  self.mud_density, self.mud_range,
                                                                                  self.maze_file,
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(
            self.nb_pieces_of_cheese, self.maze_dimension[0], self.maze_dimension[1],
            self.symmetry,
            self.player1_location,
            self.player2_location,
            self.start_random)

        # Wrapper attributes
        # Create the maze matrix
        self.maze_matrix_U = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_D = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_R = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_L = np.zeros((self.width, self.height), dtype=np.int8)
        self._maze_matrix_from_dict()
        # Create the cheese matrix
        self.cheese_matrix = np.zeros((self.width, self.height), dtype=np.int8)
        self._cheese_matrix_from_list()
        # create the player score matrices
        self.action_space = spaces.Discrete(4)

        # Define the observation space
        if not self.flatten:
            self.observation_space = spaces.Box(low=0, high=self.nb_pieces_of_cheese, shape=(10, self.width, self.height))
        else :
            self.observation_space = spaces.Box(low=0, high= self.nb_pieces_of_cheese, shape= (10*self.width*self.height,))
        # Follow the play :
        self.player1_last_move = None
        self.player2_last_move = None
        self.bg = None

    @classmethod
    def fromPickle(cls, p="./maze_files/maze.p"):
        """
        Lets you load a given maze from a previously pickled object
        Sample can be found under ./maze_files/maze.p
        :param p: the path to the maze
        :return: the object instance
        """
        return load(open(p, 'rb'))

    def step(self, action_dict):
        self.turn += 1
        # Perform both player's actions on the maze variables
        decision1, decision2 = DECISION_FROM_ACTION_DICT[action_dict[self.rat]], DECISION_FROM_ACTION_DICT[
            action_dict[self.python]]
        self.player1_last_move = decision1
        self.player2_last_move = decision2
        self._move((decision1, decision2))

        rewards = self._calculate_reward()
        rewards = {self.rat: rewards,
                   self.python: -rewards}

        # Calculate the return variables
        observations = self._observation()
        dones = self._check_done()
        dones = {"__all__": dones}
        infos = {self.rat: dict(),
                 self.python: dict()}



        return observations, rewards, dones, infos

    def reset(self):
        # reset the maze randomly
        self.random_seed = random.randint(0, sys.maxsize)
        self.turn = 0
        self.pieces_of_cheese = []
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0],
                                                                                  self.maze_dimension[1],
                                                                                  self.target_density,
                                                                                  self.connected, self.symmetry,
                                                                                  self.mud_density, self.mud_range,
                                                                                  self.maze_file,
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(
            self.nb_pieces_of_cheese,
            self.maze_dimension[
                0],
            self.maze_dimension[
                1],
            self.symmetry,
            self.player1_location,
            self.player2_location,
            self.start_random)
        # Reset player turns, score, misses
        self.player1_score, self.player2_score, self.player1_misses, self.player2_misses, self.player1_moves, self.player2_moves = 0, 0, 0, 0, 0, 0
        self.player1_last_move = None
        self.player2_last_move = None
        self.bg = None
        # Reset wrapper attributes
        # Create the maze matrix
        self.maze_matrix_U = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_D = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_R = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_L = np.zeros((self.width, self.height), dtype=np.int8)
        self._maze_matrix_from_dict()
        # Create the cheese matrix
        self.cheese_matrix = np.zeros((self.width, self.height), dtype=np.int8)
        self._cheese_matrix_from_list()

        for start in self.maze:
            for end in self.maze[start]:
                self.maze[start][end] = 1

        return self._observation()

    def render(self, mode='human'):
        if mode == 'human':
            (window_width, window_height) = cfg['resolution']
            scale, offset_x, offset_y, image_background, image_cheese, image_corner, image_moving_python, image_moving_rat, image_python, image_rat, image_wall, image_mud, image_portrait_python, image_portrait_rat, tiles, image_tile = init_coords_and_images(
                self.width, self.height, True, True, window_width, window_height)
            if self.bg is None:
                pygame.init()
                screen = pygame.display.set_mode(cfg['resolution'])
                self.bg = build_background(screen, self.maze, tiles, image_background, image_tile, image_wall,
                                           image_corner,
                                           image_mud,
                                           offset_x, offset_y, self.width, self.height, window_width, window_height,
                                           image_portrait_rat,
                                           image_portrait_python, scale, True, True)
            screen = pygame.display.get_surface()
            screen.blit(self.bg, (0, 0))
            draw_pieces_of_cheese(self.pieces_of_cheese, image_cheese, offset_x, offset_y, scale, self.width,
                                  self.height, screen,
                                  window_height)
            draw_players(self.player1_location, self.player2_location, image_rat, image_python, offset_x, offset_y,
                         scale, self.width,
                         self.height, screen, window_height)
            draw_scores("Rat", self.player1_score, image_portrait_rat, "Python", self.player2_score,
                        image_portrait_python, window_width,
                        window_height, screen, True, True, self.player1_last_move, self.player1_misses,
                        self.player2_last_move, self.player2_misses, 0,
                        0)
            pygame.display.update()
            pygame.event.get()

        elif mode == "none":
            pass

    # Utils methods

    def save_pickle(self, path="./maze_files/maze_save.p"):
        """
        pickles the maze to the given path
        :param path: the path to save to
        :return:
        """
        dump(self, open(path, "wb"))

    def _observation(self):
        canonical_board = []
        canonical_board.append(self.maze_matrix_L)
        canonical_board.append(self.maze_matrix_U)
        canonical_board.append(self.maze_matrix_R)
        canonical_board.append(self.maze_matrix_D)
        canonical_board.append(self.cheese_matrix)
        canonical_board.append(np.full_like(self.maze_matrix_R, self.player1_score))
        canonical_board.append(np.full_like(self.maze_matrix_R, self.player2_score))

        # make the player 1 position matrix
        player1_position_matrix = np.zeros_like(self.maze_matrix_L)
        player1_position_matrix[self.player1_location] = 1
        # make the player 2 position matrix
        player2_position_matrix = np.zeros_like(self.maze_matrix_L)
        player2_position_matrix[self.player2_location] = 1

        canonical_board.append(player1_position_matrix)
        canonical_board.append(player2_position_matrix)

        canonical_board_rat = canonical_board
        canonical_board_python = canonical_board.copy()

        canonical_board_rat.append(self.RAT_matrix)
        canonical_board_python.append(self.PYTHON_matrix)

        canonical_board_rat = np.array(canonical_board_rat)
        canonical_board_python = np.array(canonical_board_python)

        if self.flatten:
            canonical_board_rat = canonical_board_rat.flatten()
            canonical_board_python = canonical_board_python.flatten()

        return dict({self.rat: np.array(canonical_board_rat),
                     self.python: np.array(canonical_board_python)
                     })

    def _maze_matrix_from_dict(self):
        """
        Generates the maze matrix
        :return:
        """
        maze_dict = self.maze
        for position in maze_dict:
            for destination in maze_dict[position]:
                direction = self._calculate_direction(position, destination)
                if direction == 'U':
                    self.maze_matrix_U[position[0], position[1]] = 1
                elif direction == 'D':
                    self.maze_matrix_D[position[0], position[1]] = 1
                elif direction == 'R':
                    self.maze_matrix_R[position[0], position[1]] = 1
                elif direction == 'L':
                    self.maze_matrix_L[position[0], position[1]] = 1

    def _cheese_matrix_from_list(self):
        for cheese in self.pieces_of_cheese:
            self.cheese_matrix[cheese] = 1

    def _calculate_reward(self):
        """
        Returns the reward for the current turn and removes the potential cheeses that have been eaten
        reward is 1 if the player 1 eats a piece of cheese and player 2 doesnt, -1 if player 2 does and player 1 doesnt and 0 in all other cases
        :return: reward
        """
        reward = 0
        if self.player1_location in self.pieces_of_cheese:
            self.pieces_of_cheese.remove(self.player1_location)
            self.cheese_matrix[self.player1_location] = 0
            if self.player2_location == self.player1_location:
                self.player2_score += 0.5
                self.player1_score += 0.5
                # reward = 0
            else:
                self.player1_score += 1
                reward = 1
        if self.player2_location in self.pieces_of_cheese:
            self.pieces_of_cheese.remove(self.player2_location)
            self.cheese_matrix[self.player2_location] = 0
            self.player2_score += 1
            reward = -1
        return reward

    def _move(self, action):
        """
        imports.maze.move function wrapper
        :param action: (decision1,decision2) of both players
        """
        (decision1, decision2) = action
        self.player1_location, self.player2_location, stuck1, stuck2, self.player1_moves, self.player2_moves, self.player1_misses, self.player2_misses = move(
            decision1, decision2, self.maze, self.player1_location, self.player2_location, 0, 0, self.player1_moves,
            self.player2_moves, self.player1_misses, self.player2_misses)

    def _check_done(self):
        # noinspection PyRedundantParentheses,PyRedundantParentheses
        return (self.turn >= self.max_turns) or (self.player1_score > (self.nb_pieces_of_cheese) / 2) or (
                self.player2_score > (self.nb_pieces_of_cheese) / 2)

    def _calculate_direction(self, source, destination):
        direction = None
        delta = (destination[0] - source[0], destination[1] - source[1])
        if delta == (0, 1):
            direction = 'U'
        elif delta == (0, -1):
            direction = 'D'
        elif delta == (1, 0):
            direction = 'R'
        elif delta == (-1, 0):
            direction = 'L'
        return direction


