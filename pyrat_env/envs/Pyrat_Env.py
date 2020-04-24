# Imports

import gym
from gym import spaces
from ..pyrat import move
import numpy as np
from ..config import cfg
from pickle import load, dump
from ..imports.maze import *
from ..imports.display import *
from ..imports.parameters import *

# CONSTANTS
DECISION_FROM_ACTION_DICT = {
    0: 'L',
    1: 'U',
    2: "R",
    3: 'D'
}


class PyratEnv(gym.Env):
    # TODO : add mud
    # TODO : code a replay system and a replayer
    # TODO : recode the maze matrix to be 4 matrices indicating whether you can go up, down, to the right and to the left respectively
    """
    Description:
        2 agents compete in a maze for rewards randomly dispersed in the maze. The goal is to collect the most.
    Observation:
        Type: Dict({
            'Maze_up':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go up from case ij

            'Maze_down':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go down from case ij

            'Maze_right':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go right from case ij

            'Maze_left':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go left from case ij

            'pieces_of _cheese' :
                Type :Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix where m_ij = 0 if there is no cheese on this bloc and 1 if there is

            'player1_location' :
                Type : Tuple ( Discrete(maze_width), Discrete(maze_height))
                    The location of player 1

            'player2_location' :
                Type : Tuple ( Discrete(maze_width), Discrete(maze_height))
                    The location of player 2
        })

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

    def __init__(self, width=21, height=15, nb_pieces_of_cheese=41, max_turns=2000, target_density = 0.7, connected= True, symmetry = True, mud_density = 0.7, mud_range = 10, maze_file = "", start_random = False):
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
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0], self.maze_dimension[1], self.target_density,
                                                                                  self.connected, self.symmetry,
                                                                                  self.mud_density, self.mud_range, self.maze_file,
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
        #create the player score matrices
        self.action_space = spaces.Tuple([spaces.Discrete(4),
                                          spaces.Discrete(4)
                                          ])

        # Define the observation space
        self.observation_space = spaces.Dict({
            'Maze_up': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'Maze_down': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'Maze_right': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'Maze_left': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'pieces_of_cheese': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'player1_score': spaces.Box(low=0, high=nb_pieces_of_cheese, shape=(1,)),
            'player2_score': spaces.Box(low=0, high=nb_pieces_of_cheese, shape=(1,)),
            'player1_location': spaces.Tuple([spaces.Discrete(self.width), spaces.Discrete(self.height)]),
            'player2_location': spaces.Tuple([spaces.Discrete(self.width), spaces.Discrete(self.height)]),

        })

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

    def step(self, action):
        self.turn += 1
        # Perform both player's actions on the maze variables
        decision1, decision2 = DECISION_FROM_ACTION_DICT[action[0]], DECISION_FROM_ACTION_DICT[action[1]]
        self.player1_last_move = decision1
        self.player2_last_move = decision2
        self._move((decision1, decision2))

        reward = self._calculate_reward()

        # Calculate the return variables
        observation = self._observation()
        done = self._check_done()
        info = dict()

        return observation, reward, done, info

    def reset(self):
        # reset the maze randomly
        self.random_seed = random.randint(0, sys.maxsize)
        self.turn = 0
        self.pieces_of_cheese = []
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0],
                                                                                  self.maze_dimension[1], self.target_density,
                                                                                  self.connected, self.symmetry,
                                                                                  self.mud_density, self.mud_range, self.maze_file,
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(self.nb_pieces_of_cheese,
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

    # TODO : Rendering : maybe switch it to something better than pygame
    # TODO : Sound
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
        return dict({
            'Maze_up': self.maze_matrix_U,
            'Maze_right': self.maze_matrix_R,
            'Maze_left': self.maze_matrix_L,
            'Maze_down': self.maze_matrix_D,
            'pieces_of_cheese': self.cheese_matrix,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'player1_location': self.player1_location,
            'player2_location': self.player2_location,
        })

    def matrix_index_to_pos(self, index):
        # noinspection PyRedundantParentheses
        return (index % self.width, index // self.width)

    def pos_to_matrix_index(self, pos):
        return pos[1] * self.width + pos[0]

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
