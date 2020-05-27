###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
import time

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

ARG_TO_DIRECTION = {
    0: MOVE_LEFT,
    1: MOVE_UP,
    2: MOVE_RIGHT,
    3: MOVE_DOWN
}
DIRECTION_TO_ARG = {
    MOVE_LEFT : 0,
    MOVE_UP : 1,
    MOVE_RIGHT : 2,
    MOVE_DOWN :3
}
MCTS_CONFIG = {
    "temperature": 1,
    "add_dirichlet_noise": False,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 2.5,
    "num_simulations": 600,  # number of mcts games to simulate
    "exploit": True,
    "puct_coefficient": 2.25,
    "argmax_tree_policy": True
}

WEIGHT_FILE = "best.pth.tar"
###############################
# Please put your imports here
import numpy as np
import torch

from AIs.alphazero.PyratGame import PyratGame
from AIs.alphazero.mcts import Node, RootParentNode, MCTS
from AIs.alphazero.nn import NeuralNetWrapper
import os
###############################
# Please put your global variables here
board = None

# Evaluation model
model = NeuralNetWrapper(64, 3)
model.load_checkpoint(folder="./AIs/alphazero/weights/t0-weights", filename=WEIGHT_FILE)
model.model.nn.eval()

#MCTS variables
game = PyratGame(None)
root_parent = RootParentNode(game)
current_node = None
mcts = None

# follow the play
previous_opponent_position = None


###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def _maze_matrix_from_dict(maze, maze_matrix_L, maze_matrix_U, maze_matrix_R, maze_matrix_D):
    """
    Generates the maze matrix
    :return:
    """
    maze_dict = maze
    for position in maze_dict:
        for destination in maze_dict[position]:
            direction = _calculate_direction(position, destination)
            if direction == 'U':
                maze_matrix_U[position[0], position[1]] = maze_dict[position][destination]
            elif direction == 'D':
                maze_matrix_D[position[0], position[1]] = maze_dict[position][destination]
            elif direction == 'R':
                maze_matrix_R[position[0], position[1]] = maze_dict[position][destination]
            elif direction == 'L':
                maze_matrix_L[position[0], position[1]] = maze_dict[position][destination]


def make_state(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore,
               piecesOfCheese):
    obs = np.copy(board)
    # make the cheese matrix
    for cheese in piecesOfCheese:
        obs[4][cheese] = 1

    # make the player score matrices
    obs[5] = np.full((mazeWidth, mazeHeight), playerScore)
    obs[6] = np.full((mazeWidth, mazeHeight), opponentScore)

    # make the player position matrices
    obs[7][playerLocation[0]][playerLocation[1]] = 1
    obs[8][opponentLocation[0]][opponentLocation[1]] = 1

    return obs

def get_child_with_opponent_location(opponentLocation,current_node):
    child_list = []
    for child in current_node.children:
        if current_node.children[child].obs[8][opponentLocation[0]][opponentLocation[1]] == 1: # If the opponent is in the right location
            child_list.append((current_node.children[child],current_node.children[child].number_visits))
    child_list.sort(key= lambda a: a[1])
    return child_list[-1][0]


def _calculate_direction(source, destination):
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


def get_width_height(maze):
    width = - float("inf")
    height = - float("inf")

    for (x, y) in maze:
        if x + 1 > width:
            width = x + 1
        if y + 1 > height:
            height = y + 1
    return width, height

def display(node,tree, action):
    string = f"""Position summary :
Position evaluation : {- node.total_value / node.number_visits}
Move evaluations : """
    for child in range(4):
        if child == 0:
            s = f"Left "
        elif child == 1:
            s = f"                   Up "
        elif child == 2:
            s = f"                   Right "
        elif child == 3:
            s = f"                   Down "
        s+= f"({int(tree[child] * 1000) / 10}%)"
        if child in node.children:
            s += f" : {current_node.children[child].total_value / current_node.children[child].number_visits}"
        string += s + "\n"
    string += f"Action chosen : {action}\n"
    return string


    #     print(f"""Position summary :
    # Position evaluation : {current_node.total_value / current_node.number_visits}
    # Move evaluations : Left ({int(tree[0] * 1000) / 10}%) : {current_node.children[0].total_value / current_node.children[0].number_visits}
    #                    Up ({int(tree[1] * 1000) / 10}%) : {current_node.children[1].total_value / current_node.children[1].number_visits}
    #                    Right ({int(tree[2] * 1000) / 10}%) : {current_node.children[2].total_value / current_node.children[2].number_visits}
    #                    Down ({int(tree[3] * 1000) / 10}%) : {current_node.children[3].total_value / current_node.children[3].number_visits}
    # """

def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    # Example prints that appear in the shell only at the beginning of the game
    # Remove them when you write your own program

    """returns a game description with feature planes :
    current board : board : tensor of size (10,21,15) with each layer containing :
            0) Maze_left
            1) Maze_up
            2) Maze_right
            3) Maze_down
            4) Pieces of cheese location
            5) rat score
            6) python 2 score
            7) rat location
            8) python location
            """
    start = time.time()
    global board, model, current_node, mcts
    im_size = (10, mazeWidth, mazeHeight)
    board = np.zeros(im_size, dtype=np.float16)
    _maze_matrix_from_dict(mazeMap, board[0], board[1], board[2], board[3])



    # Create the root MCTS node
    start_state  = make_state(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, 0, 0,
               piecesOfCheese)
    mcts = MCTS(model,MCTS_CONFIG)
    current_node = Node(action=None, obs=start_state[:9], done=False, reward=0, state=start_state, player=1, mcts=mcts,
                        parent=root_parent)

    # Preprocess MCTS sims
    # mcts.num_sims = 1
    # tree, action, next_node = mcts.compute_action(current_node)

    # print(f"""Position summary :
    # Position evaluation : {current_node.total_value / current_node.number_visits}
    # Move evaluations : Left ({int(tree[0] * 1000) / 10}%) : {current_node.children[0].total_value / current_node.children[0].number_visits}
    #                    Up ({int(tree[1] * 1000) / 10}%) : {current_node.children[1].total_value / current_node.children[1].number_visits}
    #                    Right ({int(tree[2] * 1000) / 10}%) : {current_node.children[2].total_value / current_node.children[2].number_visits}
    #                    Down ({int(tree[3] * 1000) / 10}%) : {current_node.children[3].total_value / current_node.children[3].number_visits}
    # """)


###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese,
         timeAllowed):
    global current_node, mcts, previous_opponent_position
    mcts.num_sims = 600
    if previous_opponent_position != None:
        try :
            # Calculer le move de l'adversaire et se placer sur le bon noeud
            opponent_move = _calculate_direction(previous_opponent_position, opponentLocation)
            current_node = current_node.get_child(DIRECTION_TO_ARG[opponent_move])
        except KeyError:
            current_node = get_child_with_opponent_location(opponentLocation,current_node)

    # Lancer les simulations MCTS
    tree, action, next_node = mcts.compute_action(current_node)
    display_string = display(current_node,tree,action)

    print(display_string)

    current_node = next_node
    previous_opponent_position = opponentLocation

    # In this example, we always go up
    return ARG_TO_DIRECTION[action]
