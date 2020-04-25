import numpy as np

RAT = -1
PYTHON = 1

class Pyrat:
    def __init__(self,maze_width = 21, maze_height = 15, board = None):
        self.board  = board #A 9x21x15 board, as outputed by the matricized PyratEnv
        # Precompute
        self.RAT_matrix = np.full((9,21,15),RAT)
        self.PYTHON_matrix = np.full((9,21,15),PYTHON)

    def add_player_info(self,player,obs):
        if player == PYTHON :
            return np.append(obs,self.PYTHON_matrix, axis=0)
        elif player == RAT:
            return np.append(obs,self.RAT_matrix, axis=0)

    def get_next_state(self,board,player, action):
        """
        Used
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by both

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        self._make_move(board,player,action)

        if player == PYTHON:
            self._calculate_scores
