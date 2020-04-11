from ..Game import Game
from pyrat_env.wrappers import AlphaZero
import numpy as np
RAT = -1
PYTHON = 1

class PyratGame(Game):
    def __init__(self,env):
        """Takes in the Alphazero wrapped version of the pyrat env.
        If you didnt, please wrap the env using pyrat_env.wrappers.AlphaZero

            env : type : PyRatEnv

            Attributes :
                - env : the PyRat gym env
                - player : the current player's turn to play
                - current board : board (ie the maze, the pieces of cheese, the player scores and player locations, in the canonical form
                                because of the nature of the game that's not really sequential, we need to add the player's turn to play on player move"""
        assert isinstance(env,AlphaZero)
        self.env = env
        self.player = RAT
        self.current_board = None
        self.done = False
        self.rat_action = None

        # Precompute
        self.RAT_matrix = np.full((9,21,15),RAT)
        self.PYTHON_matrix = np.full((9,21,15),PYTHON)

    def add_player_info(self,player,obs):
        if player == PYTHON :
            return np.append(obs,self.PYTHON_matrix, axis=0)
        elif player == RAT:
            return np.append(obs,self.RAT_matrix, axis=0)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        initial_state = self.env.reset()
        self.current_board = initial_state
        return self.add_player_info(initial_state, RAT)


    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.env.maze_dimension

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 4

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        if self.player == PYTHON:
            obs, rew, done, _ = self.env.step((self.rat_action, action))
            self.current_board = obs
        self.player *= -1
        return self.add_player_info(self.current_board,self.player)


    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return [1,1,1,1]

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass

    def _make_move(self,obs,move):
        if