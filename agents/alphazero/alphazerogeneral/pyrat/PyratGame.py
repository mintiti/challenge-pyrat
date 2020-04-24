from ..Game import Game
import numpy as np

RAT = 1
PYTHON = -1


class PyratGame(Game):
    DECISION_FROM_ACTION_DICT = {
        0: 'L',
        1: 'U',
        2: "R",
        3: 'D'
    }
    board_transform_dict = {0 : lambda x : np.roll(x,-1, axis = 0), #move to the left
                            2 :lambda x : np.roll(x, 1,axis=0), # move to the right
                            1 : lambda x: np.roll(x, 1, axis=1), # move up
                            3 : lambda x : np.roll(x,-1, axis= 1) # move down
    }

    def __init__(self, env):
        """Takes in the Alphazero wrapped version of the pyrat env.
        If you didnt, please wrap the env using pyrat_env.wrappers.AlphaZero

            env : type : PyRatEnv

            Attributes :
                - env : the PyRat gym env
                - player : the current player's turn to play
                - current board : board : tensor of size (9,21,15) with each layer containing :
                            0) Maze_left
                            1) Maze_up
                            2) Maze_right
                            3) Maze_down
                            4) Pieces of cheese location
                            5) Player 1 score
                            6) Player 2 score
                            7) Player 1 location
                            8) Player 2 location
                            9) Player whose turn it is to play
                            10) Number of turns since the beginning
                            """
        self.env = env  # This is basically just a maze generator
        self.current_board = None
        self.rat_action = None

        # Precompute
        self.RAT_matrix = np.full((1, 21, 15), RAT)
        self.PYTHON_matrix = np.full((1, 21, 15), PYTHON)


    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        initial_state = self.env.reset()
        turns = np.zeros((1, 21, 15))
        self.current_board = np.concatenate((initial_state, self.RAT_matrix, turns))
        return self.current_board

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

    def getNextState(self, board, player, action, previous_move = None):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        self.current_board = board
        self.rat_action = previous_move
        if self.current_board[9][0][0] == PYTHON:
            self.rat_action = previous_move
            python_action = action
        elif  self.current_board[9][0][0] == RAT:
            self.rat_action = action
            python_action = previous_move

        if previous_move != None:
            # make the move with the previous rat action and the new action
            self._make_move(self.rat_action, python_action) # move, remove cheese and take care of scores
            self.current_board[10] += 1

        self.current_board[9] *= -1

        return self.current_board, - player

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
        return [1, 1, 1, 1]

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        self.board = board
        nb_turns = self.current_board[10][0][0]
        p1_score = self.current_board[5][0][0]
        p2_score = self.current_board[6][0][0]
        if player == PYTHON:
            if nb_turns >=1000:
                if p1_score > p2_score:
                    return 1
                elif p1_score == p2_score:
                    return 0.00000001
                else :
                    return -1

            else:
                if p1_score + p2_score == 41:
                    if p1_score > p2_score:
                        return 1
                    elif p1_score == p2_score:
                        return 0.00000001
                    else:
                        return -1
        return 0

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
        return board

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
        return str(board)

    def _make_move(self, rat_action, python_action):
        """Makes the players move, then checks if cheeses have been taken and
            calculate the scores accordingly
            :arg rat_action : The rat's action
            :arg python_action : the python's action

            :return : None"""
        # make the move (move the players)
        # recalculate scores
        # take out the cheeses if needed

        # move the players
        rat_position = np.where(self.current_board[7] == 1)
        if self.current_board[rat_action][rat_position[0][0]][rat_position[1][0]]:
            # Select the right board transform
            board_transform = self.board_transform_dict[rat_action]
            #apply it to the player's position
            self.current_board[7] = board_transform(self.current_board[7])

        python_position = np.where(self.current_board[8] == 1)
        if self.current_board[python_action][python_position[0][0]][python_position[1][0]]:
            # Select the right board transform
            board_transform = self.board_transform_dict[python_action]
            self.current_board[8] = board_transform(self.current_board[8])

        # recalculate the scores
        if self.current_board[4][rat_position[0][0]][rat_position[1][0]]: # check if there's a cheese on player 1
            if python_position == rat_position :
                # Add 0.5 to each player's scores
                self.current_board[5] += 0.5
                self.current_board[6] += 0.5
            else:
                self.current_board[5] +=1
            self.current_board[4][rat_position[0][0]][rat_position[1][0]] = 0

        if self.current_board[4][python_position[0][0]][python_position[1][0]]:
            self.current_board[6]+= 1
            self.current_board[4][python_position[0][0]][python_position[1][0]] = 0

        self.rat_action = None
