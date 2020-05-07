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
    board_transform_dict = {0: lambda x: np.roll(x, -1, axis=0),  # move to the left
                            2: lambda x: np.roll(x, 1, axis=0),  # move to the right
                            1: lambda x: np.roll(x, 1, axis=1),  # move up
                            3: lambda x: np.roll(x, -1, axis=1)  # move down
                            }

    def __init__(self, env):
        """Takes in the Alphazero wrapped version of the pyrat game.
        If you didnt, please wrap the game using pyrat_env.wrappers.AlphaZero

            game : type : PyRatEnv

            Attributes :
                - game : the PyRat gym game
                - player : the current player's turn to play
                - current board : board : tensor of size (9,21,15) with each layer containing :
                            0) Maze_left
                            1) Maze_up
                            2) Maze_right
                            3) Maze_down
                            4) Pieces of cheese location
                            5) Player score
                            6) Opponent score
                            7) Player location
                            8) Opponent location
                            9) Number of turns since the beginning
                            """
        self.env = env  # This is basically just a maze generator
        self.current_board = None
        self.rat_action = None


    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        initial_state = self.env.reset()
        turns = np.zeros((1, 21, 15), dtype= np.float16)
        self.current_board = np.concatenate((initial_state, turns))
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

    def getNextState(self, board, player, action, previous_move=None):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        self.current_board = np.copy(board)

        if player == -1:
            # make the move with the previous player's action and the current player's action
            self._make_move(action, previous_move)  # move, remove cheese and take care of scores
            self.current_board[9] += 1
        self._switch_perspective()

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
        self.board = np.copy(board)
        nb_turns = self.current_board[9][0][0]
        player_score = self.current_board[5][0][0]
        opponent_score = self.current_board[6][0][0]
        if player == -1:
            if nb_turns >= 200:
                if player_score > opponent_score:
                    return 1
                elif player_score == opponent_score:
                    return 0.00000001
                else:
                    return -1

            else:
                if player_score > 20.5:
                    return 1
                elif opponent_score > 20.5:
                    return -1
                elif player_score == 20.5 and opponent_score == 20.5:
                    return 0.00000001
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
        symmetries = Symmetries(board, pi)
        return symmetries()

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    def _make_move(self, player_action, opponent_action):
        """Makes the players move, then checks if cheeses have been taken and
            calculate the scores accordingly
            :arg player_action : The player's action
            :arg opponent_action : the opponent's action

            :return : None"""
        # make the move (move the players)
        # recalculate scores
        # take out the cheeses if needed

        # move the players
        player_position = np.where(self.current_board[7] == 1)
        if self.current_board[player_action][player_position[0][0]][player_position[1][0]] == 1:
            # Select the right board transform
            board_transform = self.board_transform_dict[player_action]
            # apply it to the player's position
            self.current_board[7] = board_transform(self.current_board[7])

        opponent_position = np.where(self.current_board[8] == 1)
        if self.current_board[opponent_action][opponent_position[0][0]][opponent_position[1][0]] == 1:
            # Select the right board transform
            board_transform = self.board_transform_dict[opponent_action]
            self.current_board[8] = board_transform(self.current_board[8])

        # recalculate the scores
        if self.current_board[4][player_position[0][0]][player_position[1][0]] == 1:  # check if there's a cheese on player 1
            if opponent_position == player_position:
                # Add 0.5 to each player's scores
                self.current_board[5] += 0.5
                self.current_board[6] += 0.5
            else:
                self.current_board[5] += 1
            self.current_board[4][player_position[0][0]][player_position[1][0]] = 0

        if self.current_board[4][opponent_position[0][0]][opponent_position[1][0]]:
            self.current_board[6] += 1
            self.current_board[4][opponent_position[0][0]][opponent_position[1][0]] = 0

        self.rat_action = None

    def _switch_perspective(self):
        old_player_score = np.copy(self.current_board[5])
        old_player_location = np.copy(self.current_board[7])

        # make the current player's location and score the opponent's
        self.current_board[5] = self.current_board[6]
        self.current_board[7] = self.current_board[8]

        # make the oppoennt the current player
        self.current_board[6] = old_player_score
        self.current_board[8] = old_player_location


class Symmetries:
    def __init__(self, obs, pi):
        self.original_obs = np.copy(obs)
        self.original_pi = pi.copy()

    def _rotate_right_obs(self, obs):
        obs = np.rot90(obs, -1, axes=(1, 2))
        original_L = np.copy(obs[0])
        original_U = np.copy(obs[1])
        original_R = np.copy(obs[2])
        original_D = np.copy(obs[3])
        obs[0], obs[1], obs[2], obs[3] = original_D, original_L, original_U, original_R
        return obs

    def _rotate_right_pi(self, pi):
        return_pi = [pi[3], pi[0], pi[1], pi[2]]
        return return_pi

    def _vertical_flip_obs(self, obs):
        obs = np.flip(obs, axis=2)
        original_up = np.copy(obs[1])
        original_down = np.copy(obs[3])
        obs[1], obs[3] = original_down, original_up
        return obs

    def _vertical_flip_pi(self, pi):
        return_pi = [pi[0], pi[3], pi[2], pi[1]]
        pi[0], pi[2] = pi[2], pi[0]
        return return_pi

    def _vertical_flip(self, obs, pi):
        return self._vertical_flip_obs(obs), self._vertical_flip_pi(pi)

    def _rotate_right(self, obs, pi):
        return self._rotate_right_obs(obs), self._rotate_right_pi(pi)

    def __call__(self):
        """Outputs all the symmetries of the obs, pi that were given on object instanciation
        There's 8 symmetries of the board"""
        obs, pi = self.original_obs, self.original_pi
        symmetries = [(obs, pi)]

        # Rotate once
        obs2, pi2 = obs.copy(), pi.copy()
        obs2, pi2 = self._rotate_right(obs2, pi2)
        # symmetries.append((obs2, pi2))

        # Rotate twice
        obs3, pi3 = obs.copy(), pi.copy()
        obs3, pi3 = self._rotate_right(obs3, pi3)
        obs3, pi3 = self._rotate_right(obs3, pi3)
        symmetries.append((obs3, pi3))

        # Rotate 3 times
        obs4, pi4 = obs.copy(), pi.copy()
        obs4, pi4 = self._rotate_right(obs4, pi4)
        obs4, pi4 = self._rotate_right(obs4, pi4)
        obs4, pi4 = self._rotate_right(obs4, pi4)
        # symmetries.append((obs4, pi4))

        # Flip vertically
        obs5, pi5 = obs.copy(), pi.copy()
        obs5, pi5 = self._vertical_flip(obs5, pi5)
        symmetries.append((obs5, pi5))
        # Flip vertically and rotate once
        obs6, pi6 = obs5.copy(), pi5.copy()
        obs6, pi6 = self._rotate_right(obs6, pi6)
        # symmetries.append((obs6, pi6))
        # Flip vertically and rotate twice
        obs7, pi7 = obs6.copy(), pi6.copy()
        obs7, pi7 = self._rotate_right(obs7, pi7)
        symmetries.append((obs7, pi7))
        # Flip vertically and rotate 3 times
        obs8, pi8 = obs7.copy(), pi7.copy()
        obs8, pi8 = self._rotate_right(obs8, pi8)
        # symmetries.append((obs8, pi8))

        return symmetries

