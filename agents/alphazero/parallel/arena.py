from .mcts import RootParentNode, Node
from tqdm.auto import trange
class Arena:

    def __init__(self, pmcts, nmcts, game):
        """Pits one mcts against another
            args :
                - pmcts : (.mcts.MCTS) the mcts for the previous net
                - nmcts : (.mcts.MCTS) the mcts for the new net to be tested"""

        self.player1 = pmcts
        self.player2 = nmcts
        self.game = game

        # put the mcts in eval mode
        self.player1.eval()
        self.player2.eval()

    def playGame(self):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2,0 if draw) from p1's perspective

        """
        self.player1.model.clear_cache()
        self.player2.model.clear_cache()
        p1_board = self.game.getInitBoard()
        p2_board, _ = self.game.getNextState(p1_board,1,0,previous_move = None) # just switches the perspective

        #Create the starting nodes for the mcts
        root_parent = RootParentNode(self.game)
        p1_node = Node(action=None, obs=p1_board[:9], done=False, reward=0, state=p1_board, player=1, mcts=self.player1, parent=root_parent)
        p2_node = Node(action=None, obs=p2_board[:9], done=False, reward=0, state=p2_board, player=1, mcts=self.player2, parent=root_parent)

        turn = 0

        while not p1_node.done:
            # get the player's moves and next childs
            # note : can be run in parallel
            p1_tree, p1_action, p1_next_child = self.player1.compute_action(p1_node)
            p2_tree, p2_action, p2_next_child = self.player2.compute_action(p2_node)

            # make the moves on the player nodes

            p1_node = p1_next_child.get_child(p2_action)
            p2_node = p2_next_child.get_child(p1_action)

        return p1_node.reward, p1_node

    def playGames(self, num):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in trange(num, desc = "Running evaluation game, part 1", unit = "game"):
            gameResult, final_state = self.playGame()
            if gameResult == 1:
                oneWon +=1
            elif gameResult == -1:
                twoWon += 1
            else :
                draws +=1

            print(f"""\nGame ended, winner is {gameResult}, final state is:
nb_turns : {final_state.state[9][0][0]}
p1_score : {final_state.state[5][0][0]}
p2_score : {final_state.state[6][0][0]}""")

        # Exchange players
        self.player2, self.player1 = self.player1, self.player2

        for _ in trange(num, desc="Running evaluation game, part 2", unit="game"):
            gameResult, final_state = self.playGame()
            if gameResult == - 1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            print(f"""\nGame ended, winner is {gameResult}, final state is:
nb_turns : {final_state.state[9][0][0]}
p1_score : {final_state.state[5][0][0]}
p2_score : {final_state.state[6][0][0]}""")

        return oneWon,twoWon, draws
