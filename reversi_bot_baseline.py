import copy
from datetime import datetime

import numpy as np
import random as rand
import reversi_baseline

TIME_PER_TURN = 3


class ReversiBot:

    def __init__(self, move_num):
        self.move_num = move_num
        self.turn = -1

    def make_move(self, state):
        '''
        This is the only function that needs to be implemented for the lab!
        The bot should take a game state and return a move.

        The parameter "state" is of type ReversiGameState and has two useful
        member variables. The first is "board", which is an 8x8 numpy array
        of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
        there is a 1 that means the spot has one of player 1's stones. If
        there is a 2 on the spot that means that spot has one of player 2's
        stones. The other useful member variable is "turn", which is 1 if it's
        player 1's turn and 2 if it's player 2's turn.

        ReversiGameState objects have a nice method called get_valid_moves.
        When you invoke it on a ReversiGameState object a list of valid
        moves for that state is returned in the form of a list of tuples.

        Move should be a tuple (row, col) of the move you want the bot to make.
        '''
        self.start = datetime.now()
        self.turn = state.turn
        max_score = -np.inf
        best_move = None
        depth = 2
        while (datetime.now() - self.start).seconds < TIME_PER_TURN:
            score, move = self.minimax(state, depth)
            if move is None:
                break
            if score > max_score:
                max_score = score
                best_move = move
            depth += 1

        if best_move is None:
            valid_moves = state.get_valid_moves()
            best_move = rand.choice(valid_moves)

        return best_move

    def minimax(self, start_state, total_depth):
        def dfs(state, move, depth, is_max, alpha, beta):
            # If we're out of time, abort
            if (datetime.now() - self.start).seconds >= TIME_PER_TURN:
                return -np.inf, None

            turned_tiles = self.move_state(state, move)

            # We're at the depth limit
            if depth == -1:
                score = self.evaluate_state(state)
                self.reverse_move(state, move, turned_tiles)
                return score, move

            # If there are no valid moves, we're at a leaf node
            valid_moves = state.get_valid_moves()
            if len(valid_moves) == 0:
                score = self.evaluate_state(state)
                self.reverse_move(state, move, turned_tiles)
                return score, move

            if is_max:
                bssf = -np.inf
                for next_move in valid_moves:
                    score = dfs(state, next_move, depth - 1, False, alpha, beta)[0]
                    if score > bssf:
                        bssf = score
                        if move is None:
                            move = next_move
                    alpha = max(alpha, bssf)
                    if beta <= alpha:
                        break
                self.reverse_move(state, move, turned_tiles)
                return bssf, move
            else:
                bssf = np.inf
                for next_move in valid_moves:
                    score = dfs(state, next_move, depth - 1, True, alpha, beta)[0]
                    bssf = min(bssf, score)
                    beta = min(beta, bssf)
                    if beta <= alpha:
                        break
                self.reverse_move(state, move, turned_tiles)
                return bssf, move

        return dfs(copy.deepcopy(start_state), None, total_depth, True, -np.inf, np.inf)

    def move_state(self, state, move):
        if move is None:
            return []

        turned_tiles = []
        state.board[move[0], move[1]] = state.turn

        def dfs(tile, delta_x, delta_y):
            if tile[0] < 0 or tile[0] > 7 or tile[1] < 0 or tile[1] > 7:
                return False
            if state.board[tile[0], tile[1]] == 0:
                return False
            elif state.board[tile[0], tile[1]] == state.turn:
                return True
            else:
                valid = dfs((tile[0] + delta_y, tile[1] + delta_x), delta_x, delta_y)
                if valid:
                    state.board[tile[0], tile[1]] = state.turn
                    turned_tiles.append(tile)
                return valid

        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                dfs((move[0] + y, move[1] + x), x, y)

        state.turn = 1 if state.turn == 2 else 2
        return turned_tiles

    def reverse_move(self, state, move, turned_tiles):
        if move is None:
            return []

        state.turn = 1 if state.turn == 2 else 2

        opponent = 1 if state.turn == 2 else 2
        state.board[move[0], move[1]] = 0
        for tile in turned_tiles:
            state.board[tile[0], tile[1]] = opponent

    # The heuristic evaluation function in all its glory
    def evaluate_state(self, state):
        if state.turn == self.turn:
            return len(state.get_valid_moves())
        else:
            return -len(state.get_valid_moves())
