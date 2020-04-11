import random

import numpy as np
from tqdm import tqdm


class Evaluator:

    def __init__(self, environment, player0, player1):
        """
        Class for evaluating performance of two agent's against each other
        Parameters
        ----------
        environment: piggy.environment.Environment
        player0: piggy.agent.Agent
        player1: piggy.agent.Agent
        """

        self.environment = environment
        self.player0 = player0
        self.player1 = player1
        self.player_num_to_player = {0: player1, 1: player1}

    def evaluate(self, num_games):
        """
        Play a specified number of games between the two agents - return their respective win rates
        Parameters
        ----------
        num_games: int

        Returns
        -------
        p0_win_rate: float
        p1_win_rate: float
        """
        player0_wins = []  # store results as list of bools for each game indicating if p0 won
        for _ in tqdm(range(num_games)):

            game_over = False
            current_player_idx = random.randint(0, 1)  # pick random player to start
            state = (0, 0, 0)  # state from pov of current player

            # loop until someone wins game
            while not game_over:

                # between turns switch player and make state from the new players point of view
                current_player_idx = int(not current_player_idx)  # switch player
                current_player = self.player_num_to_player[current_player_idx]
                state = (state[1], state[0], 0)  # switch your_score and opponents_score and set turn_score to 0
                go_again = True  # whether it's still current players go

                # loop until time to switch players because player decided to hold or rolled a 1
                while go_again:
                    action = current_player.select_action(state)
                    state, reward, go_again = self.environment.take_action(state, action)
                    if reward == 1:
                        player0_wins += (current_player_idx == 0)
                        game_over = True

        p0_win_rate = np.mean(player0_wins)
        p1_win_rate = 1 - p0_win_rate

        return p0_win_rate, p1_win_rate


if __name__ == '__main__':
    """ testing """
    from piggy.utils.create_policy import hold_at_n_policy
    from piggy.environment import Environment
    from piggy.agent import Agent
    env = Environment(dice_sides=6, target_score=100)
    p0 = Agent(initial_policy=hold_at_n_policy(target_score=100, hold_at=20))
    p1 = Agent(initial_policy=hold_at_n_policy(target_score=100, hold_at=20))
    _eval = Evaluator(env, p0, p1)
    _p0_win_rate, _p1_win_rate = _eval.evaluate(num_games=100)
    print('p0 win rate: {}  -  p1 win rate: {}'.format(_p0_win_rate, _p1_win_rate))
