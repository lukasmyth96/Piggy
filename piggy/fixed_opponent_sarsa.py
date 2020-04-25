import os
import random

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from piggy.utils.common import won_or_lost
from piggy.utils.io import create_directory_path_with_timestamp
from piggy.evaluator import Evaluator
from piggy.agent import Agent


class FixedOpponentSarsa:

    def __init__(self, environment, opponent, eps, alpha, decay):
        """
        Learn an optimal policy against a fixed opponent using SARSA.

        Initialize learning rate α, exploration rate ε, decay rate λ and discount factor γ
        Initialize state-action value function Q(s,a) arbitrarily for each s∈S, a∈A
        Let π(s|Q,ε) be our ε-greedy policy
        For episode in M:
            Initialize start of game state s ← (0, 0, 0)
            while game not over do:
                Sample action a~π(s|Q,ε) from policy
                Take action - observe reward r, new state s'
                Sample action a'~π(s'|Q,ε) from policy

                Q(s,a) ← Q(s,a) + α[(r + γQ(s',a')) - Q(s,a')]
                s ← s' , a ← a'

            ε ← ε·λ (decay ε)
            α ← α·λ (decay α)


        Parameters
        ----------
        environment: piggy.environment.Environment
        opponent: piggy.agent.Agent
        eps: float
        alpha: float
            learning rate
        decay: float
            factor by which we decay eps and alpha each episode e.g. ε' = decay * ε
        """

        self.environment = environment
        self.opponent = opponent

        self.eps = eps
        self.alpha = alpha
        self.decay = decay

        # Initialise state-action value function array Q(s, a) - 4D array (your_score, opponent_score, turn_score, action)
        shape = (environment.target_score,)*3 + (2,)
        self._Q = np.random.random(size=shape)

    def Q(self, s, a):
        """
        Getter for Q(s, a)
        Parameters
        ----------
        s: tuple
            (your score, opponent score, turn score)
        a: int
            1 for roll, 0 for hold

        Returns
        -------
        q: float
        """
        if won_or_lost(s, target_score=self.environment.target_score):
            q = 0  # SARSA algorithm requires that Q(s',⋅)=0 in all terminal states
        else:
            q = self._Q[s[0], s[1], s[2], a]
        return q

    def run(self, episodes, evaluate_every, output_dir):
        """
        Run SARSA algorithm
        Parameters
        ----------
        episodes: int
        evaluate_every: int
            number of episodes between successive evaluations against fixed opponent
        output_dir: str
        """
        # Metrics
        tensorboard_writer = tf.summary.create_file_writer(output_dir)
        latest_win_rate = 0

        progress_bar = tqdm(range(episodes))

        for episode in progress_bar:

            state = (0, 0, 0)

            game_over = False
            action = 1  # always roll at start of game
            while not game_over:

                progress_bar.set_description('state: {} - latest win rate: {:.1%} -'.format(state, latest_win_rate), refresh=True)

                # Take chosen action in current state
                new_state, reward, go_again = self.environment.take_action(state, action)
                game_won = bool(reward)

                # If it's no longer our turn then the opponent acts as part of environment and transitions us to the
                # next state in which it is our turn
                game_lost = False
                if not go_again and not game_won:
                    new_state, game_lost = self.opponents_turn(new_state)

                # Select action from new state - will be None if new_state is terminal
                new_action = self.select_e_greedy_action(new_state)

                # Update Q(s,a) - Note Q(s',a') is forced to 0 if new_state s' is terminal
                self._Q[state[0], state[1], state[2], action] += \
                    self.alpha * ((reward + self.Q(new_state, new_action)) - self.Q(state, action))

                # Increment
                state = new_state
                action = new_action
                game_over = game_won or game_lost

                # Periodically evaluate against fixed opponent - logs written to tensorboard
                if episode % evaluate_every == 0:
                    latest_win_rate = self.evaluate_against_fixed_opponent()
                    with tensorboard_writer.as_default():
                        tf.summary.scalar('win rate vs fixed opponent', latest_win_rate, step=episode)
                        tf.summary.scalar('exploration rate ε', self.eps, step=episode)
                        tf.summary.scalar('learning rate α', self.alpha, step=episode)


        # Decay eps and alpha
        self.eps *= self.decay
        self.alpha *= self.decay

    def select_e_greedy_action(self, state):
        if won_or_lost(state, target_score=self.environment.target_score):
            action = None  # cannot sample action in terminal state
        elif random.random() < self.eps:
            action = random.randint(0, 1)  # random action
        else:
            action = np.argmax([self.Q(state, 0), self.Q(state, 1)])  # greedy action
        return action

    def opponents_turn(self, state):
        """
        The turn of the opponents is considered to be part of the environment such that at each new iteration of the
        SARSA algorithm we are in a state where it is our turn.
        Parameters
        ----------
        state: tuple
            Note - this is from the point of view of you NOT the opponent

        Returns
        -------
        new_state: tuple
            Note - this is converted back to the point of view of you NOT the opponent
        opponent_won: bool
            Whether or not the opponent has won by end of turn
        """

        state = (state[1], state[0], 0)  # Make state from pov of opponent

        go_again = True
        opponent_won = False
        # Opponent takes turn until it wins, rolls 1 or holds
        while go_again and not opponent_won:
            action = self.opponent.select_action(state)
            state, reward, go_again = self.environment.take_action(state, action)
            opponent_won = bool(reward)

        new_state = (state[1], state[0], 0)  # Revert back to state from pov of you

        return new_state, opponent_won

    def evaluate_against_fixed_opponent(self, num_games=250):
        """
        Evaluate current policy against fixed opponent to track progress
        Returns
        -------
        win_rate: float
        """
        current_policy = np.argmax(self._Q, axis=3)  # argmax along action axis gives policy array
        agent = Agent(initial_policy=current_policy)
        evaluator = Evaluator(self.environment, player0=agent, player1=self.opponent)
        win_rate, _ = evaluator.evaluate(num_games=num_games)
        return win_rate


if __name__ == '__main__':

    """ Learn optimal policy against the optimal policy using SARSA """
    from piggy.environment import Environment
    from definition import ROOT_DIR
    optimal_policy = np.load('/home/luka/PycharmProjects/Piggy/experiment_results/standard_pig_optimal_policy.npy')
    optimal_agent = Agent(initial_policy=optimal_policy)
    env = Environment(dice_sides=6, target_score=100)
    _sarsa = FixedOpponentSarsa(environment=env, opponent=optimal_agent, eps=0.25, alpha=0.05, decay=0.99)

    output_dir = create_directory_path_with_timestamp(destination_dir=os.path.join(ROOT_DIR, 'experiment_results', 'fixed_opponent_sarsa'))
    _sarsa.run(episodes=10000, evaluate_every=10)

















