import random

import numpy as np
from tqdm import tqdm

from piggy.utils.common import won


class FixedOpponentSarsa:

    def __init__(self, environment, opponent, eps, alpha, decay):
        """
        Learn an optimal policy against a fixed opponent using SARSA.

        Initialize learning rate α, exploration rate ε and discount factor γ
        Initialize state-action value function Q(s,a) arbitrarily for each s∈S, a∈A
        Let π(s|Q,ε) be our ε-greedy policy
        For episode in M:
            Initialize start of game state s ← (0, 0, 0)

            Sample action a~π(s|Q,ε) from policy
            Take action - observe reward r, new state s'
            Sample action a'~π(s'|Q,ε) from policy

            Q(s,a) ← Q(s,a) + α[(r + γQ(s',a')) - Q(s,a')]
            s ← s' , a ← a'

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

        # Initialise State action value function array Q(s, a)
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
        if won(s, target_score=self.environment.target_score):
            q = 1
        else:
            q = self._Q[s[0], s[1], s[2], a]
        return q

    def run(self, episodes):

        for episode in tqdm(range(episodes)):

            state = (0, 0, 0)  # start of new game

            game_over = False
            action = 1  # always roll at start of game
            while not game_over:

                # Take chosen action in current state
                new_state, reward, go_again = self.environment.take_action(state, action)
                game_won = bool(reward)

                # If it's no longer our turn then the opponent acts as part of environment and transitions us to the
                # next state in which it is our turn
                game_lost = False
                if not go_again and not game_won:
                    new_state, game_lost = self.opponents_turn(state)

                # Select action from new state
                new_action = self.select_e_greedy_action(new_state)

                # Update Q(s,a)
                q_s_a = self.Q(state, action)  # Q(s, a)
                self._Q[state[0], state[1], state[2], action] += \
                    self.alpha * ((reward + self.Q(new_state, new_action)) - q_s_a)

                # Increment
                state = new_state
                action = new_action
                game_over = game_won or game_lost

        # Decay eps and alpha
        self.eps *= self.decay
        self.alpha *= self.decay

    def select_e_greedy_action(self, state):
        # select e-greedy action
        if random.random() < self.eps:
            action = random.randint(0, 1)  # random action
        else:
            action = np.argmax([self.Q(state, 0), self.Q(state, 1)])  # greedy action
        return action

    def opponents_turn(self, state):

        # Make state from pov of opponent
        state = (state[1], state[0], 0)

        go_again = True
        game_won = False
        while go_again and not game_won:
            action = self.opponent.select_action(state)
            new_state, reward, go_again = self.environment.take_action(state, action)
            game_won = bool(reward)

        # Revert back to state from pov of you
        state = (state[1], state[0], 0)

        return state, game_won


if __name__ == '__main__':

    """ Learn optimal policy against the optimal policy using SARSA """
    from piggy.agent import Agent
    from piggy.environment import Environment
    from piggy.evaluator import Evaluator
    optimal_policy = np.load('/home/luka/PycharmProjects/Piggy/experiment_results/standard_pig_optimal_policy.npy')
    optimal_agent = Agent(initial_policy=Agent)
    env = Environment(dice_sides=6, target_score=100)
    _sarsa = FixedOpponentSarsa(environment=env, opponent=optimal_agent, eps=0.5, alpha=0.05, decay=0.99)
    _sarsa.run(episodes=10000)

















