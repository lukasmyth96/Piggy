import random
import copy

import numpy as np

from piggy.utils.common import get_all_playable_states, won, lost


class ValueIteration:

    def __init__(self, environment, eps, playing_piglet=False):
        """
        For each s∈ S, initialize V(s) arbitrarily.
        Repeat
            Δ ← 0
            For each s∈ S,
                v ← V(s)
                V(s) ← max_a Σ_s' (P(s'|s,a) * [R + γV(s')])
                Δ ← max(Δ, |v - V(s)|)
        until Δ < ε

        Notes -
        - In pig - we use γ=1 with a terminal reward R=1 such that V(s) can be interpreted as the probability of winning in state s

        - In pig the value (win probability) of a new state s' you arrive in after just
        holding or rolling a 1 can only be calculated as (1 - probability of opponent winning from state
        (opponents_score, your_score, 0)

        Parameters
        ----------
        environment: piggy.environment.Environment
        eps: float
            minimum max difference between V(s) between successive iterations across all states s
        playing_piglet: bool, optional
            affects line 67 - for pig the scoring rolls are 2-num_dice_sides but for piglet rolling a head scores
            a 1 not a 2 (as it would if we considered a coin to be a two sided die with 1 and 2 on it)
        """
        self.environment = environment
        self.eps = eps
        self.playing_piglet = playing_piglet

        self.states = get_all_playable_states(target_score=environment.target_score)
        self._V = np.random.random(size=(environment.target_score+1,)*3)

    def V(self, state):
        if won(state, self.environment.target_score):
            return 1
        elif lost(state, self.environment.target_score):
            return 0
        else:
            return self._V[state[0], state[1], state[2]]

    def run(self):

        delta = self.eps
        while delta >= self.eps:
            delta = 0
            for s in self.states:

                old_v = self.V(s)

                # If you hold - prob of winning = 1 - prob opponent winning
                v_hold = 1 - self.V((s[1], s[0] + s[2], 0))

                # If you roll
                dice_sides = self.environment.dice_sides
                # Line below is to account for fact that piglet you score 1 for heads rather than 2
                scoring_rolls = list(range(1, dice_sides)) if self.playing_piglet else list(range(2, dice_sides+1))
                v_roll = (1 / dice_sides) * ((1 - self.V((s[1], s[0], 0))) +
                                             sum([self.V((s[0], s[1], s[2] + roll_score))
                                                  for roll_score in scoring_rolls]))

                new_v = max(v_hold, v_roll)
                self._V[s[0], s[1], s[2]] = new_v
                delta = max(delta, abs(new_v - old_v))

            print('max delta: {:.5f}'.format(delta))

    def get_state_to_value(self):
        """
        Returns
        -------
        state_to_value: dict
        """
        return {state: self.V(state) for state in self.states}


if __name__ == '__main__':
    """ Testing """
    from piggy.environment import Environment
    _env = Environment(dice_sides=2, target_score=2)

    # This will give the WRONG result for piglet:
    valit = ValueIteration(environment=_env, eps=0.0001, playing_piglet=False)
    valit.run()
    print('stop here')

    # This will give the CORRECT result for piglet
    valit = ValueIteration(environment=_env, eps=0.0001, playing_piglet=True)
    valit.run()
    print('stop here')





