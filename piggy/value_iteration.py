import os

import numpy as np
from tqdm import tqdm

from piggy.utils.common import get_all_playable_states, won, lost
from definition import ROOT_DIR


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
        - In pig - we use γ=1 with a terminal reward R=1 such that V(s) is the expected probability of winning from state s

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
        if environment.dice_sides == 2 and not playing_piglet:
            print('\nWarning! - Set playing_piglet=True if you are playing with piglet rules\n')

        if environment.dice_sides != 2 and playing_piglet:
            print('\nWarning! - You have set playing_piglet=True but dice_sides={}'.format(environment.dice_sides))

        self.environment = environment
        self.eps = eps
        self.playing_piglet = playing_piglet

        self.states = get_all_playable_states(target_score=environment.target_score)
        self._V = np.random.random(size=(environment.target_score+1,)*3)

        self.policy = np.random.random(size=(environment.target_score+1,)*3)

    def V(self, state):
        """
        Getter method for value function - Returns 1 if already won, 0 if lost and V(s) otherwise.
        Parameters
        ----------
        state: tuple

        Returns
        -------

        """
        if won(state, self.environment.target_score):
            return 1
        elif lost(state, self.environment.target_score):
            return 0
        else:
            return self._V[state[0], state[1], state[2]]

    def run(self):

        # Perform value iteration on disjoint subsets of states in which the sum of your score and opponents score equals
        # some value - starting with 2*(target-1) (i.e. both 1 away from winning) and working backward to 0 (start of game)
        # This technique was reported in Neller (2004) to improve convergence rate.
        for score_sum in tqdm(range(2 * (self.environment.target_score - 1), -1, -1)):

            delta = 1  # arbitrary number > eps to ensure while loop starts
            while delta >= self.eps:
                delta = 0
                for s in self.states:

                    if s[0] + s[1] == score_sum:

                        old_v = self.V(s)

                        # If you hold - prob of winning = 1 - prob opponent winning
                        v_hold = 1 - self.V((s[1], s[0] + s[2], 0))

                        # If you roll
                        dice_sides = self.environment.dice_sides
                        # Line below is to account for fact that piglet you score 1 for heads rather than 2
                        scoring_rolls = [1] if self.playing_piglet else list(range(2, dice_sides+1))
                        v_roll = (1 / dice_sides) * ((1 - self.V((s[1], s[0], 0))) +
                                                     sum([self.V((s[0], s[1], s[2] + roll_score))
                                                          for roll_score in scoring_rolls]))

                        new_v = max(v_hold, v_roll)
                        self._V[s[0], s[1], s[2]] = new_v
                        self.policy[s[0], s[1], s[2]] = np.argmax([v_hold, v_roll])
                        delta = max(delta, abs(new_v - old_v))


    def get_playable_state_to_value(self):
        """
        Returns dict mapping each playable state to it's current estimated value

        Returns
        -------
        state_to_value: dict
        """
        return {state: self.V(state) for state in self.states}

    def save(self, output_dir):
        """
        Save value function as .npy
        Parameters
        ----------
        output_dir: str
        """
        filepath_template = os.path.join(output_dir, '{}__{}_side_die__target_{}.npy'
                                         .format('{}',
                                                 self.environment.dice_sides,
                                                 self.environment.target_score))

        vf_filepath = filepath_template.format('value_func')
        np.save(vf_filepath, self._V)

        policy_filepath = filepath_template.format('policy')
        np.save(policy_filepath, self.policy)




if __name__ == '__main__':
    """ Testing """
    from piggy.environment import Environment
    _env = Environment(dice_sides=6, target_score=100)

    # This will give the CORRECT result for piglet
    valit = ValueIteration(environment=_env, eps=0.001, playing_piglet=False)
    valit.run()
    valit.save(output_dir=os.path.join(ROOT_DIR, 'experiment_results'))
    print('stop here')





