import numpy as np

from piggy.utils.common import get_all_states


class ValueIteration:

    def __init__(self, environment, eps):
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
        - In pig - we use γ=1 such that V(s) can be interpreted as the probability of winning in state s

        - In pig we have only the terminal reward R=1 on winning

        - In pig the value (win probability) from a new state s' = (your_score, opponents_score, 0) after just
        holding or rolling a 1 can only be calculated as (1 - probability of opponent winning from state
        (opponents_score, your_score, 0)

        Parameters
        ----------
        environment: piggy.environment.Environment
        eps: float
            minimum max difference between V(s) between successive iterations across all states s
        """
        self.environment = environment
        self.eps = eps

        self.states = get_all_states(target_score=environment.target_score)
        self._V = np.random.random(size=(environment.target_score,)*3)

    def run(self):

        delta = self.eps
        while delta >= self.eps:
            for s in self.states[::-1]:
                old_v = self.V[np.array(s)]

                # If you hold - prob of winning = 1 - prob opponent winning
                v_hold = 1 - self.V[s[1], s[0] + s[2], 0]

                # If you roll
                dice_sides = self.environment.dice_sides
                v_roll = (1 / dice_sides) * ((1 - self.V[s[1], s[0], 0]) +
                                             sum([self.V[s[0], s[1], s[2] + roll]
                                                  for roll in range(1, dice_sides + 1)]))

                new_v = max(v_hold, v_roll)
                self.V[np.array(s)] = new_v
                delta = max(delta, abs(new_v - old_v))


if __name__ == '__main__':
    """ Testing """
    from piggy.environment import Environment
    _env = Environment(dice_sides=2, target_score=2)
    valit = ValueIteration(environment=_env, eps=0.01)
    valit.run()
    print('stop here')





