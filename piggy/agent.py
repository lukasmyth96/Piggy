

class Agent:

    def __init__(self, initial_policy):
        """

        Parameters
        ----------
        initial_policy: np.ndarray
            [target_score, target_score, target_score] binary array in which 1 indicates a policy of rolling and 0 hold
        """
        self.policy = initial_policy

    def select_action(self, state):
        """
        Parameters
        ----------
        state: tuple
            (your_score, opponent_score, turn_score)

        Returns
        -------
        action: int
            1 for roll, 0 for hold
        """
        your_score, opponent_score, turn_score = state
        action = self.policy[your_score, opponent_score, turn_score] if turn_score > 0 else 1
        return action

    def update_policy(self, state, new_action):
        your_score, opponent_score, turn_score = state
        self.policy[your_score, opponent_score, turn_score] = new_action

