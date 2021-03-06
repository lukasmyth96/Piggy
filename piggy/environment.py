import random


class Environment:

    def __init__(self, dice_sides, target_score):
        """

        Parameters
        ----------
        dice_sides: int
            number of sides on dice
        target_score: int
        """
        self.dice_sides = dice_sides
        self.target_score = target_score

    def take_action(self, state, action):
        """
        Take an action from current state
        Parameters
        ----------
        state: tuple
            (your_score, opponent_score, turn_score)
        action: int
            1 for roll, 0 for hold

        Returns
        -------
        new_state: tuple
            new state from the point of view of the player who just played
        reward: int
            1 for win, 0 otherwise
        go_again: bool
            whether it's this players go again
        """
        if action == 1:  # if action is to roll
            dice_roll = random.randint(1, self.dice_sides)
            if dice_roll == 1:
                go_again = False
                turn_score = 0  # rolling a 1 resets turn score to zero and transitions to other players turn
            else:
                turn_score = state[2] + dice_roll  # rolling 2-6 adds that number to your turn score and you get to go again
                go_again = True
            new_state = state[:2] + (turn_score,)
        else:  # action is to hold
            go_again = False  # default value
            new_state = (state[0] + state[2], state[1], 0)  # add current turn score to total

        reward = int((new_state[0] + new_state[2]) >= self.target_score)

        return new_state, reward, go_again

