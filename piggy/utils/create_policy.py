import random

import numpy as np


""" Functions for creating simple policies"""


def random_policy(target_score):
    return np.random.choice([0, 1], size=(target_score, target_score, target_score))


def hold_at_n_policy(target_score, hold_at):
    """ Policy where you roll while turn_score < hold_at"""
    policy_array = np.zeros(shape=(target_score, target_score, target_score), dtype=int)
    policy_array[:, :, :hold_at] = 1
    return policy_array

