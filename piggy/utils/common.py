import itertools


def get_all_states(target_score, include_winning_states=True):
    """
    Returns a list of all possible (your_score, opponents_score, turn_score) state tuples
    Parameters
    ----------
    target_score: int
    include_winning_states: bool, optional
        whether to include an axis for winning states

    Returns
    -------
    states: list[tuples]
    """
    return list(itertools.product(list(range(target_score + int(include_winning_states))), repeat=3))

