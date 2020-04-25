import itertools


def won(state, target_score):
    return (state[0] + state[2]) >= target_score


def lost(state, target_score):
    return state[1] >= target_score


def won_or_lost(state, target_score):
    return won(state, target_score) or lost(state, target_score)


def get_all_playable_states(target_score):
    """
    Returns a list of all playable (your_score, opponents_score, turn_score) state tuples.

    A playable state defined as one where you have not yet won or lost.

    Parameters
    ----------
    target_score: int

    Returns
    -------
    playable_states: list[tuples]
    """
    all_states = list(itertools.product(list(range(target_score + 1)), repeat=3))
    playable_states = filter_non_playable_states(all_states, target_score)
    return playable_states


def filter_non_playable_states(states, target_score):
    """
    Filter states in which you have already won or lost
    Parameters
    ----------
    states: list[tuple]
        (your_score, opponents_score, turn_score)
    target_score: int

    Returns
    -------
    playable_states: list[tuple]
    """

    return [state for state in states if not (won(state, target_score) or lost(state, target_score))]
