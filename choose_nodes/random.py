#!/usr/bin/env python3

import numpy as np


def random(N, n):
    '''
    Chose n nodes as the measure nodes randomly

    Arguments:
    1. N:     Total number of nodes
    2. n:     Number of measure nodes

    Returns:
    1. measure_id:    Measured node indices of the original network
    2. hidden_id      Hidden node indices of the original network
    '''
    assert type(N) == int and N > 0, "N must be a positive integer"
    assert type(n) == int and (n > 0 and n < N), "n must be a positive integerless than N"

    measure_id = np.random.choice(N, n, replace=False)

    all_id = np.arange(N)
    hidden_id = np.setdiff1d(all_id, measure_id)

    return measure_id, hidden_id
