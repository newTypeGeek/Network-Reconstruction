#!/usr/bin/env python3

import numpy as np


def lazy(N, n):
    '''
    Chose first n nodes as the measure nodes

    Arguments:
    1. N:     Total number of nodes
    2. n:     Number of measure nodes

    Returns:
    1. measure_id:    Measured node indices of the original (whole) network
    2. hidden_id      Hidden node indices of the original (whole) network
    '''
    assert type(N) == int and N > 0, "N must be a positive integer"
    assert type(n) == int and (n > 0 and n < N), "n must be a positive integer less than N"

    measure_id = np.arange(n)
    hidden_id = np.arange(n, N)

    return measure_id, hidden_id
