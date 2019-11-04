#!/usr/bin/env python3

import numpy as np


def er_random(N, p):
    '''
    Construct an unweighted bi-directional ER random network without self-loop

    Arguments:
    1. N:    Total number of nodes in the network
    2. p:    Connection probability

    Returns:
    1. A:    Adjacency matrix
    '''
    assert type(N) == int and N > 0, "N must be a positive integer"
    assert (type(p) == int or type(p) == float), "p must be a real number"
    assert p >= 0 and p <= 1, "p must be within 0 and 1 (inclusive)"

    if p == 0:
        print("[WARN] p = 0, the network has no links")
        A = np.zeros((N, N))
        return A

    elif p == 1:
        print("[WARN] p = 1, the network is fully connected")
        A = np.ones((N, N))
        return A

    else:
        # Generate a random matrix with size N x N
        # Each element takes value between 0 and 1
        A = np.random.rand(N, N)

        # Set the connectivity based on input probability threshold p
        A = A < p
        A = A.astype(int)

        # Extract only the upper triangle elements
        # And set other (lower triangle + diagonal) to zero
        A = np.triu(A, 1)

        # Symmetrize the matrix
        A = A + A.T

        return A
