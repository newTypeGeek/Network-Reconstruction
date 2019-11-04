#!/usr/bin/env python3

import numpy as np


def gaussian(A, mean, std):
    '''
    Construct a weighted bi-directional network with Gaussian
    distributed coupling from the adjacency matrix

    Arguments:
    1. A:        Adjacency matrix
    2. mean:     Mean value of Gaussian distribution
    3. std:      Standard deviation of Gaussian distribution

    Returns:
    1. W:        Weighted adjacency matrix
    '''
    assert type(A) == np.ndarray, "A must be of type 'numpy.ndarray'"
    assert A.size > 0, "A must not be empty"
    assert A.dtype == int, "A must be of dtype 'int'"
    size = A.shape
    assert len(size) == 2, "A must be 2D shape"
    assert size[0] == size[1], "A must be a square matrix"
    assert np.allclose(A, A.T), "A must be symmetric"
    assert (np.diag(A) == 0).all(), "Diagonal elements of A must all be zero"
    assert np.min(A) == 0, "Elements of A must be either 0 or 1"
    assert np.max(A) <= 1, "Elements of A must be either 0 or 1"
    assert np.max(A) == 1, "Elements of A are all zero"

    assert (type(mean) == int or type(mean) == float), "mean must be of type 'int' or 'float'"
    assert np.isfinite(mean), "mean must be a finite real number"

    assert (type(std) == int or type(std) == float), "std must be of type 'int' or 'float'"
    assert np.isfinite(std) and std > 0, "std must be a finite positive number"

    # Generate a random matrix with same size as A
    # with elements follow Gaussian distribution
    G = np.random.normal(loc=mean, scale=std, size=A.shape)

    # Extract only the upper triangle elements
    # And set other (lower triangle + diagonal) to zero
    G = np.triu(G, 1)

    # Symmetrize the matrix
    G = G + G.T

    # Element-wise multiplication
    W = G * A

    return W
