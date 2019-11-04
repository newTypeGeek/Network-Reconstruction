#!/usr/bin/env python3

import numpy as np
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from utils import base
from utils import network


def inverse_covariance(cov, measure_id, hidden_id):
    '''
    Obtain 2 inverse covariance matrices (see Returns section)
    from the node indices

    Arguments:
    1. cov:             Covariance matrix obtained from simulation
    2. measure_id:      Measured node indices
    3. hidden_id:       Hidden node indices

    Returns:
    1. cov_inv_m:       Inverse of covariance matrix without hidden node effect
                          Step 1:   Invert cov
                          Step 2:   Extract the block matrix of the inverse of cov from measure_id

    2. cov_m_inv:       Inverse of covariance matrix with hidden node effect
                          Step 1:   Extract the block matrix of cov from measure_id
                          Step 2:   Invert this block matrix
    '''
    assert type(cov) == np.ndarray, "cov must be of type 'numpy.ndarray'"
    assert cov.size > 0, "cov must not be empty"
    assert cov.dtype == int or cov.dtype == float, "cov must be of dtype 'int' or 'float'"
    assert np.isfinite(cov).all(), "Elements of cov must be finite real number"
    size = cov.shape
    assert len(size) == 2, "cov must be 2D shape"
    assert size[0] == size[1], "cov must be a square matrix"

    assert type(measure_id) == np.ndarray, "measure_id must be of type 'np.ndarray'"
    assert measure_id.size > 0, "measure_id must not be empty"
    assert measure_id.dtype == int, "measure_id must be of dtype 'int'"
    assert len(measure_id.shape) == 1, "measure_id must be 1D shape"
    assert (measure_id >= 0).all(), "measure_id elements must be non-negative integers"
    assert np.max(measure_id) < size[0], "measure_id elements must be smaller than cov size"

    assert type(hidden_id) == np.ndarray, "hidden_id must be of type 'np.ndarray'"
    assert hidden_id.size > 0, "hidden_id must not be empty"
    assert hidden_id.dtype == int, "hidden_id must be of dtype 'int'"
    assert len(hidden_id.shape) == 1, "hidden_id must be 1D shape"
    assert (hidden_id >= 0).all(), "hidden_id elements must be non-negative integers"
    assert np.max(hidden_id) < size[0], "hidden_id elements must be smaller than the input matrix size"

    num_m = len(measure_id)
    num_h = len(hidden_id)
    assert size[0] == (num_m + num_h), "Total number of elements in measure_id and hidden_id does not equal the number of rows of M"

    all_id = np.unique(np.concatenate((measure_id, hidden_id)))
    assert len(all_id) == size[0], "mesure_id and hidden_id contain common elements"

    # Compute cov_inv_m
    cov_inv = base.inverse(cov)
    _, cov_inv_m, _, _, _ = base.matrix_rearrange(cov_inv, measure_id, hidden_id)

    # Compute cov_m_inv
    _, cov_m, _, _, _ = base.matrix_rearrange(cov, measure_id, hidden_id)
    cov_m_inv = base.inverse(cov_m)

    return cov_inv_m, cov_m_inv


def stationary_check(W, tol=1e-9):
    '''
    Check if the weighted adjaceny matrix fullfils
    the stationarity condition.

    This can be done by verifying if the corresponding
    weighted Laplacian matrix has negative eigenvalues

    Arguments:
    1. W:               weighted adjacency matrix

    2. tol:             tolerance value for verifying the existence of
                        negative eigenvalues (default: 1e-9)

    Returns:
    1. is_stationary:  True if it is stationary given noise is weak
                       False if it is definitely non-stationary
    '''
    assert type(W) == np.ndarray, "A must be of type 'numpy.ndarray'"
    assert W.size > 0, "A must not be empty"
    assert W.dtype == int or W.dtype == float, "A must of dtype 'int' or 'float'"
    assert np.isfinite(W).all(), "Elements of A must be finite real numbers"
    size = W.shape
    assert len(size) == 2, "A must be 2D shape"
    assert size[0] == size[1], "A must be a square matrix"
    assert (np.diag(W) == 0).all(), "A must not have self-loop"

    assert (type(tol) == int or type(tol) == float) and tol > 0, "tol must be positive real number"

    # Construct the weighted Laplacian matrix
    L = network.laplacian(W)

    # Compute the eigenvalues of L
    eig_vals = base.eigen_values(L)

    # Check if there are negative eigenvalues
    # Note that weighted Laplacian matrix ALWAYS has at least one zero eigenvalue
    # The zero eigenvalues can be negative due to numerical error
    abs_min_eig_val = min(abs(eig_vals))

    print("Absolute minimum eigenvalue of L = ", abs_min_eig_val)
    if abs_min_eig_val < tol:
        print("The dynamics would probably fluctuate around a stable fixed point")
        is_stationary = True

    else:
        print("The dynamics would blow up!")
        is_stationary = False

    return is_stationary
