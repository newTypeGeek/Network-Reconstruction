#!/usr/bin/env python3

import numpy as np
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from utils import base


def error_rates(A, A_reco):
    '''
    Obtain the error rates by comparing the reconstructed adjacency matrix A_reco
    to the actual adjacency matrix A and print the error rates to the console

    Arguments:
    1. A:       Actual adjacency matrix
    2. A_reco:  Reconstructed adjacency matrix

    Returns:
    1. fn:        Number of false negative
    2. fp:        Number of false positive
    3. num_link:  Number of bi-directional links
    '''
    assert type(A) == np.ndarray, "A must be of type 'numpy.ndarray'"
    assert A.size > 0, "A must not be empty"
    assert A.dtype == int, "Elements in A must be of dtype 'int'"
    size = A.shape
    assert len(size) == 2, "A must be of 2D shape"
    assert size[0] == size[1], "A must be a square matrix"
    assert np.allclose(A, A.T), "A must be symmetric"
    assert (np.diag(A) == 0).all(), "Diagonal elements of A must all be zero"
    assert np.min(A) == 0, "Elements in A must be either 0 or 1"
    assert np.max(A) <= 1, "Elements in A must be either 0 or 1"
    assert np.max(A) == 1, "All elements in A are zero"

    assert type(A_reco) == np.ndarray, "A_reco must be of type 'numpy.ndarray'"
    assert A_reco.size > 0, "A_reco must not be empty"
    assert A_reco.dtype == int, "Elements in A_reco must be of dtype 'int'"
    size_reco = A_reco.shape
    assert len(size_reco) == 2, "A_reco must be of 2D shape"
    assert size_reco[0] == size_reco[1], "A_reco must be a square matrix"
    assert np.allclose(A_reco, A_reco.T), "A_reco must be symmetric"
    assert (np.diag(A_reco) == 0).all(), "Diagonal elements of A_reco must all be zero"
    assert np.min(A_reco) == 0, "Elements in A_reco must be either 0 or 1"
    assert np.max(A_reco) <= 1, "Elements in A must be either 0 or 1"
    assert np.max(A_reco) == 1, "All elements in A_reco are zero"

    assert size == size_reco, "A and A_reco must have the same shape"

    # Extract only the off-diagonl (upper triangle) elements to avoid double counting
    # NOTE: This function requires both A and A_reco to be symmetric
    A_off = base.off_diag_upper(A)
    A_reco_off = base.off_diag_upper(A_reco)

    # Number of false positive fp, and false negative fn
    fp = np.sum(A_reco_off[A_off == 0] == 1)
    fn = np.sum(A_reco_off[A_off == 1] == 0)

    # Number of bi-directional links
    # NOTE: This function requires A to have at least one link
    num_link = np.sum(A_off)

    # Normalize fp and fn to get error rates
    fpr = fp/num_link
    fnr = fn/num_link
    print("Number of bidirectional links = {}".format(num_link))
    print("Number of false positive = {}".format(fp))
    print("Number of false negative = {}".format(fn))
    print("False positive rate = {:.4f}%".format(fpr*100))
    print("False negative rate = {:.4f}%".format(fnr*100))

    return fn, fp, num_link
