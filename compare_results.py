#!/usr/bin/env python3

############################
#### compare_results.py ####
############################

import tools
import numpy as np

def error_rates(A, A_reco):
    '''
    Obtain the error rates by comparing the reconstructed adjacency matrix A_reco
    to the actual adjacency matrix A

    Arguments:
    1. A:       Actual adjacency matrix
    2. A_reco:  Reconstructed adjacency matrix
    '''


    assert type(A) == np.ndarray, "The actual adjacency matrix must be of type 'numpy.ndarray'"
    assert A.size > 0, "The actual adjacency matrix must not be empty"
    assert A.dtype == int, "The actual adjacency matrix must be of dtype 'int'"
    size = A.shape
    assert len(size) == 2, "The actual adjacency matrix must be 2D shape"
    assert size[0] == size[1], "The actual adjacency matrix must be a square matrix"
    assert np.allclose(A, A.T) == True, "The actual adjacency matrix must be symmetric"
    assert (np.diag(A) == 0).all() == True, "The actual adjacency matrix must not have self-loop"
    assert np.min(A) == 0, "The actual adjacency matrix elements must be non-negative"
    assert np.max(A) <= 1, "The actual adjacency matrix here does not allow multiple edges"
    assert np.max(A) == 1, "The actual adjacency matrix has no links"


    assert type(A_reco) == np.ndarray, "The reconstricted adjacency matrix must be of type 'numpy.ndarray'"
    assert A_reco.size > 0, "The reconstructed adjacency matrix must not be empty"
    assert A_reco.dtype == int, "The reconstructed adjacency matrix must be of dtype 'int'"
    size_reco = A_reco.shape
    assert len(size_reco) == 2, "The reconstructed adjacency matrix must be 2D shape"
    assert size_reco[0] == size_reco[1], "The reconstructed adjacency matrix must be a square matrix"
    assert np.allclose(A_reco, A_reco.T) == True, "The reconstructed adjacency matrix must be symmetric"
    assert (np.diag(A_reco) == 0).all() == True, "The reconstructed adjacency matrix must not have self-loop"
    assert np.min(A_reco) == 0, "The reconstructed adjacency matrix elements must be non-negative"
    assert np.max(A_reco) <= 1, "The reconstructed adjacency matrix here does not allow multiple edges"


    assert size == size_reco, "The actual and reconstructed adjacency matrices must be the same size"


    # Extract only the off-diagonl (upper triangle) elements to avoid double counting
    # NOTE: This function requires both A and A_reco to be symmetric
    A_off = tools.off_diag_upper(A)
    A_reco_off = tools.off_diag_upper(A_reco)


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
