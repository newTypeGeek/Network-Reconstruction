#!/usr/bin/env python3

import numpy as np


def eigen_values(M):
    '''
    Compute the eigenvalues of a general square matrix M

    Arguments:
    1. M:         A general square matrix

    Returns:
    1. eig_vals:   Eigen-values of matrix M
    '''
    assert type(M) == np.ndarray, "M must be of type 'numpy.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"

    eig_vals, _ = np.linalg.eig(M)

    return eig_vals


def inverse(M, tol=1e5):
    '''
    Compute the inverse of a square matrix M

    Arguments:
    1. M:       A general square matrix
    2. tol:     Tolerance value of condition number (default: 1e5)

    Returns:
    1. M_inv:   Inverse of the matrix M
    '''
    assert type(M) == np.ndarray, "M must be of type 'numpy.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"
    assert (type(tol) == int or type(tol) == float) and np.isfinite(tol) and tol >= 0, "tol must be a non-negative number"

    cond_num = np.linalg.cond(M)

    M_inv = None
    if cond_num < tol:
        M_inv = np.linalg.inv(M)
    else:
        print("M is highly singular")

    return M_inv


def off_diag_upper(M):
    '''
    Extract the off-diagonal elements (upper triangle) of a square matrix

    Arguments:
    1. M:       input matrix

    Returns:
    1. off:     off-diagonal elements (upper triangle) of input matrix M
    '''

    assert type(M) == np.ndarray, "M must be of type 'np.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"

    off_upper = M[np.triu(np.ones(size), 1) == 1]

    return off_upper


def index_recover(n):
    '''
    Recover the original row and column indices of the
    flatten upper triangle vector using triu method (see off_diag_upper)

    Arguments:
    1. n:    The original matrix size

    Returns:
    1. row:  Row indices
    2. col:  Column indices
    '''

    assert type(n) == int and n > 0, "n must be a positive integer"

    v_size = int(n*(n-1)/2)

    # Recover the row index and column index
    row = np.zeros((v_size,))
    col = np.zeros((v_size,))

    for i in range(n-1):
        bin_size = n-1-i

        if i == 0:
            start = 0
            end = bin_size
        else:
            start = end
            end = start + bin_size

        row[start:end] = i * np.ones((bin_size,))
        col[start:end] = sorted(np.arange(n-1, i, -1))

    return row, col



def block_diag_up(M, measure_id):
    '''
    Extract the block matrix from matrix M with row and column 
    correspond to measured nodes

    Arguments:
    1. M:            The original square matrix M
    2. measure_id:   Measured node indices of the original matrix M

    Returns:
    1. B:          Block matrix with elements equal to the original matrix
                   formed among the measure nodes
    '''
    assert type(M) == np.ndarray, "M must be of type 'np.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"

    assert type(measure_id) == np.ndarray, "measure_id must be of type 'np.ndarray'"
    assert measure_id.size > 0, "measure_id must not be empty"
    assert measure_id.dtype == int, "measure_id must be of dtype 'int'"
    assert len(measure_id.shape) == 1, "measure_id must be 1D shape"
    assert (measure_id >= 0).all(), "measure_id elements must be non-negative integers"
    assert np.max(measure_id) < size[0], "measure_id elements must be smaller than the input matrix size"

    n = len(measure_id)
    B = np.zeros((n, n))

    if np.allclose(M, M.T):
        for i in range(n):
            B[i, i] = M[measure_id[i], measure_id[i]]

            for j in range(i+1, n):
                B[i, j] = M[measure_id[i], measure_id[j]]
                B[j, i] = B[i, j]
    else:
        for i in range(n):
            for j in range(n):
                B[i, j] = M[measure_id[i], measure_id[j]]

    return B


def block_diag_low(M, hidden_id):
    '''
    Extract the block matrix from matrix M with row and column 
    correspond to hidden nodes

    Arguments:
    1. M:            The original square matrix M
    2. hidden_id:    Hidden node indices of the original matrix M

    Returns:
    1. B:          Block matrix with elements equal to the original matrix
                   formed among the hidden nodes
    '''
    assert type(M) == np.ndarray, "M must be of type 'np.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"

    assert type(hidden_id) == np.ndarray, "hidden_id must be of type 'np.ndarray'"
    assert hidden_id.size > 0, "hidden_id must not be empty"
    assert hidden_id.dtype == int, "hidden_id must be of dtype 'int'"
    assert len(hidden_id.shape) == 1, "hidden_id must be 1D shape"
    assert (hidden_id >= 0).all(), "hidden_id elements must be non-negative integers"
    assert np.max(hidden_id) < size[0], "hidden_id elements must be smaller than the input matrix size"

    n = len(hidden_id)
    B = np.zeros((n, n))

    if np.allclose(M, M.T):
        for i in range(n):
            B[i, i] = M[hidden_id[i], hidden_id[i]]

            for j in range(i+1, n):
                B[i, j] = M[hidden_id[i], hidden_id[j]]
                B[j, i] = B[i, j]
    else:
        for i in range(n):
            for j in range(n):
                B[i, j] = M[hidden_id[i], hidden_id[j]]

    return B


def block_off_up(M, measure_id, hidden_id):
    '''
    Extract the block matrix from matrix M with row corresponds to measured nodes 
    and column corresponds to hidden nodes

    Arguments:
    1. M:            The original square matrix M
    2. measure_id:   Measured node indices of the original matrix M
    3. hidden_id:    Hidden node indices of the original matrix M

    Returns:
    1. B:          Block matrix with elements equal to the original matrix
                   formed with row corresponds to measured nodes and column
                   corresponds to hidden nodes
    '''
    assert type(M) == np.ndarray, "M must be of type 'np.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"

    assert type(measure_id) == np.ndarray, "measure_id must be of type 'np.ndarray'"
    assert measure_id.size > 0, "measure_id must not be empty"
    assert measure_id.dtype == int, "measure_id must be of dtype 'int'"
    assert len(measure_id.shape) == 1, "measure_id must be 1D shape"
    assert (measure_id >= 0).all(), "measure_id elements must be non-negative integers"
    assert np.max(measure_id) < size[0], "measure_id elements must be smaller than the input matrix size"

    assert type(hidden_id) == np.ndarray, "hidden_id must be of type 'np.ndarray'"
    assert hidden_id.size > 0, "hidden_id must not be empty"
    assert hidden_id.dtype == int, "hidden_id must be of dtype 'int'"
    assert len(hidden_id.shape) == 1, "hidden_id must be 1D shape"
    assert (hidden_id >= 0).all(), "hidden_id elements must be non-negative integers"
    assert np.max(hidden_id) < size[0], "hidden_id elements must be smaller than the input matrix size"

    n_m = len(measure_id)
    n_h = len(hidden_id)
    assert size[0] == (n_m + n_h), "Total number of elements in measure_id and hidden_id does not equal the number of rows of M"

    all_id = np.unique(np.concatenate((measure_id, hidden_id)))
    assert len(all_id) == size[0], "mesure_id and hidden_id contain common elements"

    for i in range(n_m):
        for j in range(n_h):
            B[i, j] = M[measure_id[i], hidden_id[j]]

    return B


def block_off_low(M, measure_id, hidden_id):
    '''
    Extract the block matrix from matrix M with row corresponds to hidden nodes 
    and column corresponds to measured nodes

    Arguments:
    1. M:            The original square matrix M
    2. measure_id:   Measured node indices of the original matrix M
    3. hidden_id:    Hidden node indices of the original matrix M

    Returns:
    1. B:          Block matrix with elements equal to the original matrix
                   formed with row corresponds to hidden nodes and column
                   corresponds to column nodes
    '''
    assert type(M) == np.ndarray, "M must be of type 'np.ndarray'"
    assert M.size > 0, "M must not be empty"
    assert M.dtype == int or M.dtype == float, "M must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all(), "Elements of M must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "M must be 2D shape"
    assert size[0] == size[1], "M must be a square matrix"

    assert type(measure_id) == np.ndarray, "measure_id must be of type 'np.ndarray'"
    assert measure_id.size > 0, "measure_id must not be empty"
    assert measure_id.dtype == int, "measure_id must be of dtype 'int'"
    assert len(measure_id.shape) == 1, "measure_id must be 1D shape"
    assert (measure_id >= 0).all(), "measure_id elements must be non-negative integers"
    assert np.max(measure_id) < size[0], "measure_id elements must be smaller than the input matrix size"

    assert type(hidden_id) == np.ndarray, "hidden_id must be of type 'np.ndarray'"
    assert hidden_id.size > 0, "hidden_id must not be empty"
    assert hidden_id.dtype == int, "hidden_id must be of dtype 'int'"
    assert len(hidden_id.shape) == 1, "hidden_id must be 1D shape"
    assert (hidden_id >= 0).all(), "hidden_id elements must be non-negative integers"
    assert np.max(hidden_id) < size[0], "hidden_id elements must be smaller than the input matrix size"

    n_m = len(measure_id)
    n_h = len(hidden_id)
    assert size[0] == (n_m + n_h), "Total number of elements in measure_id and hidden_id does not equal the number of rows of M"

    all_id = np.unique(np.concatenate((measure_id, hidden_id)))
    assert len(all_id) == size[0], "mesure_id and hidden_id contain common elements"

    for i in range(n_h):
        for j in range(n_m):
            B[i, j] = M[hidden_id[i], measure_id[j]]

    return B



def matrix_rearrange(M, measure_id, hidden_id):
    '''
    Re-arrange the square matrix according to which nodes
    are measured nodes / hidden nodes

    Arguments:
    1. M:            The original square matrix M
    2. measure_id:   Measured node indices of the original matrix M
    3. hidden_id:    Hidden node indices of the original matrix M

    Returns:
    1. M_perm:       The permutated matrix and would be in
                     this format:

                     | M_m    M_u |
                     | M_l    M_h |

    2. M_m:          Block matrix with elements equal to the original matrix
                     formed among the measure nodes

    3. M_h:          Block matrix with elements equal to the original matrix
                     formed among the hidden nodes

    4. M_u:          Block matrix with elements equal to the original matrix
                     formed between measure nodes and hidden nodes

    5. M_l:          Block matrix with elements equal to the original matrix
                     formed between measure nodes and hidden nodes
    '''
    #NOTE: The functions named block_ below consists of assertion checking
    #      So we do not intend to add assertions in this function   
    M_m = block_diag_up(M, measure_id)
    M_h = block_diag_low(M, hidden_id)
    M_u = block_off_up(M, measure_id, hidden_id)

    if np.allclose(M, M.T):
        M_l = M_u.T
    else:
        M_l = block_off_low(M, measure_id, hidden_id)

    # Combine all 4 block matrices
    M_perm = np.block([[M_m, M_u], [M_l, M_h]])

    return M_perm, M_m, M_u, M_l, M_h
