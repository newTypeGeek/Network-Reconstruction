#!/usr/bin/env python3


##################
#### tools.py ####
##################

import numpy as np


def eigen_values(M):
    '''
    Compute the eigenvalues of a general square matrix M
    
    Arguments:
    1. M:         A general square matrix

    Returns:
    1. eig_vals:   Eigen-values of matrix M
    '''
    assert type(M) == np.ndarray, "The input matrix must be of type 'numpy.ndarray'"
    assert M.size > 0, "The input matrix must not be empty"
    assert M.dtype == int or M.dtype == float, "The input matrix must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all() == True, "The input matrix elements must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "The input matrix must be 2D shape"
    assert size[0] == size[1], "The input matrix must be a square matrix"

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

    assert type(M) == np.ndarray, "The input matrix must be of type 'numpy.ndarray'"
    assert M.size > 0, "The input matrix must not be empty"
    assert M.dtype == int or M.dtype == float, "The input matrix must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all() == True, "The input matrix elements must be finite real numbers" 
    size = M.shape
    assert len(size) == 2, "The input matrix must be 2D shape"
    assert size[0] == size[1], "The input matrix must be a square matrix"

    assert (type(tol) == int or type(tol) == float) and np.isfinite(tol) == True and tol >= 0, "Tolerance value must be a non-negative number"



    cond_num = np.linalg.cond(M)

    M_inv = None
    if cond_num < tol:
        M_inv = np.linalg.inv(M)
    else:
        print("The input matrix is highly singular")

    return M_inv




def off_diag_upper(M):
    '''
    Extract the off-diagonal elements (upper triangle) of a square matrix
    
    Arguments:
    1. M:       input matrix
    
    Returns:
    1. off:     off-diagonal elements (upper triangle) of input matrix M
    '''
    
    assert type(M) == np.ndarray, "The input matrix must be of type 'np.ndarray'"
    assert M.size > 0, "The input matrix must not be empty"
    assert M.dtype == int or M.dtype == float, "The input matrix must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all() == True, "The input matrix elements must be finite real numbers" 
    size = M.shape
    assert len(size), "The input matrix must be 2D shape"
    assert size[0] == size[1], "The input matrix must be a square matrix"
   
    
    off_upper = M[np.triu(np.ones(size), 1) == 1]
    
    return off_upper




def index_recover(n):
    '''
    Recover the original row and colum indices of the
    flatten upper triangle vector using triu method (see off_diag_upper)

    Arguments:
    1. n:    The original matrix size

    Returns:
    1. row:  Row indices
    2. col:  Column indices
    '''

    assert type(n) == int and n > 0, "The original matrix size n must be a positive integer"

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

        row[start : end] = i * np.ones((bin_size,))
        col[start : end] = sorted(np.arange(n-1, i, -1))

    return row, col







def matrix_rearrange(M, measure_id, hidden_id):
    '''
    Re-arrange the square matrix according to which nodes
    are measure nodes / hidden nodes

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


    assert type(M) == np.ndarray, "The input matrix must be of type 'np.ndarray'"
    assert M.size > 0, "The input matrix must not be empty"
    assert M.dtype == int or M.dtype == float, "The input matrix must be of dtype 'int' or 'float'"
    assert np.isfinite(M).all() == True, "The input matrix elements must be finite real numbers"
    size = M.shape
    assert len(size) == 2, "The input matrix must be 2D shape"
    assert size[0] == size[1], "The input matrix must be a square matrix"

    assert type(measure_id) == np.ndarray, "The measure_id must be of type 'np.ndarray'"
    assert measure_id.size > 0, "The measure_id must not be empty"
    assert measure_id.dtype == int, "The measure_id must be of dtype 'int'"
    assert len(measure_id.shape) == 1, "The measure_id must be 1D shape"
    assert (measure_id >= 0).all() == True, "The measure_id elements must be non-negative integers"
    assert np.max(measure_id) < size[0], "The measure_id elements must be smaller than the input matrix size"


    assert type(hidden_id) == np.ndarray, "The hidden_id must be of type 'np.ndarray'"
    assert hidden_id.size > 0, "The hidden_id must not be empty"
    assert hidden_id.dtype == int, "The hidden_id must be of dtype 'int'"
    assert len(hidden_id.shape) == 1, "The hidden_id must be 1D shape"
    assert (hidden_id >= 0).all() == True, "The hidden_id elements must be non-negative integers"
    assert np.max(hidden_id) < size[0], "The hidden_id elements must be smaller than the input matrix size"

    num_m = len(measure_id)
    num_h = len(hidden_id)
    assert size[0] == (num_m + num_h), "The total length of input indices does not match the weighted adjacency matrix size"

    # Initialize 4 block matrices
    M_m = np.zeros((num_m, num_m))
    M_h = np.zeros((num_h, num_h))
    M_u = np.zeros((num_m, num_h))
    M_l = np.zeros((num_h, num_m))


    # For symmetric M, we can iterate fewer times
    if np.allclose(M, M.T) == True:

        # Construct M_m
        for i in range(num_m):
            M_m[i,i] = M[measure_id[i], measure_id[i]]
            for j in range(i+1, num_m):
               M_m[i,j] = M[measure_id[i], measure_id[j]]
               M_m[j,i] = M_m[i,j]


        # Construct M_h
        for i in range(num_h):
            M_h[i,i] = M[hidden_id[i], hidden_id[i]]
            for j in range(i+1, num_h):
                M_h[i,j] = M[hidden_id[i], hidden_id[j]]
                M_h[j,i] = M_h[i,j]

        # Construct M_u
        for i in range(num_m):
            for j in range(num_h):
                M_u[i,j] = M[measure_id[i], hidden_id[j]]
        
        # Construct M_l
        M_l = M_u.T


    # Non-symmetric M, need to loop all elements
    else:
        # Construct M_m
        for i in range(num_m):
            for j in range(num_m):
                M_m[i,j] = M[measure_id[i], measure_id[j]]
        
        # Construct M_h
        for i in range(num_h):
            for j in range(num_h):
                M_h[i,j] = M[hidden_id[i], hidden_id[j]]

        # Construct M_u
        for i in range(num_m):
            for j in range(num_h):
                M_u[i,j] = M[measure_id[i], hidden_id[j]]

        # Construct M_l
        for i in range(num_h):
            for j in range(num_m):
                M_l[i,j] = M[hidden_id[i], measure_id[j]]


    # Combine all 4 block matrices
    M_perm = np.block([ [M_m, M_u], [M_l, M_h] ])


    return M_perm, M_m, M_u, M_l, M_h

