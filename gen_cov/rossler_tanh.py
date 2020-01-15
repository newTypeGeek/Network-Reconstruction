#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm


def tanh_couple(W, state, N):
    '''
    Facilitate the computation of tanh diffusive coupling function
    sum_{j neq i} g_{ij} A_{ij} tanh( state[j] - state[i] ) (Eq. (*))

    Arguments:
    1. W:       Weighted adjacency matrix of the whole network
    2. state:   State variable of a single dimension of all nodes
    3. N:       Total number of nodes

    Returns
    1. interaction:   The interaction between node i and others along a single dimension
                        The expression is given by Eq. (*) above
    '''
    # NOTE: Since this function is expected to be called from rossler_tanh,
    # we do not intend to add assertions as it is checked in rossler_tanh already

    # Initialization
    interaction = np.zeros((N, ))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            interaction[i] += W[i, j] * np.tanh(state[j] - state[i])

    return interaction


def rossler_tanh(W, c1, c2, c3, sigma, int_dt, sample_dt, sample_start, data_num, get_ts=False):
    '''
    Simulate the coupled SDEs with
      - Rossler dynamics (c1, c2, c3) are those standard parameters
      - tanh diffusive coupling function along each three dimensional state variables
      - Only the x state is injected with white noise

    and obtain the covariance matrix of the whole network.
    NOTE: only the x state is used for covariance computation.

    Arguments:
    1. W:               Weighted adjacency matrix of the whole network
    2. c1:              Rossler parameter
    3. c2:              Rossler parameter
    4. c3:              Rossler parameter
    5. sigma:           Noise strength (standard deviation of Gaussian distribution)
    6. int_dt:          Integration time step
    7. sample_dt:       Sampling time step
    8, start_sample:    Time step to start sampling
    9. data_num:        Total number of sampled data for covariance matrix computation
    10. get_ts:         To sample time series of the first node or not (default: False)

    Returns:
    1. cov:        Covariance matrix of the whole network
    2. x_ts:       Sampled time series of the first node of the x-state
    3. y_ts:       Sampled time series of the first node of the y-state
    4. z_ts:       Sampled time series of the first node of the z-state
    '''
    assert type(W) == np.ndarray, "W must be of type 'numpy.ndarray'"
    assert W.size > 0, "W must not be empty"
    assert W.dtype == int or W.dtype == float, "W must of dtype 'int' or 'float'"
    assert np.isfinite(W).all(), "Elements in W must be finite real numbers"
    size = W.shape
    assert len(size) == 2, "W must be 2D shape"
    assert size[0] == size[1], "W must be a square matrix"
    assert (np.diag(W) == 0).all(), "Diagonal elements in W must all be zero"
    assert (type(c1) == int or type(c1) == float) and np.isfinite(c1), "c1 must be a real number"
    assert (type(c2) == int or type(c2) == float) and np.isfinite(c2), "c2 must be a real number"
    assert (type(c3) == int or type(c3) == float) and np.isfinite(c3), "c3 must be a real number"

    assert (type(sigma) == int or type(sigma) == float) and np.isfinite(sigma) and sigma > 0, "sigma must be a positive real number"
    assert (type(int_dt) == int or type(int_dt) == float) and np.isfinite(int_dt) and int_dt > 0, "int_dt must be a positive real number"
    assert (type(sample_dt) == int or type(sample_dt) == float) and np.isfinite(sample_dt) and sample_dt > int_dt, "sample_dt must be a positive real number, and greater than int_dt"
    assert type(sample_start) == int and sample_start >= 0, "sample_start must be a non-negative integer"
    assert type(data_num) == int and data_num > sample_dt, "data_num must be a positive integer, and greater than sample_dt"

    assert type(get_ts) == bool, "get_ts must be boolean"

    # Sampling time interval
    sample_inter = int(sample_dt/int_dt)

    # Total number of iteration
    T = int((data_num) * sample_inter + sample_start)

    # Initialize the current state of N nodes
    N = size[0]
    x = np.random.normal(loc=0.5, scale=0.01, size=(N,))
    y = np.random.normal(loc=0.5, scale=0.01, size=(N,))
    z = np.random.normal(loc=0.5, scale=0.01, size=(N,))

    # Initialize the 1st and 2nd moment matrix of the state vector x
    # They are used to compute the covariance matrix
    m_01 = np.zeros((N,))
    m_02 = np.zeros((N, N))

    # Initialize the sampled time series of the first node
    if get_ts:
        x_ts = np.zeros((int(T/sample_inter),))
        y_ts = np.zeros((int(T/sample_inter),))
        z_ts = np.zeros((int(T/sample_inter),))
        i = 0
    else:
        x_ts = None
        y_ts = None
        z_ts = None

    # Solve the coupled SDEs using Euler-Maruyama method
    for t in tqdm(range(T)):
        eta = np.random.normal(size=(N,))

        interaction_x = tanh_couple(W, x, N)
        interaction_y = tanh_couple(W, y, N)
        interaction_z = tanh_couple(W, z, N)

        x_old = x
        x += (-y - z + interaction_x) * int_dt + sigma*np.sqrt(int_dt)*eta
        y += (x_old + c1*y + interaction_y) * int_dt
        z += (c2 + z*(x_old - c3) + interaction_z) * int_dt

        # Sample the node states
        if t % sample_inter == 0:

            # Sample dynamics of the first node
            if get_ts:
                x_ts[i] = x[0]
                y_ts[i] = x[0]
                z_ts[i] = x[0]
                i += 1

            # Sample 1st and 2nd moment
            if t >= sample_start:
                m_01 += x/data_num
                m_02 += np.outer(x, x)/data_num

    # Compute the covariance matrix of the whole network
    cov = m_02 - np.outer(m_01, m_01)

    return cov, x_ts, y_ts, z_ts
