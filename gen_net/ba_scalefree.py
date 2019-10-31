#!/usr/bin/env python3
import numpy as np

def ba_scalefree(N, m0, m):
    '''
    Construct an unweighted bi-directional BA scale-free network without self-loop
    
    Arguments:
    1. N:    Total number of nodes in the network
    2. m0:   Initial number of existing nodes that are fully connected among themselves
    3. m:    Number of existing nodes to be connected by a new node
    
    Returns:
    1. A:    Adjacency matrix
    '''
    
    assert type(N) == int and N > 0, "Number of node must be a positive integer"
    assert type(m0) == int and m0 > 0, "Initial number of existing nodes must be a positive integer" 
    assert type(m) == int and m > 0 and m0 <= m, "Number of existing nodes to be connected by a new node must be a positive integer, and not greater than m0"

 
    # Initialize the adjacency matrix
    A = np.zeros((N, N))
    A[0:m0, 0:m0] = 1
    np.fill_diagonal(A, 0)

    # Initialize the node candidates (i.e. existing nodes) indices
    candidates = list(np.arange(0, m0))


    # Grow the network with new node curr
    for curr in range(m0, N):
        # Update the candidates list

        # Compute the probability criteria of the existing nodes
        k = np.sum(A, 1)[0:curr]
        prob = k / np.sum(k)

        # Select m nodes from candidates list without replacement
        # according to the probability prob
        targets = np.random.choice(candidates, size=m, replace=False, p=prob)

        # Connect the current new node (curr) to the selected nodes (targets)
        A[targets, curr] = 1

        # Symmetrize the matrix
        A[curr, targets] = 1

        # Update the candidates list
        candidates.append(curr+1)

    return A
