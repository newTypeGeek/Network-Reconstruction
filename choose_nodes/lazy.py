#!/usr/bin/env python3

#######################
### choose_nodes.py ###
#######################


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
    
    assert type(N) == int and N > 0, "Total number of node must be a positive integer"
    assert type(n) == int and (n > 0 and n < N), "Number of measure node must be a positive integer and less than N"

    measure_id = np.arange(n)
    hidden_id = np.arange(n, N)

    return measure_id, hidden_id



def random(N, n):
    '''
    Chose n nodes as the measure nodes randomly

    Arguments:
    1. N:     Total number of nodes
    2. n:     Number of measure nodes

    Returns:
    1. measure_id:    Measured node indices of the original (whole) network
    2. hidden_id      Hidden node indices of the original (whole) network
    '''
    
    assert type(N) == int and N > 0, "Total number of node must be a positive integer"
    assert type(n) == int and (n > 0 and n < N), "Number of measure node must be a positive integer and less than N"


    measure_id = np.random.choice(N, n, replace=False)


    all_id = np.arange(N)
    hidden_id = np.array( list( set(all_id).difference(set(measure_id)) ) )

    return measure_id, hidden_id



#######################################################
#  You add other methods to choose measured nodes    ##
#  and hidden nodes by following the above examples  ##
#######################################################
