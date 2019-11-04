#!/usr/bin/env python3
'''
reconstruct package contains functions to perform network reconstruction
Every method should consist of two key argument:
    1. data:   a 1D numpy array that is extracted from either the upper or lower
               off-diagonal elements of a matrix

    2. n:      the number of nodes in a network that you are going to reconstruct

Every method should consist of one key return:
    1. A_reco: the reconstructed adjacency matrix in 2D numpy array with dtype int
'''
from reconstruct.kmeans import kmeans
