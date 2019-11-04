#!/usr/bin/env python3
'''
evaluate package contains functions to examine the network reconstruction performance
All functions take these two arguments:
    1. A           The actual adjacency matrix
    2. A_reco      The reconstructed adjacency matrix
'''
from evaluate.error_rates import error_rates
