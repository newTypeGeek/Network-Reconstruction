#!/usr/bin/env python3
'''
choose_nodes package contains functions to select nodes as measured nodes and hidden nodes
All functions take two key arguments:
    1. N:    Total number of nodes
    2. n:    Number of measured nodes

Each of the functions return two numpy arrays:
    1. measure_id:   The measured node indices correspond to the whole network
    2. hidden_id:    The hidden node indices correspond to the whole network
'''

from choose_nodes.lazy import lazy
from choose_nodes.random import random
