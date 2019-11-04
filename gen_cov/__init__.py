#!/usr/bin/env python3
'''
gen_cov package simulates a set of coupled SDEs using Euler-Maruyama method
The time series of node (see the details) at each sampling step is used to 
compute the covariance matrix corresponding to all N nodes
'''
from gen_cov.logistic_diffusive import logistic_diffusive
from gen_cov.fhn_diffusive import fhn_diffusive
from gen_cov.rossler_tanh import rossler_tanh
