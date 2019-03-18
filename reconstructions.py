#!/usr/bin/env python3

############################
#### reconstructions.py ####
############################

import numpy as np
import tools
from sklearn.cluster import KMeans


def reconstruct(n, row_conn_triu, col_conn_triu):
    '''
    Reconstruct the symmetric adjacency matrix

    Arguments:
    1. n:           Number of measured node
    2. row_conn_triu:    Upper triangle row indices that connects to col_conn indices
    3. col_conn_triu:    Upper triangle column indices that connects to row_conn indices

    Returns:
    1. A_reco:      Reconstructed adjacency matrix

    '''
    assert type(n) == int and n > 0, "Number of measured node must be positive integer"
    
    assert type(row_conn_triu) == np.ndarray, "row_conn_triu must be of type 'np.ndarray'"
    assert row_conn_triu.size > 0, "row_conn_triu must not be empty"
    assert row_conn_triu.dtype == int, "row_conn_triu must of dtype 'int'"
    size = row_conn_triu.shape
    assert len(size) == 1, "row_conn_triu must be of 1D shape"
    assert (row_conn_triu >= 0).all() == True, "row_conn_triu elements must be non-negative integers"

    assert type(col_conn_triu) == np.ndarray, "col_conn_triu must be of type 'np.ndarray'"
    assert col_conn_triu.size > 0, "col_conn_triu must not be empty"
    assert col_conn_triu.dtype == int, "col_conn_triu must of dtype 'int'"
    size = col_conn_triu.shape
    assert len(size) == 1, "col_conn_triu must be of 1D shape"
    assert (col_conn_triu >= 0).all() == True, "col_conn_triu elements must be non-negative integers"

    assert len(np.unique(np.concatenate((row_conn_triu, col_conn_triu)))) == n, "The number of unique node indices in row_conn_triu + col_conn_triu does not match the number of measured node"



    # Initialize the reconstructed adjacency matrix
    # with zero elements
    A_reco = np.zeros((n,n))

    # Assign links according to row_conn, and col_conn
    # NOTE: only upper triangle elements are assigned
    #       (see off_diag_upper() in tools.py)
    A_reco[row_conn_triu, col_conn_triu] = 1

    # Symmetrize the reconstructed adjacency matrix
    A_reco = A_reco + A_reco.T

    A_reco = A_reco.astype('int')

    return A_reco








def kmeans(data, k=2):
    '''
    Cluster the data using k-means clustering

    Arguments:
    1. data:        1D array data. Expected to be one of the following data
                    i.      off-diagonal elements of cov_inv_m
                    ii.     off-diagonal elements of cov_m_inv
                    (see inverse_covariance() in dynamics_tools.py)

    2. k:           Number of clusters (default: 2)

    
    Returns:
    1. row_conn_triu:    Upper triangle row indices that connects to col_conn indices
    2. col_conn_triu:    Upper triangle column indices that connects to row_conn indices
    '''

    assert type(data) == np.ndarray, "Input data must be of type 'numpy.ndarray'"
    assert data.size > 0, "Input data must not be empty"
    assert data.dtype == int or data.dtype == float, "Input data must be of dtype 'int' or 'float'"
    assert np.isfinite(data).all() == True, "Input data elements must be real numbers"
    size = data.shape
    assert len(size) == 1, "Input data must be 1D shape"
    assert size[0] > 1, "Input data must have at least two elements for clustering"
    assert type(k) == int and k > 1, "Number of cluster must be positive integer and not less than 2"

    
    # Convert data to a shape for clustering
    data = data.reshape(-1, 1)

    # Create k-means clustering object
    kmeans = KMeans(n_clusters=k, random_state=29)
    
    # Perform k-means clustering on the data
    kmeans.fit(data)

    # Get the reconstructed indices (should 0's and 1's for n_clusters = 2)
    reco_indices = kmeans.labels_

    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Data with centroid absolute value close to zero are regarded as
    # unconnected pairs
    ucon_id = np.argmin(abs(centroids))

    # Recover the number of nodes from number of data points
    n_data = data.shape[0]
    n_node = int(0.5*(1 + np.sqrt(1 + 8*n_data)))
    row, col = tools.index_recover(n_node)

    # Get the connected pair indices
    row_conn_triu = row[reco_indices != ucon_id].astype("int")
    col_conn_triu = col[reco_indices != ucon_id].astype("int")


    return row_conn_triu, col_conn_triu



################################################
#  You can add other clustering methods below  #
#  by following the kmeans example             #
################################################



