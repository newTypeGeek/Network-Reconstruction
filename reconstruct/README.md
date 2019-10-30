# Reconstruct the network connectivity (adjacency matrix) from the given data
This `reconstruct` package consists of different methods to reconstruct the adjacency matrix from the data.<br>
The data is assumed to be the upper triangular elements of an inverse covariance matrix.<br>
1. Each python file consists of one function only
2. The function name inside the file is same as its file name
3. Each function must return with the first slot as `A_reco`, a 2D square numpy array, which is the reconstructed adjacency matrix


# Reconstruction Methods
1. `kmeans.py`<br>
   k-means clustering (unsupervised learning method) with k clusters is used to reconstruct the adjacency matrix

# Development
If you would like to add a new reconstruction method, please follow the convention and edit `__init__.py`
