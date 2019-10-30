# Generate a weighted bi-directional network from a chosen model
This `gen_net` package consists of different network models to generate weighted bi-directional network. <br>
Weighted adjacency matrix, generated from `gen_net` package, is required as an input for these scripts.
1. Each python file consists of one function only
2. The function name inside the file is same as its file name


# Network Models
1. `er_random.py`
    Generate an unweighted bi-directional Erdős–Rényi (ER) random network

2. `gaussian.py`
    Assign Gaussian distributed weights to an unweighted bi-directional network

# Development
If you would like to add a new network model, please follow the convention and edit `__init__.py`
