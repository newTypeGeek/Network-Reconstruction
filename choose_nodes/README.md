# Select nodes as measured nodes and hidden nodes
This `choose_nodes` package consists of different methods to select nodes as measured nodes and hidden nodes.

1. Each python file consists of one node selection scheme only
2. The node selection scheme name inside is same as its file name
3. Each scheme outputs two numpy arrays of node indices (`meausre_id` and `hidden_id`)  


# Definition
1. `N`: total number of nodes in the whole network
2. `n`: number of measured nodes


# Scheme of node selection
1. `lazy.py`<br>
   Select the first `n` nodes as measured nodes

2. `random.py`<br>
   Randomly select `n` nodes as measured nodes


# Development
If you would like to add new selection scheme, please follow the convention and edit `__init__.py`
