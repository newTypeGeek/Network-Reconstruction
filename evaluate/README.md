# Evaluate the network reconstruction result
This `evaluate` package consists of different methods to evaluate the network reconstruction result.
1. Each python file consists of one function only
2. The function name inside the file is same as its file name
3. Each function must accept at least two arguments
   - `A`: the actual adjacency matrix
   - `A_reco`: the reconstructed adjacency matrix
   Both of them are 2D square numpy array with the same shape


# Evaluation Metrics
1. `error_rates.py`<br>
   Compute the false negative and false positive error rates


# Development
If you would like to add a new evaluation metrics, please follow the convention and edit `__init__.py`
