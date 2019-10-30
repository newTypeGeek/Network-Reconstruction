# Generate covariance matrix from a chosen dynamical system
This `gen_cov` package consists of different dynamical systems to generate the corresponding covariance matrix. <br>
Weighted adjacency matrix, generated from `gen_net` package, is required as an input for these scripts.
1. Each python file consists of one function only
2. The function name inside the file is same as its file name
3. Each function must return with the first slot as `cov`, a 2D square numpy array, which is the covariance matrix of the whole network


# Dynamical Systems
1. `logistic_diffusive.py`
   - Intrinsic dynamics: `f(x) = rx(1-x)` with `r > 0`
   - Coupling function: `h(y-x) = y-x`
   - The dynamics is solved by Euler-Maruyama method
   - For details, please read the source code and `demo.ipynb` in the root directory


# Development
If you would like to add a new dynamical system , please follow the convention and edit `__init__.py`
