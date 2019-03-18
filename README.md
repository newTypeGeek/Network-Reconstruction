# Effect of hidden nodes on the reconstruction on bidirectional networks

Based on Emily S. C Ching, P. H. Tam, "[Effects of hidden nodes on the reconstruction of bidirectional networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.062318)", Phys. Rev. E 98, 062318 (2018), the network reconstruction method is demonstrated with Python scripts.

# Pre-requisite
1. Python 3
2. Jupyter Notebook
3. numpy
4. sklearn
5. matplotlib
6. tqdm


# Understanding the files
1. `Demo.ipynb`<br>
   Jupyter notebook to demonstrate network reconstruction

2. `adjacency.py`<br>
   Generate unweighted bi-directional network

3. `assign_weights.py`<br>
   Generate weighted bi-directional network from adjacency matrix

4. `choose_nodes.py`<br>
   Functions to select measured nodes and hidden nodes

5. `compare_results.py`<br>
   Evaluate the reconstruction performance by comparing
   the actual adjacency matrix with the reconstructed adjacency matrix

6.  `dynamics_tools.py`<br>
    Tools required for simulating the networked dynamics

7. `network_tools.py`<br>
   Tools required for studying networks

8. `reconstructions.py`<br>
   Functions to reconstruct an adjacency matrix given the data

9. `simulations.py`<br>
   Generate dynamics of the nodes by solving the coupled SDEs
   and obtain the covariance matrix

10. `tools.py`<br>
    General tools for matrix analysis


# Citation
```
@article{PhysRevE.98.062318,
  title = {Effects of hidden nodes on the reconstruction of bidirectional networks},
  author = {Ching, Emily S. C. and Tam, P. H.},
  journal = {Phys. Rev. E},
  volume = {98},
  issue = {6},
  pages = {062318},
  numpages = {9},
  year = {2018},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.98.062318},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.98.062318}
}

```
