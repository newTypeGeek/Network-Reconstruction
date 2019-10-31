# Effect of hidden nodes on the reconstruction on bidirectional networks

Emily S. C Ching, P. H. Tam, "[Effects of hidden nodes on the reconstruction of bidirectional networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.062318)", Phys. Rev. E 98, 062318 (2018)<br>

To demonstrate the method used in this journal publication using Python.


# Pre-requisite
1. Python 3
2. Jupyter Notebook
3. numpy
4. sklearn
5. matplotlib
6. tqdm


# Project structure
1. `Demo.ipynb`<br>
   Jupyter notebook to demonstrate network reconstruction

2. `utils`<br>
   A package with general tools for computations 

3. `gen_net`<br>
   A package to generate weighted adjacency matrix from a network model

4. `gen_cov`<br>
   A package to generate covariance matrix from a dynamical system given the weighted adjacency matrix

5. `choose_nodes`<br>
   A package to select nodes as measured nodes and hidden nodes

6. `reconstruct`<br>
   A package to reconstruct adjacency matrix from a reconstruction method given the data

8. `evaluate`<br>
   A package to evaluate the reconstruction performance


# TODO
1. Extend the `Demo.ipynb` with more details


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
