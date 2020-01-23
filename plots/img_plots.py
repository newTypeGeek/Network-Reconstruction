import numpy as np
import matplotlib.pyplot as plt

def weighted_adj_plot(W):
    '''
    Image plot the weighted adjacency matrix with colourbar 
    showing the weights (coupling strength) of the edges

    Arguments:
    1. W:  Weighted adjacency matrix
    '''
  
    N = W.shape[0]

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(W, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label("Coupling strength", fontsize=16)
    plt.title("Weighted adjacency matrix", fontsize=16)
    plt.xlabel("Node index", fontsize=16)
    plt.ylabel("Node index", fontsize=16)
    plt.xticks(np.arange(0, N+20, 20), fontsize=16)
    plt.yticks(np.arange(0, N+20, 20), fontsize=16)


