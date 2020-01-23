import numpy as np
import matplotlib.pyplot as plt

def covinv_plots(ucon_nohidden, conn_nohidden, ucon_hidden, conn_hidden,
                num_bin, x_min, x_max):
    '''
    Plot TWO distributions from off-diagonal elements of inverse covariance matrix. 
    Both plots correspond to the same underlying network connectivity except that
    the LEFT plot does not have the effect of hidden nodes and the RIGHT plot
    includes the effect of hidden nodes

    Arguments:
    Denote off-diagonal elements of inverse covariance matrix as X[i,j], and
           the off-diagonal elements of adjacency matrix as A[i,j]

    1. ucon_nohidden: X[i,j] with A[i,j] = 0, without the hidden node effects
    2. conn_nohidden: X[i,j] with A[i,j] = 1, without the hidden node effects
    3. ucon_hidden:   X[i,j] with A[i,j] = 0, with the hidden node effects
    4. conn_hidden:   X[i,j] with A[i,j] = 1, with the hidden node effects

    5. num_bin:       number of bin for BOTH distribution plots
    6. x_min:         min of x-axis for BOTH plots
    7. x_max:         max of x-axis for BOTH plots
    '''

    # Setup the bin edges and bin centres
    bin_lims = np.linspace(x_min, x_max, num_bin+1)
    bin_centre = 0.5*(bin_lims[:-1]+bin_lims[1:])

    # Get the count (frequency) in the histogram
    ucon_nohidden_count, _ = np.histogram(ucon_nohidden, bins=bin_lims)
    conn_nohidden_count, _ = np.histogram(conn_nohidden, bins=bin_lims)

    ucon_hidden_count, _ = np.histogram(ucon_hidden, bins=bin_lims)
    conn_hidden_count, _ = np.histogram(conn_hidden, bins=bin_lims)

    # Remove zero count elements
    ucon_nohidden_bin_centre = bin_centre[ucon_nohidden_count != 0]
    conn_nohidden_bin_centre = bin_centre[conn_nohidden_count != 0]
    ucon_nohidden_count = ucon_nohidden_count[ucon_nohidden_count != 0]
    conn_nohidden_count = conn_nohidden_count[conn_nohidden_count != 0]

    ucon_hidden_bin_centre = bin_centre[ucon_hidden_count != 0]
    conn_hidden_bin_centre = bin_centre[conn_hidden_count != 0]
    ucon_hidden_count = ucon_hidden_count[ucon_hidden_count != 0]
    conn_hidden_count = conn_hidden_count[conn_hidden_count != 0]

    # Normalize the histogram AS A WHOLE
    nohidden_total_count =  np.sum(ucon_nohidden_count) + np.sum(conn_nohidden_count)
    ucon_nohidden_dens = ucon_nohidden_count / nohidden_total_count
    conn_nohidden_dens = conn_nohidden_count / nohidden_total_count

    hidden_total_count =  np.sum(ucon_hidden_count) + np.sum(conn_hidden_count)
    ucon_hidden_dens = ucon_hidden_count / hidden_total_count
    conn_hidden_dens = conn_hidden_count / hidden_total_count

    # Plot the distribution
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(ucon_nohidden_bin_centre, ucon_nohidden_dens, s=100, color='red', marker='o', facecolor='None')
    plt.scatter(conn_nohidden_bin_centre, conn_nohidden_dens, s=100, color='blue', marker='^', facecolor='None')
    plt.legend([r'$A_{ij} = 0$', r'$A_{ij} = 1$'], fontsize=24)
    plt.xlim(-45, 5)
    plt.xlabel(r'$(\Sigma^{-1})_{ij}$', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title('Without the effect of hidden nodes', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.scatter(ucon_hidden_bin_centre, ucon_hidden_dens, s=100, color='red', marker='o', facecolor='None')
    plt.scatter(conn_hidden_bin_centre, conn_hidden_dens, s=100, color='blue', marker='^', facecolor='None')
    plt.legend([r'$A_{ij} = 0$', r'$A_{ij} = 1$'], fontsize=24)
    plt.xlim(x_min, x_max)
    plt.xlabel(r'$(\Sigma_{m}^{-1})_{ij}$', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title('With the effect of hidden nodes', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)




def hidden_node_effect_plot(C_ucon, C_conn, num_bin, x_min, x_max):
    '''
    Plot the distribution of off-diagonal elements of C matrix (effect of hidden nodes)

    Arguments:
    Denote off-diagonal elements of C matrix as C[i,j], and
           the off-diagonal elements of adjacency matrix as A[i,j]

    1. C_ucon:  C[i,j] with A[i,j] = 0
    2. C_conn:  C[i,j] with A[i,j] = 1
    3. num_bin:       number of bin
    4. x_min:         min of x-axis
    5. x_max:         max of x-axis
    '''

    # Setup the bin edges and bin centres
    bin_lims = np.linspace(x_min, x_max, num_bin+1)
    bin_centre = 0.5*(bin_lims[:-1]+bin_lims[1:])
    
    # Get the count (frequency) in the histogram
    C_ucon_count, _ = np.histogram(C_ucon, bins=bin_lims)
    C_conn_count, _ = np.histogram(C_conn, bins=bin_lims)

    # Remove zero count elements
    C_ucon_bin_centre = bin_centre[C_ucon_count != 0]
    C_conn_bin_centre = bin_centre[C_conn_count != 0]
    C_ucon_count = C_ucon_count[C_ucon_count != 0]
    C_conn_count = C_conn_count[C_conn_count != 0]

    # Normalize the histogram SEPARATELY
    C_ucon_dens = C_ucon_count / np.sum(C_ucon_count)
    C_conn_dens = C_conn_count / np.sum(C_conn_count)

    # Plot the distribution
    plt.figure(figsize=(12, 8))
    plt.scatter(C_ucon_bin_centre, C_ucon_dens, s=150, color='red', marker='o', facecolor='None')
    plt.scatter(C_conn_bin_centre, C_conn_dens, s=150, color='blue', marker='^', facecolor='None')
    plt.legend([r'$A_{ij} = 0$', r'$A_{ij} = 1$'], fontsize=28)
    plt.xlim(x_min, x_max)
    plt.xlabel(r'$C_{ij}$', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
