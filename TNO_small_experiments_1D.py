#! /usr/bin/env python3

# David Mathas - TNO

# imports:
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
from typing import Tuple

### COMPUTATION AND HELPER FUNCTIONS: 
def spatial_corr_sim_1D(L: int, 
                        num_locations: int, 
                        print_results: bool = True) -> np.ndarray:
    """
    Calculates and returns the spatial correlation matrix, standard deviation matrix, 
    and covariance matrix for a given set of locations and the spatial correlation length L.

    Args:
        L (int): Correlation length.
        num_locations (int): Number of locations.
        print_results (bool): Boolean to specify wether intermediate resulting matrices (d, C and B) are printed. Defaults to True.

    Returns:
        np.ndarray: Spatial correlation matrix C.
    """
    locations = np.arange(num_locations) # locations vector 

    # Use L2 norm for distance and comput d (differences between locations):
    d = cdist(np.array(locations).reshape(-1, 1), np.array(locations).reshape(-1, 1), metric='euclidean')
    if print_results: 
        print("Distances d: \n", np.array_str(d, precision = 3, suppress_small = True), '\n')

    # Compute and show C (spatial correlation matrix):
    if L > 0:
        C = np.exp(-d / L)
        epsilon = 1e-10  #helps in ensuring positive definiteness
        np.fill_diagonal(C, 1 + epsilon)
    elif d.any() > 0: 
        C = np.eye(d.shape[0])
    else: #since exp(0) only gives ones
        C = np.ones(d.shape)
    
    if print_results:
        print("Spatial correlation between cells matrix: \n", np.array_str(C, precision = 4, suppress_small = False), '\n')
    assert C.shape == d.shape, "Shapes of C and d do not match" # quick assertion check
    
    return C

def sorted_evalues_evectors(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes and returns sorted positive eigenvalues and corresponding eigenvectors.

    Args:
        C (np.ndarray): Input matrix of simulated patial correlation matrix C for eigenvalue decomposition.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Sorted positive eigenvalues v ctor,
                                                               unsorted eigenvalues vector,
                                                               corresponding sorted eigenvectors matrix,
                                                               unsorted eigenvectors matrix.
    """
    # Compute eigenvalues (Λ) and eigenvectors (Q):
    Λ, Q = np.linalg.eigh(C)

    # Sort eigenvalues and eigenvectors in descending order:
    sorted_indices = np.argsort(Λ)[::-1]
    sorted_Λ = Λ[sorted_indices]
    sorted_Q = Q[:, sorted_indices]
    
    # Ensure the eigenvalues are positive:
    positive_indices = [i for i, val in enumerate(sorted_Λ) if val > 0]
    positive_sorted_Λ = sorted_Λ[positive_indices]
    corresponding_Q = sorted_Q[:, positive_indices]

    return positive_sorted_Λ, Λ, corresponding_Q, Q

def compute_B(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Computes the covariance matrix B.

    Args:
        S (np.ndarray): The standard deviation matrix.
        C (np.ndarray): The spatial correlation matrix.

    Returns:
        np.ndarray: The covariance matrix B.
    """
    B = S @ C @ S.T
    return B

# function to generate S:
def st_dev_matrix(num_locations: int,
                  rho_sigma_factor: float,
                  cons_emission_val: float) -> np.ndarray:
    """
    Construct the standard deviation matrix (S) with a constant value on the diagonal.

    Args:
        num_locations (int): Number of locations (diagonal elements).
        rho_sigma_factor (float): Factor to scale the standard deviation.
        cons_emission_val (float): Constant emission value for scaling.

    Returns:
        np.ndarray: Diagonal matrix with standard deviation values.
    """
    # Build the st. dev. matrix S with st. dev. on the diagonal:
    sigma_e = cons_emission_val * rho_sigma_factor
    S = np.diag(np.full(num_locations, sigma_e))
    
    return S

def aggregated_unc(a_coord: list,
                   S_sim: np.ndarray,
                   C_sim: np.ndarray) -> float:
    """Calculate the standard deviation of the aggregated uncertainty over specified coordinates.

    Args:
        a_coord (list): The range of coordinates to aggregate.
        S_sim (np.ndarray): The standard deviation matrix.
        C_sim (np.ndarray): The correlation matrix.

    Returns:
        float: The standard deviation of the aggregated emissions over the specified coordinates.
    """
    # set up locations and emissions:
    num_locations = S_sim.shape[0]
    emission_value = S_sim[0][0] 
    locations = np.arange(num_locations)
    emissions = np.array([emission_value] * num_locations).reshape(-1, 1) 

    # accumulation vector:
    a = np.array([1 if a_coord[0] <= i < a_coord[1] else 0 for i in locations]).reshape(-1, 1)
    eta = a.T @ emissions 

    # print and return st. dev. of spatial sum:
    B_sim = compute_B(S_sim, C_sim)
    sigma_eta = np.sqrt(a.T @ B_sim @ a)

    return sigma_eta


## Original C and Q @ Q.transposed plot
def heatmap_C_QQ(C: np.ndarray, 
                 Q_positive_sorted: np.ndarray, 
                 high_dpi: int = None) -> None:
    """
    Plot heatmaps for the original correlation matrix (C) and the product of the sorted positive eigenvectors (Q).

    Args:
        C (np.ndarray): Original correlation matrix.
        Q_positive_sorted (np.ndarray): Product of the sorted positive eigenvectors.
        high_dpi (int): If specified, sets the DPI of the plot for higher resolution. Defaults to None.

    Returns:
        None
    """
    dpi_setting = 300 if high_dpi else None
    fig = plt.figure(figsize=(12, 6), dpi = dpi_setting)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    # Plot for the original C matrix
    ax1 = fig.add_subplot(gs[0])
    cax1 = ax1.imshow(C, cmap='plasma', interpolation='nearest')
    ax1.set_title('Original $C$ Matrix Heatmap', size=16)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot for Q^T @ Q
    ax2 = fig.add_subplot(gs[1])
    cax2 = ax2.imshow(Q_positive_sorted.T @ Q_positive_sorted, cmap='plasma', interpolation='nearest')
    ax2.set_title('$Q^T \\cdot Q$ Heatmap', size=16)
    ax2.set_xticks([])
    ax2.set_yticks([])

    cb_ax = fig.add_subplot(gs[2])
    fig.colorbar(cax2, cax=cb_ax)

    plt.tight_layout()
    plt.show()

## Descending pos. e.values and e.vectors plot
def plot_eigenvalues_and_eigenvectors(positive_sorted_Λ: np.ndarray, 
                                      positive_sorted_Q: np.ndarray, 
                                      num_eigenvectors: int = 10,
                                      high_dpi: int = None) -> None:
    """
    Plots the eigenvalues in descending order, the corresponding eigenvectors, and the cumulative explained percentage.

    Args:
        positive_sorted_Λ (np.ndarray): Array of positive sorted eigenvalues.
        positive_sorted_Q (np.ndarray): Matrix with columns as positive sorted eigenvectors.
        num_eigenvectors (int): Number of eigenvectors to plot. Default is 10.
        high_dpi (int): If specified, sets the DPI of the plot for higher resolution. Default is None.
    
    Returns:
        None    
    """
    print(f'Plotting the (positive) eigenvalues, eigenvectors, and cumulative explained percentage. \n Full number of eigenvalues/vectors: {len(positive_sorted_Λ)}')
    dpi_setting = 300 if high_dpi else None

    # Calculate the percentage of C explained by the cumulative eigenvalue sum:
    cumulative_eigenvalues = np.cumsum(positive_sorted_Λ)
    percentage_explained = (cumulative_eigenvalues / np.sum(positive_sorted_Λ)) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi_setting)

    tick_interval = 10 if num_eigenvectors > 10 else 1
    range_x = np.arange(0, num_eigenvectors + 1, tick_interval)
    range_x[0] = 1
    
    # Eigenvalues plot:
    axes[0].plot(range(1, num_eigenvectors + 1), positive_sorted_Λ[: num_eigenvectors], marker='o')
    axes[0].set_xticks(range_x)
    axes[0].set_xlabel('$\\lambda$ Index')
    axes[0].set_ylabel('Eigenvalue ($\\lambda_i$)')
    axes[0].set_title('(Positive) Eigenvalues in Descending Order', size=15)
    axes[0].grid(True, linestyle=':')
    axes[0].tick_params(axis='x', labelsize=10)
    # axes[0].set_yscale('log')

    # Eigenvectors plot:
    nr_evectors = int(num_eigenvectors / 2) if num_eigenvectors <= 16 else 5
    for i in range(nr_evectors): 
        axes[1].plot(positive_sorted_Q[:, i], label=f'Eigenvector {i+1}')
    axes[1].set_xlabel('Cell Index')
    axes[1].set_ylabel('Eigenvector Value')
    axes[1].set_title('Corresponding Eigenvectors', size=15)
    axes[1].legend(fontsize='x-small')
    axes[1].grid(True, linestyle=':')

    # Cumulative explained percentage plot
    axes[2].plot(range(1, num_eigenvectors + 1), percentage_explained[: num_eigenvectors], marker='o')
    axes[2].set_xticks(range_x)
    axes[2].set_xlabel('$\\lambda$ and $q$ Index')
    axes[2].set_ylabel('Cumulative Explained Percentage')
    axes[2].set_title('Cumulative Explained Percentage by Eigenvectors', size=15)
    axes[2].grid(True, linestyle=':')
    axes[2].tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    plt.show()

## E.vectors and C = sum(e.value * e.vectors sqaured) subplot 
def plot_eigenvectors_cumulative(positive_sorted_Λ: np.ndarray, 
                                 positive_sorted_Q: np.ndarray, 
                                 num_eigenvectors: int = 5,
                                 high_dpi: int = None) -> None:
    """
    Plot eigenvectors and cumulative eigenvectors.

    Args:
        positive_sorted_Λ (np.ndarray): Array of positive sorted eigenvalues.
        positive_sorted_Q (np.ndarray): Array of positive sorted eigenvectors.
        num_eigenvectors (int): Number of eigenvectors to plot. Default is to 5.
        high_dpi (int): If specified, sets the DPI of the plot for higher resolution. Defaults is None.

    Returns:
        None  
    """
    print('Plotting eigenvectors and cumulative eigenvectors...')
    dpi_setting = 300 if high_dpi else None
    fig = plt.figure(figsize=(15, 5), dpi = dpi_setting)
    gs = fig.add_gridspec(2, num_eigenvectors + 1, width_ratios=[0.5] * num_eigenvectors + [0.04])

    cumulative_eigenvalues = np.cumsum(positive_sorted_Λ)
    percentage_explained = (cumulative_eigenvalues / np.sum(positive_sorted_Λ)) * 100
    cumulative_outer_product_scaled = np.zeros_like(np.outer(positive_sorted_Q[:, 0], positive_sorted_Q[:, 0]))

    # Plot subplots:
    for i in range(num_eigenvectors):
        outer_product = np.outer(positive_sorted_Q[:, i], positive_sorted_Q[:, i])
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(outer_product, cmap='plasma', interpolation='nearest')
        ax.set_title(f'Eigenvector {i+1} ($q_{i+1}$) outer product', size=8)
        ax.set_xticks([])
        ax.set_yticks([])

        cumulative_outer_product_scaled += positive_sorted_Λ[i] * outer_product
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(cumulative_outer_product_scaled, cmap='plasma', interpolation='nearest')
        ax.set_title(f'Cum. to $q_{i+1}$ (scaled by $\\lambda_{{i+1}}$ = {np.round(positive_sorted_Λ[i], 2)}) - {percentage_explained[i]:.2f}\\% explained', 
                     size=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar:
    cb_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.ax.tick_params(labelsize=8) 

    plt.tight_layout()
    plt.show()

def plot_L_comparison_experiment(L_values: list,
                      num_locations: int = 100, 
                      num_eigen: int = 10,
                      high_dpi: int = None) -> None:
    """
    Plotting eigenvalue comparison for different correlation length (L) values.
    
    Args:
        L_values (list): A list of correlation lengths (L) to be compared.
        num_locations (int, optional): Number of locations to simulate in the spatial correlation matrix. Default is 100.
        num_eigen (int, optional): Number of top eigenvalues/eigenvectors to plot. Default is 10.
        high_dpi (int, optional): If specified, sets the DPI of the plot for higher resolution. Default is None.

    Returns:
        None: 
    """
    print('Plotting eigenvalue comparison for different correlation length (L) values...')

    dpi_setting = 300 if high_dpi else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi_setting)
    # Eigenvalues plot:
    for L in L_values:
        C_sim = spatial_corr_sim_1D(L = L, 
                                    num_locations=num_locations,
                                    print_results=False)
        positive_sorted_Λ, _, _, _ = sorted_evalues_evectors(C_sim)
        axes[0].plot(range(1, num_eigen + 1), positive_sorted_Λ[: num_eigen], marker='o',label=f'L = {L}')
        cumulative_eigenvalues = np.cumsum(positive_sorted_Λ)
        percentage_explained = (cumulative_eigenvalues / np.sum(positive_sorted_Λ)) * 100 
        axes[1].plot(range(1, num_eigen + 1), percentage_explained[: num_eigen], marker='o', label=f'L = {L}') 

    axes[0].set_xticks(np.arange(1, num_eigen + 1, 1))
    # axes[0].set_xlabel('Eigenvalue/Location Index')
    axes[0].set_xlabel('$\\lambda$ Index')
    axes[0].set_ylabel('Eigenvalue magnitude')
    axes[0].set_title('(Positive) Eigenvalues in Descending Order', size=15)
    axes[0].grid(True, linestyle=':')
    axes[0].legend()

    axes[1].set_xticks(np.arange(1, num_eigen + 1, 1))
    # axes[1].set_xlabel('Number of Eigenvectors')
    axes[1].set_xlabel('$\\lambda$ Index')
    axes[1].set_ylabel('Cumulative Explained Percentage')
    axes[1].set_title('Cumulative Explained Percentage by Eigenvectors', size=15)
    axes[1].grid(True, linestyle=':')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def spatial_corr_convergence_experiment(positive_sorted_Λ: np.ndarray,
                                        corresp_sorted_Q: np.ndarray,
                                        N_sim: list,
                                        N_locations: int,
                                        C_sim: np.ndarray,
                                        zeroing: bool = True,
                                        high_dpi: int = None) -> None:
    """
    Conducts a convergence experiment for spatial correlation matrices.

    Args:
        positive_sorted_Λ (np.ndarray): Sorted positive eigenvalues of the spatial correlation matrix.
        corresp_sorted_Q (np.ndarray): Corresponding sorted eigenvectors of the spatial correlation matrix.
        N_sim (list): A list of sample sizes to simulate.
        N_locations (int): Number of locations for the spatial correlation matrix.
        C_sim (np.ndarray): Original spatial correlation matrix for comparison.
        zeroing (bool): Flag to apply zeroing to small negative eigenvalues, default it set to True. 
        high_dpi (int): If specified, sets the DPI of the plot for higher resolution. Defaults is None.

    Returns:
        None    
    """
    print('Plotting convergence comparison of mu and C for different simulation sizes...')

    mu_norms = []
    frobenius_norms = []

    Λ_matrix = np.diag(positive_sorted_Λ)

    # determine number of significant eigenvalues and employ "zeroing" for potential increased computational efficiency: 
    if zeroing:
        relative_threshold = 0.001  # 0.1% of the largest eigenvalue
        cutoff_threshold = relative_threshold * positive_sorted_Λ[0]
        significant_eigenvalues = positive_sorted_Λ > cutoff_threshold 
        positive_sorted_Λ[~significant_eigenvalues] = 0

    # Construct the square root of C:
    C_sqrt = corresp_sorted_Q @ np.sqrt(Λ_matrix) @ corresp_sorted_Q.T

    # Perform simulations for different sample sizes N_sim:
    for iN in N_sim:
        W = np.random.normal(0, 1, (N_locations, iN))
        V = C_sqrt @ W

        # Compute ensemble stats and store them:
        mu = np.mean(V, axis=1)
        C_tilde = (V @ V.T) / (iN - 1)
        mu_norms.append(np.linalg.norm(mu))
        frobenius_norms.append(np.linalg.norm(C_tilde - C_sim, 'fro'))

    # Plots:
    dpi_setting = 300 if high_dpi else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi_setting)

    # Plot for mean convergence:
    axes[0].plot(N_sim, mu_norms, marker='o', label='Euclidean norm of $\\mu$', color='blue')
    axes[0].axhline(y=0, color='black', linestyle='dotted', linewidth=1.2, label='$\\mu$ is equal to zero-vector')
    axes[0].set_xticks(N_sim)
    axes[0].set_xscale('log')
    # axes[0].set_yscale('log')
    axes[0].set_xlabel('Sample size (N)')
    axes[0].set_ylabel('Euclidean norm of $\\mu$')
    axes[0].set_title('Convergence of Mean')
    axes[0].grid(True, linestyle=':')
    axes[0].legend()

    # Plot for covariance matrix convergence
    axes[1].plot(N_sim, frobenius_norms, marker='o', label='Frobenius norm of C difference', color='green')
    axes[1].axhline(y=0, color='black', linestyle='dotted', linewidth=1.2, label=r'$C$ is equal to $\tilde{C}$')
    axes[1].set_xticks(N_sim)
    axes[1].set_xscale('log')
    # axes[1].set_yscale('log')
    axes[1].set_xlabel('Sample size (N)')
    axes[1].set_ylabel('Frobenius norm of $C$ difference')
    axes[1].set_title('Convergence of Covariance Matrix')
    axes[1].grid(True, linestyle=':')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_sigmas_experiment(L_values: list, 
                         accum_coord: list,
                         rho_values: list,
                         sigma_eta_values: list,
                         constant_emission_val: float = 10.,
                         num_locations: int = 100,
                         plot_option: int = 1,
                         high_dpi: bool = False) -> None:
    """
    Plots the relationship between rho_sigma factor, L value, and aggregated uncertainty.

    Args:
        L_values (list): A list of L values to be considered.
        accum_coord (list): Coordinates for accumulation vector.
        rho_values (list): A list of rho_sigma factors to be considered.
        sigma_eta_values (list): A list of sigma_eta values to be considered for the second plot.
        constant_emission_val (float): Constant value for emissions. Defaults to 10.
        num_locations (int): Number of locations. Defaults to 100.
        plot_option (int): Determines which plots to show. 1 for the first plot, 2 for the second, 3 for both.
        high_dpi (bool): If set to True, the plot will have high DPI. Defaults to False.

    Returns:
        None.    
    """

    dpi_setting = 300 if high_dpi else None
    # Calculation of alpha values:
    arbitrary_rho = 0.1
    alpha_coefficients = [
        aggregated_unc(
            a_coord=accum_coord,
            S_sim=st_dev_matrix(
                num_locations=num_locations, 
                rho_sigma_factor=arbitrary_rho, 
                cons_emission_val=constant_emission_val
            ),
            C_sim=spatial_corr_sim_1D(
                L=L, 
                num_locations=num_locations, 
                print_results=False
            )
        )[0][0] / arbitrary_rho for L in L_values]

    if plot_option == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=dpi_setting)

    if plot_option in [1, 3]:
        # First plot:
        if plot_option == 1:
            plt.figure(figsize=(10, 6), dpi=dpi_setting)
            ax1 = plt.gca()
        for L in L_values:
            C_sim = spatial_corr_sim_1D(L=L, num_locations=num_locations, print_results=False)
            aggr_unc_list = [aggregated_unc(accum_coord, S_sim=st_dev_matrix(num_locations, rho, constant_emission_val), C_sim=C_sim)[0][0] for rho in rho_values]
            ax1.plot(rho_values, aggr_unc_list, label=f'L = {int(L-0.01)}', linestyle=':')
        ax1.set_xlabel('$\\rho_\\sigma$ factor')
        ax1.set_ylabel('Aggregated Uncertainty ($\\sigma_\\eta$)')
        ax1.set_title('Aggregated Uncertainty vs. $\\rho_\\sigma$ factor for different L values')
        ax1.legend()
        ax1.grid(True, linestyle=':')

    if plot_option in [2, 3]:
        # Second plot:
        if plot_option == 2:
            plt.figure(figsize=(10, 6), dpi=dpi_setting)
            ax2 = plt.gca()
        for sigma_eta in sigma_eta_values:
            rho_sigma_list = [sigma_eta / alpha for alpha in alpha_coefficients]
            ax2.plot(L_values, rho_sigma_list, label=f'$\\sigma_\\eta$ = {sigma_eta:.2f}')
        ax2.set_xlabel('L value')
        ax2.set_ylabel('$\\rho_\\sigma$ factor')
        ax2.set_title('Relationship between L and $\\rho_\\sigma$ for different $\\sigma_\\eta$ values')
        ax2.legend()
        ax2.grid(True, linestyle=':')

    if plot_option in [1, 2]:
        plt.tight_layout()
        plt.show()

    
################ MAIN: ################
def main():
    """
    Main function to execute the spatial correlation experiments and plotting.
    """ 
    start_time = time.time() # start of timer 

    # Constants:
    NUM_LOCATIONS = 100
    L = 20
    L_VALUES_LIST = np.arange(0, 26, 5)  # Define which values of L should be compared in the plot for experiment 1 and 2 
    SIMULATION_SIZES = [int(size) for size in  [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]] # for CLT/LLN convergence experiment

    # Simulate matrices:
    C_sim = spatial_corr_sim_1D(L=L, num_locations=NUM_LOCATIONS)  # Simulate spatial corr. matrix C
    sorted_pos_lambda, lambda_vector, corresp_sorted_Q, Q_matrix = sorted_evalues_evectors(C_sim)  # Sort and perform e.value decomp.

    # Plots:
    heatmap_C_QQ(C_sim, corresp_sorted_Q, high_dpi=0)
    plot_eigenvalues_and_eigenvectors(sorted_pos_lambda, corresp_sorted_Q, high_dpi=0)
    plot_eigenvectors_cumulative(sorted_pos_lambda, corresp_sorted_Q, high_dpi=0)

    # # Small scale simulated experiments:
    plot_L_comparison_experiment(L_values=L_VALUES_LIST, high_dpi=0)
    spatial_corr_convergence_experiment(sorted_pos_lambda, corresp_sorted_Q, SIMULATION_SIZES, NUM_LOCATIONS, C_sim, high_dpi=0)
    test_coord = [30, 70]
    rho_sigma_factor_LIST = np.arange(0.01, 0.21, 0.01)
    sigma_eta_values_LIST = np.linspace(0.01, 100, 6)
    plot_sigmas_experiment(L_values=L_VALUES_LIST,
                        accum_coord=test_coord,
                        rho_values=rho_sigma_factor_LIST,
                        sigma_eta_values=sigma_eta_values_LIST,
                        plot_option=3,
                        high_dpi=0)

    # End timer and print execution time:
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()