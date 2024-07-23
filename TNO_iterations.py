#! /usr/bin/env python3

# David Mathas - TNO 

# imports: 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.style.use('uba')
import xarray as xr
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import os
from pathlib import Path
import sys
import platform
import scipy as sp
import time
from typing import Tuple

# paths:
YEAR = 2018
pdir = Path(os.path.join( os.environ['PROJECTS'], 'EU/CoCO2' ))
path_ds_unc_TNO = os.path.join( pdir, 'WP2-priors', 'Task5-uncertainties', 'data', 'output', 'TNO_GHGco_v4_0_%i_uncertainties_v2_0_COCO2.nc' % YEAR)

basepath = Path(os.environ['MODAS_SHARED'])
final_code_dir = basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Final_code'
min_L_values_dir = basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Minimum_CorrLengths.csv'
optimized_params_dir =  basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Updated_corr_lengths_gamma'
sys.path.insert(0, str(final_code_dir))  # for adding existing final code scripts 

# own:
import TNO_processing_classes # assuming TNO_processing_classes.py is in the current dir
import tools
from tools import ll_distance, make_distance_matrix, make_distance_matrix_vec
import TNO_optim_L_gamma
from TNO_small_experiments_1D import *
import TNO_small_experiments_1D


## class
class IterationsAnalysis:
    def __init__(self, optimized_params_dir, full_grid, countries, specie, category, save_path):
        self.optim_dir = optimized_params_dir
        self.grid = full_grid
        self.countries = countries
        self.specie = specie
        self.category = category
        self.save_path = save_path
        # per country saves:
        self.unc_matrices = {}
        self.L_optims = {}
        self.nr_blocks = None
        self.size = None

        self.results = pd.DataFrame()

    def load_data(self):
        for country in self.countries:
            file_path = self.optim_dir / f'Updated_corr_lengths_gamma_{country}.csv'
            data = pd.read_csv(file_path)
            filtered_data = data[(data['Species'] == self.specie) & (data['Emission Category'] == self.category)]
            L_optim = filtered_data['Optimized L (km)'].values[0] * 1000  # Convert km to meters

            unc_matrix_path = self.optim_dir / f'Uncertainty_Matrix_{country}.csv'
            columns = ['longitude_source', 'latitude_source', f'distr_{self.specie}', f'unc_distr_{self.specie}', 'Emission Category']
            unc_matrix = pd.read_csv(unc_matrix_path)[columns]
            unc_matrix = unc_matrix[unc_matrix['Emission Category'] == self.category].dropna()

            self.unc_matrices[country] = unc_matrix
            self.L_optims[country] = L_optim

    def calculate_C_spec(self, country, B_bool: bool = False):
        if country not in self.unc_matrices:
            self.load_data()  
        unc_matrix = self.unc_matrices[country]
        L_optim = self.L_optims[country]

        coords_spec = unc_matrix[['longitude_source', 'latitude_source']].values
        distance_matrix = make_distance_matrix_vec(coords_spec)
        self.size = distance_matrix.shape[0]
        C = np.exp(-(distance_matrix / L_optim)) if L_optim > 0 else np.eye(len(distance_matrix))
        
        if B_bool:
            abs_sigma = optimized_unc_matrix[f'unc_distr_{specie}'] * optimized_unc_matrix[f'distr_{specie}']
            S = np.diag(abs_sigma)
            B = S @ C @ S.T
            return C, B
        else:
            return C, None

    def calculate_C_square(self, country, B_bool: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        full_grid = self.grid
        if country not in self.unc_matrices:
            self.load_data()  
        unc_matrix = self.unc_matrices[country]
        L_optim = self.L_optims[country]
        coords_spec = unc_matrix[['longitude_source', 'latitude_source']].values

        min_long = np.min(unc_matrix.longitude_source)
        max_long = np.max(unc_matrix.longitude_source)
        min_lat = np.min(unc_matrix.latitude_source)
        max_lat = np.max(unc_matrix.latitude_source)

        long_spacing = np.diff(full_grid.longitude.values).mean()
        lat_spacing = np.diff(full_grid.latitude.values).mean()

        longitude = np.arange(min_long, max_long + long_spacing, long_spacing)
        latitude = np.arange(min_lat, max_lat + lat_spacing, lat_spacing)

        grid_longitude, grid_latitude = np.meshgrid(longitude, latitude)
        coords_square = np.stack([grid_longitude.ravel(), grid_latitude.ravel()], axis=1)
        distance_matrix = make_distance_matrix_vec(coords_square)
        C_square = np.exp(-(distance_matrix / L_optim)) if L_optim > 0 else np.eye(distance_matrix.shape[0])

        p = int(grid_latitude.shape[0])
        m = C_square.shape[0] / p
        self.nr_blocks = p

        B_square = None
        if B_bool:
            S_vector = np.zeros(coords_square.shape[0])
            coords_spec_str = [str(coord_sp) for coord_sp in coords_spec]
            coords_square_str = [str(coord_sq) for coord_sq in coords_spec]

            abs_sigma = optimized_unc_matrix[f'unc_distr_{specie}'] * optimized_unc_matrix[f'distr_{specie}']
            spec_sigma_map = dict(zip(coords_spec_str, abs_sigma))

            for i, coord_str in enumerate(coords_square_str):
                if coord_str in spec_sigma_map:
                    S_vector[i] = spec_sigma_map[coord_str]

            S = np.diag(S_vector)
            B_square = S @ C_square @ S.T # dubbel check dit later nog even --> werkt S gebruiken om C_square te subsetten naar behoren ?

        return C_square, B_square
    

    def plot_matrix(self, country, use_square, B_bool=False, high_dpi=0):
        if not use_square: 
            matrices = self.calculate_C_spec(country, B_bool)
        if use_square:
            matrices = self.calculate_C_square(country, B_bool)

        mtypes = ['Correlation (C)', 'Covariance (B)']
        labels = ['Correlation', 'Covariance']

        dpi_setting = 300 if high_dpi else None

        for i, matrix in enumerate(matrices):
            if matrix is not None:
                plt.figure(dpi=dpi_setting)
                plt.imshow(matrix, cmap='plasma', norm=LogNorm())
                plt.colorbar(label=labels[i])
                plt.title(f'Spatial {mtypes[i]} Matrix for {country}')
                plt.grid(True)  
                plt.show()
            else:
                print("Matrix not available.")

    @staticmethod
    def calculate_direct_C_sqrt_eigen(C: np.ndarray):
        """
        Calculate the direct square root of the correlation matrix using eigenvalue decomposition.

        Args:
            country (str): The country for which to perform the calculation.
            use_square (bool): If True, use C_square matrix; otherwise, use C_spec matrix.

        Returns:
            tuple: (C_sqrt, compute_time)
                C_sqrt (np.ndarray): The square root of the correlation matrix.
                compute_time (float): The time taken to compute the square root.
        """
        st = time.time()
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # sqrt_eigenvalues = np.sqrt(np.clip(eigenvalues, a_min=0, a_max=None))
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        C_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
        et = time.time()

        compute_time = round(et - st, 3)
        return C_sqrt, compute_time

    @staticmethod
    def calculate_direct_C_sqrt_chol(C: np.ndarray):
        """
        Calculate the direct square root of the correlation matrix using Cholesky decomposition.

        Args:
            C (np.ndarray): Correlation matrix.

        Returns:
            tuple: (C_sqrt, compute_time)
                C_sqrt (np.ndarray): The square root of the correlation matrix.
                compute_time (float): The time taken to compute the square root.
        """
        st = time.time()
        C_sqrt = np.linalg.cholesky(C)
        et = time.time()
        
        compute_time = round(et - st, 3)
        return C_sqrt, compute_time
    

    def arnoldi_iteration(self, C: np.ndarray, n_steps: int = 350, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """DOCSTRINGS
        """
        st = time.time()
        # initializations:
        n = C.shape[0]
        Q = np.zeros((n, n_steps), dtype=C.dtype)  # make the datatypes match for numba!
        H = np.zeros((n_steps, n_steps), dtype=C.dtype)  
        b = np.random.rand(n).astype(C.dtype)
        q = b / np.linalg.norm(b)
        Q[:, 0] = q # first q vector ..
        
        # Regularize C:
        C += eps * np.eye(n, dtype=C.dtype)

        for i in range(n_steps - 1):
            v = C @ np.ascontiguousarray(Q[:, i]) # C: n x n; q_i: n x 1
            for j in range(i + 1):
                qj = np.ascontiguousarray(Q[:, j]) 
                H[j, i] = qj.T @ v # becomes 1 x 1 
                v -= H[j, i] * qj # n x 1

            # Re orthogonalize:
            for j in range(i + 1):
                qj = np.ascontiguousarray(Q[:, j])
                correction = qj @ v
                v -= correction * qj

            H[i + 1, i] = np.linalg.norm(v) # also 1 x 1
            if H[i + 1, i] > 0:
                Q[:, i + 1] = v / H[i + 1, i] # also n x 1
        
        arnoldi_eigenvalues, V = np.linalg.eigh(H)
        arnoldi_eigenvalues[arnoldi_eigenvalues < 0] = 0
        # Sort the eigenvalues and corresponding eigenvectors V:
        idx = np.argsort(arnoldi_eigenvalues)[::-1]
        arnoldi_eigenvalues = arnoldi_eigenvalues[idx]
        sorted_V = V[:, idx]

        Lambda_sqrt = np.sqrt(np.diag(arnoldi_eigenvalues))
        C_sqrt_arnoldi = Q @ sorted_V @ Lambda_sqrt @ sorted_V.T @ Q.T
        LB = 0.01
        C_sqrt_arnoldi[C_sqrt_arnoldi < LB] = 0

        et = time.time()
        
        compute_time = round(et - st, 3)
        return C_sqrt_arnoldi, compute_time

    def lanczos_iteration(self, C: np.ndarray, n_steps: int = 350, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the Lanczos iteration to approximate the eigenvalues and eigenvectors of a matrix.

        Args:
            C (np.ndarray): The matrix for which to compute the eigenvalues and eigenvectors.
            n_steps (int): The number of iterations to perform.
            tol (float): The tolerance for early termination based on vector norm.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns the orthogonal basis vectors (Q),
                                                    the tridiagonal matrix (T), and the beta values.
        """
        st = time.time()
        n = C.shape[0]  # Dimension of the matrix
        Q = np.zeros((n, n_steps), dtype=C.dtype)  # Orthogonal vectors
        T = np.zeros((n_steps, n_steps), dtype=C.dtype)  # Tridiagonal matrix
        beta = np.zeros(n_steps, dtype=C.dtype)  # Beta values

        # Start with a random vector and normalize it
        b = np.random.rand(n).astype(C.dtype)
        q = b / np.linalg.norm(b)
        Q[:, 0] = q  # Set the first column to the normalized vector

        for i in range(1, n_steps):
            v = C @ Q[:, i-1]  # Compute v = C * q_i
            alpha = Q[:, i-1].T @ v  # Compute alpha = q_i' * v
            v -= alpha * Q[:, i-1] + (beta[i-1] * Q[:, i-2] if i > 1 else 0)  # Orthogonalize against previous vectors

            # Re-orthogonalization to improve numerical stability
            for j in range(i):
                v -= (Q[:, j].T @ v) * Q[:, j]

            beta[i] = np.linalg.norm(v)  # Compute the norm of v
            if beta[i] < tol:  # Check convergence
                print("[Lanczos] Converged at iteration", i)
                break

            Q[:, i] = v / beta[i]  # Normalize v to get the next orthogonal vector
            T[i-1, i] = T[i, i-1] = beta[i]  # Fill the tridiagonal matrix T
            T[i-1, i-1] = alpha

        Q_lanczos, T =  Q[:, :i + 1], T[:i + 1, :i + 1]

        lanczos_eigenvalues, V = np.linalg.eigh(T)
        lanczos_eigenvalues[lanczos_eigenvalues < 0] = 0
        # Sort the eigenvalues and corresponding eigenvectors V:
        idx = np.argsort(lanczos_eigenvalues)[::-1]
        lanczos_eigenvalues = lanczos_eigenvalues[idx]
        sorted_V = V[:, idx]

        Lambda_sqrt = np.sqrt(np.diag(lanczos_eigenvalues))
        C_sqrt_lanczos = Q_lanczos @ sorted_V @ Lambda_sqrt @ sorted_V.T @ Q_lanczos.T
        LB = 0.01
        C_sqrt_lanczos[C_sqrt_lanczos < LB] = 0

        et = time.time()

        compute_time = round(et - st, 3)
        return C_sqrt_lanczos, compute_time

    @staticmethod
    def compute_J(m: int) -> np.ndarray:
        """Compute the signature matrix J for given block size m."""
        J = np.zeros((2 * m, 2 * m), dtype=np.float32)
        J[:m, :m] = np.eye(m)
        J[m:2*m, m:2*m] = -np.eye(m)
        return J

    @staticmethod
    def compute_G(C: np.ndarray, m: int, p: int) -> np.ndarray:
        """Compute the initial G matrix using the Cholesky factor of T0."""
        G = np.zeros((2*m, p*m), dtype=np.float32)
        L0 = np.linalg.cholesky(C[:m, :m])
        L0_inv = np.linalg.inv(L0)
        T0 = L0.T
        G[:m, :m] = T0
        for j in range(1, p):
            T_j = C[:m, m * j:m * (j + 1)]
            transformed_T_j = L0_inv @ T_j
            G[:m, m * j:m * (j + 1)] = transformed_T_j
            G[m:2*m, m * j:m * (j + 1)] = transformed_T_j
        return G, T0

    @staticmethod
    def compute_rho(G: np.ndarray, 
                    T0: np.ndarray, 
                    i: int, 
                    m: int, 
                    k: int) -> float:
        """Compute rho and T_i1 for given indices."""
        scaled_factor = .25
        T_i1 = G[m:m*2, m*i:m*(i+1)][:, k] * scaled_factor # to ensure valid square roots 
        T_i1_sum = np.sum(T_i1**2)
        rho = np.sign(T0[k, k]) * np.sqrt(T0[k, k]**2 - T_i1_sum)
        return rho, T_i1

    @staticmethod
    def compute_H(rho: float, 
                T_i1: np.ndarray, 
                J: np.ndarray, 
                T0: np.ndarray, 
                k: int, 
                m: int) -> np.ndarray:
        """Compute the Householder matrix H using rho and T_i1."""
        x = np.zeros(m*2, dtype=np.float32).reshape(-1, 1)
        x[k-1] = (T0[k, k] + rho) #/ (15*k**2)
        x[m:, 0] = T_i1
        H = J - 2* x @ x.T / (x.T @ J @ x)
        return H
    
    def FBTBR(self, C: np.ndarray) -> np.ndarray:
        """Performs the Fast Block Toeplitz Cholesky Decomposition.
        m is the block size.
        p is the number of blocks.
        """
        st = time.time()
        red_factor = 2. # -> reduction factor 
        p = self.nr_blocks
        assert p, "FBTBR algorithm is only compatible with square C matrix (needs known constant block size) \n>> Make sure to use calculate_C_square(country)"
        m = int(C.shape[0] / p)  
        R = np.zeros_like(C, dtype=np.float32)
        J = self.compute_J(m)
        G, T0 = self.compute_G(C, m, p)

        # Construct other block rows of tr. factor R:
        for i in range(p):
            if i == 0:
                R[:m, :] = G[:m, :].copy()
                G[:m, m:] = G[:m, :-m] # shift to the right 
                G[:m, :m] = 0

            else:
                for k in range(1, m):
                    rho, T_i1 = self.compute_rho(G, T0, i, m, k)
                    G[m:2*m, (i-1)*m:(i)*m] = 0
                    H = self.compute_H(rho, T_i1, J, T0, k, m)
                    H *= H

                G = H @ G
                G[m:2*m, i*m:(i+1)*m] = 0 

                G[:m, m*i:m*(i+1)] = np.triu(G[:m, m*i:m*(i+1)]) # ensure triangularization
                R[i*m:(i+1)*m, :] = G[:m, :] / red_factor # ensure the values don't explode!
                G[:m, (i+1)*m:] = G[:m, i*m:-m] # shift to the right
                G[:2*m, :(i+1)*m] = 0
        et = time.time()
    
        compute_time = round(et - st, 3)
        return R.T, compute_time

    @staticmethod
    def compare_sample_stats(C_true_sqrt, C_estimated_sqrt, N_sim = 5000):
        size = C_true_sqrt.shape[0]
        W = np.random.normal(0, 1, (size, N_sim))
        # Generate samples:
        V_true = C_true_sqrt @ W
        V_estimated = C_estimated_sqrt @ W
        
        C_true_sample = np.cov(V_true)
        C_estimated_sample = np.cov(V_estimated)
        # extract variances:
        variances_true = np.diag(C_true_sample)
        variances_estimated = np.diag(C_estimated_sample)
        
        C_fro_norm_diff = np.linalg.norm(C_true_sample - C_estimated_sample, 'fro')
        C_true_fro_norm = np.linalg.norm(C_true_sample, 'fro')
        C_rel_fro_norm = C_fro_norm_diff / C_true_fro_norm

        variance_diff = variances_estimated - variances_true
        rel_variance_norm = np.linalg.norm(variance_diff) / np.linalg.norm(variances_true)

        return round(rel_variance_norm, 3), round(C_rel_fro_norm, 3)


    def compare_norms(self, C_true, C_true_sqrt, C_estimated_sqrt):
        # sing norms:
        singular_norm_diff = np.linalg.norm(C_true_sqrt - C_estimated_sqrt, ord=2)
        relative_singular_norm_diff = round((singular_norm_diff / np.linalg.norm(C_true_sqrt, ord=2)), 3)

        # frob norms:
        frob_norm_diff = np.linalg.norm(C_true_sqrt - C_estimated_sqrt, 'fro')
        frob_norm_C_diff = np.linalg.norm(C_true - C_estimated_sqrt @ C_estimated_sqrt.T, 'fro')
        relative_frob_norm_C_sqrt = round((frob_norm_diff / np.linalg.norm(C_true_sqrt, 'fro')), 3)
        relative_frob_norm_C = round((frob_norm_C_diff / np.linalg.norm(C_true, 'fro')), 3)

        # random sampled norms:
        rel_variance_norm, C_rel_fro_norm = self.compare_sample_stats(C_true_sqrt, C_estimated_sqrt)

        return {'Rel. sing. norm C_sqrt': relative_singular_norm_diff,
                # 'Rel. frob. norm C_sqrt': relative_frob_norm_C_sqrt,
                'Rel. frob. norm C': relative_frob_norm_C,
                'Rel. eucl. norm sample var.': rel_variance_norm,
                'Rel. frob. norm sample cov.': C_rel_fro_norm}


    def compare_methods(self, nr_tests: int = 4):
        self.load_data()  # Ensure all data is loaded first!
        results = {}  
        for country in self.countries:
            print(f'Analyzing {country} computation time and norms...')
            averages = {
            'time_eigen': 0,
            'time_arnoldi': 0,
            'time_lanczos': 0,
            'time_chol': 0,
            'time_FBTBR': 0,
            'norms_arnoldi': {'Rel. sing. norm C_sqrt': [], 'Rel. frob. norm C': [], 'Rel. eucl. norm sample var.': [], 'Rel. frob. norm sample cov.': []},
            'norms_lanczos': {'Rel. sing. norm C_sqrt': [], 'Rel. frob. norm C': [], 'Rel. eucl. norm sample var.': [], 'Rel. frob. norm sample cov.': []},
            'norms_FBTBR':   {'Rel. sing. norm C_sqrt': [], 'Rel. frob. norm C': [], 'Rel. eucl. norm sample var.': [], 'Rel. frob. norm sample cov.': []}
            # 'norms_arnoldi': {'Rel. sing. norm C_sqrt': [], 'Rel. frob. norm C_sqrt': [], 'Rel. frob. norm C': [], 'Rel. eucl. norm sample var.': [], 'Rel. frob. norm sample cov.': []},
            # 'norms_lanczos': {'Rel. sing. norm C_sqrt': [], 'Rel. frob. norm C_sqrt': [], 'Rel. frob. norm C': [], 'Rel. eucl. norm sample var.': [], 'Rel. frob. norm sample cov.': []},
            # 'norms_FBTBR':   {'Rel. sing. norm C_sqrt': [], 'Rel. frob. norm C_sqrt': [], 'Rel. frob. norm C': [], 'Rel. eucl. norm sample var.': [], 'Rel. frob. norm sample cov.': []}
        }

            C_spec, _ = self.calculate_C_spec(country)
            C_square, _ = self.calculate_C_square(country)
            for _ in range(nr_tests):
                # direct C_sqrt eigen:
                C_sqrt_true_eigen, t_true_eigen = self.calculate_direct_C_sqrt_eigen(C_spec)
                averages['time_eigen'] += t_true_eigen

                # Arnoldi & norms:
                C_sqrt_arnoldi, t_arnoldi = self.arnoldi_iteration(C_spec)
                norms_arnoldi = self.compare_norms(C_spec, C_sqrt_true_eigen, C_sqrt_arnoldi)
                averages['time_arnoldi'] += t_arnoldi
                for key in norms_arnoldi:
                    averages['norms_arnoldi'][key].append(norms_arnoldi[key])

                # Lanczos & norms:
                C_sqrt_lanczos, t_lanczos = self.lanczos_iteration(C_spec)
                norms_lanczos = self.compare_norms(C_spec, C_sqrt_true_eigen, C_sqrt_lanczos)
                averages['time_lanczos'] += t_lanczos
                for key in norms_lanczos:
                    averages['norms_lanczos'][key].append(norms_lanczos[key])

                # Direct C_sqrt chol:
                C_sqrt_true_chol, t_true_chol = self.calculate_direct_C_sqrt_chol(C_square)
                averages['time_chol'] += t_true_chol

                # FBTBR & norms:
                C_sqrt_FBTBR, t_FBTBR = self.FBTBR(C_square)
                norms_FBTBR = self.compare_norms(C_square, C_sqrt_true_chol, C_sqrt_FBTBR)
                averages['time_FBTBR'] += t_FBTBR
                for key in norms_FBTBR:
                    averages['norms_FBTBR'][key].append(norms_FBTBR[key])

            # Calculate averages:
            for key in averages:
                if 'time' in key:
                    averages[key] = round((averages[key] / nr_tests), 3)
                elif 'norms' in key:
                    for norm_key in averages[key]:
                        averages[key][norm_key] = round((sum(averages[key][norm_key]) / nr_tests), 3)

            country_size = self.size
            formatted_country = f"{country} ({country_size})"
            results[formatted_country] = averages
        return results
            

    def plot_results(self, results, save_plot=False, high_dpi=True):
        """
        Plot the results of computation times and norms for each country.

        Args:
            results (dict): The dictionary containing all computation times and norm results.
            save_plot (bool): If True, save the plot to disk.
            high_dpi (bool): If True, increase the resolution of the plot.
        """
        if save_plot:
            plt.ioff()  # Turn off interactive plotting to avoid showing it inline if saving.

        dpi_setting = 300 if high_dpi else None
        
        # First figure: Computation Times
        fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8), dpi=dpi_setting)
        methods = ['time_eigen', 'time_arnoldi', 'time_lanczos', 'time_chol', 'time_FBTBR']
        baseline_times = {method: results[list(results.keys())[0]][method] + 1e-5 for method in methods}
        
        # Absolute times plot
        ax1 = axes1[0]
        ax1.set_yscale('log')
        line_styles = {'time_eigen': 'dashed', 'time_arnoldi': 'solid', 'time_lanczos': 'solid', 'time_chol': 'dashed', 'time_FBTBR': 'solid'}
        labels = []
        for method in methods:
            times = [results[country][method] + 1e-5 for country in results] 
            label = method.replace('time_', 'Numpy ' if 'eigen' in method or 'chol' in method else '').capitalize()
            ax1.plot(list(results.keys()), times, marker='o', linestyle=line_styles[method], label=label)
            labels.append(label)
        ax1.set_ylabel('Average Computation Time [s]')
        ax1.set_title('Absolute Computation Times by Country')
        ax1.grid(True, linestyle=':')

        # Relative times plot
        ax2 = axes1[1]
        ax2.set_yscale('log')
        for method in methods:
            relative_times = [100 * (results[country][method] + 1e-5) / baseline_times[method] for country in results]
            ax2.plot(list(results.keys()), relative_times, marker='o', linestyle=line_styles[method])
        ax2.set_ylabel(r'Relative Computation Time [\%]')
        ax2.set_title('Relative Computation Times by Country')
        ax2.grid(True, linestyle=':')

        # Legend for the first figure:
        fig1.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.4, 0.055), fontsize=13, fancybox=True, shadow=True, ncol=5)
        # plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.tight_layout(rect=[0.01, 0.07, 1, 1]) 
        # OLD SETTINGS # fig1.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.435, 0.075), fontsize=13, fancybox=True, shadow=True, ncol=5)
        # # plt.tight_layout(rect=[0, 0.05, 1, 1])
        # plt.tight_layout(rect=[0.01, 0.06, 1, 1]) 


        if save_plot:
            filename = 'computation_times_plot_TESTTTT.png'
            full_path = os.path.join(self.save_path, filename)
            plt.savefig(full_path)
            print(f"\n Plot saved to {full_path}")
        else:
            plt.show()  
        plt.close(fig1)

        # Second figure: Norm Metrics
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), dpi=dpi_setting)
        norm_types = ['norms_arnoldi', 'norms_lanczos', 'norms_FBTBR']
        norms = ['Rel. sing. norm C_sqrt', 'Rel. frob. norm C', 'Rel. eucl. norm sample var.', 'Rel. frob. norm sample cov.']
        line_styles = ['--', '--', '-', '-', '-']
        norm_labels = []

        for ax, norm_type in zip(axes2, norm_types):
            for norm, style in zip(norms, line_styles):
                norm_values = [results[country][norm_type][norm] for country in results]
                label = norm
                ax.plot(list(results.keys()), norm_values, marker='o', label=label, linestyle=style)
                if label not in norm_labels:
                    norm_labels.append(label)
            ax.set_title(f'{norm_type.split("_")[1].upper()} Norm Metrics') 
            ax.set_ylabel('Relative Norm Value')
            ax.grid(True, linestyle=':')

        # Legend for the second figure:
        fig2.legend(labels=norm_labels, loc='upper center', bbox_to_anchor=(0.45, 0.055),  fontsize=14, fancybox=True, shadow=True, ncol=3)
        # plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.tight_layout(rect=[0.01, 0.08, 1, 1]) 

        if save_plot:
            filename = 'norm_metrics_plot_FINAL.png'
            full_path = os.path.join(self.save_path, filename)
            plt.savefig(full_path)
            print(f"\n Norm metrics plot saved to {full_path}\n")
        else:
            plt.show()  
        plt.close(fig2)


    def create_summary_table(self, results):
        data = []
        methods = ['eigen', 'arnoldi', 'lanczos', 'chol', 'FBTBR']
        norm_types = ['norms_arnoldi', 'norms_lanczos', 'norms_FBTBR']
        norms = ['Rel. sing. norm C_sqrt', 'Rel. frob. norm C', 'Rel. eucl. norm sample var.', 'Rel. frob. norm sample cov.']

        # Set a baseline for rel comparison:
        baseline_times = {f'time_{method}': results[list(results.keys())[0]][f'time_{method}'] + 1e-5 for method in methods}

        for country in results:
            for method in methods:
                entry = {
                    'Country-Method': f'{country} - {method.capitalize()}',
                    'Abs. Time': results[country][f'time_{method}'],
                    'Rel. Time': 100 * (results[country][f'time_{method}'] + 1e-5) / baseline_times[f'time_{method}']
                }
                # Add norms
                if method in ['arnoldi', 'lanczos', 'FBTBR']:
                    for norm in norms:
                        entry[norm] = results[country][f'norms_{method}'][norm]
                else:  # Numpy eigen / chol are assumed to be 'true':
                    for norm in norms:
                        entry[norm] = None
                data.append(entry)

        df = pd.DataFrame(data)
        df.set_index(['Country-Method'], inplace=True)

        filename = 'Computation_Times_and_Norms_FINAL.csv'
        full_path = os.path.join(self.save_path, filename)
        df.to_csv(full_path)
        print(f"Table saved to {full_path}\n")

        return df

    def exploratory_plot_C_sqrt(self, example_country='NLD', save_plot=True, high_dpi=True):
        if save_plot:
            plt.ioff()  # Turn off interactive plotting to avoid showing it inline if saving.

        C_spec, _ = self.calculate_C_spec(example_country)
        C_square, _ = self.calculate_C_square(example_country)

        C_sqrt_true_eigen, _ = self.calculate_direct_C_sqrt_eigen(C_spec)
        C_sqrt_arnoldi, _ = self.arnoldi_iteration(C_spec)
        C_sqrt_lanczos, _ = self.lanczos_iteration(C_spec)
        C_sqrt_true_chol, _ = self.calculate_direct_C_sqrt_chol(C_square)
        C_sqrt_FBTBR, _ = self.FBTBR(C_square)

        # Square roots and C reconstructions:
        sqrt_methods = [C_sqrt_true_eigen, C_sqrt_arnoldi, C_sqrt_lanczos, C_sqrt_true_chol, C_sqrt_FBTBR]
        method_names = ['NUMPY Eigen', 'Arnoldi', 'Lanczos', 'NUMPY Chol (using C. sq. m.)', 'FBTBR (using C. sq. m.)']
        reconstructed = [m @ m.T for m in sqrt_methods]

        # Plot:
        dpi_setting = 300 if high_dpi else None
        fig, axes = plt.subplots(2, 5, figsize=(15, 6), dpi=dpi_setting)

        for i, (sqrt, recon) in enumerate(zip(sqrt_methods, reconstructed)):
            ax = axes[0, i]
            im = ax.imshow(sqrt, cmap='plasma', interpolation='nearest', norm=LogNorm())
            ax.set_title(f'{method_names[i]} $C^{{1/2}}$', fontsize=9.5)
            ax.axis('off')

            ax = axes[1, i]
            ax.imshow(recon, cmap='plasma', interpolation='nearest', norm=LogNorm())
            ax.set_title(f'{method_names[i]} $C$', fontsize=9.5)
            ax.axis('off')

        # Add colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12) 
        fig.colorbar(im, cax=cbar_ax)

        if save_plot:
            filename = 'C_sqrt_comparison_plot_FINAL.png'
            full_path = os.path.join(self.save_path, filename)
            plt.savefig(full_path)#, bbox_inches='tight')
            print(f"\nPlot saved to {full_path}\n")
        else:
            plt.show()
        plt.close(fig)



################ MAIN: ################
def main():
    """Main function to execute the iterations time and norm comparisons and plotting.""" 
    # load in data: 
    ds_unc_TNO = xr.open_dataset(path_ds_unc_TNO)
    df_unc_TNO = TNO_processing_classes.Emis_unc_DF(ds_unc_TNO)
    min_L_values = pd.read_csv(min_L_values_dir, delimiter=';')
    print("Loaded in all data.\n")
    save_path = '/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/Internship/David/Plots/Iteration_plots'

    countries = ['LUX', 'SVN', 'NLD', 'AUT', 'GRC', 'ROU', 'POL', 'ESP'] # logical size comparison
    specie = 'CO2'
    cat = 'F1-F2-F3-F4'

    full_grid = df_unc_TNO.grid
    iterations = IterationsAnalysis(optimized_params_dir, full_grid, countries, specie, cat, save_path)
    
    # Exploratory plots:
    # iterations.exploratory_plot_C_sqrt()

    results = iterations.compare_methods()
    iterations.plot_results(results, save_plot=1)

    summary_table = iterations.create_summary_table(results)
    latex_table = summary_table.to_latex(index=True) 
    print(latex_table, '\n\n')

if __name__ == "__main__":
    main()