#! /usr/bin/env python3
#! /usr/bin/env python3

# David Mathas - TNO 

# imports: 
import numpy as np
import pandas as pd 
import xarray as xr 
from tqdm import tqdm
import os
from pathlib import Path
import sys
import time
import scipy as sp
from numba import jit
from typing import Tuple

# paths:
YEAR = 2018
pdir = Path(os.path.join( os.environ['PROJECTS'], 'EU/CoCO2' ))
path_ds_unc_TNO = os.path.join( pdir, 'WP2-priors', 'Task5-uncertainties', 'data', 'output', 'TNO_GHGco_v4_0_%i_uncertainties_v2_0_COCO2.nc' % YEAR)

basepath = Path(os.environ['MODAS_SHARED'])
final_code_dir = basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Final_code'
min_L_values_dir = basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Minimum_CorrLengths.csv'
sys.path.insert(0, str(final_code_dir))  

# own:
import TNO_processing_classes # assuming TNO_processing_classes.py is in the current dir
from tools import ll_distance, make_distance_matrix, make_distance_matrix_vec


## Functions for optim.:
def calculate_aggregated_sigma(L: float, 
                               unc_distr_matrix: pd.DataFrame, 
                               specie: str, 
                               distance_matrix: np.ndarray, 
                               cat_selection_vector: np.ndarray, 
                               too_large_bool: bool) -> float:
    """
    Calculate the grid-aggregated standard deviation of emissions across a specified category and species,
    taking into account the correlation between emission sources based on their distances.

    Parameters:
        L (float): Characteristic length scale for spatial correlation decay. If zero, spatial correlation is ignored.
        unc_distr_matrix (pd.DataFrame): DataFrame containing emissions data and uncertainties for different species.
                                     It should have columns named 'unc_distr_{specie}' and 'distr_{specie}' where {specie}
                                     is replaced by the actual species name.
        specie (str): Name of the species for which the aggregation is calculated.
        distance_matrix (np.ndarray): A 2D numpy array representing the pairwise distances between emission sources.
        cat_selection_vector (np.ndarray): A boolean array used to filter the distance matrix and sigma values based
                                        on some category selection criteria.
        too_large_bool (bool): A flag (0 or 1) indicating whether the distance matrix is too large to handle without
                              filtering. If 0, the distance matrix and sigma values are filtered by `cat_selection_vector`.

    Returns:
        float: The aggregated standard deviation normalized by the total emissions of the specified species, giving
               a measure of relative uncertainty. If total emissions are zero, returns 0 to avoid division by zero.
    """
    abs_sigma = unc_distr_matrix[f'unc_distr_{specie}'].astype(np.float32) * unc_distr_matrix[f'distr_{specie}'].astype(np.float32)

    if too_large_bool == 0:
        distance_matrix = distance_matrix[cat_selection_vector][:, cat_selection_vector]

    if L == 0:
        sigma_eta_aggr = np.sum(abs_sigma**2)**.5
    else:    
        C = np.exp(-0.5 * (distance_matrix.astype(np.float32) / L)**2)
        S = np.diag(abs_sigma) # matrix with std devs
        B = S @ C @ S
        a = np.ones(B.shape[0], dtype=np.float32)
        sigma_eta_aggr = (a @ B @ a)**.5  # aggregated absolute sigma eta 
    total_emissions = np.sum(unc_distr_matrix[f'distr_{specie}'])
    return float(sigma_eta_aggr / total_emissions) if total_emissions != 0 else 0.


def optimize_uncertainties(unc_matrix_data: pd.DataFrame, 
                           min_L_values: pd.DataFrame, 
                           country: str, 
                           specie: str, 
                           emis_cat: str, 
                           reported_sigma: float, 
                           dist_matrix: np.ndarray, 
                           too_large_bool: bool) -> Tuple[float, float, pd.DataFrame]:
    """
    Optimizes the uncertainty distribution based on the grid-aggregated and reported standard deviation of emissions
    and adjusts the correlation length or gamma uncertainty factor to match the uncertainty levels.

    Parameters:
        unc_matrix_data (pd.DataFrame): DataFrame containing emissions data including uncertainties and geographical information.
        min_L_values (pd.DataFrame): DataFrame containing the minimum correlation lengths for different categories.
        country (str): The country for which the uncertainties are being optimized.
        specie (str): The species for which the uncertainties are being optimized.
        emis_cat (str): The emission category code for which the uncertainties are being optimized.
        reported_sigma (float): The reported standard deviation of emissions for the species and category.
        dist_matrix (np.ndarray): A 2D numpy array of distances between emission sources used to calculate spatial correlations.
        too_large_bool (bool): A boolean flag indicating if optimizations need to be restricted due to large data size.

    Returns:
        Tuple:
            - L_value (float): The optimized correlation length in meters.
            - gamma_unc_factor (float): The scaling factor applied to the uncertainties to match the reported sigma.
            - unc_distr_matrix (pd.DataFrame): The updated DataFrame with adjusted uncertainties.
    """
    L_value = min_L_values[min_L_values['AggSectorCode'] == emis_cat]['MinCorrLength'].values[0] * 1000

    if too_large_bool:
        unc_distr_matrix_original = unc_matrix_data[[f'distr_{specie}', f'unc_distr_{specie}', \
                                                                          'longitude_source', 'latitude_source', \
                                                                          'longitude_index', 'latitude_index'
                                                                          ]]
        cat_selection_vector = None
    
    else:
        cat_selection_vector = (unc_matrix_data['emis_cat_code'] == emis_cat).values
        unc_distr_matrix_original = unc_matrix_data[cat_selection_vector][[f'distr_{specie}', f'unc_distr_{specie}', \
                                                                          'longitude_source', 'latitude_source', \
                                                                          'longitude_index', 'latitude_index'
                                                                          ]] 

    unc_distr_matrix = unc_distr_matrix_original.copy() 
    
    aggregated_sigma = calculate_aggregated_sigma(L_value, unc_distr_matrix, specie, dist_matrix, cat_selection_vector, too_large_bool)
    gamma_unc_factor = 0

    # Optimization:
    if reported_sigma > 0 and aggregated_sigma == 0: 
        mean_std_dev = np.mean(unc_matrix_data[f'unc_distr_{specie}']) # mean relative standard deviation over all sectors for country-specie combination
        unc_distr_matrix.loc[:, f'unc_distr_{specie}'] = mean_std_dev if mean_std_dev > 0 else 0.01 
        unc_distr_matrix_original.loc[:, f'unc_distr_{specie}'] = mean_std_dev if mean_std_dev > 0 else 0.01
        aggregated_sigma = calculate_aggregated_sigma(L_value, unc_distr_matrix, specie, dist_matrix, cat_selection_vector, too_large_bool) 
        gamma_unc_factor = 1.

    if reported_sigma > 0 and aggregated_sigma > 0:

        L_stepsize = 5000 # 5 km in meters 
        UB_L = 500000
        gamma_unc_factor = 1.

        if aggregated_sigma < reported_sigma:
            while aggregated_sigma < reported_sigma and gamma_unc_factor < 5.:
                if  L_value + L_stepsize/2 <= UB_L:
                    L_value += L_stepsize
                    aggregated_sigma = calculate_aggregated_sigma(L_value, unc_distr_matrix, specie, dist_matrix, cat_selection_vector, too_large_bool)

                else:
                    gamma_unc_factor += 0.005
                    adjusted_unc_distr = unc_distr_matrix_original[f'unc_distr_{specie}'] * gamma_unc_factor
                    unc_distr_matrix.loc[:, f'unc_distr_{specie}'] = adjusted_unc_distr
                    aggregated_sigma = calculate_aggregated_sigma(L_value, unc_distr_matrix, specie, dist_matrix, cat_selection_vector, too_large_bool)

        elif aggregated_sigma > reported_sigma:
            while aggregated_sigma > reported_sigma:
                gamma_unc_factor -= 0.005 
                adjusted_unc_distr = unc_distr_matrix_original[f'unc_distr_{specie}'] * gamma_unc_factor
                unc_distr_matrix.loc[:, f'unc_distr_{specie}'] = adjusted_unc_distr
                aggregated_sigma = calculate_aggregated_sigma(L_value, unc_distr_matrix, specie, dist_matrix, cat_selection_vector, too_large_bool)

        return L_value, gamma_unc_factor, unc_distr_matrix

    else:
        unc_distr_matrix.loc[:, f'unc_distr_{specie}'] *= gamma_unc_factor
        return L_value, gamma_unc_factor, unc_distr_matrix



def run_optimization_for_country(country: str, 
                                 species: str, 
                                 categories: list, 
                                 data_df: pd.DataFrame, 
                                 min_L_values: pd.DataFrame) -> pd.DataFrame:
    """
    Runs an optimization process across multiple species and emission categories for a specific country,
    aiming to align reported uncertainties with calculated ones by adjusting correlation lengths and uncertainty factors.

    Parameters:
        country (str): The country for which the optimization is being run.
        species (str): The species or pollutants for which uncertainties are being optimized.
        categories (list): List of emission categories for which the optimization is performed.
        data_df (pd.DataFrame): DataFrame containing all necessary emissions data.
        min_L_values (pd.DataFrame): DataFrame with minimum correlation lengths for different categories.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the optimizations, including the new gamma factors,
                      optimized correlation lengths, and comparisons of the reported and calculated uncertainties.
    """
    results = []
    full_matrix_data = pd.DataFrame()

    # define these outside the s-c loop for speed increase
    unc_matrix_data = data_df.extract_subset(country=[country])
    coords = unc_matrix_data[['longitude_source', 'latitude_source']].values.astype(np.float32)
    size_cap = 50000
    too_large_bool = coords.shape[0] > size_cap 
    print('determined large bool:', too_large_bool)

    if too_large_bool:
        unc_matrix_data = data_df
        sel_vector = None
    else:    
        distance_matrix = make_distance_matrix_vec(coords)

    for specie in species:
        for emis_cat in tqdm(categories, desc=f'Optimizing for {country}, {specie}'):
            
            reported_sigma = data_df.extract_country_lvl(country=country, species=specie, cat=emis_cat)['uncertainty'].values[0].astype(np.float32)

            if too_large_bool:
                unc_matrix_data = data_df.extract_subset(country=[country], cat=[emis_cat])
                coords = unc_matrix_data[['longitude_source', 'latitude_source']].values
                distance_matrix = make_distance_matrix_vec(coords)
            else: 
                sel_vector = (unc_matrix_data['emis_cat_code'] == emis_cat).values # for new aggr and diff calc in results table

            # Call optimizer: 
            optimized_L, gamma_factor, unc_distr_matrix = optimize_uncertainties(unc_matrix_data, min_L_values, country, specie, emis_cat, reported_sigma, distance_matrix, too_large_bool)
            # unc_distr_matrix_original[f'unc_distr_{specie}'] = unc_distr_matrix_original[f'unc_distr_{specie}'] * gamma_factor
            
            results.append({
                'Country': country,
                'Species': specie,
                'Emission Category': emis_cat,
                'Original minimal L (km)': min_L_values[min_L_values['AggSectorCode'] == emis_cat]['MinCorrLength'].values[0],
                'Optimized L (km)': optimized_L / 1000,
                'Gamma Factor': gamma_factor,
                'Reported Sigma': reported_sigma,
                'New aggregated Sigma': calculate_aggregated_sigma(optimized_L, unc_distr_matrix, specie, distance_matrix, sel_vector, too_large_bool), 
                'Diff.': np.abs(calculate_aggregated_sigma(optimized_L, unc_distr_matrix, specie, distance_matrix, sel_vector, too_large_bool) - reported_sigma)
            })

            unc_distr_matrix['Country'] = country
            unc_distr_matrix['Emission Category'] = emis_cat

            full_matrix_data = pd.concat([full_matrix_data, unc_distr_matrix], ignore_index=True)

    matrix_data_path = f'/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/Internship/David/Code/Updated_corr_lengths_gamma/Uncertainty_Matrix_{country}.csv'
    full_matrix_data.to_csv(matrix_data_path, index=False)

    # Overview table:
    results_df = pd.DataFrame(results)
    # To csv:
    save_path = f'/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/Internship/David/Code/Updated_corr_lengths_gamma/Updated_corr_lengths_gamma_{country}.csv'
    results_df.to_csv(save_path, index=False)
    # To latex:
    # latex_output = results_df.to_latex(index=False, caption="Optimization Results", label="tab:optim_results", longtable=True, column_format='|c|c|c|c|c|c|')
    # print(latex_output)
    
    return results_df




### MAIN:
def main():
    """
    Main function to import, convert and plot uncertainty data.
    """ 
    start_time = time.time() # start of timer 

    # load in data: 
    ds_unc_TNO = xr.open_dataset(path_ds_unc_TNO)
    df_unc_TNO = TNO_processing_classes.Emis_unc_DF(ds_unc_TNO)
    min_L_values = pd.read_csv(min_L_values_dir, delimiter=';')
    print("Loaded in all data. \n")

    # Run optimization for each country:      
    emis_cats = df_unc_TNO.cats['emis_cat_code'].values  
    countries = [os.environ['COUNTRY']] 
    
    # Run optimization:
    for country in countries:
        results_df = run_optimization_for_country(country, df_unc_TNO.species, emis_cats, df_unc_TNO, min_L_values)
        print(f'\n Results for {country} have been saved.')

    end_time = time.time()
    print(f"\n Execution time: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()