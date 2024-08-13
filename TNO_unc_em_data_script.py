#! /usr/bin/env python3

# David Mathas - TNO

# imports:
import numpy as np
import xarray as xr
import os
from pathlib import Path
import time 
from TNO_processing_classes import Emis_unc_DF, Data_analyzer

################ MAIN: ################ 
def main():
    """Main function to import, convert and plot uncertainty data.""" 
    start_time = time.time() # start of timer 

    # Paths:
    YEAR = 2018
    pdir = Path(os.path.join(os.environ['PROJECTS'], 'EU/CoCO2'))
    path_ds_unc_TNO = os.path.join( pdir, 'WP2-priors', 'Task5-uncertainties', 'data', 'output', 'TNO_GHGco_v4_0_%i_uncertainties_v2_0_COCO2.nc' % YEAR) # change path here to other path if runninng locally.
    basepath = Path(os.environ['MODAS_SHARED']) # change path here to other path if runninng locally.
    final_code_dir = basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Final code' 

    # Load in data:
    print('Loading in data... \n')
    ds_unc_TNO = xr.open_dataset(path_ds_unc_TNO)
    print('NetCDF file: \n', ds_unc_TNO, '\n')

    # Convert:
    df_unc_TNO = Emis_unc_DF(ds_unc_TNO)
    print('Converted NetCDF to dataframe. \n')

    # Plotting/analyzer class:
    analyzer_unc = Data_analyzer(df_unc_TNO, basepath)

    # Specific single plot:
    single_plot = True # OFF ON SWITCH
    if single_plot:
        specie = 'CO2'
        ctry_acro = ['NLD', 'DEU']    
        cat = 'F1-F2-F3-F4'        
        analyzer_unc.plot_map(species = f'unc_distr_{specie}', 
                                country_acro = ctry_acro,
                                emis_cat = [cat], 
                                high_dpi = 1,
                                save_plot = 1)

    # All maps plot loop: 
    multi_plot_loop = False # OFF ON SWITCH
    if multi_plot_loop:
        progress = 0
        for cat in df_unc_TNO.cats['emis_cat_code']:
            for specie in df_unc_TNO.species:
                progress += 1
                progress_perc = np.round(progress / (len(df_unc_TNO.cats['emis_cat_code']) * len(df_unc_TNO.species)) * 100, 2)
                print('Progess:', progress_perc, '%') # to display some progress statistic

                analyzer_unc.plot_map(species = f'unc_distr_{specie}', 
                            country_acro = None,
                            emis_cat = [cat], 
                            high_dpi = 1,
                            save_plot = 1)

    ## Comparison of the reported uncertainty and the self-aggregated uncertainty: 
    unc_comp_loop = False
    if unc_comp_loop: # OFF ON SWITCH
        for cat in df_unc_TNO.cats['emis_cat_code']:
            analyzer_unc.plot_uncertainty_comparison(cat)
    

    # End timer and print execution time:
    end_time = time.time()
    print(f"\n Execution time: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()