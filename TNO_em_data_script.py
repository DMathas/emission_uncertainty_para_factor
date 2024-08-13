# David Mathas - TNO

# imports:
import numpy as np
import xarray as xr
import os
from pathlib import Path
import time 
from TNO_processing_classes import EmisDF, Data_analyzer

################ MAIN: ################ 
def main():
    """
    Main function to import, convert and plot emission data.
    """ 
    start_time = time.time() # start of timer 
        
    # load in data:
    print('Loading in data... \n')
    basepath = Path(os.environ['MODAS_SHARED'])
    pdir = Path(os.path.join(str(basepath), 'ProjectData/Internship/David/Code' ))
    print('Path: \n', pdir, '\n')

    path_data_TNO_2018 = pdir / 'TNO_GHGco_v4_0_year2018.nc'
    ds_TNO_2018 = xr.open_dataset(path_data_TNO_2018) # change path here to other path if runninng locally.
    print('NetCDF file: \n', ds_TNO_2018, '\n')

    # convert:
    df_TNO_2018 = EmisDF(ds_TNO_2018)
    print('Converted dataframe sources: \n', df_TNO_2018.sources, '\n')

    # Plotting/analyzer class instantiate:
    analyzer = Data_analyzer(df_TNO_2018, basepath)
     
    analyzer.plot_map(species = 'co2_ff', 
                      country_acro = ['NLD', 'DEU'], # --> None gives Europe 
                      emis_cat = ['A'],  
                      high_dpi = 1,
                      save_plot = 1)
    analyzer.plot_bar(species_list = ['co2_ff'], 
                      country_acro = ['NLD'], 
                      high_dpi = 1)
    
    # loop over all species and emis cats and make EU plots:
    progress = 0
    for cat in df_TNO_2018.cats['emis_cat_code']:
        for specie in df_TNO_2018.species:
            progress += 1
            progress_perc = np.round(progress / (len(df_TNO_2018.cats['emis_cat_code']) * len(df_TNO_2018.species)) * 100, 2)
            print('Progess:', progress_perc, '%') # to display some progress statistic

            analyzer.plot_map(species = specie, 
                        country_acro = None, # None: results in plots of europe default countries     
                        emis_cat = [cat], 
                        high_dpi = 1,
                        save_plot = 1)

    # End timer and print execution time:
    end_time = time.time()
    print(f"\n Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()