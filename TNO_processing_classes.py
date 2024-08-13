#! /usr/bin/env python3

# David Mathas - TNO 

# imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('uba')
import matplotlib
matplotlib.colors.LogNorm
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import xarray as xr
import os
from pathlib import Path
import time 
from typing import Tuple

# own:
from tools import make_distance_matrix_vec # make sure tools.py is in same dir 

print("\nLoaded TNO_processing_classes @._.@ \n")

### ### ###

### FOR EMISSION DATA --> DF CLASS ###
# convert to multiple pd.DataFrames () --> code inspiration and begin: Arjo Segers
class EmisDF(object):
    
    def __init__(self, ds: xr.core.dataset.Dataset):
        """
        Convert content of emission dataset into dataframes.

        Content of ds:
            country_id ('country',)
            country_name ('country',)
            emis_cat_code ('emis_cat',)
            emis_cat_name ('emis_cat',)
            source_type_code ('source_type',)
            source_type_name ('source_type',)
            longitude_bounds ('longitude', 'bound')
            latitude_bounds ('latitude', 'bound')
            area ('latitude', 'longitude')
            longitude_source ('source',)
            latitude_source ('source',)
            longitude_index ('source',)
            latitude_index ('source',)
            country_index ('source',)
            emission_category_index ('source',)
            source_type_index ('source',)
            co2_ff ('source',)
            co2_bf ('source',)
            co_ff ('source',)
            co_bf ('source',)
            nox ('source',)
            ch4 ('source',)
            nmvoc ('source',)
        
        Args:
            ds (xarray.core.dataset.Dataset): The NetCDF / xarray dataset containing the emission data. 
        """

        # Initialize empty dataframes:
        self.sources = pd.DataFrame()
        self.countries = pd.DataFrame()
        self.cats = pd.DataFrame() 

        # stypes:
        self.source_types = pd.DataFrame({'source_type_code': np.char.decode(ds.source_type_code, 'utf-8'),
                                          'source_type_name': np.char.decode(ds.source_type_name, 'utf-8')}, index=[1, 2])

        # loop over datasset variables:
        for key in ds.keys():
            if "source" in ds[key].dims:
                self.sources[key] = ds[key].values               
            elif "country" in ds[key].dims:
                self.countries[key] = np.char.decode(ds[key].values, 'utf-8') if ds[key].dtype.kind in {'U', 'S'} else ds[key].values
            elif "emis_cat" in ds[key].dims:
                self.cats[key] = np.char.decode(ds[key].values, 'utf-8') if ds[key].dtype.kind in {'U', 'S'} else ds[key].values

        # Set columns to names instead of indices per variable:
        self.sources['country_code'] = self.countries['country_id'].values[self.sources['country_index'].values-1]
        self.sources['emis_cat_code'] = self.cats['emis_cat_code'].values[self.sources['emission_category_index'].values-1]
        self.sources['source_type_code'] = self.source_types['source_type_code'].values[self.sources['source_type_index'].values-1]
        
        self.sources.drop(['country_index', 'emission_category_index', 'source_type_index'], axis=1, inplace=True) # drop old..

        # known species ...
        self.species = ['co2_ff', 'co2_bf', 'co_ff', 'co_bf', 'nox', 'ch4','nmvoc']
        # storage for info per species:
        self.specinfo = pd.DataFrame()
        # loop over species:
        for specie in self.species:
            # loop over attributes to be copied:
            for aname in ['units','long_units','long_name'] :
                self.specinfo.at[specie,aname] = ds[specie].attrs[aname]


    def extract_subset(self, 
                    country: list = None, 
                    cat=None, 
                    stype=None,
                    ):
    
        """
        Filters the emission data by country, category, and source type.

        Args:
            country (list, optional): List of country identifiers for filtering.
            cat (list, optional): List of emission category codes for filtering.
            stype_index (int or list, optional): Source type index(es) for filtering.

        Returns:
            pd.DataFrame: Filtered emissions data.
        """
        mask = np.ones(len(self.sources), dtype=bool)

        if country is not None:
            country_mask = self.sources['country_code'].isin(country)
            mask &= country_mask
        else: # the countries that are not default have invalid/incomplete data 
            default_countries = ['ALB', 'AUT', 'BEL', 'BGR', 'BIH', 'BLR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'KOS', 'LTU', 'LUX', 'LVA', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR']
            country_mask = self.sources['country_code'].isin(default_countries)
            mask &= country_mask

        if cat:
            cat_mask = self.sources['emis_cat_code'].isin(cat)
            mask &= cat_mask

        if stype:
            stype_mask = self.sources['source_type_code'].isin(stype)
            mask &= stype_mask

        subset_sources = self.sources[mask]
        return subset_sources


### ### ###

### FOR UNCERTAINTY EMISSION DATA --> DF CLASS ###
class Emis_unc_DF(object):
    
    def __init__(self, ds: xr.core.dataset.Dataset):
        """
        Convert content of emission (uncertainty) dataset into dataframes.

        Content of ds:
            country_id: |S3 ('country',)
            country_name: |S41 ('country',)
            
            pollutant_name: |S3 ('pollutant',)

            emis_cat_code: |S13 ('emis_cat',)
            emis_cat_name: |S20 ('emis_cat',)
            corr_length: int32 ('emis_cat',)

            source_type_code: |S1 ('source_type',)
            source_type_name: |S5 ('source_type',)

            emissions: float64 ('country', 'pollutant', 'emis_cat')
            unc_emis: float32 ('country', 'pollutant', 'emis_cat')

            longitude_source: float32 ('source',)
            latitude_source: float32 ('source',)
            longitude_index: int32 ('source',)
            latitude_index: int32 ('source',)
            country_index: int32 ('source',)
            source_type_index: int32 ('source',)  
            emission_category_index: int32 ('source',)

            distribution: float64 ('source', 'pollutant')
            unc_dist: float32 ('source', 'pollutant')

            temporal_profile: float64 ('time', 'pollutant', 'emis_cat')
            low_lim_profile: float64 ('time', 'pollutant', 'emis_cat')
            up_lim_profile: float64 ('time', 'pollutant', 'emis_cat')

        Args:
            ds (xarray.core.dataset.Dataset): The NetCDF / xarray dataset containing the emission uncertainty data. 
        """
        self.ds = ds
        # Initialize empty dataframes:
        self.sources = pd.DataFrame()
        self.countries = pd.DataFrame()
        self.cats = pd.DataFrame()
        self.source_types = pd.DataFrame()

        # loop over dataset variables & decode:
        for key in ds.keys():
            if ds[key].dims == ("source",): # SOURCE
                    self.sources[key] = np.char.decode(ds[key].values, 'utf-8') if ds[key].dtype.kind in {'U', 'S'} else ds[key].values

            elif ds[key].dims == ("country",):# COUNTRY
                self.countries[key] = np.char.decode(ds[key].values, 'utf-8') if ds[key].dtype.kind in {'U', 'S'} else ds[key].values

            elif ds[key].dims == ("emis_cat",): # EMIS_CAT
                self.cats[key] = np.char.decode(ds[key].values, 'utf-8') if ds[key].dtype.kind in {'U', 'S'} else ds[key].values

            elif ds[key].dims == ("source_type",): # POLLUTANT
                    self.source_types[key] = np.char.lower(np.char.decode(ds[key].values, 'utf-8')) if ds[key].dtype.kind in {'U', 'S'} else np.char.lower(ds[key].values)
                    
        # Set columns to names instead of indices per variable:
        self.sources['country_code'] = self.countries['country_id'].values[self.sources['country_index'].values - 1]
        self.sources['emis_cat_code'] = self.cats['emis_cat_code'].values[self.sources['emission_category_index'].values - 1]
        self.sources['source_type_code'] = self.source_types['source_type_code'].values[self.sources['source_type_index'].values - 1]
        
        self.sources.drop(columns=['country_index', 'emission_category_index', 'source_type_index'], inplace=True)


        self.species = np.char.decode(ds.pollutant_name , 'utf-8') # SPECIES
        
        for ispecie, specie in enumerate(self.species): # (UNC) DISTRIBUTION 
            self.sources[f"distr_{specie}"] = ds["distribution"].values[:, ispecie]
            self.sources[f"unc_distr_{specie}"] = ds["unc_dist"].values[:, ispecie]

        # extract regular grid:
        # self.grid = ds[['longitude','latitude','longitude_bounds','latitude_bounds','area']] 
        self.grid = ds[['longitude','latitude']]

    def extract_country_lvl(self,
                            country: list = None,
                            cat = None, 
                            species = None
                            ):
        """
        DOCSTRINGS
        """
        selection_dict = {} # to select
    
        # COUNTRY
        if country:
            country_idx = np.where(self.countries['country_id'] == country)[0][0]
            selection_dict['country'] = country_idx
        
        # EMIS_CAT
        if cat:
            cat_indx = np.where(self.cats['emis_cat_code'] == cat)[0][0]
            selection_dict['emis_cat'] = cat_indx
        
        # POLLUTANT
        if species:
            species_idx = np.where(self.species == species)[0][0]
            selection_dict['pollutant'] = species_idx
        
        # To DataFrame:
        data_array_unc = self.ds.unc_emis.sel(**selection_dict)
        data_array_emis = self.ds.emissions.sel(**selection_dict)

        if data_array_unc.shape:
            df_unc = data_array_unc.to_dataframe(name='uncertainty').reset_index()
            df_emis = data_array_emis.to_dataframe(name='emission').reset_index()
            df = pd.merge(df_unc, df_emis)
        else:
            unc = float(data_array_unc.values)
            emis = float(data_array_emis.values)
            df = pd.DataFrame({'uncertainty': [unc], 'emission': [emis]}, index = [''])

        return df
        

    def extract_subset(self, 
                        country: list = None, 
                        cat=None, 
                        stype=None,
                        specie: list = None,
                        domain: list = None):
    
        """
        Filters the emission data by country, category, and source type.

        Args:
            country (list, optional): List of country identifiers for filtering.
            cat (list, optional): List of emission category codes for filtering.
            stype_index (int or list, optional): Source type index(es) for filtering.
            specie (int or list, optional): Pollutant type for filtering.

        Returns:
            pd.DataFrame: Filtered emissions data.
        """
        mask = np.ones(len(self.sources), dtype=bool)

        if country is not None:
            country_mask = self.sources['country_code'].isin(country)
            mask &= country_mask
        else: # the countries that are not default have invalid/incomplete data 
            default_countries = ['ALB', 'AUT', 'BEL', 'BGR', 'BIH', 'BLR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'KOS', 'LTU', 'LUX', 'LVA', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR']
            country_mask = self.sources['country_code'].isin(default_countries)
            mask &= country_mask

        if cat:
            cat_mask = self.sources['emis_cat_code'].isin(cat)
            mask &= cat_mask

        if stype:
            stype_mask = self.sources['source_type_code'].isin(stype)
            mask &= stype_mask

        if domain:
            assert len(domain) == 4, 'Length domain has to be four (coordinates) --> two lon, two lat.'
            domain_mask = (self.sources['longitude_source'] > domain[0]) & (self.sources['longitude_source'] <= domain[1]) & \
                            (self.sources['latitude_source'] > domain[2]) & (self.sources['latitude_source'] <= domain[3])
            mask &= domain_mask
        
        
        return self.sources[mask]


### ### ###

### DATA ANALYZER for both emission and uncertainty dataframes ### 
class Data_analyzer(object):
    """
    Class to analyze and generate plots.
    """

    def __init__(self, emission_df, basepath):
        self.data = emission_df 
        self.basepath = basepath
        self.min_L_values = pd.read_csv(basepath / 'ProjectData' / 'Internship' / 'David' / 'Code' / 'Minimum_CorrLengths.csv', delimiter=';')

    def plot_map(self, 
                 species: str,
                 emis_cat: list = None, 
                 country_acro: str = None,
                 max_value: float = None,
                 high_dpi: int = None,
                 save_plot: bool = True) -> None:
        """
        Plots a geographical map of emissions for a given species, optionally filtering by country and emission category.

        Args:
            species (str): The species of emissions to plot (e.g., 'co2_ff').
            emis_cat (list, optional): List of categories to include in the plot.
            country_acro (str, optional): Country acronym to focus the plot on; defaults to Europe if None.
            max_value (float, optional): Max value for the heatmap; defaults to a value of None, determined by data thus by default.  
            high_dpi (int, optional): DPI setting for high-resolution output; if None, defaults to standard resolution.
            save_plot (bool): If True, saves the plot to a designated directory; otherwise, displays the plot.
        
        Returns: 
            None
        """
        if save_plot:
            plt.ioff() # to make sure no plots are shown or "pop up" when chosing to merely save plot.

        is_uncertainty = 'unc' in str(self.data.__class__) # --> check if it's a regular emission dataframe or uncertainty..!
        data_type = 'Standard Deviation' if is_uncertainty else 'Emissions'
        print(f'\n Subsetting {data_type} data and plotting specified mapped data...')

        # choose and subset country:
        subset_emissions = self.data.extract_subset(country_acro, cat=emis_cat)

        # error raising:
        if emis_cat is not None and subset_emissions.empty:
            raise ValueError(f"No data available for the specified emission categorie(s): {emis_cat}")

        # choose species and aggregate over grids:
        aggr_emissions = subset_emissions.groupby(['longitude_index', 'latitude_index'])[species].sum() * 100 if is_uncertainty else subset_emissions.groupby(['longitude_index', 'latitude_index'])[species].sum()
        # fill up the (longitude then latitude) gaps:
        min_index_lon = aggr_emissions.index.get_level_values('longitude_index').min()
        max_index_lon = aggr_emissions.index.get_level_values('longitude_index').max()

        full_index_range_lon = np.arange(min_index_lon, max_index_lon + 1)
        min_index_lat = aggr_emissions.index.get_level_values('latitude_index').min()
        max_index_lat = aggr_emissions.index.get_level_values('latitude_index').max()
        full_index_range_lat = np.arange(min_index_lat, max_index_lat + 1)

        # re index:
        new_index = pd.MultiIndex.from_product([full_index_range_lon, full_index_range_lat], names=['longitude_index', 'latitude_index'])
        aggr_emissions = aggr_emissions.reindex(new_index, fill_value=0)
        emission_grid = aggr_emissions.unstack(fill_value=0)

        # xr with corrected longs/lats:
        min_longitude = np.min(subset_emissions['longitude_source'])
        min_latitude = np.min(subset_emissions['latitude_source'])

        # lon/lat full range for xr:
        longitude = min_longitude + 0.1 * np.arange(full_index_range_lon.size)
        latitude = min_latitude + 0.05 * np.arange(full_index_range_lat.size)
        emission_grid_xr = xr.DataArray(
            emission_grid.values.T,
            dims=['latitude', 'longitude'],
            coords={
                'longitude': longitude,
                'latitude': latitude
            },
            attrs={
                'description': 'Aggregated Emissions',
                'units': 'kg/year'
            }
        )

        #plotting:
        dpi_setting = 300 if high_dpi else None
        fig, ax = plt.subplots(figsize=(12, 7.2), dpi=dpi_setting, subplot_kw={'projection': ccrs.PlateCarree()})
        colors = ["white", "lightgray", "#0000FF", "gray", "lightgreen", "yellow", "orange", "#FFCCCC", "#8B0000"]  
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", colors, N=256)
        cmap.set_under('white')
        max_value_final = max_value if max_value else np.max(aggr_emissions) / (2 if is_uncertainty else 4)
        cbar_plot = emission_grid_xr.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmax=max_value_final, add_colorbar=False)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS) 
        # cbar_plot = emission_grid_xr.plot(cmap = 'plasma', vmax = max_value, ax=ax, add_colorbar = False)
        # cbar_plot = emission_grid_xr.plot(norm=LogNorm(vmin=1e0), cmap='plasma', ax=ax, add_colorbar = False) 
        ax.grid(False)

        emis_cat_name = self.data.cats[self.data.cats['emis_cat_code'] == emis_cat[0]].iloc[0]['emis_cat_name']

        def clean_country_acro(country_acro):
            if country_acro is not None:
                #Clearn up the string for proper file naming:
                return str(country_acro).replace('[', '').replace(']', '').replace("'", '').replace(',', '_').replace(' ', '')
            return "europe"

        ax.set_title(f'Gridded {species} {data_type.lower()} over {"Europe" if country_acro is None else clean_country_acro(country_acro)} - {"All" if emis_cat is None else emis_cat_name}', fontsize = 14)  
        # ax.set_xlabel('Longitude (degrees)', fontsize = 8.5)  
        # ax.set_ylabel('Latitude (degrees)', fontsize = 8.5)

        # ax.set_xticks(np.round(np.linspace(longitude.min(), longitude.max(), num=5), 2)) 
        # ax.set_yticks(np.round(np.linspace(latitude.min(), latitude.max(), num=5), 2))  
        # cbar = plt.colorbar(cbar_plot, ax=ax, fraction=0.0222, pad=0.04)
        ax.tick_params(axis='both', which='major', labelsize=7)

        cbar = fig.colorbar(cbar_plot, orientation='horizontal', pad=0.025, aspect=40, fraction=0.0355) #, fraction=0.055)
        cbar.ax.tick_params(labelsize=9.5)
        # cbar.set_label(f'{data_type} (kg/year)', fontsize )
        cbar.set_label( f'{data_type} {"[%]" if is_uncertainty else "[kg/year]"}', fontsize = 12)

        plt.tight_layout()

        if save_plot:
            save_path = Path(os.path.join(str(self.basepath), 'ProjectData/Internship/David/Plots/Em_plots'))
            cleaned_country_acro = clean_country_acro(country_acro)
            filename = f'{data_type.lower().replace(" ", "_")}_map_plot_{cleaned_country_acro}_{emis_cat_name.replace(" ", "_")}_{species}.png'
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path)  # , quality = 100
            print(f"\n Plot saved to {full_path}")
        else:
            plt.show()  # if running locally...
        plt.close(fig)

    
    def plot_bar(self, 
                 species_list: list,
                 country_acro: str = None,
                 high_dpi: int = None,
                 save_plot: bool = True) -> None:
        """
        Generates a horizontal bar chart of total emissions by category for specified species, optionally filtered by country.
        It's assumed here that either emissions data is used or data containing uncertainties (in the latter case, ensure the "unc" are contained somewhere in the dataframe's name).

        Args:
            country_acro (str, optional): Country acronym to focus the plot on; defaults to plotting Europe if None.
            species_list (list, optional): List of species for which emissions will be plotted. Default is all common emissions species.
            high_dpi (int, optional): DPI setting for high-resolution output; if None, defaults to standard resolution.
            save_plot (bool, optional): If True, saves the plot to a designated directory; otherwise, displays the plot.

        Reurns: 
            None
        """
        if save_plot:
            plt.ioff() # to make sure no plots are shown or "pop up" when chosing to merely save plot.
        is_uncertainty = 'unc' in str(self.data.__class__) # --> again, check if it's a regular emission dataframe or uncertainty..!
        data_type = 'Standard Deviations' if is_uncertainty else 'Emissions'
        print(f'\n Subsetting {data_type} data and plotting specified bar graph...')

        # choose and subset country:
        subset = self.data.extract_subset(country_acro)

        # bar charts plot:
        subset_cat_sum = subset.groupby('emis_cat_code').sum()

        dpi_setting = 300 if high_dpi else None
        fig, ax = plt.subplots(figsize=(12, 8), dpi = dpi_setting)
        # some nice colors depending on the type of data:
        colors = plt.cm.Oranges(np.linspace(0.3, 1, len(species_list))) if is_uncertainty else plt.cm.Greens(np.linspace(0.3, 1, len(species_list)))
        for i, spec in enumerate(species_list):
            ax.barh(subset_cat_sum.index, subset_cat_sum[spec], color=colors[i], label=spec)

        ax.set_xlabel(f'Annual Total {data_type} {"[%]" if is_uncertainty else "[kg/year]"}') 
        ax.set_ylabel('Emission Category')
        # ax.set_xscale('log')  
        ax.set_title(f'Total {data_type} by Emission Category for {"Europe" if country_acro is None else country_acro}')

        category_labels = [f"{self.data.cats[self.data.cats['emis_cat_code'] == cat].iloc[0]['emis_cat_name']} ({cat})" for cat in subset_cat_sum.index]
        ax.set_yticks(np.arange(len(category_labels)))
        ax.set_yticklabels(category_labels, fontsize=10)

        ax.set_yticks(np.arange(len(category_labels))) # due to 1-based indexing in nc file
        ax.set_yticklabels(category_labels, fontsize=10)

        # Add gridlines:
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', color='grey', linestyle=':', linewidth=0.25)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(AutoMinorLocator(.5)) #no minor ticks on y-axis

        ax.legend(bbox_to_anchor=(.45, -0.2))

        plt.tight_layout()

        if save_plot:
            save_path = Path(os.path.join(str(self.basepath), 'ProjectData/Internship/David/Plots' ))  
            filename = f'{data_type.lower().replace(" ", "_")}_bar_plot_{country_acro}_{species_list}.png'  
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path)
            print(f"\n Plot saved to {full_path}")
        else:
            plt.show() # if running locally...
        plt.close(fig)
    

    def sigmas_helper_zero(self, country: str, specie: str, cat: str) -> Tuple[float, float]:
        """Extracts the original uncertainty and calculates relative uncertainty for comparisons ONLY FOR L=0.
        
        Args:
            country (str): Country acronym to focus caclulation on.
            specie (str): Species for which caclulation will be performed.
            cat (str): Emission category for which caclulation will be performed.
        
        Returns:
            Tuple of sigma_eta_reported (float) and sigma_eta_relative (float).
        """
        sigma_eta_reported = self.data.extract_country_lvl(country=country, species=specie, cat=cat)['uncertainty'].values[0]

        unc_distr_matrix = self.data.extract_subset(country=[country], cat=[cat]).copy()
        abs_unc_col = unc_distr_matrix[f'unc_distr_{specie}'] * unc_distr_matrix[f'distr_{specie}']
        sigma_eta = np.sum(abs_unc_col**2)**.5
        total_emissions = np.sum(unc_distr_matrix[f'distr_{specie}'])
        sigma_eta_relative = sigma_eta / total_emissions if total_emissions != 0 else 0
        
        return sigma_eta_reported, sigma_eta_relative    


    def calculate_aggregated_sigma(self, L: float, country: str, specie: str, cat: str) -> float:
        """DOCSTRINGS
        """
        sigma_eta_reported = self.data.extract_country_lvl(country=country, species=specie, cat=cat)['uncertainty'].values[0]
        unc_distr_matrix = self.data.extract_subset(country=[country], cat=[cat]).copy()

        abs_sigma = unc_distr_matrix[f'unc_distr_{specie}'] * unc_distr_matrix[f'distr_{specie}']
        if L == 0:
            sigma_eta_aggr = np.sum(abs_sigma**2)**.5
        else:    
            coords = unc_distr_matrix[['longitude_source', 'latitude_source']].values
            distance_matrix = make_distance_matrix_vec(coords) ## SEPERATE FUNCTION!! 
            C = np.exp(-0.5 * (distance_matrix / L)**2)
            S = np.diag(abs_sigma) # matrix with std devs...
            B = S @ C @ S.T # full covariance matrix B
            a = np.ones(B.shape[0])
            sigma_eta_aggr = (a @ B @ a.T)**.5  # aggregated absolute sigma eta 
        total_emissions = np.sum(unc_distr_matrix[f'distr_{specie}'])
        sigma_relative_aggregated = float(sigma_eta_aggr / total_emissions)if total_emissions != 0 else 0.

        return sigma_eta_reported, sigma_relative_aggregated


    def plot_uncertainty_comparison(self, emis_cat: str, high_dpi: bool = True, save_plot: bool = True) -> None:
        """DOCSTRINGS
        """
        if save_plot:
            plt.ioff() # to make sure no plots are shown or "pop up" when chosing to merely save plot.
        emis_cat_name = self.data.cats[self.data.cats['emis_cat_code'] == emis_cat].iloc[0]['emis_cat_name']
        print(f'Plotting uncertainty comparison for {emis_cat_name}...')

        dpi_setting = 300 if high_dpi else None
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=dpi_setting)
        fig.subplots_adjust(left=0.55)

        country_sel = ['ALB', 'AUT', 'BEL', 'BGR', 'BIH', 'BLR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'KOS', 'LTU', 'LUX', 'LVA', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR']
        mask = self.data.countries['country_id'].isin(country_sel)
        # mask = self.data.countries['country_id'].isin(['DZA', 'EGY', 'LBY', 'MAR', 'TUN', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN', 'SAU', 'SYR', 'PSE', 'ARM', 'AZE', 'GEO']) # exclude these countries in this analysis since coming from global em distr....
        countries = self.data.countries['country_id'][mask].tolist()#[:10]
        species_list = self.data.species.tolist()

        for ax, specie in zip(axes.flatten(), species_list):
            country_labels = [] # only take the countries where aggregated sigma and aggregated sigma are not zero (e.g. roadtransport in the sea "coutries")
            L_min_value = self.min_L_values[self.min_L_values['AggSectorCode'] == emis_cat]['MinCorrLength'].values[0]

            for country in countries:
                sigma_eta_reported, sigma_eta_calc = self.sigmas_helper_zero(country, specie, emis_cat)
                # sigma_eta_reported, sigma_eta_calc = self.calculate_aggregated_sigma(L_min_value, country, specie, emis_cat)

                # if (sigma_eta_orig > 0 and sigma_eta_calc > 0) or \
                #    (sigma_eta_orig > 0 and sigma_eta_calc == 0) or \
                #    (sigma_eta_orig == 0 and sigma_eta_calc > 0): # leave the cases where both sigmas are zero for further research...           
                    
                country_labels.append(country)

                    # Color for sigma_eta_original:
                color_orig = 'blue' if sigma_eta_reported == 0 else 'black'
                    # Color for sigma_eta_calc:
                if sigma_eta_calc == 0:
                    color_calc = 'blue'
                else:
                    color_calc = 'green' if sigma_eta_reported > sigma_eta_calc else 'red'
                    
                    marker_size = 26
                    ax.scatter(country_labels.index(country), sigma_eta_reported*100, color=color_orig, marker='*', s=marker_size + 4) # *100 so they are valid percentages 
                    ax.scatter(country_labels.index(country), sigma_eta_calc*100, color=color_calc, marker='s', s=marker_size)

            emis_cat_name = self.data.cats[self.data.cats['emis_cat_code'] == emis_cat].iloc[0]['emis_cat_name']
            # ax.set_title(f'Relative Standard Deviation Comparison for {specie} - Category: {emis_cat_name} ({emis_cat}) with minimal corr. L value ({L_min_value})', fontsize=18)
            ax.set_title(f'Relative Standard Deviation Comparison for {specie} {emis_cat_name} ({emis_cat}) with L = 0', fontsize=17)
            
            ax.set_xticks(range(len(country_labels)))
            ax.set_xticklabels(country_labels, rotation=45, fontsize=8)
            ax.grid(True, linestyle=':')
            ax.tick_params(axis='y', labelsize=9)

            # legend_handles = [
        #     mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=5.5, label=f'Aggregated $\sigma_{{\eta, L=0}}$ ({specie}), above reported'),
        #     mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=5.5, label=f'Aggregated $\sigma_{{\eta, L=0}}$ ({specie}), within reported range'),
        #     mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=5.5, label=f'Aggregated $\sigma_{{\eta, L=0}}$ ({specie}), value zero'),
        #     mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=5.5, label=f'Reported $\sigma_{{\eta}}$ ({specie})'),
        #     mlines.Line2D([], [], color='blue', marker='*', linestyle='None', markersize=5.5, label=f'Reported $\sigma_{{\eta}}$ ({specie}), value zero'
        #     )
        # ]
        # # ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
        # ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1., 1.155), fontsize=6.5, ncol=3)

        legend_handles = [
                mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=5.5, label=f'Aggregated $\sigma_{{\eta}}$, above reported'),
                mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=5.5, label=f'Aggregated $\sigma_{{\eta}}$, below reported'),
                mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=5.5, label=f'Aggregated $\sigma_{{\eta}}$, value zero and below reported'),
                mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=5.5, label=f'Reported $\sigma_{{\eta}}$'),
                mlines.Line2D([], [], color='blue', marker='*', linestyle='None', markersize=5.5, label=f'Reported $\sigma_{{\eta}}$, value zero'
                )
            ]
            # ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
        # fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(.91, 0.0), fontsize=10.5, ncol=5)
        # fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.4, 0.1), fontsize=10, ncol=5)
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.435, 0.035), fontsize=10, ncol=5)

        fig.text(0.003, 0.5, r'$\mathbf{Relative\ Standard\ Deviation\ [\%]}$', va='center', rotation='vertical', fontsize=16)
        # plt.tight_layout(rect=[0, 0.85, 1, 1])
        plt.tight_layout(rect=[0.01, 0.015, 1, 1]) 

        if save_plot:
            save_path = Path(os.path.join(str(self.basepath), 'ProjectData/Internship/David/Plots/Unc_comp_plots' ))  
            filename = f'uncertainties_{emis_cat_name}_comparison.png'  
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path)
            print(f"\n Plot saved to {full_path}")
        else:
            plt.show() # if running locally...
        plt.close(fig)

### ### ###