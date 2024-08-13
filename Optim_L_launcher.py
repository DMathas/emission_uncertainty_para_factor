#! /usr/bin/env python3

### LAUNCHER for running L optimization procedure parallel on cluster ###

import os
import subprocess

# Check with countries still need to be optimized and select those:
directory_path = '/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/Internship/David/Code/Updated_corr_lengths_gamma'
countries = ['ALB', 'AUT', 'BEL', 'BGR', 'BIH', 'BLR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IRL', 'ISL', \
'ITA', 'KOS', 'LTU', 'LUX', 'LVA', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR']
print('Number of countries:', len(countries))
print('Number of optimized countries:', len(os.listdir(directory_path))//2)

countries_without_files = []

# Check for each country with loop:
for country in countries:
    pattern = f"Uncertainty_Matrix_{country}.csv" or f"Updated_corr_lengths_gamma_{country}.csv"
    
    if pattern not in os.listdir(directory_path):
        countries_without_files.append(country)

if countries_without_files:
    print(f"Countries ({len(countries_without_files)}) without CSV files:", ', '.join(countries_without_files), '\n')
else:
    print("All countries have corresponding CSV files in the directory.", '\n')
## ! ##
countries = countries_without_files
## ! ##

countries = ['SWE', 'DEU', 'NOR', 'ESP']
# countries = ['RUS']
for country in countries:
    os.environ['COUNTRY'] = country
    cpus = "2" if country == "RUS" else "1" 
    mem = "72GB" if country == "RUS" else "24GB"  
    output_file = f"/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/Internship/David/Code/Final_code/optim-L-{country}-%j.out"  

    subprocess.check_call(["sbatch",
                           "--job-name=optim-L-{}".format(country),
                           "--cpus-per-task={}".format(cpus),
                           "--mem={}".format(mem),
                           "--output={}".format(output_file),
                           "--wrap=./TNO_optim_L_gamma.py",
                           "--time=7-0:00:00",
                           "--partition=longq"])