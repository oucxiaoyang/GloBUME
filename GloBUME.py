
"""
October 2020

This code retrieves the calculation of building material use and embodied GHG emissions in 26 global regions. For the original code & latest updates, see: https://github.com/oucxiaoyang/BUME
The dynamic stock model is based on the ODYM model developed by Stefan Pauliuk, Uni Freiburg, Germany. For the original code & latest updates, see: https://github.com/IndEcol/ODYM
The dynamic material model is based on the BUMA model developed by Deetman Sebastiaan, Leiden University, the Netherlands. For the original code & latest updates, see: https://github.com/SPDeetman/BUMA

*NOTE: Insert location of GloBME-main folder in 'dir_path' (line 27) before running the code. 

Software version: Python 3.7

@author: Sebastiaan Deetman; deetman@cml.leidenuniv.nl
         Xiaoyang Zhong; x.zhong@cml.leidenuniv.nl
contributions from: Glenn Aguilar
                    Sylvia Marinova
"""

#%% GENERAL SETTING & STATEMENTS
import pandas as pd
import numpy as np
import os
import ctypes     
import math

# set current directory
dir_path = ""
os.chdir(dir_path)   

# Set general constants
regions = 26        #26 IMAGE regions
res_building_types = 4  #4 residential building types: detached, semi-detached, appartments & high-rise 
area = 2            #2 areas: rural & urban
materials = 7       #7 materials: Steel, brick, Concrete, Wood, Copper, Aluminium, Glass
inflation = 1.2423  # gdp/cap inflation correction between 2005 (IMAGE data) & 2016 (commercial calibration) according to https://www.bls.gov/data/inflation_calculator.htm

# Set Flags for sensitivity analysis
flag_alpha = 0      # switch for the sensitivity analysis on alpha, if 1 the maximum alpha is 10% above the maximum found in the data
flag_ExpDec = 0     # switch to choose between Gompertz and Exponential Decay function for commercial floorspace demand (0 = Gompertz, 1 = Expdec)
flag_Normal = 0     # switch to choose between Weibull and Normal lifetime distributions (0 = Weibull, 1 = Normal)
flag_Mean   = 0     # switch to choose between material intensity settings (0 = regular regional, 1 = mean, 2 = high, 3 = low, 4 = median)

#%%Load files & arrange tables ----------------------------------------------------

if flag_Mean == 0:
    file_addition = ''
elif flag_Mean == 1:
    file_addition = '_mean'
elif flag_Mean ==2:
    file_addition = '_high'
elif flag_Mean ==3:
    file_addition = '_low'
else:
    file_addition = '_median'

# Load Population, Floor area, and Service value added (SVA) Database csv-files
pop = pd.read_csv('files_population/pop.csv', index_col = [0])                                   # Pop; unit: million of people; meaning: global population (over time, by region)             
rurpop = pd.read_csv('files_population/rurpop.csv', index_col = [0])                             # rurpop; unit: %; meaning: the share of people living in rural areas (over time, by region)
housing_type = pd.read_csv('files_population\Housing_type.csv')               # Housing_type; unit: %; meaning: the share of the NUMBER OF PEOPLE living in a particular building type (by region & by area) 
floorspace = pd.read_csv('files_floor_area/res_Floorspace.csv')                                  # Floorspace; unit: m2/capita; meaning: the average m2 per capita (over time, by region & area)
floorspace = floorspace[floorspace.Region != regions + 1]                                # Remove empty region 27
avg_m2_cap = pd.read_csv('files_floor_area\Average_m2_per_cap.csv')           # Avg_m2_cap; unit: m2/capita; meaning: average square meters per person (by region & area (rural/urban) & building type) 
sva_pc_2005 = pd.read_csv('files_GDP/sva_pc.csv', index_col = [0])
sva_pc = sva_pc_2005 * inflation                                                            # we use the inflation corrected SVA to adjust for the fact that IMAGE provides gdp/cap in 2005 US$

# load material density data csv-files
building_materials_steel = pd.read_csv('files_material_density\Building_materials_steel' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
building_materials_concrete = pd.read_csv('files_material_density\Building_materials_concrete' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
building_materials_wood = pd.read_csv('files_material_density\Building_materials_wood' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
building_materials_copper = pd.read_csv('files_material_density\Building_materials_copper' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
building_materials_aluminium = pd.read_csv('files_material_density\Building_materials_aluminium' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
building_materials_glass = pd.read_csv('files_material_density\Building_materials_glass' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)

materials_commercial_steel = pd.read_csv('files_material_density\materials_commercial_steel' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type) 
materials_commercial_concrete = pd.read_csv('files_material_density\materials_commercial_concrete' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type)
materials_commercial_wood = pd.read_csv('files_material_density\materials_commercial_wood' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type)
materials_commercial_copper = pd.read_csv('files_material_density\materials_commercial_copper' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type)
materials_commercial_aluminium = pd.read_csv('files_material_density\materials_commercial_aluminium' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type)
materials_commercial_glass = pd.read_csv('files_material_density\materials_commercial_glass' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type)

material_brick_rural = pd.read_csv('files_material_density//Building_materials_brick_rural.csv')
material_brick_urban = pd.read_csv('files_material_density//Building_materials_brick_urban.csv')
materials_commercial_brick = pd.read_csv('files_material_density//materials_commercial_brick.csv')

# Load fitted regression parameters for comercial floor area estimate
if flag_alpha == 0:
    gompertz = pd.read_csv('files_floor_area//files_commercial/Gompertz_parameters.csv', index_col = [0])
else:
    gompertz = pd.read_csv('files_floor_area//files_commercial/Gompertz_parameters_alpha.csv', index_col = [0])

# Ensure full time series  for pop & rurpop (interpolation, some years are missing)
rurpop2 = rurpop.reindex(list(range(1970,2061,1))).interpolate()
pop2 = pop.reindex(list(range(1970,2061,1))).interpolate()

# Remove 1st year, to ensure same Table size as floorspace data (from 1971)
pop2 = pop2.iloc[1:]
rurpop2 = rurpop2.iloc[1:]

#pre-calculate urban population
urbpop = 1 - rurpop2                                                           # urban population is 1 - the fraction of people living in rural areas (rurpop)
        
# Restructure the tables to regions as columns; for floorspace
floorspace_rur = floorspace.pivot(index="t", columns="Region", values="Rural")
floorspace_urb = floorspace.pivot(index="t", columns="Region", values="Urban")

# Restructuring for square meters (m2/cap)
avg_m2_cap_urb = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Urban'].drop('Area', 1).T  # Remove area column & Transpose
avg_m2_cap_urb.columns = list(map(int,avg_m2_cap_urb.iloc[0]))                      # name columns according to the row containing the region-labels
avg_m2_cap_urb2 = avg_m2_cap_urb.drop(['Region'])                                 # Remove idle row 

avg_m2_cap_rur = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Rural'].drop('Area', 1).T  # Remove area column & Transpose
avg_m2_cap_rur.columns = list(map(int,avg_m2_cap_rur.iloc[0]))                      # name columns according to the row containing the region-labels
avg_m2_cap_rur2 = avg_m2_cap_rur.drop(['Region'])                                 # Remove idle row 

# Restructuring for the Housing types (% of population living in them)
housing_type_urb = housing_type.loc[housing_type['Area'] == 'Urban'].drop('Area', 1).T  # Remove area column & Transpose
housing_type_urb.columns = list(map(int,housing_type_urb.iloc[0]))                      # name columns according to the row containing the region-labels
housing_type_urb2 = housing_type_urb.drop(['Region'])                                 # Remove idle row 

housing_type_rur = housing_type.loc[housing_type['Area'] == 'Rural'].drop('Area', 1).T  # Remove area column & Transpose
housing_type_rur.columns = list(map(int,housing_type_rur.iloc[0]))                      # name columns according to the row containing the region-labels
housing_type_rur2 = housing_type_rur.drop(['Region'])                                 # Remove idle row 

#%% COMMERCIAL building space demand (stock) calculated from Gomperz curve (fitted, using separate regression model)

# Select gompertz curve paramaters for the total commercial m2 demand (stock)
alpha = gompertz['All']['a'] if flag_ExpDec == 0 else 25.601
beta =  gompertz['All']['b'] if flag_ExpDec == 0 else 28.431
gamma = gompertz['All']['c'] if flag_ExpDec == 0 else 0.0415

# find the total commercial m2 stock (in Millions of m2)
commercial_m2_cap = pd.DataFrame(index=range(1971,2061), columns=range(1,27))
for year in range(1971,2061):
    for region in range(1,27):
        if flag_ExpDec == 0:
            commercial_m2_cap[region][year] = alpha * math.exp(-beta * math.exp((-gamma/1000) * sva_pc[str(region)][year]))
        else:
            commercial_m2_cap[region][year] = max(0.542, alpha - beta * math.exp((-gamma/1000) * sva_pc[str(region)][year]))

# Subdivide the total across Offices, Retail+, Govt+ & Hotels+
commercial_m2_cap_office = pd.DataFrame(index=range(1971,2061), columns=range(1,27))    # Offices
commercial_m2_cap_retail = pd.DataFrame(index=range(1971,2061), columns=range(1,27))    # Retail & Warehouses
commercial_m2_cap_hotels = pd.DataFrame(index=range(1971,2061), columns=range(1,27))    # Hotels & Restaurants
commercial_m2_cap_govern = pd.DataFrame(index=range(1971,2061), columns=range(1,27))    # Hospitals, Education, Government & Transportation

minimum_com_office = 25
minimum_com_retail = 25
minimum_com_hotels = 25
minimum_com_govern = 25

for year in range(1971,2061):
    for region in range(1,27):
        
        # get the square meter per capita floorspace for 4 commercial applications
        office = gompertz['Office']['a'] * math.exp(-gompertz['Office']['b'] * math.exp((-gompertz['Office']['c']/1000) * sva_pc[str(region)][year]))
        retail = gompertz['Retail+']['a'] * math.exp(-gompertz['Retail+']['b'] * math.exp((-gompertz['Retail+']['c']/1000) * sva_pc[str(region)][year]))
        hotels = gompertz['Hotels+']['a'] * math.exp(-gompertz['Hotels+']['b'] * math.exp((-gompertz['Hotels+']['c']/1000) * sva_pc[str(region)][year]))
        govern = gompertz['Govt+']['a'] * math.exp(-gompertz['Govt+']['b'] * math.exp((-gompertz['Govt+']['c']/1000) * sva_pc[str(region)][year]))

        #calculate minimum values for later use in historic tail(Region 20: China @ 134 $/cap SVA)
        minimum_com_office = office if office < minimum_com_office else minimum_com_office      
        minimum_com_retail = retail if retail < minimum_com_retail else minimum_com_retail
        minimum_com_hotels = hotels if hotels < minimum_com_hotels else minimum_com_hotels
        minimum_com_govern = govern if govern < minimum_com_govern else minimum_com_govern
        
        # Then use the ratio's to subdivide the total commercial floorspace into 4 categories      
        commercial_sum = office + retail + hotels + govern
        
        commercial_m2_cap_office[region][year] = commercial_m2_cap[region][year] * (office/commercial_sum)
        commercial_m2_cap_retail[region][year] = commercial_m2_cap[region][year] * (retail/commercial_sum)
        commercial_m2_cap_hotels[region][year] = commercial_m2_cap[region][year] * (hotels/commercial_sum)
        commercial_m2_cap_govern[region][year] = commercial_m2_cap[region][year] * (govern/commercial_sum)

#%% Add historic tail (1720-1970) + 100 yr initial --------------------------------------------

# load historic population development
hist_pop = pd.read_csv('files_initial_stock\hist_pop.csv', index_col = [0])  # initial population as a percentage of the 1970 population; unit: %; according to the Maddison Project Database (MPD) 2018 (Groningen University)

# Determine the historical average global trend in floorspace/cap  & the regional rural population share based on the last 10 years of IMAGE data
floorspace_urb_trend_by_region = [0 for j in range(0,26)]
floorspace_rur_trend_by_region = [0 for j in range(0,26)]
rurpop_trend_by_region = [0 for j in range(0,26)]
commercial_m2_cap_office_trend = [0 for j in range(0,26)]
commercial_m2_cap_retail_trend = [0 for j in range(0,26)]
commercial_m2_cap_hotels_trend = [0 for j in range(0,26)]
commercial_m2_cap_govern_trend = [0 for j in range(0,26)]

# For the RESIDENTIAL & COMMERCIAL floorspace: Derive the annual trend (in m2/cap) over the initial 10 years of IMAGE data
for region in range(1,27):
    floorspace_urb_trend_by_year = [0 for i in range(0,10)]
    floorspace_rur_trend_by_year = [0 for i in range(0,10)]
    commercial_m2_cap_office_trend_by_year = [0 for j in range(0,10)]    
    commercial_m2_cap_retail_trend_by_year = [0 for i in range(0,10)]   
    commercial_m2_cap_hotels_trend_by_year = [0 for j in range(0,10)]
    commercial_m2_cap_govern_trend_by_year = [0 for i in range(0,10)]
    
    # Get the growth by year (for the first 10 years)
    for year in range(1970,1980):
        floorspace_urb_trend_by_year[year-1970] = floorspace_urb[region][year+1]/floorspace_urb[region][year+2]
        floorspace_rur_trend_by_year[year-1970] = floorspace_rur[region][year+1]/floorspace_rur[region][year+2]
        commercial_m2_cap_office_trend_by_year[year-1970] = commercial_m2_cap_office[region][year+1]/commercial_m2_cap_office[region][year+2]
        commercial_m2_cap_retail_trend_by_year[year-1970] = commercial_m2_cap_retail[region][year+1]/commercial_m2_cap_retail[region][year+2] 
        commercial_m2_cap_hotels_trend_by_year[year-1970] = commercial_m2_cap_hotels[region][year+1]/commercial_m2_cap_hotels[region][year+2]
        commercial_m2_cap_govern_trend_by_year[year-1970] = commercial_m2_cap_govern[region][year+1]/commercial_m2_cap_govern[region][year+2]
        
    rurpop_trend_by_region[region-1] = ((1-(rurpop[str(region)][1980]/rurpop[str(region)][1970]))/10)*100
    floorspace_urb_trend_by_region[region-1] = sum(floorspace_urb_trend_by_year)/10
    floorspace_rur_trend_by_region[region-1] = sum(floorspace_rur_trend_by_year)/10
    commercial_m2_cap_office_trend[region-1] = sum(commercial_m2_cap_office_trend_by_year)/10
    commercial_m2_cap_retail_trend[region-1] = sum(commercial_m2_cap_retail_trend_by_year)/10
    commercial_m2_cap_hotels_trend[region-1] = sum(commercial_m2_cap_hotels_trend_by_year)/10
    commercial_m2_cap_govern_trend[region-1] = sum(commercial_m2_cap_govern_trend_by_year)/10

# Average global annual decline in floorspace/cap in %, rural: 1%; urban 1.2%;  commercial: 1.26-2.18% /yr   
floorspace_urb_trend_global = (1-(sum(floorspace_urb_trend_by_region)/26))*100              # in % decrease per annum
floorspace_rur_trend_global = (1-(sum(floorspace_rur_trend_by_region)/26))*100              # in % decrease per annum
commercial_m2_cap_office_trend_global = (1-(sum(commercial_m2_cap_office_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_retail_trend_global = (1-(sum(commercial_m2_cap_retail_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_hotels_trend_global = (1-(sum(commercial_m2_cap_hotels_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_govern_trend_global = (1-(sum(commercial_m2_cap_govern_trend)/26))*100    # in % decrease per annum

# define historic floorspace (1820-1970) in m2/cap
floorspace_urb_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=floorspace_urb.columns)
floorspace_rur_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=floorspace_rur.columns)
rurpop_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=rurpop.columns)
pop_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=pop2.columns)
commercial_m2_cap_office_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_office.columns)
commercial_m2_cap_retail_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_retail.columns)
commercial_m2_cap_hotels_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_hotels.columns)
commercial_m2_cap_govern_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_govern.columns)

# Find minumum or maximum values in the original IMAGE data (Just for residential, commercial minimum values have been calculated above)
minimum_urb_fs = floorspace_urb.values.min()    # Region 20: China
minimum_rur_fs = floorspace_rur.values.min()    # Region 20: China
maximum_rurpop = rurpop.values.max()            # Region 9 : Eastern Africa

# Calculate the actual values used between 1820 & 1970, given the trends & the min/max values
for region in range(1,regions+1):
    for year in range(1820,1971):
        # MAX of 1) the MINimum value & 2) the calculated value
        floorspace_urb_1820_1970[region][year] = max(minimum_urb_fs, floorspace_urb[region][1971] * ((100-floorspace_urb_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        floorspace_rur_1820_1970[region][year] = max(minimum_rur_fs, floorspace_rur[region][1971] * ((100-floorspace_rur_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_office_1820_1970[region][year] = max(minimum_com_office, commercial_m2_cap_office[region][1971] * ((100-commercial_m2_cap_office_trend_global)/100)**(1971-year))  # single global value for average annual Decrease  
        commercial_m2_cap_retail_1820_1970[region][year] = max(minimum_com_retail, commercial_m2_cap_retail[region][1971] * ((100-commercial_m2_cap_retail_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_hotels_1820_1970[region][year] = max(minimum_com_hotels, commercial_m2_cap_hotels[region][1971] * ((100-commercial_m2_cap_hotels_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_govern_1820_1970[region][year] = max(minimum_com_govern, commercial_m2_cap_govern[region][1971] * ((100-commercial_m2_cap_govern_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        # MIN of 1) the MAXimum value & 2) the calculated value        
        rurpop_1820_1970[str(region)][year] = min(maximum_rurpop, rurpop[str(region)][1970] * ((100+rurpop_trend_by_region[region-1])/100)**(1970-year))  # average annual INcrease by region
        # just add the tail to the population (no min/max & trend is pre-calculated in hist_pop)        
        pop_1820_1970[str(region)][year] = hist_pop[str(region)][year] * pop[str(region)][1970]

urbpop_1820_1970 = 1 - rurpop_1820_1970

# To avoid full model setup in 1820 (all required stock gets built in yr 1) we assume another tail that linearly increases to the 1820 value over a 100 year time period, so 1720 = 0
floorspace_urb_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=floorspace_urb.columns)
floorspace_rur_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=floorspace_rur.columns)
rurpop_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=rurpop.columns)
urbpop_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=urbpop.columns)
pop_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=pop2.columns)
commercial_m2_cap_office_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_office.columns)
commercial_m2_cap_retail_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_retail.columns)
commercial_m2_cap_hotels_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_hotels.columns)
commercial_m2_cap_govern_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_govern.columns)

for region in range(1,27):
    for time in range(1721,1820):
        #                                                        MAX(0,...) Because of floating point deviations, leading to negative stock in some cases
        floorspace_urb_1721_1820[int(region)][time]            = max(0.0, floorspace_urb_1820_1970[int(region)][1820] - (floorspace_urb_1820_1970[int(region)][1820]/100)*(1820-time))
        floorspace_rur_1721_1820[int(region)][time]            = max(0.0, floorspace_rur_1820_1970[int(region)][1820] - (floorspace_rur_1820_1970[int(region)][1820]/100)*(1820-time))
        rurpop_1721_1820[str(region)][time]                    = max(0.0, rurpop_1820_1970[str(region)][1820] - (rurpop_1820_1970[str(region)][1820]/100)*(1820-time))
        urbpop_1721_1820[str(region)][time]                    = max(0.0, urbpop_1820_1970[str(region)][1820] - (urbpop_1820_1970[str(region)][1820]/100)*(1820-time))
        pop_1721_1820[str(region)][time]                       = max(0.0, pop_1820_1970[str(region)][1820] - (pop_1820_1970[str(region)][1820]/100)*(1820-time))
        commercial_m2_cap_office_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_office_1820_1970[region][1820] - (commercial_m2_cap_office_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_retail_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_retail_1820_1970[region][1820] - (commercial_m2_cap_retail_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_hotels_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_hotels_1820_1970[region][1820] - (commercial_m2_cap_hotels_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_govern_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_govern_1820_1970[region][1820] - (commercial_m2_cap_govern_1820_1970[region][1820]/100)*(1820-time))

# combine historic with IMAGE data here
rurpop_tail                     = rurpop_1820_1970.append(rurpop2, ignore_index=False)
urbpop_tail                     = urbpop_1820_1970.append(urbpop, ignore_index=False)
pop_tail                        = pop_1820_1970.append(pop2, ignore_index=False)
floorspace_urb_tail             = floorspace_urb_1820_1970.append(floorspace_urb, ignore_index=False)
floorspace_rur_tail             = floorspace_rur_1820_1970.append(floorspace_rur, ignore_index=False)
commercial_m2_cap_office_tail   = commercial_m2_cap_office_1820_1970.append(commercial_m2_cap_office, ignore_index=False)
commercial_m2_cap_retail_tail   = commercial_m2_cap_retail_1820_1970.append(commercial_m2_cap_retail, ignore_index=False)
commercial_m2_cap_hotels_tail   = commercial_m2_cap_hotels_1820_1970.append(commercial_m2_cap_hotels, ignore_index=False)
commercial_m2_cap_govern_tail   = commercial_m2_cap_govern_1820_1970.append(commercial_m2_cap_govern, ignore_index=False)

rurpop_tail                     = rurpop_1721_1820.append(rurpop_1820_1970.append(rurpop2, ignore_index=False), ignore_index=False)
urbpop_tail                     = urbpop_1721_1820.append(urbpop_1820_1970.append(urbpop, ignore_index=False), ignore_index=False)
pop_tail                        = pop_1721_1820.append(pop_1820_1970.append(pop2, ignore_index=False), ignore_index=False)
floorspace_urb_tail             = floorspace_urb_1721_1820.append(floorspace_urb_1820_1970.append(floorspace_urb, ignore_index=False), ignore_index=False)
floorspace_rur_tail             = floorspace_rur_1721_1820.append(floorspace_rur_1820_1970.append(floorspace_rur, ignore_index=False), ignore_index=False)
commercial_m2_cap_office_tail   = commercial_m2_cap_office_1721_1820.append(commercial_m2_cap_office_1820_1970.append(commercial_m2_cap_office, ignore_index=False), ignore_index=False)
commercial_m2_cap_retail_tail   = commercial_m2_cap_retail_1721_1820.append(commercial_m2_cap_retail_1820_1970.append(commercial_m2_cap_retail, ignore_index=False), ignore_index=False)
commercial_m2_cap_hotels_tail   = commercial_m2_cap_hotels_1721_1820.append(commercial_m2_cap_hotels_1820_1970.append(commercial_m2_cap_hotels, ignore_index=False), ignore_index=False)
commercial_m2_cap_govern_tail   = commercial_m2_cap_govern_1721_1820.append(commercial_m2_cap_govern_1820_1970.append(commercial_m2_cap_govern, ignore_index=False), ignore_index=False)

#%% SQUARE METER Calculations -----------------------------------------------------------

# adjust the share for urban/rural only (shares in csv are as percantage of the total(Rur + Urb), we needed to adjust the urban shares to add up to 1, same for rural)
housing_type_rur3 = housing_type_rur2/housing_type_rur2.sum()
housing_type_urb3 = housing_type_urb2/housing_type_urb2.sum()

# calculte the total rural/urban population (pop2 = millions of people, rurpop2 = % of people living in rural areas)
people_rur = pd.DataFrame(rurpop_tail.values*pop_tail.values, columns=pop_tail.columns, index=pop_tail.index)
people_urb = pd.DataFrame(urbpop_tail.values*pop_tail.values, columns=pop_tail.columns, index=pop_tail.index)

# calculate the total number of people (urban/rural) BY HOUSING TYPE (the sum of det,sem,app & hig equals the total population e.g. people_rur)
people_det_rur = pd.DataFrame(housing_type_rur3.iloc[0].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_sem_rur = pd.DataFrame(housing_type_rur3.iloc[1].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_app_rur = pd.DataFrame(housing_type_rur3.iloc[2].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_hig_rur = pd.DataFrame(housing_type_rur3.iloc[3].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)

people_det_urb = pd.DataFrame(housing_type_urb3.iloc[0].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_sem_urb = pd.DataFrame(housing_type_urb3.iloc[1].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_app_urb = pd.DataFrame(housing_type_urb3.iloc[2].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_hig_urb = pd.DataFrame(housing_type_urb3.iloc[3].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)

# calculate the total m2 (urban/rural) BY HOUSING TYPE (= nr. of people * OWN avg m2, so not based on IMAGE)
m2_unadjusted_det_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[0].values * people_det_rur.values, columns=people_det_rur.columns, index=people_det_rur.index)
m2_unadjusted_sem_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[1].values * people_sem_rur.values, columns=people_sem_rur.columns, index=people_sem_rur.index)
m2_unadjusted_app_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[2].values * people_app_rur.values, columns=people_app_rur.columns, index=people_app_rur.index)
m2_unadjusted_hig_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[3].values * people_hig_rur.values, columns=people_hig_rur.columns, index=people_hig_rur.index)

m2_unadjusted_det_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[0].values * people_det_urb.values, columns=people_det_urb.columns, index=people_det_urb.index)
m2_unadjusted_sem_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[1].values * people_sem_urb.values, columns=people_sem_urb.columns, index=people_sem_urb.index)
m2_unadjusted_app_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[2].values * people_app_urb.values, columns=people_app_urb.columns, index=people_app_urb.index)
m2_unadjusted_hig_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[3].values * people_hig_urb.values, columns=people_hig_urb.columns, index=people_hig_urb.index)

# Define empty dataframes for m2 adjustments
total_m2_adj_rur = pd.DataFrame(index=m2_unadjusted_det_rur.index, columns=m2_unadjusted_det_rur.columns)
total_m2_adj_urb = pd.DataFrame(index=m2_unadjusted_det_urb.index, columns=m2_unadjusted_det_urb.columns)

# Sum all square meters in Rural area
for j in range(1721,2061,1):
    for i in range(1,27,1):
        total_m2_adj_rur.loc[j,str(i)] = m2_unadjusted_det_rur.loc[j,str(i)] + m2_unadjusted_sem_rur.loc[j,str(i)] + m2_unadjusted_app_rur.loc[j,str(i)] + m2_unadjusted_hig_rur.loc[j,str(i)]

# Sum all square meters in Urban area
for j in range(1721,2061,1):
    for i in range(1,27,1):
        total_m2_adj_urb.loc[j,str(i)] = m2_unadjusted_det_urb.loc[j,str(i)] + m2_unadjusted_sem_urb.loc[j,str(i)] + m2_unadjusted_app_urb.loc[j,str(i)] + m2_unadjusted_hig_urb.loc[j,str(i)]

# average square meter per person implied by our OWN data
avg_m2_cap_adj_rur = pd.DataFrame(total_m2_adj_rur.values / people_rur.values, columns=people_rur.columns, index=people_rur.index) 
avg_m2_cap_adj_urb = pd.DataFrame(total_m2_adj_urb.values / people_urb.values, columns=people_urb.columns, index=people_urb.index)

# factor to correct square meters per capita so that we respect the IMAGE data in terms of total m2, but we use our own distinction between Building types
m2_cap_adj_fact_rur = pd.DataFrame(floorspace_rur_tail.values / avg_m2_cap_adj_rur.values, columns=floorspace_rur_tail.columns, index=floorspace_rur_tail.index)
m2_cap_adj_fact_urb = pd.DataFrame(floorspace_urb_tail.values / avg_m2_cap_adj_urb.values, columns=floorspace_urb_tail.columns, index=floorspace_urb_tail.index)

# All m2 by region (in millions), Building_type & year (using the correction factor, to comply with IMAGE avg m2/cap)
m2_det_rur = pd.DataFrame(m2_unadjusted_det_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)
m2_sem_rur = pd.DataFrame(m2_unadjusted_sem_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)
m2_app_rur = pd.DataFrame(m2_unadjusted_app_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)
m2_hig_rur = pd.DataFrame(m2_unadjusted_hig_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)

m2_det_urb = pd.DataFrame(m2_unadjusted_det_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
m2_sem_urb = pd.DataFrame(m2_unadjusted_sem_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
m2_app_urb = pd.DataFrame(m2_unadjusted_app_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
m2_hig_urb = pd.DataFrame(m2_unadjusted_hig_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)

# Add a checksum to see if calculations based on adjusted OWN avg m2 (by building type) now match the total m2 according to IMAGE. 
m2_sum_rur_OWN = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur
m2_sum_rur_IMAGE = pd.DataFrame(floorspace_rur_tail.values*people_rur.values, columns=m2_sum_rur_OWN.columns, index=m2_sum_rur_OWN.index)
m2_checksum = m2_sum_rur_OWN - m2_sum_rur_IMAGE
if m2_checksum.sum().sum() > 0.0000001 or m2_checksum.sum().sum() < -0.0000001:
    ctypes.windll.user32.MessageBoxW(0, "IMAGE & OWN m2 sums do not match", "Warning", 1)

# total RESIDENTIAL square meters by region
m2 = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur + m2_det_urb + m2_sem_urb + m2_app_urb + m2_hig_urb

# Total m2 for COMMERCIAL Buildings
commercial_m2_office = pd.DataFrame(commercial_m2_cap_office_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
commercial_m2_retail = pd.DataFrame(commercial_m2_cap_retail_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
commercial_m2_hotels = pd.DataFrame(commercial_m2_cap_hotels_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
commercial_m2_govern = pd.DataFrame(commercial_m2_cap_govern_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)

#%% MATERIAL STOCK CALCULATIONS

#rural steel stock
material_steel_det=building_materials_steel.loc[(building_materials_steel['Building_type']=='Detached')]
material_steel_det=material_steel_det.set_index('Region')
material_steel_det=material_steel_det.drop(['Building_type'],axis=1)
material_steel_det=pd.DataFrame(material_steel_det.values.T, index=material_steel_det.columns, columns=material_steel_det.index)
a=m2_det_rur.index
material_steel_det=material_steel_det.set_index(a)
kg_det_rur_steel=m2_det_rur*material_steel_det

material_steel_sem=building_materials_steel.loc[(building_materials_steel['Building_type']=='Semi-detached')]
material_steel_sem=material_steel_sem.set_index('Region')
material_steel_sem=material_steel_sem.drop(['Building_type'],axis=1)
material_steel_sem=pd.DataFrame(material_steel_sem.values.T, index=material_steel_sem.columns, columns=material_steel_sem.index)
a=m2_sem_rur.index
material_steel_sem=material_steel_sem.set_index(a)
kg_sem_rur_steel=m2_sem_rur*material_steel_sem

material_steel_app=building_materials_steel.loc[(building_materials_steel['Building_type']=='Appartments')]
material_steel_app=material_steel_app.set_index('Region')
material_steel_app=material_steel_app.drop(['Building_type'],axis=1)
material_steel_app=pd.DataFrame(material_steel_app.values.T, index=material_steel_app.columns, columns=material_steel_app.index)
a=m2_app_rur.index
material_steel_app=material_steel_app.set_index(a)
kg_app_rur_steel=m2_app_rur*material_steel_app

material_steel_hig=building_materials_steel.loc[(building_materials_steel['Building_type']=='High-rise')]
material_steel_hig=material_steel_hig.set_index('Region')
material_steel_hig=material_steel_hig.drop(['Building_type'],axis=1)
material_steel_hig=pd.DataFrame(material_steel_hig.values.T, index=material_steel_hig.columns, columns=material_steel_hig.index)
a=m2_hig_rur.index
material_steel_hig=material_steel_hig.set_index(a)
kg_hig_rur_steel=m2_hig_rur*material_steel_hig

#urban steel stock
material_steel_det=building_materials_steel.loc[(building_materials_steel['Building_type']=='Detached')]
material_steel_det=material_steel_det.set_index('Region')
material_steel_det=material_steel_det.drop(['Building_type'],axis=1)
material_steel_det=pd.DataFrame(material_steel_det.values.T, index=material_steel_det.columns, columns=material_steel_det.index)
a=m2_det_urb.index
material_steel_det=material_steel_det.set_index(a)
kg_det_urb_steel=m2_det_urb*material_steel_det

material_steel_sem=building_materials_steel.loc[(building_materials_steel['Building_type']=='Semi-detached')]
material_steel_sem=material_steel_sem.set_index('Region')
material_steel_sem=material_steel_sem.drop(['Building_type'],axis=1)
material_steel_sem=pd.DataFrame(material_steel_sem.values.T, index=material_steel_sem.columns, columns=material_steel_sem.index)
a=m2_sem_urb.index
material_steel_sem=material_steel_sem.set_index(a)
kg_sem_urb_steel=m2_sem_urb*material_steel_sem

material_steel_app=building_materials_steel.loc[(building_materials_steel['Building_type']=='Appartments')]
material_steel_app=material_steel_app.set_index('Region')
material_steel_app=material_steel_app.drop(['Building_type'],axis=1)
material_steel_app=pd.DataFrame(material_steel_app.values.T, index=material_steel_app.columns, columns=material_steel_app.index)
a=m2_app_urb.index
material_steel_app=material_steel_app.set_index(a)
kg_app_urb_steel=m2_app_urb*material_steel_app

material_steel_hig=building_materials_steel.loc[(building_materials_steel['Building_type']=='High-rise')]
material_steel_hig=material_steel_hig.set_index('Region')
material_steel_hig=material_steel_hig.drop(['Building_type'],axis=1)
material_steel_hig=pd.DataFrame(material_steel_hig.values.T, index=material_steel_hig.columns, columns=material_steel_hig.index)
a=m2_hig_urb.index
material_steel_hig=material_steel_hig.set_index(a)
kg_hig_urb_steel=m2_hig_urb*material_steel_hig

#rural brick stock
material_brick_rural = pd.read_csv('files_material_density//Building_materials_brick_rural.csv')
material_brick_urban = pd.read_csv('files_material_density//Building_materials_brick_urban.csv')
material_brick_rural = material_brick_rural.set_index('Region')
material_brick_urban = material_brick_urban.set_index('Region')
material_brick_rural=material_brick_rural.T
material_brick_urban=material_brick_urban.T
material_brick_rural.index=m2_det_rur.index
material_brick_urban.index=m2_det_urb.index

kg_det_rur_brick=m2_det_rur*material_brick_rural
kg_sem_rur_brick=m2_sem_rur*material_brick_rural
kg_app_rur_brick=m2_app_rur*material_brick_rural
kg_hig_rur_brick=m2_hig_rur*material_brick_rural

#urban brick stock
kg_det_urb_brick=m2_det_urb*material_brick_urban
kg_sem_urb_brick=m2_sem_urb*material_brick_urban
kg_app_urb_brick=m2_app_urb*material_brick_urban
kg_hig_urb_brick=m2_hig_urb*material_brick_urban

#rural concrete stock
material_concrete_det=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='Detached')]
material_concrete_det=material_concrete_det.set_index('Region')
material_concrete_det=material_concrete_det.drop(['Building_type'],axis=1)
material_concrete_det=pd.DataFrame(material_concrete_det.values.T, index=material_concrete_det.columns, columns=material_concrete_det.index)
a=m2_det_rur.index
material_concrete_det=material_concrete_det.set_index(a)
kg_det_rur_concrete=m2_det_rur*material_concrete_det

material_concrete_sem=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='Semi-detached')]
material_concrete_sem=material_concrete_sem.set_index('Region')
material_concrete_sem=material_concrete_sem.drop(['Building_type'],axis=1)
material_concrete_sem=pd.DataFrame(material_concrete_sem.values.T, index=material_concrete_sem.columns, columns=material_concrete_sem.index)
a=m2_sem_rur.index
material_concrete_sem=material_concrete_sem.set_index(a)
kg_sem_rur_concrete=m2_sem_rur*material_concrete_sem

material_concrete_app=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='Appartments')]
material_concrete_app=material_concrete_app.set_index('Region')
material_concrete_app=material_concrete_app.drop(['Building_type'],axis=1)
material_concrete_app=pd.DataFrame(material_concrete_app.values.T, index=material_concrete_app.columns, columns=material_concrete_app.index)
a=m2_app_rur.index
material_concrete_app=material_concrete_app.set_index(a)
kg_app_rur_concrete=m2_app_rur*material_concrete_app

material_concrete_hig=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='High-rise')]
material_concrete_hig=material_concrete_hig.set_index('Region')
material_concrete_hig=material_concrete_hig.drop(['Building_type'],axis=1)
material_concrete_hig=pd.DataFrame(material_concrete_hig.values.T, index=material_concrete_hig.columns, columns=material_concrete_hig.index)
a=m2_hig_rur.index
material_concrete_hig=material_concrete_hig.set_index(a)
kg_hig_rur_concrete=m2_hig_rur*material_concrete_hig

#urban concrete stock
material_concrete_det=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='Detached')]
material_concrete_det=material_concrete_det.set_index('Region')
material_concrete_det=material_concrete_det.drop(['Building_type'],axis=1)
material_concrete_det=pd.DataFrame(material_concrete_det.values.T, index=material_concrete_det.columns, columns=material_concrete_det.index)
a=m2_det_urb.index
material_concrete_det=material_concrete_det.set_index(a)
kg_det_urb_concrete=m2_det_urb*material_concrete_det

material_concrete_sem=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='Semi-detached')]
material_concrete_sem=material_concrete_sem.set_index('Region')
material_concrete_sem=material_concrete_sem.drop(['Building_type'],axis=1)
material_concrete_sem=pd.DataFrame(material_concrete_sem.values.T, index=material_concrete_sem.columns, columns=material_concrete_sem.index)
a=m2_sem_urb.index
material_concrete_sem=material_concrete_sem.set_index(a)
kg_sem_urb_concrete=m2_sem_urb*material_concrete_sem

material_concrete_app=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='Appartments')]
material_concrete_app=material_concrete_app.set_index('Region')
material_concrete_app=material_concrete_app.drop(['Building_type'],axis=1)
material_concrete_app=pd.DataFrame(material_concrete_app.values.T, index=material_concrete_app.columns, columns=material_concrete_app.index)
a=m2_app_urb.index
material_concrete_app=material_concrete_app.set_index(a)
kg_app_urb_concrete=m2_app_urb*material_concrete_app

material_concrete_hig=building_materials_concrete.loc[(building_materials_concrete['Building_type']=='High-rise')]
material_concrete_hig=material_concrete_hig.set_index('Region')
material_concrete_hig=material_concrete_hig.drop(['Building_type'],axis=1)
material_concrete_hig=pd.DataFrame(material_concrete_hig.values.T, index=material_concrete_hig.columns, columns=material_concrete_hig.index)
a=m2_hig_urb.index
material_concrete_hig=material_concrete_hig.set_index(a)
kg_hig_urb_concrete=m2_hig_urb*material_concrete_hig

#rural wood stock
material_wood_det=building_materials_wood.loc[(building_materials_wood['Building_type']=='Detached')]
material_wood_det=material_wood_det.set_index('Region')
material_wood_det=material_wood_det.drop(['Building_type'],axis=1)
material_wood_det=pd.DataFrame(material_wood_det.values.T, index=material_wood_det.columns, columns=material_wood_det.index)
a=m2_det_rur.index
material_wood_det=material_wood_det.set_index(a)
kg_det_rur_wood=m2_det_rur*material_wood_det

material_wood_sem=building_materials_wood.loc[(building_materials_wood['Building_type']=='Semi-detached')]
material_wood_sem=material_wood_sem.set_index('Region')
material_wood_sem=material_wood_sem.drop(['Building_type'],axis=1)
material_wood_sem=pd.DataFrame(material_wood_sem.values.T, index=material_wood_sem.columns, columns=material_wood_sem.index)
a=m2_sem_rur.index
material_wood_sem=material_wood_sem.set_index(a)
kg_sem_rur_wood=m2_sem_rur*material_wood_sem

material_wood_app=building_materials_wood.loc[(building_materials_wood['Building_type']=='Appartments')]
material_wood_app=material_wood_app.set_index('Region')
material_wood_app=material_wood_app.drop(['Building_type'],axis=1)
material_wood_app=pd.DataFrame(material_wood_app.values.T, index=material_wood_app.columns, columns=material_wood_app.index)
a=m2_app_rur.index
material_wood_app=material_wood_app.set_index(a)
kg_app_rur_wood=m2_app_rur*material_wood_app

material_wood_hig=building_materials_wood.loc[(building_materials_wood['Building_type']=='High-rise')]
material_wood_hig=material_wood_hig.set_index('Region')
material_wood_hig=material_wood_hig.drop(['Building_type'],axis=1)
material_wood_hig=pd.DataFrame(material_wood_hig.values.T, index=material_wood_hig.columns, columns=material_wood_hig.index)
a=m2_hig_rur.index
material_wood_hig=material_wood_hig.set_index(a)
kg_hig_rur_wood=m2_hig_rur*material_wood_hig

#urban wood stock
material_wood_det=building_materials_wood.loc[(building_materials_wood['Building_type']=='Detached')]
material_wood_det=material_wood_det.set_index('Region')
material_wood_det=material_wood_det.drop(['Building_type'],axis=1)
material_wood_det=pd.DataFrame(material_wood_det.values.T, index=material_wood_det.columns, columns=material_wood_det.index)
a=m2_det_urb.index
material_wood_det=material_wood_det.set_index(a)
kg_det_urb_wood=m2_det_urb*material_wood_det

material_wood_sem=building_materials_wood.loc[(building_materials_wood['Building_type']=='Semi-detached')]
material_wood_sem=material_wood_sem.set_index('Region')
material_wood_sem=material_wood_sem.drop(['Building_type'],axis=1)
material_wood_sem=pd.DataFrame(material_wood_sem.values.T, index=material_wood_sem.columns, columns=material_wood_sem.index)
a=m2_sem_urb.index
material_wood_sem=material_wood_sem.set_index(a)
kg_sem_urb_wood=m2_sem_urb*material_wood_sem

material_wood_app=building_materials_wood.loc[(building_materials_wood['Building_type']=='Appartments')]
material_wood_app=material_wood_app.set_index('Region')
material_wood_app=material_wood_app.drop(['Building_type'],axis=1)
material_wood_app=pd.DataFrame(material_wood_app.values.T, index=material_wood_app.columns, columns=material_wood_app.index)
a=m2_app_urb.index
material_wood_app=material_wood_app.set_index(a)
kg_app_urb_wood=m2_app_urb*material_wood_app

material_wood_hig=building_materials_wood.loc[(building_materials_wood['Building_type']=='High-rise')]
material_wood_hig=material_wood_hig.set_index('Region')
material_wood_hig=material_wood_hig.drop(['Building_type'],axis=1)
material_wood_hig=pd.DataFrame(material_wood_hig.values.T, index=material_wood_hig.columns, columns=material_wood_hig.index)
a=m2_hig_urb.index
material_wood_hig=material_wood_hig.set_index(a)
kg_hig_urb_wood=m2_hig_urb*material_wood_hig

#rural copper stock
material_copper_det=building_materials_copper.loc[(building_materials_copper['Building_type']=='Detached')]
material_copper_det=material_copper_det.set_index('Region')
material_copper_det=material_copper_det.drop(['Building_type'],axis=1)
material_copper_det=pd.DataFrame(material_copper_det.values.T, index=material_copper_det.columns, columns=material_copper_det.index)
a=m2_det_rur.index
material_copper_det=material_copper_det.set_index(a)
kg_det_rur_copper=m2_det_rur*material_copper_det

material_copper_sem=building_materials_copper.loc[(building_materials_copper['Building_type']=='Semi-detached')]
material_copper_sem=material_copper_sem.set_index('Region')
material_copper_sem=material_copper_sem.drop(['Building_type'],axis=1)
material_copper_sem=pd.DataFrame(material_copper_sem.values.T, index=material_copper_sem.columns, columns=material_copper_sem.index)
a=m2_sem_rur.index
material_copper_sem=material_copper_sem.set_index(a)
kg_sem_rur_copper=m2_sem_rur*material_copper_sem

material_copper_app=building_materials_copper.loc[(building_materials_copper['Building_type']=='Appartments')]
material_copper_app=material_copper_app.set_index('Region')
material_copper_app=material_copper_app.drop(['Building_type'],axis=1)
material_copper_app=pd.DataFrame(material_copper_app.values.T, index=material_copper_app.columns, columns=material_copper_app.index)
a=m2_app_rur.index
material_copper_app=material_copper_app.set_index(a)
kg_app_rur_copper=m2_app_rur*material_copper_app

material_copper_hig=building_materials_copper.loc[(building_materials_copper['Building_type']=='High-rise')]
material_copper_hig=material_copper_hig.set_index('Region')
material_copper_hig=material_copper_hig.drop(['Building_type'],axis=1)
material_copper_hig=pd.DataFrame(material_copper_hig.values.T, index=material_copper_hig.columns, columns=material_copper_hig.index)
a=m2_hig_rur.index
material_copper_hig=material_copper_hig.set_index(a)
kg_hig_rur_copper=m2_hig_rur*material_copper_hig

#urban copper stock
material_copper_det=building_materials_copper.loc[(building_materials_copper['Building_type']=='Detached')]
material_copper_det=material_copper_det.set_index('Region')
material_copper_det=material_copper_det.drop(['Building_type'],axis=1)
material_copper_det=pd.DataFrame(material_copper_det.values.T, index=material_copper_det.columns, columns=material_copper_det.index)
a=m2_det_urb.index
material_copper_det=material_copper_det.set_index(a)
kg_det_urb_copper=m2_det_urb*material_copper_det

material_copper_sem=building_materials_copper.loc[(building_materials_copper['Building_type']=='Semi-detached')]
material_copper_sem=material_copper_sem.set_index('Region')
material_copper_sem=material_copper_sem.drop(['Building_type'],axis=1)
material_copper_sem=pd.DataFrame(material_copper_sem.values.T, index=material_copper_sem.columns, columns=material_copper_sem.index)
a=m2_sem_urb.index
material_copper_sem=material_copper_sem.set_index(a)
kg_sem_urb_copper=m2_sem_urb*material_copper_sem

material_copper_app=building_materials_copper.loc[(building_materials_copper['Building_type']=='Appartments')]
material_copper_app=material_copper_app.set_index('Region')
material_copper_app=material_copper_app.drop(['Building_type'],axis=1)
material_copper_app=pd.DataFrame(material_copper_app.values.T, index=material_copper_app.columns, columns=material_copper_app.index)
a=m2_app_urb.index
material_copper_app=material_copper_app.set_index(a)
kg_app_urb_copper=m2_app_urb*material_copper_app

material_copper_hig=building_materials_copper.loc[(building_materials_copper['Building_type']=='High-rise')]
material_copper_hig=material_copper_hig.set_index('Region')
material_copper_hig=material_copper_hig.drop(['Building_type'],axis=1)
material_copper_hig=pd.DataFrame(material_copper_hig.values.T, index=material_copper_hig.columns, columns=material_copper_hig.index)
a=m2_hig_urb.index
material_copper_hig=material_copper_hig.set_index(a)
kg_hig_urb_copper=m2_hig_urb*material_copper_hig

#rural aluminium stock
material_aluminium_det=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='Detached')]
material_aluminium_det=material_aluminium_det.set_index('Region')
material_aluminium_det=material_aluminium_det.drop(['Building_type'],axis=1)
material_aluminium_det=pd.DataFrame(material_aluminium_det.values.T, index=material_aluminium_det.columns, columns=material_aluminium_det.index)
a=m2_det_rur.index
material_aluminium_det=material_aluminium_det.set_index(a)
kg_det_rur_aluminium=m2_det_rur*material_aluminium_det

material_aluminium_sem=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='Semi-detached')]
material_aluminium_sem=material_aluminium_sem.set_index('Region')
material_aluminium_sem=material_aluminium_sem.drop(['Building_type'],axis=1)
material_aluminium_sem=pd.DataFrame(material_aluminium_sem.values.T, index=material_aluminium_sem.columns, columns=material_aluminium_sem.index)
a=m2_sem_rur.index
material_aluminium_sem=material_aluminium_sem.set_index(a)
kg_sem_rur_aluminium=m2_det_rur*material_aluminium_sem

material_aluminium_app=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='Appartments')]
material_aluminium_app=material_aluminium_app.set_index('Region')
material_aluminium_app=material_aluminium_app.drop(['Building_type'],axis=1)
material_aluminium_app=pd.DataFrame(material_aluminium_app.values.T, index=material_aluminium_app.columns, columns=material_aluminium_app.index)
a=m2_app_rur.index
material_aluminium_app=material_aluminium_app.set_index(a)
kg_app_rur_aluminium=m2_det_rur*material_aluminium_app

material_aluminium_hig=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='High-rise')]
material_aluminium_hig=material_aluminium_hig.set_index('Region')
material_aluminium_hig=material_aluminium_hig.drop(['Building_type'],axis=1)
material_aluminium_hig=pd.DataFrame(material_aluminium_hig.values.T, index=material_aluminium_hig.columns, columns=material_aluminium_hig.index)
a=m2_hig_rur.index
material_aluminium_hig=material_aluminium_hig.set_index(a)
kg_hig_rur_aluminium=m2_det_rur*material_aluminium_hig

#urban aluminium stock
material_aluminium_det=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='Detached')]
material_aluminium_det=material_aluminium_det.set_index('Region')
material_aluminium_det=material_aluminium_det.drop(['Building_type'],axis=1)
material_aluminium_det=pd.DataFrame(material_aluminium_det.values.T, index=material_aluminium_det.columns, columns=material_aluminium_det.index)
a=m2_det_urb.index
material_aluminium_det=material_aluminium_det.set_index(a)
kg_det_urb_aluminium=m2_det_urb*material_aluminium_det

material_aluminium_sem=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='Semi-detached')]
material_aluminium_sem=material_aluminium_sem.set_index('Region')
material_aluminium_sem=material_aluminium_sem.drop(['Building_type'],axis=1)
material_aluminium_sem=pd.DataFrame(material_aluminium_sem.values.T, index=material_aluminium_sem.columns, columns=material_aluminium_sem.index)
a=m2_sem_urb.index
material_aluminium_sem=material_aluminium_sem.set_index(a)
kg_sem_urb_aluminium=m2_det_urb*material_aluminium_sem

material_aluminium_app=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='Appartments')]
material_aluminium_app=material_aluminium_app.set_index('Region')
material_aluminium_app=material_aluminium_app.drop(['Building_type'],axis=1)
material_aluminium_app=pd.DataFrame(material_aluminium_app.values.T, index=material_aluminium_app.columns, columns=material_aluminium_app.index)
a=m2_app_urb.index
material_aluminium_app=material_aluminium_app.set_index(a)
kg_app_urb_aluminium=m2_det_urb*material_aluminium_app

material_aluminium_hig=building_materials_aluminium.loc[(building_materials_aluminium['Building_type']=='High-rise')]
material_aluminium_hig=material_aluminium_hig.set_index('Region')
material_aluminium_hig=material_aluminium_hig.drop(['Building_type'],axis=1)
material_aluminium_hig=pd.DataFrame(material_aluminium_hig.values.T, index=material_aluminium_hig.columns, columns=material_aluminium_hig.index)
a=m2_hig_urb.index
material_aluminium_hig=material_aluminium_hig.set_index(a)
kg_hig_urb_aluminium=m2_det_urb*material_aluminium_hig

#rural glass stock
material_glass_det=building_materials_glass.loc[(building_materials_glass['Building_type']=='Detached')]
material_glass_det=material_glass_det.set_index('Region')
material_glass_det=material_glass_det.drop(['Building_type'],axis=1)
material_glass_det=pd.DataFrame(material_glass_det.values.T, index=material_glass_det.columns, columns=material_glass_det.index)
a=m2_det_rur.index
material_glass_det=material_glass_det.set_index(a)
kg_det_rur_glass=m2_det_rur*material_glass_det

material_glass_sem=building_materials_glass.loc[(building_materials_glass['Building_type']=='Semi-detached')]
material_glass_sem=material_glass_sem.set_index('Region')
material_glass_sem=material_glass_sem.drop(['Building_type'],axis=1)
material_glass_sem=pd.DataFrame(material_glass_sem.values.T, index=material_glass_sem.columns, columns=material_glass_sem.index)
a=m2_sem_rur.index
material_glass_sem=material_glass_sem.set_index(a)
kg_sem_rur_glass=m2_det_rur*material_glass_sem

material_glass_app=building_materials_glass.loc[(building_materials_glass['Building_type']=='Appartments')]
material_glass_app=material_glass_app.set_index('Region')
material_glass_app=material_glass_app.drop(['Building_type'],axis=1)
material_glass_app=pd.DataFrame(material_glass_app.values.T, index=material_glass_app.columns, columns=material_glass_app.index)
a=m2_app_rur.index
material_glass_app=material_glass_app.set_index(a)
kg_app_rur_glass=m2_det_rur*material_glass_app

material_glass_hig=building_materials_glass.loc[(building_materials_glass['Building_type']=='High-rise')]
material_glass_hig=material_glass_hig.set_index('Region')
material_glass_hig=material_glass_hig.drop(['Building_type'],axis=1)
material_glass_hig=pd.DataFrame(material_glass_hig.values.T, index=material_glass_hig.columns, columns=material_glass_hig.index)
a=m2_hig_rur.index
material_glass_hig=material_glass_hig.set_index(a)
kg_hig_rur_glass=m2_det_rur*material_glass_hig

#urban glass stock
material_glass_det=building_materials_glass.loc[(building_materials_glass['Building_type']=='Detached')]
material_glass_det=material_glass_det.set_index('Region')
material_glass_det=material_glass_det.drop(['Building_type'],axis=1)
material_glass_det=pd.DataFrame(material_glass_det.values.T, index=material_glass_det.columns, columns=material_glass_det.index)
a=m2_det_urb.index
material_glass_det=material_glass_det.set_index(a)
kg_det_urb_glass=m2_det_urb*material_glass_det

material_glass_sem=building_materials_glass.loc[(building_materials_glass['Building_type']=='Semi-detached')]
material_glass_sem=material_glass_sem.set_index('Region')
material_glass_sem=material_glass_sem.drop(['Building_type'],axis=1)
material_glass_sem=pd.DataFrame(material_glass_sem.values.T, index=material_glass_sem.columns, columns=material_glass_sem.index)
a=m2_sem_urb.index
material_glass_sem=material_glass_sem.set_index(a)
kg_sem_urb_glass=m2_det_urb*material_glass_sem

material_glass_app=building_materials_glass.loc[(building_materials_glass['Building_type']=='Appartments')]
material_glass_app=material_glass_app.set_index('Region')
material_glass_app=material_glass_app.drop(['Building_type'],axis=1)
material_glass_app=pd.DataFrame(material_glass_app.values.T, index=material_glass_app.columns, columns=material_glass_app.index)
a=m2_app_urb.index
material_glass_app=material_glass_app.set_index(a)
kg_app_urb_glass=m2_det_urb*material_glass_app

material_glass_hig=building_materials_glass.loc[(building_materials_glass['Building_type']=='High-rise')]
material_glass_hig=material_glass_hig.set_index('Region')
material_glass_hig=material_glass_hig.drop(['Building_type'],axis=1)
material_glass_hig=pd.DataFrame(material_glass_hig.values.T, index=material_glass_hig.columns, columns=material_glass_hig.index)
a=m2_hig_urb.index
material_glass_hig=material_glass_hig.set_index(a)
kg_hig_urb_glass=m2_det_urb*material_glass_hig


# Commercial Building materials (in Million kg)

#commercial steel stock
materials_steel_office=materials_commercial_steel.loc[(materials_commercial_steel['Building_type']=='Offices')]
materials_steel_office=materials_steel_office.drop(['Building_type'],axis=1)
materials_steel_office=pd.DataFrame(materials_steel_office.values.T, index=materials_steel_office.columns, columns=materials_steel_office.index)
a= commercial_m2_office.index
materials_steel_office=materials_steel_office.set_index(a)
kg_office_steel=commercial_m2_office*materials_steel_office

materials_steel_retail=materials_commercial_steel.loc[(materials_commercial_steel['Building_type']=='Retail+')]
materials_steel_retail=materials_steel_retail.drop(['Building_type'],axis=1)
materials_steel_retail=pd.DataFrame(materials_steel_retail.values.T, index=materials_steel_retail.columns, columns=materials_steel_retail.index)
a= commercial_m2_retail.index
materials_steel_retail=materials_steel_retail.set_index(a)
kg_retail_steel=commercial_m2_retail*materials_steel_retail

materials_steel_hotels=materials_commercial_steel.loc[(materials_commercial_steel['Building_type']=='Hotels+')]
materials_steel_hotels=materials_steel_hotels.drop(['Building_type'],axis=1)
materials_steel_hotels=pd.DataFrame(materials_steel_hotels.values.T, index=materials_steel_hotels.columns, columns=materials_steel_hotels.index)
a= commercial_m2_hotels.index
materials_steel_hotels=materials_steel_hotels.set_index(a)
kg_hotels_steel=commercial_m2_hotels*materials_steel_hotels

materials_steel_govern=materials_commercial_steel.loc[(materials_commercial_steel['Building_type']=='Govt+')]
materials_steel_govern=materials_steel_govern.drop(['Building_type'],axis=1)
materials_steel_govern=pd.DataFrame(materials_steel_govern.values.T, index=materials_steel_govern.columns, columns=materials_steel_govern.index)
a= commercial_m2_govern.index
materials_steel_govern=materials_steel_govern.set_index(a)
kg_govern_steel=commercial_m2_govern*materials_steel_govern

#commercial brick stock
materials_commercial_brick = materials_commercial_brick.set_index('Region')

materials_brick_office=materials_commercial_brick.loc[(materials_commercial_brick['Building_type']=='Offices')]
materials_brick_office=materials_commercial_brick.loc[(materials_commercial_brick['Building_type']=='Offices')]
materials_brick_office=materials_brick_office.drop(['Building_type'],axis=1)
materials_brick_office=pd.DataFrame(materials_brick_office.values.T, index=materials_brick_office.columns, columns=materials_brick_office.index)
a= commercial_m2_office.index
materials_brick_office=materials_brick_office.set_index(a)
kg_office_brick=commercial_m2_office*materials_brick_office

materials_brick_retail=materials_commercial_brick.loc[(materials_commercial_brick['Building_type']=='Retail+')]
materials_brick_retail=materials_brick_retail.drop(['Building_type'],axis=1)
materials_brick_retail=pd.DataFrame(materials_brick_retail.values.T, index=materials_brick_retail.columns, columns=materials_brick_retail.index)
a= commercial_m2_retail.index
materials_brick_retail=materials_brick_retail.set_index(a)
kg_retail_brick=commercial_m2_retail*materials_brick_retail

materials_brick_hotels=materials_commercial_brick.loc[(materials_commercial_brick['Building_type']=='Hotels+')]
materials_brick_hotels=materials_brick_hotels.drop(['Building_type'],axis=1)
materials_brick_hotels=pd.DataFrame(materials_brick_hotels.values.T, index=materials_brick_hotels.columns, columns=materials_brick_hotels.index)
a= commercial_m2_hotels.index
materials_brick_hotels=materials_brick_hotels.set_index(a)
kg_hotels_brick=commercial_m2_hotels*materials_brick_hotels

materials_brick_govern=materials_commercial_brick.loc[(materials_commercial_brick['Building_type']=='Govt+')]
materials_brick_govern=materials_brick_govern.drop(['Building_type'],axis=1)
materials_brick_govern=pd.DataFrame(materials_brick_govern.values.T, index=materials_brick_govern.columns, columns=materials_brick_govern.index)
a= commercial_m2_govern.index
materials_brick_govern=materials_brick_govern.set_index(a)
kg_govern_brick=commercial_m2_govern*materials_brick_govern

#commercial concrete stock
materials_concrete_office=materials_commercial_concrete.loc[(materials_commercial_concrete['Building_type']=='Offices')]
materials_concrete_office=materials_concrete_office.drop(['Building_type'],axis=1)
materials_concrete_office=pd.DataFrame(materials_concrete_office.values.T, index=materials_concrete_office.columns, columns=materials_concrete_office.index)
a= commercial_m2_office.index
materials_concrete_office=materials_concrete_office.set_index(a)
kg_office_concrete=commercial_m2_office*materials_concrete_office

materials_concrete_retail=materials_commercial_concrete.loc[(materials_commercial_concrete['Building_type']=='Retail+')]
materials_concrete_retail=materials_concrete_retail.drop(['Building_type'],axis=1)
materials_concrete_retail=pd.DataFrame(materials_concrete_retail.values.T, index=materials_concrete_retail.columns, columns=materials_concrete_retail.index)
a= commercial_m2_retail.index
materials_concrete_retail=materials_concrete_retail.set_index(a)
kg_retail_concrete=commercial_m2_retail*materials_concrete_retail

materials_concrete_hotels=materials_commercial_concrete.loc[(materials_commercial_concrete['Building_type']=='Hotels+')]
materials_concrete_hotels=materials_concrete_hotels.drop(['Building_type'],axis=1)
materials_concrete_hotels=pd.DataFrame(materials_concrete_hotels.values.T, index=materials_concrete_hotels.columns, columns=materials_concrete_hotels.index)
a= commercial_m2_hotels.index
materials_concrete_hotels=materials_concrete_hotels.set_index(a)
kg_hotels_concrete=commercial_m2_hotels*materials_concrete_hotels

materials_concrete_govern=materials_commercial_concrete.loc[(materials_commercial_concrete['Building_type']=='Govt+')]
materials_concrete_govern=materials_concrete_govern.drop(['Building_type'],axis=1)
materials_concrete_govern=pd.DataFrame(materials_concrete_govern.values.T, index=materials_concrete_govern.columns, columns=materials_concrete_govern.index)
a= commercial_m2_govern.index
materials_concrete_govern=materials_concrete_govern.set_index(a)
kg_govern_concrete=commercial_m2_govern*materials_concrete_govern

#commercial wood stock
materials_wood_office=materials_commercial_wood.loc[(materials_commercial_wood['Building_type']=='Offices')]
materials_wood_office=materials_wood_office.drop(['Building_type'],axis=1)
materials_wood_office=pd.DataFrame(materials_wood_office.values.T, index=materials_wood_office.columns, columns=materials_wood_office.index)
a= commercial_m2_office.index
materials_wood_office=materials_wood_office.set_index(a)
kg_office_wood=commercial_m2_office*materials_wood_office

materials_wood_retail=materials_commercial_wood.loc[(materials_commercial_wood['Building_type']=='Retail+')]
materials_wood_retail=materials_wood_retail.drop(['Building_type'],axis=1)
materials_wood_retail=pd.DataFrame(materials_wood_retail.values.T, index=materials_wood_retail.columns, columns=materials_wood_retail.index)
a= commercial_m2_retail.index
materials_wood_retail=materials_wood_retail.set_index(a)
kg_retail_wood=commercial_m2_retail*materials_wood_retail

materials_wood_hotels=materials_commercial_wood.loc[(materials_commercial_wood['Building_type']=='Hotels+')]
materials_wood_hotels=materials_wood_hotels.drop(['Building_type'],axis=1)
materials_wood_hotels=pd.DataFrame(materials_wood_hotels.values.T, index=materials_wood_hotels.columns, columns=materials_wood_hotels.index)
a= commercial_m2_hotels.index
materials_wood_hotels=materials_wood_hotels.set_index(a)
kg_hotels_wood=commercial_m2_hotels*materials_wood_hotels

materials_wood_govern=materials_commercial_wood.loc[(materials_commercial_wood['Building_type']=='Govt+')]
materials_wood_govern=materials_wood_govern.drop(['Building_type'],axis=1)
materials_wood_govern=pd.DataFrame(materials_wood_govern.values.T, index=materials_wood_govern.columns, columns=materials_wood_govern.index)
a= commercial_m2_govern.index
materials_wood_govern=materials_wood_govern.set_index(a)
kg_govern_wood=commercial_m2_govern*materials_wood_govern

#commercial copper stock
materials_copper_office=materials_commercial_copper.loc[(materials_commercial_copper['Building_type']=='Offices')]
materials_copper_office=materials_copper_office.drop(['Building_type'],axis=1)
materials_copper_office=pd.DataFrame(materials_copper_office.values.T, index=materials_copper_office.columns, columns=materials_copper_office.index)
a= commercial_m2_office.index
materials_copper_office=materials_copper_office.set_index(a)
kg_office_copper=commercial_m2_office*materials_copper_office

materials_copper_retail=materials_commercial_copper.loc[(materials_commercial_copper['Building_type']=='Retail+')]
materials_copper_retail=materials_copper_retail.drop(['Building_type'],axis=1)
materials_copper_retail=pd.DataFrame(materials_copper_retail.values.T, index=materials_copper_retail.columns, columns=materials_copper_retail.index)
a= commercial_m2_retail.index
materials_copper_retail=materials_copper_retail.set_index(a)
kg_retail_copper=commercial_m2_retail*materials_copper_retail

materials_copper_hotels=materials_commercial_copper.loc[(materials_commercial_copper['Building_type']=='Hotels+')]
materials_copper_hotels=materials_copper_hotels.drop(['Building_type'],axis=1)
materials_copper_hotels=pd.DataFrame(materials_copper_hotels.values.T, index=materials_copper_hotels.columns, columns=materials_copper_hotels.index)
a= commercial_m2_hotels.index
materials_copper_hotels=materials_copper_hotels.set_index(a)
kg_hotels_copper=commercial_m2_hotels*materials_copper_hotels

materials_copper_govern=materials_commercial_copper.loc[(materials_commercial_copper['Building_type']=='Govt+')]
materials_copper_govern=materials_copper_govern.drop(['Building_type'],axis=1)
materials_copper_govern=pd.DataFrame(materials_copper_govern.values.T, index=materials_copper_govern.columns, columns=materials_copper_govern.index)
a= commercial_m2_govern.index
materials_copper_govern=materials_copper_govern.set_index(a)
kg_govern_copper=commercial_m2_govern*materials_copper_govern

#commercial aluminium stock
materials_aluminium_office=materials_commercial_aluminium.loc[(materials_commercial_aluminium['Building_type']=='Offices')]
materials_aluminium_office=materials_aluminium_office.drop(['Building_type'],axis=1)
materials_aluminium_office=pd.DataFrame(materials_aluminium_office.values.T, index=materials_aluminium_office.columns, columns=materials_aluminium_office.index)
a= commercial_m2_office.index
materials_aluminium_office=materials_aluminium_office.set_index(a)
kg_office_aluminium=commercial_m2_office*materials_aluminium_office

materials_aluminium_retail=materials_commercial_aluminium.loc[(materials_commercial_aluminium['Building_type']=='Retail+')]
materials_aluminium_retail=materials_aluminium_retail.drop(['Building_type'],axis=1)
materials_aluminium_retail=pd.DataFrame(materials_aluminium_retail.values.T, index=materials_aluminium_retail.columns, columns=materials_aluminium_retail.index)
a= commercial_m2_retail.index
materials_aluminium_retail=materials_aluminium_retail.set_index(a)
kg_retail_aluminium=commercial_m2_retail*materials_aluminium_retail

materials_aluminium_hotels=materials_commercial_aluminium.loc[(materials_commercial_aluminium['Building_type']=='Hotels+')]
materials_aluminium_hotels=materials_aluminium_hotels.drop(['Building_type'],axis=1)
materials_aluminium_hotels=pd.DataFrame(materials_aluminium_hotels.values.T, index=materials_aluminium_hotels.columns, columns=materials_aluminium_hotels.index)
a= commercial_m2_hotels.index
materials_aluminium_hotels=materials_aluminium_hotels.set_index(a)
kg_hotels_aluminium=commercial_m2_hotels*materials_aluminium_hotels

materials_aluminium_govern=materials_commercial_aluminium.loc[(materials_commercial_aluminium['Building_type']=='Govt+')]
materials_aluminium_govern=materials_aluminium_govern.drop(['Building_type'],axis=1)
materials_aluminium_govern=pd.DataFrame(materials_aluminium_govern.values.T, index=materials_aluminium_govern.columns, columns=materials_aluminium_govern.index)
a= commercial_m2_govern.index
materials_aluminium_govern=materials_aluminium_govern.set_index(a)
kg_govern_aluminium=commercial_m2_govern*materials_aluminium_govern

#commercial glass stock
materials_glass_office=materials_commercial_glass.loc[(materials_commercial_glass['Building_type']=='Offices')]
materials_glass_office=materials_glass_office.drop(['Building_type'],axis=1)
materials_glass_office=pd.DataFrame(materials_glass_office.values.T, index=materials_glass_office.columns, columns=materials_glass_office.index)
a= commercial_m2_office.index
materials_glass_office=materials_glass_office.set_index(a)
kg_office_glass=commercial_m2_office*materials_glass_office

materials_glass_retail=materials_commercial_glass.loc[(materials_commercial_glass['Building_type']=='Retail+')]
materials_glass_retail=materials_glass_retail.drop(['Building_type'],axis=1)
materials_glass_retail=pd.DataFrame(materials_glass_retail.values.T, index=materials_glass_retail.columns, columns=materials_glass_retail.index)
a= commercial_m2_retail.index
materials_glass_retail=materials_glass_retail.set_index(a)
kg_retail_glass=commercial_m2_retail*materials_glass_retail

materials_glass_hotels=materials_commercial_glass.loc[(materials_commercial_glass['Building_type']=='Hotels+')]
materials_glass_hotels=materials_glass_hotels.drop(['Building_type'],axis=1)
materials_glass_hotels=pd.DataFrame(materials_glass_hotels.values.T, index=materials_glass_hotels.columns, columns=materials_glass_hotels.index)
a= commercial_m2_hotels.index
materials_glass_hotels=materials_glass_hotels.set_index(a)
kg_hotels_glass=commercial_m2_hotels*materials_glass_hotels

materials_glass_govern=materials_commercial_glass.loc[(materials_commercial_glass['Building_type']=='Govt+')]
materials_glass_govern=materials_glass_govern.drop(['Building_type'],axis=1)
materials_glass_govern=pd.DataFrame(materials_glass_govern.values.T, index=materials_glass_govern.columns, columns=materials_glass_govern.index)
a= commercial_m2_govern.index
materials_glass_govern=materials_glass_govern.set_index(a)
kg_govern_glass=commercial_m2_govern*materials_glass_govern

# Summing commercial material stock (Million kg)
kg_steel_comm       = kg_office_steel + kg_retail_steel + kg_hotels_steel + kg_govern_steel
kg_brick_comm      = kg_office_brick + kg_retail_brick + kg_hotels_brick + kg_govern_brick
kg_concrete_comm    = kg_office_concrete + kg_retail_concrete + kg_hotels_concrete + kg_govern_concrete
kg_wood_comm        = kg_office_wood + kg_retail_wood + kg_hotels_wood + kg_govern_wood
kg_copper_comm      = kg_office_copper + kg_retail_copper + kg_hotels_copper + kg_govern_copper
kg_aluminium_comm   = kg_office_aluminium + kg_retail_aluminium + kg_hotels_aluminium + kg_govern_aluminium
kg_glass_comm       = kg_office_glass + kg_retail_glass + kg_hotels_glass + kg_govern_glass

# Summing across RESIDENTIAL building types (millions of kg, in stock)
kg_steel_urb = kg_hig_urb_steel + kg_app_urb_steel + kg_sem_urb_steel + kg_det_urb_steel 
kg_steel_rur = kg_hig_rur_steel + kg_app_rur_steel + kg_sem_rur_steel + kg_det_rur_steel 

kg_brick_urb = kg_hig_urb_brick + kg_app_urb_brick + kg_sem_urb_brick + kg_det_urb_brick 
kg_brick_rur = kg_hig_rur_brick + kg_app_rur_brick + kg_sem_rur_brick + kg_det_rur_brick

kg_concrete_urb = kg_hig_urb_concrete + kg_app_urb_concrete + kg_sem_urb_concrete + kg_det_urb_concrete 
kg_concrete_rur = kg_hig_rur_concrete + kg_app_rur_concrete + kg_sem_rur_concrete + kg_det_rur_concrete

kg_wood_urb = kg_hig_urb_wood + kg_app_urb_wood + kg_sem_urb_wood + kg_det_urb_wood 
kg_wood_rur = kg_hig_rur_wood + kg_app_rur_wood + kg_sem_rur_wood + kg_det_rur_wood

kg_copper_urb = kg_hig_urb_copper + kg_app_urb_copper + kg_sem_urb_copper + kg_det_urb_copper 
kg_copper_rur = kg_hig_rur_copper + kg_app_rur_copper + kg_sem_rur_copper + kg_det_rur_copper

kg_aluminium_urb = kg_hig_urb_aluminium + kg_app_urb_aluminium + kg_sem_urb_aluminium + kg_det_urb_aluminium 
kg_aluminium_rur = kg_hig_rur_aluminium + kg_app_rur_aluminium + kg_sem_rur_aluminium + kg_det_rur_aluminium

kg_glass_urb = kg_hig_urb_glass + kg_app_urb_glass + kg_sem_urb_glass + kg_det_urb_glass 
kg_glass_rur = kg_hig_rur_glass + kg_app_rur_glass + kg_sem_rur_glass + kg_det_rur_glass

# Sums for total building material use (in-stock, millions of kg)
kg_steel    = kg_steel_urb + kg_steel_rur + kg_steel_comm
kg_brick   = kg_brick_urb + kg_brick_rur + kg_brick_comm
kg_concrete = kg_concrete_urb + kg_concrete_rur + kg_concrete_comm
kg_wood     = kg_wood_urb + kg_wood_rur + kg_wood_comm
kg_copper   = kg_copper_urb + kg_copper_rur + kg_copper_comm
kg_aluminium = kg_aluminium_urb + kg_aluminium_rur + kg_aluminium_comm
kg_glass   = kg_glass_urb + kg_glass_rur + kg_glass_comm


#%% INFLOW & OUTFLOW

import sys 
sys.path.append(dir_path)
import dynamic_stock_model   
from dynamic_stock_model import DynamicStockModel as DSM
idx = pd.IndexSlice   # needed for slicing multi-index

#if flag_Normal == 0:
#    lifetimes_DB = pd.read_csv('files_lifetimes\lifetimes.csv')  # Weibull parameter database (shape & scale parameters given by region, area & building-type)
#else:
#    lifetimes_DB = pd.read_csv('files_lifetimes\lifetimes_normal.csv')  # Normal distribution database (Mean & StDev parameters given by region, area & building-type, though only defined by region for now)
lifetimes_DB_shape = pd.read_csv(dir_path + '/files_lifetimes/lifetimes_shape.csv')
lifetimes_DB_scale= pd.read_csv(dir_path + '/files_lifetimes/lifetimes_scale.csv')
# actual inflow calculations

def inflow_outflown(shape, scale, stock, length):            # length is the number of years in the entire period
    out_oc_reg = pd.DataFrame(index=range(1721,2061), columns= pd.MultiIndex.from_product([list(range(1,27)), list(range(1721,2061))]))  # Multi-index columns (region & years), to contain a matrix of years*years for each region
    out_i_reg = pd.DataFrame(index=range(1721,2061), columns=range(1,27))
    out_s_reg = pd.DataFrame(index=range(1721,2061), columns=range(1,27))
    out_o_reg = pd.DataFrame(index=range(1721,2061), columns=range(1,27))
    
    for region in range(1,27):
        shape_list = shape.loc[region]
        scale_list = scale.loc[region]
        
        if flag_Normal == 0:
            DSMforward = DSM(t = np.arange(0,length,1), s=np.array(stock[region]), lt = {'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})
        else:
            DSMforward = DSM(t = np.arange(0,length,1), s=np.array(stock[region]), lt = {'Type': 'FoldNorm', 'Mean': np.array(shape_list), 'StdDev': np.array(scale_list)}) # shape & scale list are actually Mean & StDev here
        
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)
        
        out_i_reg[region] = out_i        
        out_oc[out_oc < 0] = 0 # remove negative outflow, replace by 0
        out_oc_reg.loc[:,idx[region,:]]  = out_oc
        
        
        # If you are only interested in the total outflow, you can sum the outflow by cohort
        out_o_reg[region] = out_oc.sum(axis=1)
        out_o_reg_corr = out_o_reg._get_numeric_data()        
        out_o_reg_corr[out_o_reg_corr < 0] = 0            
        out_s_reg[region] = out_sc.sum(axis=1) #Stock 
        
        
    return out_i_reg, out_oc_reg


length = len(m2_hig_urb[1])  # = 340
nindex=np.arange(0,26)

shape_selection_m2_det_rur = lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Rural') & (lifetimes_DB_shape['Type'] == 'Detached')]
scale_selection_m2_det_rur = lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Rural') & (lifetimes_DB_scale['Type'] == 'Detached')]
shape_selection_m2_sem_rur = lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Rural') & (lifetimes_DB_shape['Type'] == 'Semi-detached')]
scale_selection_m2_sem_rur = lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Rural') & (lifetimes_DB_scale['Type'] == 'Semi-detached')]
shape_selection_m2_app_rur = lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Rural') & (lifetimes_DB_shape['Type'] == 'Appartments')]
scale_selection_m2_app_rur = lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Rural') & (lifetimes_DB_scale['Type'] == 'Appartments')]
shape_selection_m2_hig_rur = lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Rural') & (lifetimes_DB_shape['Type'] == 'High-rise')]
scale_selection_m2_hig_rur = lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Rural') & (lifetimes_DB_scale['Type'] == 'High-rise')]

shape_selection_m2_det_rur=shape_selection_m2_det_rur.set_index('Region')
shape_selection_m2_det_rur=shape_selection_m2_det_rur.drop(['Type', 'Area'],axis=1)
scale_selection_m2_det_rur=scale_selection_m2_det_rur.set_index('Region')
scale_selection_m2_det_rur=scale_selection_m2_det_rur.drop(['Type', 'Area'],axis=1)

shape_selection_m2_sem_rur=shape_selection_m2_sem_rur.set_index('Region')
shape_selection_m2_sem_rur=shape_selection_m2_sem_rur.drop(['Type', 'Area'],axis=1)
scale_selection_m2_sem_rur=scale_selection_m2_sem_rur.set_index('Region')
scale_selection_m2_sem_rur=scale_selection_m2_sem_rur.drop(['Type', 'Area'],axis=1)

shape_selection_m2_app_rur=shape_selection_m2_app_rur.set_index('Region')
shape_selection_m2_app_rur=shape_selection_m2_app_rur.drop(['Type', 'Area'],axis=1)
scale_selection_m2_app_rur=scale_selection_m2_app_rur.set_index('Region')
scale_selection_m2_app_rur=scale_selection_m2_app_rur.drop(['Type', 'Area'],axis=1)

shape_selection_m2_hig_rur=shape_selection_m2_hig_rur.set_index('Region')
shape_selection_m2_hig_rur=shape_selection_m2_hig_rur.drop(['Type', 'Area'],axis=1)
scale_selection_m2_hig_rur=scale_selection_m2_hig_rur.set_index('Region')
scale_selection_m2_hig_rur=scale_selection_m2_hig_rur.drop(['Type', 'Area'],axis=1)

shape_selection_m2_det_urb=lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Urban') & (lifetimes_DB_shape['Type'] == 'Detached')]
scale_selection_m2_det_urb=lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Urban') & (lifetimes_DB_scale['Type'] == 'Detached')]
shape_selection_m2_sem_urb = lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Urban') & (lifetimes_DB_shape['Type'] == 'Semi-detached')]
scale_selection_m2_sem_urb = lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Urban') & (lifetimes_DB_scale['Type'] == 'Semi-detached')]
shape_selection_m2_app_urb =lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Urban') & (lifetimes_DB_shape['Type'] == 'Appartments')]
scale_selection_m2_app_urb =lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Urban') & (lifetimes_DB_scale['Type'] == 'Appartments')]
shape_selection_m2_hig_urb = lifetimes_DB_shape.loc[(lifetimes_DB_shape['Area'] == 'Urban') & (lifetimes_DB_shape['Type'] == 'High-rise')]
scale_selection_m2_hig_urb = lifetimes_DB_scale.loc[(lifetimes_DB_scale['Area'] == 'Urban') & (lifetimes_DB_scale['Type'] == 'High-rise')]

shape_selection_m2_det_urb=shape_selection_m2_det_urb.set_index('Region')
shape_selection_m2_det_urb=shape_selection_m2_det_urb.drop(['Type', 'Area'],axis=1)
scale_selection_m2_det_urb=scale_selection_m2_det_urb.set_index('Region')
scale_selection_m2_det_urb=scale_selection_m2_det_urb.drop(['Type', 'Area'],axis=1)

shape_selection_m2_sem_urb=shape_selection_m2_sem_urb.set_index('Region')
shape_selection_m2_sem_urb=shape_selection_m2_sem_urb.drop(['Type', 'Area'],axis=1)
scale_selection_m2_sem_urb=scale_selection_m2_sem_urb.set_index('Region')
scale_selection_m2_sem_urb=scale_selection_m2_sem_urb.drop(['Type', 'Area'],axis=1)

shape_selection_m2_app_urb=shape_selection_m2_app_urb.set_index('Region')
shape_selection_m2_app_urb=shape_selection_m2_app_urb.drop(['Type', 'Area'],axis=1)
scale_selection_m2_app_urb=scale_selection_m2_app_urb.set_index('Region')
scale_selection_m2_app_urb=scale_selection_m2_app_urb.drop(['Type', 'Area'],axis=1)

shape_selection_m2_hig_urb=shape_selection_m2_hig_urb.set_index('Region')
shape_selection_m2_hig_urb=shape_selection_m2_hig_urb.drop(['Type', 'Area'],axis=1)
scale_selection_m2_hig_urb=scale_selection_m2_hig_urb.set_index('Region')
scale_selection_m2_hig_urb=scale_selection_m2_hig_urb.drop(['Type', 'Area'],axis=1)

##
# Hardcoded lifetime parameters for COMMERCIAL building lifetime (avg. lt = 45 yr)

lifetimes_comm_shape = pd.read_csv(dir_path + '/files_lifetimes/lifetimes_shape_comm.csv')
lifetimes_comm_scale = pd.read_csv(dir_path + '/files_lifetimes/lifetimes_scale_comm.csv')
shape_comm = lifetimes_comm_shape.set_index('Region')
scale_comm = lifetimes_comm_scale.set_index('Region')

# calculating material outflow (by cohort)
def material_outflow(m2_outflow_cohort,material_density):
    emp =[]
    for i in range(0,26):
        md = material_density.iloc[:,i]
        m2 = m2_outflow_cohort.loc[:,(i+1,1721):(i+1,2060)]
        m2.columns = md.index
        material_outflow_cohort =  m2*md
        material_outflow_cohort_sum = material_outflow_cohort.sum(1)
        emp.append(material_outflow_cohort_sum)
    result = pd.DataFrame(emp)
    result.index = range(1, 27)
    return result.T

# call the actual stock model to derive inflow & outflow based on stock & lifetime
m2_det_rur_i, m2_det_rur_oc = inflow_outflown(shape_selection_m2_det_rur, scale_selection_m2_det_rur, m2_det_rur, length)
m2_sem_rur_i, m2_sem_rur_oc = inflow_outflown(shape_selection_m2_sem_rur, scale_selection_m2_sem_rur, m2_sem_rur, length)
m2_app_rur_i, m2_app_rur_oc = inflow_outflown(shape_selection_m2_app_rur, scale_selection_m2_app_rur, m2_app_rur, length)
m2_hig_rur_i, m2_hig_rur_oc = inflow_outflown(shape_selection_m2_hig_rur, scale_selection_m2_hig_rur, m2_hig_rur, length)

m2_det_urb_i, m2_det_urb_oc = inflow_outflown(shape_selection_m2_det_urb, scale_selection_m2_det_urb, m2_det_urb, length)
m2_sem_urb_i, m2_sem_urb_oc = inflow_outflown(shape_selection_m2_sem_urb, scale_selection_m2_sem_urb, m2_sem_urb, length)
m2_app_urb_i, m2_app_urb_oc = inflow_outflown(shape_selection_m2_app_urb, scale_selection_m2_app_urb, m2_app_urb, length)
m2_hig_urb_i, m2_hig_urb_oc = inflow_outflown(shape_selection_m2_hig_urb, scale_selection_m2_hig_urb, m2_hig_urb, length)

m2_office_i, m2_office_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_office, length)
m2_retail_i, m2_retail_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_retail, length)
m2_hotels_i, m2_hotels_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_hotels, length)
m2_govern_i, m2_govern_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_govern, length)

# total MILLIONS of square meters inflow
m2_res_i = m2_det_rur_i + m2_sem_rur_i + m2_app_rur_i + m2_hig_rur_i + m2_det_urb_i + m2_sem_urb_i + m2_app_urb_i + m2_hig_urb_i
m2_comm_i = m2_office_i + m2_retail_i + m2_hotels_i + m2_govern_i


#%% Material inflow & outflow
#% Material inflow
# RURAL material inflow (Millions of kgs = *1000 tons)
kg_det_rur_steel_i    = m2_det_rur_i * material_steel_det
kg_det_rur_brick_i   = m2_det_rur_i * material_brick_rural
kg_det_rur_concrete_i = m2_det_rur_i * material_concrete_det
kg_det_rur_wood_i     = m2_det_rur_i * material_wood_det
kg_det_rur_copper_i   = m2_det_rur_i * material_copper_det
kg_det_rur_aluminium_i = m2_det_rur_i * material_aluminium_det
kg_det_rur_glass_i    = m2_det_rur_i * material_glass_det

kg_sem_rur_steel_i    = m2_sem_rur_i * material_steel_sem
kg_sem_rur_brick_i   = m2_sem_rur_i * material_brick_rural
kg_sem_rur_concrete_i = m2_sem_rur_i * material_concrete_sem
kg_sem_rur_wood_i     = m2_sem_rur_i * material_wood_sem
kg_sem_rur_copper_i   = m2_sem_rur_i * material_copper_sem
kg_sem_rur_aluminium_i = m2_det_rur_i * material_aluminium_sem
kg_sem_rur_glass_i    = m2_det_rur_i * material_glass_sem

kg_app_rur_steel_i    = m2_app_rur_i * material_steel_app
kg_app_rur_brick_i   = m2_app_rur_i * material_brick_rural
kg_app_rur_concrete_i = m2_app_rur_i * material_concrete_app
kg_app_rur_wood_i     = m2_app_rur_i * material_wood_app
kg_app_rur_copper_i   = m2_app_rur_i * material_copper_app
kg_app_rur_aluminium_i = m2_det_rur_i * material_aluminium_app
kg_app_rur_glass_i    = m2_det_rur_i * material_glass_app

kg_hig_rur_steel_i    = m2_hig_rur_i * material_steel_hig
kg_hig_rur_brick_i   = m2_hig_rur_i * material_brick_rural
kg_hig_rur_concrete_i = m2_hig_rur_i * material_concrete_hig
kg_hig_rur_wood_i     = m2_hig_rur_i * material_wood_hig
kg_hig_rur_copper_i   = m2_hig_rur_i * material_copper_hig
kg_hig_rur_aluminium_i = m2_det_rur_i * material_aluminium_hig
kg_hig_rur_glass_i    = m2_det_rur_i * material_glass_hig

# URBAN material inflow (millions of kgs)
kg_det_urb_steel_i    = m2_det_urb_i * material_steel_det
kg_det_urb_brick_i   = m2_det_urb_i * material_brick_urban
kg_det_urb_concrete_i = m2_det_urb_i * material_concrete_det
kg_det_urb_wood_i     = m2_det_urb_i * material_wood_det
kg_det_urb_copper_i   = m2_det_urb_i * material_copper_det
kg_det_urb_aluminium_i  = m2_det_urb_i * material_aluminium_det
kg_det_urb_glass_i   = m2_det_urb_i * material_glass_det

kg_sem_urb_steel_i    = m2_sem_urb_i * material_steel_sem
kg_sem_urb_brick_i   = m2_sem_urb_i * material_brick_urban
kg_sem_urb_concrete_i = m2_sem_urb_i * material_concrete_sem
kg_sem_urb_wood_i     = m2_sem_urb_i * material_wood_sem
kg_sem_urb_copper_i   = m2_sem_urb_i * material_copper_sem
kg_sem_urb_aluminium_i  = m2_det_urb_i * material_aluminium_sem
kg_sem_urb_glass_i    = m2_det_urb_i * material_glass_sem

kg_app_urb_steel_i    = m2_app_urb_i * material_steel_app
kg_app_urb_brick_i   = m2_app_urb_i * material_brick_urban
kg_app_urb_concrete_i = m2_app_urb_i * material_concrete_app
kg_app_urb_wood_i     = m2_app_urb_i * material_wood_app
kg_app_urb_copper_i   = m2_app_urb_i * material_copper_app
kg_app_urb_aluminium_i  = m2_det_urb_i * material_aluminium_app
kg_app_urb_glass_i   = m2_det_urb_i * material_glass_app

kg_hig_urb_steel_i    = m2_hig_urb_i * material_steel_hig
kg_hig_urb_brick_i   = m2_hig_urb_i * material_brick_urban
kg_hig_urb_concrete_i = m2_hig_urb_i * material_concrete_hig
kg_hig_urb_wood_i     = m2_hig_urb_i * material_wood_hig
kg_hig_urb_copper_i   = m2_hig_urb_i * material_copper_hig
kg_hig_urb_aluminium_i  = m2_det_urb_i * material_aluminium_hig
kg_hig_urb_glass_i   = m2_det_urb_i * material_glass_hig

# Commercial Building materials INFLOW (in Million kg)
kg_office_steel_i     = m2_office_i * materials_steel_office
kg_office_brick_i    = m2_office_i * materials_brick_office
kg_office_concrete_i  = m2_office_i * materials_concrete_office
kg_office_wood_i      = m2_office_i * materials_wood_office
kg_office_copper_i    = m2_office_i * materials_copper_office
kg_office_aluminium_i = m2_office_i * materials_aluminium_office
kg_office_glass_i     = m2_office_i * materials_glass_office

kg_retail_steel_i     = m2_retail_i * materials_steel_retail
kg_retail_brick_i    = m2_retail_i * materials_brick_retail
kg_retail_concrete_i  = m2_retail_i * materials_concrete_retail
kg_retail_wood_i      = m2_retail_i * materials_wood_retail
kg_retail_copper_i    = m2_retail_i * materials_copper_retail
kg_retail_aluminium_i = m2_retail_i * materials_aluminium_retail
kg_retail_glass_i     = m2_retail_i * materials_glass_retail

kg_hotels_steel_i     = m2_hotels_i * materials_steel_hotels
kg_hotels_brick_i    = m2_hotels_i * materials_brick_hotels
kg_hotels_concrete_i  = m2_hotels_i * materials_concrete_hotels
kg_hotels_wood_i      = m2_hotels_i * materials_wood_hotels
kg_hotels_copper_i    = m2_hotels_i * materials_copper_hotels
kg_hotels_aluminium_i = m2_hotels_i * materials_aluminium_hotels
kg_hotels_glass_i     = m2_hotels_i * materials_glass_hotels

kg_govern_steel_i     = m2_govern_i * materials_steel_govern
kg_govern_brick_i    = m2_govern_i * materials_brick_govern
kg_govern_concrete_i  = m2_govern_i * materials_concrete_govern
kg_govern_wood_i      = m2_govern_i * materials_wood_govern
kg_govern_copper_i    = m2_govern_i * materials_copper_govern
kg_govern_aluminium_i = m2_govern_i * materials_aluminium_govern
kg_govern_glass_i     = m2_govern_i * materials_glass_govern

#% Material outflow
# RURAL material OUTflow (Millions of kgs = *1000 tons)
kg_det_rur_steel_o = material_outflow(m2_det_rur_oc, material_steel_det)
kg_det_rur_brick_o = material_outflow(m2_det_rur_oc, material_brick_rural)
kg_det_rur_concrete_o = material_outflow(m2_det_rur_oc,material_concrete_det)
kg_det_rur_wood_o = material_outflow(m2_det_rur_oc, material_wood_det)
kg_det_rur_copper_o = material_outflow(m2_det_rur_oc, material_copper_det)
kg_det_rur_aluminium_o = material_outflow(m2_det_rur_oc, material_aluminium_det)
kg_det_rur_glass_o = material_outflow(m2_det_rur_oc, material_glass_det)

kg_sem_rur_steel_o = material_outflow(m2_sem_rur_oc, material_steel_sem)
kg_sem_rur_brick_o = material_outflow(m2_sem_rur_oc, material_brick_rural)
kg_sem_rur_concrete_o = material_outflow(m2_sem_rur_oc, material_concrete_sem)
kg_sem_rur_wood_o = material_outflow(m2_sem_rur_oc, material_wood_sem)
kg_sem_rur_copper_o = material_outflow(m2_sem_rur_oc, material_copper_sem)
kg_sem_rur_aluminium_o = material_outflow(m2_det_rur_oc, material_aluminium_sem)
kg_sem_rur_glass_o = material_outflow(m2_det_rur_oc, material_glass_sem)

kg_app_rur_steel_o = material_outflow(m2_app_rur_oc, material_steel_app)
kg_app_rur_brick_o = material_outflow(m2_app_rur_oc, material_brick_rural)
kg_app_rur_concrete_o = material_outflow(m2_app_rur_oc, material_concrete_app)
kg_app_rur_wood_o = material_outflow(m2_app_rur_oc, material_wood_app)
kg_app_rur_copper_o = material_outflow(m2_app_rur_oc, material_copper_app)
kg_app_rur_aluminium_o = material_outflow(m2_det_rur_oc, material_aluminium_app)
kg_app_rur_glass_o = material_outflow(m2_det_rur_oc, material_glass_app)

kg_hig_rur_steel_o = material_outflow(m2_hig_rur_oc, material_steel_hig)
kg_hig_rur_brick_o = material_outflow(m2_hig_rur_oc, material_brick_rural)
kg_hig_rur_concrete_o = material_outflow(m2_hig_rur_oc, material_concrete_hig)
kg_hig_rur_wood_o = material_outflow(m2_hig_rur_oc, material_wood_hig)
kg_hig_rur_copper_o = material_outflow(m2_hig_rur_oc, material_copper_hig)
kg_hig_rur_aluminium_o = material_outflow(m2_det_rur_oc, material_aluminium_hig)
kg_hig_rur_glass_o = material_outflow(m2_det_rur_oc, material_glass_hig)

# URBAN material OUTflow (millions of kgs)
kg_det_urb_steel_o = material_outflow(m2_det_urb_oc, material_steel_det)
kg_det_urb_brick_o = material_outflow(m2_det_urb_oc, material_brick_urban)
kg_det_urb_concrete_o = material_outflow(m2_det_urb_oc, material_concrete_det)
kg_det_urb_wood_o = material_outflow(m2_det_urb_oc, material_wood_det)
kg_det_urb_copper_o = material_outflow(m2_det_urb_oc, material_copper_det)
kg_det_urb_aluminium_o = material_outflow(m2_det_urb_oc, material_aluminium_det)
kg_det_urb_glass_o = material_outflow(m2_det_urb_oc, material_glass_det)

kg_sem_urb_steel_o = material_outflow(m2_sem_urb_oc, material_steel_sem)
kg_sem_urb_brick_o = material_outflow(m2_sem_urb_oc, material_brick_urban)
kg_sem_urb_concrete_o = material_outflow(m2_sem_urb_oc, material_concrete_sem)
kg_sem_urb_wood_o = material_outflow(m2_sem_urb_oc, material_wood_sem)
kg_sem_urb_copper_o = material_outflow(m2_sem_urb_oc, material_copper_sem)
kg_sem_urb_aluminium_o = material_outflow(m2_det_urb_oc, material_aluminium_sem)
kg_sem_urb_glass_o = material_outflow(m2_det_urb_oc, material_glass_sem)

kg_app_urb_steel_o = material_outflow(m2_app_urb_oc, material_steel_app)
kg_app_urb_brick_o = material_outflow(m2_app_urb_oc, material_brick_urban)
kg_app_urb_concrete_o = material_outflow(m2_app_urb_oc, material_concrete_app)
kg_app_urb_wood_o = material_outflow(m2_app_urb_oc, material_wood_app)
kg_app_urb_copper_o = material_outflow(m2_app_urb_oc, material_copper_app)
kg_app_urb_aluminium_o = material_outflow(m2_det_urb_oc, material_aluminium_app)
kg_app_urb_glass_o = material_outflow(m2_det_urb_oc, material_glass_app)

kg_hig_urb_steel_o = material_outflow(m2_hig_urb_oc, material_steel_hig)
kg_hig_urb_brick_o= material_outflow(m2_hig_urb_oc, material_brick_urban)
kg_hig_urb_concrete_o = material_outflow(m2_hig_urb_oc, material_concrete_hig)
kg_hig_urb_wood_o = material_outflow(m2_hig_urb_oc, material_wood_hig)
kg_hig_urb_copper_o = material_outflow(m2_hig_urb_oc, material_copper_hig)
kg_hig_urb_aluminium_o = material_outflow(m2_det_urb_oc, material_aluminium_hig)
kg_hig_urb_glass_o = material_outflow(m2_det_urb_oc, material_glass_hig)

# Commercial Building materials OUTFLOW (in Million kg)
kg_office_steel_o = material_outflow(m2_office_oc, materials_steel_office)
kg_office_brick_o = material_outflow(m2_office_oc, materials_brick_office)
kg_office_concrete_o = material_outflow(m2_office_oc, materials_concrete_office)
kg_office_wood_o = material_outflow(m2_office_oc, materials_wood_office)
kg_office_copper_o = material_outflow(m2_office_oc, materials_copper_office)
kg_office_aluminium_o = material_outflow(m2_office_oc, materials_aluminium_office)
kg_office_glass_o = material_outflow(m2_office_oc, materials_glass_office)

kg_retail_steel_o = material_outflow(m2_retail_oc, materials_steel_retail)
kg_retail_brick_o = material_outflow(m2_retail_oc, materials_brick_retail)
kg_retail_concrete_o = material_outflow(m2_retail_oc, materials_concrete_retail)
kg_retail_wood_o = material_outflow(m2_retail_oc, materials_wood_retail)
kg_retail_copper_o = material_outflow(m2_retail_oc, materials_copper_retail)
kg_retail_aluminium_o = material_outflow(m2_retail_oc, materials_aluminium_retail)
kg_retail_glass_o = material_outflow(m2_retail_oc, materials_glass_retail)

kg_hotels_steel_o = material_outflow(m2_hotels_oc, materials_steel_hotels)
kg_hotels_brick_o = material_outflow(m2_hotels_oc, materials_brick_hotels)
kg_hotels_concrete_o = material_outflow(m2_hotels_oc, materials_concrete_hotels)
kg_hotels_wood_o = material_outflow(m2_hotels_oc, materials_wood_hotels)
kg_hotels_copper_o = material_outflow(m2_hotels_oc, materials_copper_hotels)
kg_hotels_aluminium_o = material_outflow(m2_hotels_oc, materials_aluminium_hotels)
kg_hotels_glass_o = material_outflow(m2_hotels_oc, materials_glass_hotels)

kg_govern_steel_o = material_outflow(m2_govern_oc, materials_steel_govern)
kg_govern_brick_o = material_outflow(m2_govern_oc, materials_brick_govern)
kg_govern_concrete_o = material_outflow(m2_govern_oc, materials_concrete_govern)
kg_govern_wood_o = material_outflow(m2_govern_oc, materials_wood_govern)
kg_govern_copper_o = material_outflow(m2_govern_oc, materials_copper_govern)
kg_govern_aluminium_o = material_outflow(m2_govern_oc, materials_aluminium_govern)
kg_govern_glass_o = material_outflow(m2_govern_oc, materials_glass_govern)


#%% CSV output (material stock & m2 stock)

length = 3
tag = ['stock', 'inflow', 'outflow']
  
# RURAL
kg_det_rur_steel_out  = [[]] * length
kg_det_rur_steel_out[0]  = kg_det_rur_steel.transpose()
kg_det_rur_steel_out[1]  = kg_det_rur_steel_i.transpose()
kg_det_rur_steel_out[2]  = kg_det_rur_steel_o.transpose()
for item in range(0,length):
    kg_det_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_det_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
    kg_det_rur_steel_out[item].insert(0,'type', ['detached'] * 26)
    kg_det_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
    
kg_det_rur_brick_out      = [[]] * length  
kg_det_rur_brick_out[0]   = kg_det_rur_brick.transpose() 
kg_det_rur_brick_out[1]   = kg_det_rur_brick_i.transpose() 
kg_det_rur_brick_out[2]   = kg_det_rur_brick_o.transpose() 
for item in range(0,length):
        kg_det_rur_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_det_rur_brick_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_brick_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_rur_concrete_out      = [[]] * length  
kg_det_rur_concrete_out[0]   = kg_det_rur_concrete.transpose() 
kg_det_rur_concrete_out[1]   = kg_det_rur_concrete_i.transpose() 
kg_det_rur_concrete_out[2]   = kg_det_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_det_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_det_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_concrete_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_rur_wood_out      = [[]] * length  
kg_det_rur_wood_out[0]   = kg_det_rur_wood.transpose() 
kg_det_rur_wood_out[1]   = kg_det_rur_wood_i.transpose() 
kg_det_rur_wood_out[2]   = kg_det_rur_wood_o.transpose() 
for item in range(0,length):
        kg_det_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_det_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_wood_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
   
kg_det_rur_copper_out      = [[]] * length  
kg_det_rur_copper_out[0]   = kg_det_rur_copper.transpose() 
kg_det_rur_copper_out[1]   = kg_det_rur_copper_i.transpose() 
kg_det_rur_copper_out[2]   = kg_det_rur_copper_o.transpose() 
for item in range(0,length):
        kg_det_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_det_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_copper_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_rur_aluminium_out      = [[]] * length  
kg_det_rur_aluminium_out[0]   = kg_det_rur_aluminium.transpose() 
kg_det_rur_aluminium_out[1]   = kg_det_rur_aluminium_i.transpose() 
kg_det_rur_aluminium_out[2]   = kg_det_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_det_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_det_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_aluminium_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_rur_glass_out      = [[]] * length  
kg_det_rur_glass_out[0]   = kg_det_rur_glass.transpose() 
kg_det_rur_glass_out[1]   = kg_det_rur_glass_i.transpose() 
kg_det_rur_glass_out[2]   = kg_det_rur_glass_o.transpose() 
for item in range(0,length):
        kg_det_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_det_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_glass_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)

kg_sem_rur_steel_out      = [[]] * length  
kg_sem_rur_steel_out[0]   = kg_sem_rur_steel.transpose() 
kg_sem_rur_steel_out[1]   = kg_sem_rur_steel_i.transpose() 
kg_sem_rur_steel_out[2]   = kg_sem_rur_steel_o.transpose() 
for item in range(0,length):
        kg_sem_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_sem_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_steel_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_brick_out      = [[]] * length  
kg_sem_rur_brick_out[0]   = kg_sem_rur_brick.transpose() 
kg_sem_rur_brick_out[1]   = kg_sem_rur_brick_i.transpose() 
kg_sem_rur_brick_out[2]   = kg_sem_rur_brick_o.transpose() 
for item in range(0,length):
        kg_sem_rur_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_sem_rur_brick_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_brick_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_concrete_out      = [[]] * length  
kg_sem_rur_concrete_out[0]   = kg_sem_rur_concrete.transpose() 
kg_sem_rur_concrete_out[1]   = kg_sem_rur_concrete_i.transpose() 
kg_sem_rur_concrete_out[2]   = kg_sem_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_sem_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_sem_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_concrete_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_wood_out      = [[]] * length  
kg_sem_rur_wood_out[0]   = kg_sem_rur_wood.transpose() 
kg_sem_rur_wood_out[1]   = kg_sem_rur_wood_i.transpose() 
kg_sem_rur_wood_out[2]   = kg_sem_rur_wood_o.transpose() 
for item in range(0,length):
        kg_sem_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_sem_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_wood_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_copper_out      = [[]] * length  
kg_sem_rur_copper_out[0]   = kg_sem_rur_copper.transpose() 
kg_sem_rur_copper_out[1]   = kg_sem_rur_copper_i.transpose() 
kg_sem_rur_copper_out[2]   = kg_sem_rur_copper_o.transpose() 
for item in range(0,length):
        kg_sem_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_sem_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_copper_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_aluminium_out      = [[]] * length  
kg_sem_rur_aluminium_out[0]   = kg_sem_rur_aluminium.transpose() 
kg_sem_rur_aluminium_out[1]   = kg_sem_rur_aluminium_i.transpose() 
kg_sem_rur_aluminium_out[2]   = kg_sem_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_sem_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_sem_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_aluminium_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_glass_out      = [[]] * length  
kg_sem_rur_glass_out[0]   = kg_sem_rur_glass.transpose() 
kg_sem_rur_glass_out[1]   = kg_sem_rur_glass_i.transpose() 
kg_sem_rur_glass_out[2]   = kg_sem_rur_glass_o.transpose() 
for item in range(0,length):
        kg_sem_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_sem_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_glass_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)
 
#
kg_app_rur_steel_out      = [[]] * length  
kg_app_rur_steel_out[0]   = kg_app_rur_steel.transpose() 
kg_app_rur_steel_out[1]   = kg_app_rur_steel_i.transpose() 
kg_app_rur_steel_out[2]   = kg_app_rur_steel_o.transpose() 
for item in range(0,length):
        kg_app_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_app_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_steel_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_brick_out      = [[]] * length  
kg_app_rur_brick_out[0]   = kg_app_rur_brick.transpose() 
kg_app_rur_brick_out[1]   = kg_app_rur_brick_i.transpose() 
kg_app_rur_brick_out[2]   = kg_app_rur_brick_o.transpose() 
for item in range(0,length):
        kg_app_rur_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_app_rur_brick_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_brick_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_concrete_out      = [[]] * length  
kg_app_rur_concrete_out[0]   = kg_app_rur_concrete.transpose() 
kg_app_rur_concrete_out[1]   = kg_app_rur_concrete_i.transpose() 
kg_app_rur_concrete_out[2]   = kg_app_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_app_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_app_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_concrete_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_wood_out      = [[]] * length  
kg_app_rur_wood_out[0]   = kg_app_rur_wood.transpose() 
kg_app_rur_wood_out[1]   = kg_app_rur_wood_i.transpose() 
kg_app_rur_wood_out[2]   = kg_app_rur_wood_o.transpose() 
for item in range(0,length):
        kg_app_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_app_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_wood_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_copper_out      = [[]] * length  
kg_app_rur_copper_out[0]   = kg_app_rur_copper.transpose() 
kg_app_rur_copper_out[1]   = kg_app_rur_copper_i.transpose() 
kg_app_rur_copper_out[2]   = kg_app_rur_copper_o.transpose() 
for item in range(0,length):
        kg_app_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_app_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_copper_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_aluminium_out      = [[]] * length  
kg_app_rur_aluminium_out[0]   = kg_app_rur_aluminium.transpose() 
kg_app_rur_aluminium_out[1]   = kg_app_rur_aluminium_i.transpose() 
kg_app_rur_aluminium_out[2]   = kg_app_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_app_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_app_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_aluminium_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_glass_out      = [[]] * length  
kg_app_rur_glass_out[0]   = kg_app_rur_glass.transpose() 
kg_app_rur_glass_out[1]   = kg_app_rur_glass_i.transpose() 
kg_app_rur_glass_out[2]   = kg_app_rur_glass_o.transpose() 
for item in range(0,length):
        kg_app_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_app_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_glass_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)
        
kg_hig_rur_steel_out      = [[]] * length  
kg_hig_rur_steel_out[0]   = kg_hig_rur_steel.transpose() 
kg_hig_rur_steel_out[1]   = kg_hig_rur_steel_i.transpose() 
kg_hig_rur_steel_out[2]   = kg_hig_rur_steel_o.transpose() 
for item in range(0,length):
        kg_hig_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_hig_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_steel_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_brick_out      = [[]] * length  
kg_hig_rur_brick_out[0]   = kg_hig_rur_brick.transpose() 
kg_hig_rur_brick_out[1]   = kg_hig_rur_brick_i.transpose() 
kg_hig_rur_brick_out[2]   = kg_hig_rur_brick_o.transpose() 
for item in range(0,length):
        kg_hig_rur_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_hig_rur_brick_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_brick_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_concrete_out      = [[]] * length  
kg_hig_rur_concrete_out[0]   = kg_hig_rur_concrete.transpose() 
kg_hig_rur_concrete_out[1]   = kg_hig_rur_concrete_i.transpose() 
kg_hig_rur_concrete_out[2]   = kg_hig_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_hig_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_hig_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_concrete_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_wood_out      = [[]] * length  
kg_hig_rur_wood_out[0]   = kg_hig_rur_wood.transpose() 
kg_hig_rur_wood_out[1]   = kg_hig_rur_wood_i.transpose() 
kg_hig_rur_wood_out[2]   = kg_hig_rur_wood_o.transpose() 
for item in range(0,length):
        kg_hig_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_hig_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_wood_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_copper_out      = [[]] * length  
kg_hig_rur_copper_out[0]   = kg_hig_rur_copper.transpose() 
kg_hig_rur_copper_out[1]   = kg_hig_rur_copper_i.transpose() 
kg_hig_rur_copper_out[2]   = kg_hig_rur_copper_o.transpose() 
for item in range(0,length):
        kg_hig_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_hig_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_copper_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_aluminium_out      = [[]] * length  
kg_hig_rur_aluminium_out[0]   = kg_hig_rur_aluminium.transpose() 
kg_hig_rur_aluminium_out[1]   = kg_hig_rur_aluminium_i.transpose() 
kg_hig_rur_aluminium_out[2]   = kg_hig_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_hig_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_hig_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_aluminium_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_glass_out      = [[]] * length  
kg_hig_rur_glass_out[0]   = kg_hig_rur_glass.transpose() 
kg_hig_rur_glass_out[1]   = kg_hig_rur_glass_i.transpose() 
kg_hig_rur_glass_out[2]   = kg_hig_rur_glass_o.transpose() 
for item in range(0,length):
        kg_hig_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_hig_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_glass_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# URBAN 
kg_det_urb_steel_out  = [[]] * length
kg_det_urb_steel_out[0]  = kg_det_urb_steel.transpose()
kg_det_urb_steel_out[1]  = kg_det_urb_steel_i.transpose()
kg_det_urb_steel_out[2]  = kg_det_urb_steel_o.transpose()
for item in range(0,length):
    kg_det_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_det_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
    kg_det_urb_steel_out[item].insert(0,'type', ['detached'] * 26)
    kg_det_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
    
kg_det_urb_brick_out      = [[]] * length  
kg_det_urb_brick_out[0]   = kg_det_urb_brick.transpose() 
kg_det_urb_brick_out[1]   = kg_det_urb_brick_i.transpose() 
kg_det_urb_brick_out[2]   = kg_det_urb_brick_o.transpose() 
for item in range(0,length):
        kg_det_urb_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_det_urb_brick_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_brick_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_urb_concrete_out      = [[]] * length  
kg_det_urb_concrete_out[0]   = kg_det_urb_concrete.transpose() 
kg_det_urb_concrete_out[1]   = kg_det_urb_concrete_i.transpose() 
kg_det_urb_concrete_out[2]   = kg_det_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_det_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_det_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_concrete_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_urb_wood_out      = [[]] * length  
kg_det_urb_wood_out[0]   = kg_det_urb_wood.transpose() 
kg_det_urb_wood_out[1]   = kg_det_urb_wood_i.transpose() 
kg_det_urb_wood_out[2]   = kg_det_urb_wood_o.transpose() 
for item in range(0,length):
        kg_det_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_det_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_wood_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
   
kg_det_urb_copper_out      = [[]] * length  
kg_det_urb_copper_out[0]   = kg_det_urb_copper.transpose() 
kg_det_urb_copper_out[1]   = kg_det_urb_copper_i.transpose() 
kg_det_urb_copper_out[2]   = kg_det_urb_copper_o.transpose() 
for item in range(0,length):
        kg_det_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_det_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_copper_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_urb_aluminium_out      = [[]] * length  
kg_det_urb_aluminium_out[0]   = kg_det_urb_aluminium.transpose() 
kg_det_urb_aluminium_out[1]   = kg_det_urb_aluminium_i.transpose() 
kg_det_urb_aluminium_out[2]   = kg_det_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_det_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_det_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_aluminium_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_urb_glass_out      = [[]] * length  
kg_det_urb_glass_out[0]   = kg_det_urb_glass.transpose() 
kg_det_urb_glass_out[1]   = kg_det_urb_glass_i.transpose() 
kg_det_urb_glass_out[2]   = kg_det_urb_glass_o.transpose() 
for item in range(0,length):
        kg_det_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_det_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_glass_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)

kg_sem_urb_steel_out      = [[]] * length  
kg_sem_urb_steel_out[0]   = kg_sem_urb_steel.transpose() 
kg_sem_urb_steel_out[1]   = kg_sem_urb_steel_i.transpose() 
kg_sem_urb_steel_out[2]   = kg_sem_urb_steel_o.transpose() 
for item in range(0,length):
        kg_sem_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_sem_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_steel_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_brick_out      = [[]] * length  
kg_sem_urb_brick_out[0]   = kg_sem_urb_brick.transpose() 
kg_sem_urb_brick_out[1]   = kg_sem_urb_brick_i.transpose() 
kg_sem_urb_brick_out[2]   = kg_sem_urb_brick_o.transpose() 
for item in range(0,length):
        kg_sem_urb_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_sem_urb_brick_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_brick_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_concrete_out      = [[]] * length  
kg_sem_urb_concrete_out[0]   = kg_sem_urb_concrete.transpose() 
kg_sem_urb_concrete_out[1]   = kg_sem_urb_concrete_i.transpose() 
kg_sem_urb_concrete_out[2]   = kg_sem_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_sem_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_sem_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_concrete_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_wood_out      = [[]] * length  
kg_sem_urb_wood_out[0]   = kg_sem_urb_wood.transpose() 
kg_sem_urb_wood_out[1]   = kg_sem_urb_wood_i.transpose() 
kg_sem_urb_wood_out[2]   = kg_sem_urb_wood_o.transpose() 
for item in range(0,length):
        kg_sem_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_sem_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_wood_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_copper_out      = [[]] * length  
kg_sem_urb_copper_out[0]   = kg_sem_urb_copper.transpose() 
kg_sem_urb_copper_out[1]   = kg_sem_urb_copper_i.transpose() 
kg_sem_urb_copper_out[2]   = kg_sem_urb_copper_o.transpose() 
for item in range(0,length):
        kg_sem_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_sem_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_copper_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_aluminium_out      = [[]] * length  
kg_sem_urb_aluminium_out[0]   = kg_sem_urb_aluminium.transpose() 
kg_sem_urb_aluminium_out[1]   = kg_sem_urb_aluminium_i.transpose() 
kg_sem_urb_aluminium_out[2]   = kg_sem_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_sem_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_sem_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_aluminium_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_glass_out      = [[]] * length  
kg_sem_urb_glass_out[0]   = kg_sem_urb_glass.transpose() 
kg_sem_urb_glass_out[1]   = kg_sem_urb_glass_i.transpose() 
kg_sem_urb_glass_out[2]   = kg_sem_urb_glass_o.transpose() 
for item in range(0,length):
        kg_sem_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_sem_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_glass_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)
 
#
kg_app_urb_steel_out      = [[]] * length  
kg_app_urb_steel_out[0]   = kg_app_urb_steel.transpose() 
kg_app_urb_steel_out[1]   = kg_app_urb_steel_i.transpose() 
kg_app_urb_steel_out[2]   = kg_app_urb_steel_o.transpose() 
for item in range(0,length):
        kg_app_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_app_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_steel_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_brick_out      = [[]] * length  
kg_app_urb_brick_out[0]   = kg_app_urb_brick.transpose() 
kg_app_urb_brick_out[1]   = kg_app_urb_brick_i.transpose() 
kg_app_urb_brick_out[2]   = kg_app_urb_brick_o.transpose() 
for item in range(0,length):
        kg_app_urb_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_app_urb_brick_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_brick_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_concrete_out      = [[]] * length  
kg_app_urb_concrete_out[0]   = kg_app_urb_concrete.transpose() 
kg_app_urb_concrete_out[1]   = kg_app_urb_concrete_i.transpose() 
kg_app_urb_concrete_out[2]   = kg_app_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_app_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_app_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_concrete_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_wood_out      = [[]] * length  
kg_app_urb_wood_out[0]   = kg_app_urb_wood.transpose() 
kg_app_urb_wood_out[1]   = kg_app_urb_wood_i.transpose() 
kg_app_urb_wood_out[2]   = kg_app_urb_wood_o.transpose() 
for item in range(0,length):
        kg_app_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_app_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_wood_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_copper_out      = [[]] * length  
kg_app_urb_copper_out[0]   = kg_app_urb_copper.transpose() 
kg_app_urb_copper_out[1]   = kg_app_urb_copper_i.transpose() 
kg_app_urb_copper_out[2]   = kg_app_urb_copper_o.transpose() 
for item in range(0,length):
        kg_app_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_app_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_copper_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_aluminium_out      = [[]] * length  
kg_app_urb_aluminium_out[0]   = kg_app_urb_aluminium.transpose() 
kg_app_urb_aluminium_out[1]   = kg_app_urb_aluminium_i.transpose() 
kg_app_urb_aluminium_out[2]   = kg_app_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_app_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_app_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_aluminium_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_glass_out      = [[]] * length  
kg_app_urb_glass_out[0]   = kg_app_urb_glass.transpose() 
kg_app_urb_glass_out[1]   = kg_app_urb_glass_i.transpose() 
kg_app_urb_glass_out[2]   = kg_app_urb_glass_o.transpose() 
for item in range(0,length):
        kg_app_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_app_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_glass_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)
        
kg_hig_urb_steel_out      = [[]] * length  
kg_hig_urb_steel_out[0]   = kg_hig_urb_steel.transpose() 
kg_hig_urb_steel_out[1]   = kg_hig_urb_steel_i.transpose() 
kg_hig_urb_steel_out[2]   = kg_hig_urb_steel_o.transpose() 
for item in range(0,length):
        kg_hig_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_hig_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_steel_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_brick_out      = [[]] * length  
kg_hig_urb_brick_out[0]   = kg_hig_urb_brick.transpose() 
kg_hig_urb_brick_out[1]   = kg_hig_urb_brick_i.transpose() 
kg_hig_urb_brick_out[2]   = kg_hig_urb_brick_o.transpose() 
for item in range(0,length):
        kg_hig_urb_brick_out[item].insert(0,'material', ['brick'] * 26)
        kg_hig_urb_brick_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_brick_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_brick_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_concrete_out      = [[]] * length  
kg_hig_urb_concrete_out[0]   = kg_hig_urb_concrete.transpose() 
kg_hig_urb_concrete_out[1]   = kg_hig_urb_concrete_i.transpose() 
kg_hig_urb_concrete_out[2]   = kg_hig_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_hig_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_hig_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_concrete_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_wood_out      = [[]] * length  
kg_hig_urb_wood_out[0]   = kg_hig_urb_wood.transpose() 
kg_hig_urb_wood_out[1]   = kg_hig_urb_wood_i.transpose() 
kg_hig_urb_wood_out[2]   = kg_hig_urb_wood_o.transpose() 
for item in range(0,length):
        kg_hig_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_hig_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_wood_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_copper_out      = [[]] * length  
kg_hig_urb_copper_out[0]   = kg_hig_urb_copper.transpose() 
kg_hig_urb_copper_out[1]   = kg_hig_urb_copper_i.transpose() 
kg_hig_urb_copper_out[2]   = kg_hig_urb_copper_o.transpose() 
for item in range(0,length):
        kg_hig_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_hig_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_copper_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_aluminium_out      = [[]] * length  
kg_hig_urb_aluminium_out[0]   = kg_hig_urb_aluminium.transpose() 
kg_hig_urb_aluminium_out[1]   = kg_hig_urb_aluminium_i.transpose() 
kg_hig_urb_aluminium_out[2]   = kg_hig_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_hig_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_hig_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_aluminium_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_glass_out      = [[]] * length  
kg_hig_urb_glass_out[0]   = kg_hig_urb_glass.transpose() 
kg_hig_urb_glass_out[1]   = kg_hig_urb_glass_i.transpose() 
kg_hig_urb_glass_out[2]   = kg_hig_urb_glass_o.transpose() 
for item in range(0,length):
        kg_hig_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_hig_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_glass_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# COMMERCIAL ------------------------------------------------------------------

# offices
kg_office_steel_out  = [[]] * length
kg_office_steel_out[0]  = kg_office_steel.transpose()
kg_office_steel_out[1]  = kg_office_steel_i.transpose()
kg_office_steel_out[2]  = kg_office_steel_o.transpose()
for item in range(0,length):
    kg_office_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_office_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_steel_out[item].insert(0,'type', ['office'] * 26)
    kg_office_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_brick_out  = [[]] * length
kg_office_brick_out[0]  = kg_office_brick.transpose()
kg_office_brick_out[1]  = kg_office_brick_i.transpose()
kg_office_brick_out[2]  = kg_office_brick_o.transpose()
for item in range(0,length):
    kg_office_brick_out[item].insert(0,'material', ['brick'] * 26)
    kg_office_brick_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_brick_out[item].insert(0,'type', ['office'] * 26)
    kg_office_brick_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_concrete_out  = [[]] * length
kg_office_concrete_out[0]  = kg_office_concrete.transpose()
kg_office_concrete_out[1]  = kg_office_concrete_i.transpose()
kg_office_concrete_out[2]  = kg_office_concrete_o.transpose()
for item in range(0,length):
    kg_office_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_office_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_concrete_out[item].insert(0,'type', ['office'] * 26)
    kg_office_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_wood_out  = [[]] * length
kg_office_wood_out[0]  = kg_office_wood.transpose()
kg_office_wood_out[1]  = kg_office_wood_i.transpose()
kg_office_wood_out[2]  = kg_office_wood_o.transpose()
for item in range(0,length):
    kg_office_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_office_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_wood_out[item].insert(0,'type', ['office'] * 26)
    kg_office_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_copper_out  = [[]] * length
kg_office_copper_out[0]  = kg_office_copper.transpose()
kg_office_copper_out[1]  = kg_office_copper_i.transpose()
kg_office_copper_out[2]  = kg_office_copper_o.transpose()
for item in range(0,length):
    kg_office_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_office_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_copper_out[item].insert(0,'type', ['office'] * 26)
    kg_office_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_aluminium_out  = [[]] * length
kg_office_aluminium_out[0]  = kg_office_aluminium.transpose()
kg_office_aluminium_out[1]  = kg_office_aluminium_i.transpose()
kg_office_aluminium_out[2]  = kg_office_aluminium_o.transpose()
for item in range(0,length):
    kg_office_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_office_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_aluminium_out[item].insert(0,'type', ['office'] * 26)
    kg_office_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_glass_out  = [[]] * length
kg_office_glass_out[0]  = kg_office_glass.transpose()
kg_office_glass_out[1]  = kg_office_glass_i.transpose()
kg_office_glass_out[2]  = kg_office_glass_o.transpose()
for item in range(0,length):
    kg_office_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_office_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_glass_out[item].insert(0,'type', ['office'] * 26)
    kg_office_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# shops & retail
kg_retail_steel_out  = [[]] * length
kg_retail_steel_out[0]  = kg_retail_steel.transpose()
kg_retail_steel_out[1]  = kg_retail_steel_i.transpose()
kg_retail_steel_out[2]  = kg_retail_steel_o.transpose()
for item in range(0,length):
    kg_retail_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_retail_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_steel_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_brick_out  = [[]] * length
kg_retail_brick_out[0]  = kg_retail_brick.transpose()
kg_retail_brick_out[1]  = kg_retail_brick_i.transpose()
kg_retail_brick_out[2]  = kg_retail_brick_o.transpose()
for item in range(0,length):
    kg_retail_brick_out[item].insert(0,'material', ['brick'] * 26)
    kg_retail_brick_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_brick_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_brick_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_concrete_out  = [[]] * length
kg_retail_concrete_out[0]  = kg_retail_concrete.transpose()
kg_retail_concrete_out[1]  = kg_retail_concrete_i.transpose()
kg_retail_concrete_out[2]  = kg_retail_concrete_o.transpose()
for item in range(0,length):
    kg_retail_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_retail_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_concrete_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_wood_out  = [[]] * length
kg_retail_wood_out[0]  = kg_retail_wood.transpose()
kg_retail_wood_out[1]  = kg_retail_wood_i.transpose()
kg_retail_wood_out[2]  = kg_retail_wood_o.transpose()
for item in range(0,length):
    kg_retail_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_retail_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_wood_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_copper_out  = [[]] * length
kg_retail_copper_out[0]  = kg_retail_copper.transpose()
kg_retail_copper_out[1]  = kg_retail_copper_i.transpose()
kg_retail_copper_out[2]  = kg_retail_copper_o.transpose()
for item in range(0,length):
    kg_retail_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_retail_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_copper_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_aluminium_out  = [[]] * length
kg_retail_aluminium_out[0]  = kg_retail_aluminium.transpose()
kg_retail_aluminium_out[1]  = kg_retail_aluminium_i.transpose()
kg_retail_aluminium_out[2]  = kg_retail_aluminium_o.transpose()
for item in range(0,length):
    kg_retail_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_retail_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_aluminium_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_glass_out  = [[]] * length
kg_retail_glass_out[0]  = kg_retail_glass.transpose()
kg_retail_glass_out[1]  = kg_retail_glass_i.transpose()
kg_retail_glass_out[2]  = kg_retail_glass_o.transpose()
for item in range(0,length):
    kg_retail_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_retail_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_glass_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# hotels & restaurants

kg_hotels_steel_out  = [[]] * length
kg_hotels_steel_out[0]  = kg_hotels_steel.transpose()
kg_hotels_steel_out[1]  = kg_hotels_steel_i.transpose()
kg_hotels_steel_out[2]  = kg_hotels_steel_o.transpose()
for item in range(0,length):
    kg_hotels_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_hotels_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_steel_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_brick_out  = [[]] * length
kg_hotels_brick_out[0]  = kg_hotels_brick.transpose()
kg_hotels_brick_out[1]  = kg_hotels_brick_i.transpose()
kg_hotels_brick_out[2]  = kg_hotels_brick_o.transpose()
for item in range(0,length):
    kg_hotels_brick_out[item].insert(0,'material', ['brick'] * 26)
    kg_hotels_brick_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_brick_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_brick_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_concrete_out  = [[]] * length
kg_hotels_concrete_out[0]  = kg_hotels_concrete.transpose()
kg_hotels_concrete_out[1]  = kg_hotels_concrete_i.transpose()
kg_hotels_concrete_out[2]  = kg_hotels_concrete_o.transpose()
for item in range(0,length):
    kg_hotels_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_hotels_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_concrete_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_wood_out  = [[]] * length
kg_hotels_wood_out[0]  = kg_hotels_wood.transpose()
kg_hotels_wood_out[1]  = kg_hotels_wood_i.transpose()
kg_hotels_wood_out[2]  = kg_hotels_wood_o.transpose()
for item in range(0,length):
    kg_hotels_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_hotels_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_wood_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_copper_out  = [[]] * length
kg_hotels_copper_out[0]  = kg_hotels_copper.transpose()
kg_hotels_copper_out[1]  = kg_hotels_copper_i.transpose()
kg_hotels_copper_out[2]  = kg_hotels_copper_o.transpose()
for item in range(0,length):
    kg_hotels_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_hotels_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_copper_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_aluminium_out  = [[]] * length
kg_hotels_aluminium_out[0]  = kg_hotels_aluminium.transpose()
kg_hotels_aluminium_out[1]  = kg_hotels_aluminium_i.transpose()
kg_hotels_aluminium_out[2]  = kg_hotels_aluminium_o.transpose()
for item in range(0,length):
    kg_hotels_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_hotels_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_aluminium_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_glass_out  = [[]] * length
kg_hotels_glass_out[0]  = kg_hotels_glass.transpose()
kg_hotels_glass_out[1]  = kg_hotels_glass_i.transpose()
kg_hotels_glass_out[2]  = kg_hotels_glass_o.transpose()
for item in range(0,length):
    kg_hotels_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_hotels_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_glass_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# government (schools, government, public transport, hospitals)
kg_govern_steel_out  = [[]] * length
kg_govern_steel_out[0]  = kg_govern_steel.transpose()
kg_govern_steel_out[1]  = kg_govern_steel_i.transpose()
kg_govern_steel_out[2]  = kg_govern_steel_o.transpose()
for item in range(0,length):
    kg_govern_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_govern_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_steel_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_brick_out  = [[]] * length
kg_govern_brick_out[0]  = kg_govern_brick.transpose()
kg_govern_brick_out[1]  = kg_govern_brick_i.transpose()
kg_govern_brick_out[2]  = kg_govern_brick_o.transpose()
for item in range(0,length):
    kg_govern_brick_out[item].insert(0,'material', ['brick'] * 26)
    kg_govern_brick_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_brick_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_brick_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_concrete_out  = [[]] * length
kg_govern_concrete_out[0]  = kg_govern_concrete.transpose()
kg_govern_concrete_out[1]  = kg_govern_concrete_i.transpose()
kg_govern_concrete_out[2]  = kg_govern_concrete_o.transpose()
for item in range(0,length):
    kg_govern_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_govern_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_concrete_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_wood_out  = [[]] * length
kg_govern_wood_out[0]  = kg_govern_wood.transpose()
kg_govern_wood_out[1]  = kg_govern_wood_i.transpose()
kg_govern_wood_out[2]  = kg_govern_wood_o.transpose()
for item in range(0,length):
    kg_govern_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_govern_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_wood_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_copper_out  = [[]] * length
kg_govern_copper_out[0]  = kg_govern_copper.transpose()
kg_govern_copper_out[1]  = kg_govern_copper_i.transpose()
kg_govern_copper_out[2]  = kg_govern_copper_o.transpose()
for item in range(0,length):
    kg_govern_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_govern_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_copper_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_aluminium_out  = [[]] * length
kg_govern_aluminium_out[0]  = kg_govern_aluminium.transpose()
kg_govern_aluminium_out[1]  = kg_govern_aluminium_i.transpose()
kg_govern_aluminium_out[2]  = kg_govern_aluminium_o.transpose()
for item in range(0,length):
    kg_govern_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_govern_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_aluminium_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_glass_out  = [[]] * length
kg_govern_glass_out[0]  = kg_govern_glass.transpose()
kg_govern_glass_out[1]  = kg_govern_glass_i.transpose()
kg_govern_glass_out[2]  = kg_govern_glass_o.transpose()
for item in range(0,length):
    kg_govern_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_govern_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_glass_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_glass_out[item].insert(0,'flow', [tag[item]] * 26) 


# stack into 1 dataframe
frames =    [kg_det_rur_steel_out[0], kg_det_rur_brick_out[0], kg_det_rur_concrete_out[0], kg_det_rur_wood_out[0], kg_det_rur_copper_out[0], kg_det_rur_aluminium_out[0], kg_det_rur_glass_out[0],    
             kg_sem_rur_steel_out[0], kg_sem_rur_brick_out[0], kg_sem_rur_concrete_out[0], kg_sem_rur_wood_out[0], kg_sem_rur_copper_out[0], kg_sem_rur_aluminium_out[0], kg_sem_rur_glass_out[0],    
             kg_app_rur_steel_out[0], kg_app_rur_brick_out[0], kg_app_rur_concrete_out[0], kg_app_rur_wood_out[0], kg_app_rur_copper_out[0], kg_app_rur_aluminium_out[0], kg_app_rur_glass_out[0],    
             kg_hig_rur_steel_out[0], kg_hig_rur_brick_out[0], kg_hig_rur_concrete_out[0], kg_hig_rur_wood_out[0], kg_hig_rur_copper_out[0], kg_hig_rur_aluminium_out[0], kg_hig_rur_glass_out[0],    
             kg_det_urb_steel_out[0], kg_det_urb_brick_out[0], kg_det_urb_concrete_out[0], kg_det_urb_wood_out[0], kg_det_urb_copper_out[0], kg_det_urb_aluminium_out[0], kg_det_urb_glass_out[0],    
             kg_sem_urb_steel_out[0], kg_sem_urb_brick_out[0], kg_sem_urb_concrete_out[0], kg_sem_urb_wood_out[0], kg_sem_urb_copper_out[0], kg_sem_urb_aluminium_out[0], kg_sem_urb_glass_out[0],    
             kg_app_urb_steel_out[0], kg_app_urb_brick_out[0], kg_app_urb_concrete_out[0], kg_app_urb_wood_out[0], kg_app_urb_copper_out[0], kg_app_urb_aluminium_out[0], kg_app_urb_glass_out[0],    
             kg_hig_urb_steel_out[0], kg_hig_urb_brick_out[0], kg_hig_urb_concrete_out[0], kg_hig_urb_wood_out[0], kg_hig_urb_copper_out[0], kg_hig_urb_aluminium_out[0], kg_hig_urb_glass_out[0],
             kg_office_steel_out[0],  kg_office_brick_out[0],  kg_office_concrete_out[0],  kg_office_wood_out[0],  kg_office_copper_out[0],  kg_office_aluminium_out[0],  kg_office_glass_out[0],
             kg_retail_steel_out[0],  kg_retail_brick_out[0],  kg_retail_concrete_out[0],  kg_retail_wood_out[0],  kg_retail_copper_out[0],  kg_retail_aluminium_out[0],  kg_retail_glass_out[0],
             kg_hotels_steel_out[0],  kg_hotels_brick_out[0],  kg_hotels_concrete_out[0],  kg_hotels_wood_out[0],  kg_hotels_copper_out[0],  kg_hotels_aluminium_out[0],  kg_hotels_glass_out[0],
             kg_govern_steel_out[0],  kg_govern_brick_out[0],  kg_govern_concrete_out[0],  kg_govern_wood_out[0],  kg_govern_copper_out[0],  kg_govern_aluminium_out[0],  kg_govern_glass_out[0],
                                                           
             kg_det_rur_steel_out[1], kg_det_rur_brick_out[1], kg_det_rur_concrete_out[1], kg_det_rur_wood_out[1], kg_det_rur_copper_out[1], kg_det_rur_aluminium_out[1], kg_det_rur_glass_out[1],    
             kg_sem_rur_steel_out[1], kg_sem_rur_brick_out[1], kg_sem_rur_concrete_out[1], kg_sem_rur_wood_out[1], kg_sem_rur_copper_out[1], kg_sem_rur_aluminium_out[1], kg_sem_rur_glass_out[1],    
             kg_app_rur_steel_out[1], kg_app_rur_brick_out[1], kg_app_rur_concrete_out[1], kg_app_rur_wood_out[1], kg_app_rur_copper_out[1], kg_app_rur_aluminium_out[1], kg_app_rur_glass_out[1],    
             kg_hig_rur_steel_out[1], kg_hig_rur_brick_out[1], kg_hig_rur_concrete_out[1], kg_hig_rur_wood_out[1], kg_hig_rur_copper_out[1], kg_hig_rur_aluminium_out[1], kg_hig_rur_glass_out[1],    
             kg_det_urb_steel_out[1], kg_det_urb_brick_out[1], kg_det_urb_concrete_out[1], kg_det_urb_wood_out[1], kg_det_urb_copper_out[1], kg_det_urb_aluminium_out[1], kg_det_urb_glass_out[1],    
             kg_sem_urb_steel_out[1], kg_sem_urb_brick_out[1], kg_sem_urb_concrete_out[1], kg_sem_urb_wood_out[1], kg_sem_urb_copper_out[1], kg_sem_urb_aluminium_out[1], kg_sem_urb_glass_out[1],    
             kg_app_urb_steel_out[1], kg_app_urb_brick_out[1], kg_app_urb_concrete_out[1], kg_app_urb_wood_out[1], kg_app_urb_copper_out[1], kg_app_urb_aluminium_out[1], kg_app_urb_glass_out[1],    
             kg_hig_urb_steel_out[1], kg_hig_urb_brick_out[1], kg_hig_urb_concrete_out[1], kg_hig_urb_wood_out[1], kg_hig_urb_copper_out[1], kg_hig_urb_aluminium_out[1], kg_hig_urb_glass_out[1],
             kg_office_steel_out[1],  kg_office_brick_out[1],  kg_office_concrete_out[1],  kg_office_wood_out[1],  kg_office_copper_out[1],  kg_office_aluminium_out[1],  kg_office_glass_out[1],
             kg_retail_steel_out[1],  kg_retail_brick_out[1],  kg_retail_concrete_out[1],  kg_retail_wood_out[1],  kg_retail_copper_out[1],  kg_retail_aluminium_out[1],  kg_retail_glass_out[1],
             kg_hotels_steel_out[1],  kg_hotels_brick_out[1],  kg_hotels_concrete_out[1],  kg_hotels_wood_out[1],  kg_hotels_copper_out[1],  kg_hotels_aluminium_out[1],  kg_hotels_glass_out[1],
             kg_govern_steel_out[1],  kg_govern_brick_out[1],  kg_govern_concrete_out[1],  kg_govern_wood_out[1],  kg_govern_copper_out[1],  kg_govern_aluminium_out[1],  kg_govern_glass_out[1],
            
             kg_det_rur_steel_out[2], kg_det_rur_brick_out[2], kg_det_rur_concrete_out[2], kg_det_rur_wood_out[2], kg_det_rur_copper_out[2], kg_det_rur_aluminium_out[2], kg_det_rur_glass_out[2],    
             kg_sem_rur_steel_out[2], kg_sem_rur_brick_out[2], kg_sem_rur_concrete_out[2], kg_sem_rur_wood_out[2], kg_sem_rur_copper_out[2], kg_sem_rur_aluminium_out[2], kg_sem_rur_glass_out[2],    
             kg_app_rur_steel_out[2], kg_app_rur_brick_out[2], kg_app_rur_concrete_out[2], kg_app_rur_wood_out[2], kg_app_rur_copper_out[2], kg_app_rur_aluminium_out[2], kg_app_rur_glass_out[2],    
             kg_hig_rur_steel_out[2], kg_hig_rur_brick_out[2], kg_hig_rur_concrete_out[2], kg_hig_rur_wood_out[2], kg_hig_rur_copper_out[2], kg_hig_rur_aluminium_out[2], kg_hig_rur_glass_out[2],    
             kg_det_urb_steel_out[2], kg_det_urb_brick_out[2], kg_det_urb_concrete_out[2], kg_det_urb_wood_out[2], kg_det_urb_copper_out[2], kg_det_urb_aluminium_out[2], kg_det_urb_glass_out[2],    
             kg_sem_urb_steel_out[2], kg_sem_urb_brick_out[2], kg_sem_urb_concrete_out[2], kg_sem_urb_wood_out[2], kg_sem_urb_copper_out[2], kg_sem_urb_aluminium_out[2], kg_sem_urb_glass_out[2],    
             kg_app_urb_steel_out[2], kg_app_urb_brick_out[2], kg_app_urb_concrete_out[2], kg_app_urb_wood_out[2], kg_app_urb_copper_out[2], kg_app_urb_aluminium_out[2], kg_app_urb_glass_out[2],    
             kg_hig_urb_steel_out[2], kg_hig_urb_brick_out[2], kg_hig_urb_concrete_out[2], kg_hig_urb_wood_out[2], kg_hig_urb_copper_out[2], kg_hig_urb_aluminium_out[2], kg_hig_urb_glass_out[2], 
             kg_office_steel_out[2],  kg_office_brick_out[2],  kg_office_concrete_out[2],  kg_office_wood_out[2],  kg_office_copper_out[2],  kg_office_aluminium_out[2],  kg_office_glass_out[2],
             kg_retail_steel_out[2],  kg_retail_brick_out[2],  kg_retail_concrete_out[2],  kg_retail_wood_out[2],  kg_retail_copper_out[2],  kg_retail_aluminium_out[2],  kg_retail_glass_out[2],
             kg_hotels_steel_out[2],  kg_hotels_brick_out[2],  kg_hotels_concrete_out[2],  kg_hotels_wood_out[2],  kg_hotels_copper_out[2],  kg_hotels_aluminium_out[2],  kg_hotels_glass_out[2],
             kg_govern_steel_out[2],  kg_govern_brick_out[2],  kg_govern_concrete_out[2],  kg_govern_wood_out[2],  kg_govern_copper_out[2],  kg_govern_aluminium_out[2],  kg_govern_glass_out[2]   ]

material_output = pd.concat(frames)
material_output.to_csv('output_material\\material_output_upload.csv')

##emissions from materials production
building_materials = pd.read_csv('output_material//material_output_upload.csv')
building_materials_inflow = building_materials.loc[(building_materials['flow']=='inflow')]
building_materials_inflow = building_materials_inflow.set_index('Unnamed: 0')
building_materials_outflow = building_materials.loc[(building_materials['flow']=='outflow')]
building_materials_outflow = building_materials_outflow.set_index('Unnamed: 0')

# recovery and reuse
recovery_rate = pd.read_csv('files_recovery_rate//recovery_rate.csv')
recovery_rate = recovery_rate.set_index('Unnamed: 0')
reuse_rate = pd.read_csv('files_recovery_rate//reuse_rate.csv')
reuse_rate = reuse_rate.set_index('Unnamed: 0')
materials_recovery_avaliable = building_materials_outflow.iloc[:,4:] * recovery_rate.iloc[:,4:]
materials_reuse_avaliable = building_materials_outflow.iloc[:,4:] * reuse_rate.iloc[:,4:]

# secondary material
a = building_materials_inflow.iloc[:,4:].values
b = materials_recovery_avaliable.values
c = materials_reuse_avaliable.values

materials_recovery = pd.DataFrame(np.where(a < b, a, b), index=materials_recovery_avaliable.index, columns=materials_recovery_avaliable.columns)
materials_reuse = pd.DataFrame(np.where(a < c, a, c), index=materials_reuse_avaliable.index, columns=materials_reuse_avaliable.columns)
materials_primary = building_materials_inflow - materials_recovery
materials_secondary = materials_recovery - materials_reuse

# primary material input eqauls the inflow minus recovery
materials_primary = building_materials_inflow.iloc[:,4:]-materials_recovery
emission_primary_per_kg = pd.read_csv('files_emission_factor//GHG_primary_per_kg.csv')
emission_primary_per_kg = emission_primary_per_kg.set_index('Unnamed: 0')
emission_secondary_per_kg = pd.read_csv('files_emission_factor//GHG_secondary_per_kg.csv')
emission_secondary_per_kg = emission_secondary_per_kg.set_index('Unnamed: 0')
index_emission = building_materials_inflow[['flow','type','area','material']]

#primary & secondary emission
emission_primary = materials_primary * emission_primary_per_kg.iloc[:,4:]
emission_secondary = materials_secondary * emission_secondary_per_kg.iloc[:,4:]
emission_primary = pd.concat([index_emission,emission_primary], axis=1)
emission_secondary = pd.concat([index_emission,emission_secondary], axis=1)
emission_primary = emission_primary.reset_index()
emission_primary.rename(columns={'Unnamed: 0': 'region'}, inplace=True)
emission_secondary = emission_secondary.reset_index()
emission_secondary.rename(columns={'Unnamed: 0': 'region'}, inplace=True)

#total emission
emission_primary_grouped=emission_primary.groupby(by=['region'])
emission_primary_sum=emission_primary_grouped.sum()
emission_secondary_grouped=emission_secondary.groupby(by=['region'])
emission_secondary_sum=emission_secondary_grouped.sum()
emission_total = emission_primary_sum+emission_secondary_sum

#emission data output
emission_total.to_csv('output_emission\GHG_total_upload.csv')
