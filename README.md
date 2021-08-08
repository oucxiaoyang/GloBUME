# GloBUME
GloBUME models scenarios of global building material use and embodied GHG emissions between 2020-2060. It transfers the social economic scenarios in 26 global regions into the use of building materials and greenhouse gas emissions of producing these materials. It models 7 materials for 4 residential building types in two areas (rural/urban) and 4 commercial building types. More details are avaliable in the paper: Global greenhouse gas emissions from residential and commercial building materials and mitigation strategies to 2060.

Corresponding author: x.zhong@cml.leidenuniv.nl; deetman@cml.leidenuniv.nl

The dynamic material model is based on the BUMA model developed by Deetman Sebastiaan, Leiden University, the Netherlands. For the original code & latest updates, see: https://github.com/SPDeetman/BUMA

The dynamic stock model is based on the ODYM model developed by Stefan Pauliuk, Uni Freiburg, Germany. For the original code & latest updates, see: https://github.com/IndEcol/ODYM

In order to run the model please specify location of the GloBUME-main folder in 'dir_path'. Scenario analysis can be easily done in Python or Excel by customizing values of specific variables that have been well structured.

# dynamic_stock_model.py
It includes methods for efficient handling of dynamic stock models (DSMs), developed by Stefan Pauliuk, Uni Freiburg, Germany. For the original code & latest updates, see: https://github.com/IndEcol/ODYM

# GloBUME.py
It transfers the social economic scenarios in global regions into the use of building materials and emissions from the production of these materials. This is developed on the basis of the BUMA model @https://github.com/SPDeetman/BUMA.

# files_population
It includes:

* Population during 1970-2060 in 26 global regions (pop.csv)
* Rural population during 1970-2060 in 26 global regions (rurpop.csv)
* Population split by housing types and area (rural/urban) in 26 global regions (Housing_type.csv)
# files_GDP
It includes:

* GDP per capita during 1970-2060 in 26 global regions (gdp_pc.csv)
* Service value added during 1970-2060 in 26 global regions (sva_pc.csv)
# files_floor_area
It includes:

* Housing floor area per capita by region (res_Floorspace.csv)
* Housing floor area per capita by building type and region (Average_m2_per_cap.csv)
* Regression parameters for comercial floor area estimate (Gompertz_parameters.csv)
# files_lifetimes
It includes:

* Scale and shape parameters used in the weilull distribution of the residential buildings' lifetime (lifetimes.csv)
* Scale and shape parameters used in the weilull distribution of the commercial buildings' lifetime (lifetimes_comm.csv)
# files_material_density
It includes:

* Material use density in residential buildings (Building_materials. csv)
* Material use density in commercial buildings (materials_commercial_steel. csv)
# files_recovery_rate
It includes:

* Recycling rate of material scraps for secondary production (recycling_rate. csv)
* Reuse rate of material scraps (reuse_rate. csv)
# files_emission_factor
It includes:

* GHG emission factor of primary material production (GHG_primary_per_kg. csv)
* GHG emission factor of secondary material production (GHG_secondary_per_kg. csv)
* CO2 emission factor of primary material production (CO2_primary_per_kg. csv)
* CO2 emission factor of secondary material production (CO2_secondary_per_kg. csv)
# files_initial_stock
It includes:

* Assumption on the historic population development used in this model to generate the historic tail (hist_pop. csv)
# output_material
It includes the material output from running this model.

# output_emission
It includes the emission output from running this model.
