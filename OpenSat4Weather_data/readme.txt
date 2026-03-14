-------
The OpenSat4Weather dataset
-------
Version: 1.0 (created: 2025-07-28)
DOI: 10.5281/zenodo.16530167
Licence: https://creativecommons.org/licenses/by-sa/4.0

--------
Description of files


------
sml_data_2022.nc

The Satellite Microwave Link (SML) data are provided in a NetCDF file based on OpenSense format conventions described in the paper
Fencl M, Nebuloni R, C. M. Andersson J et al. Data formats and standards for opportunistic rainfall sensors 
[version 2; peer review: 2 approved]. Open Res Europe 2024, 3:169 (https://doi.org/10.12688/openreseurope.16068.2)

The file has two dimensions: SML identifier (sml_id) and time stamp  (time) 

Variables:
 
rsl  = 2D array (sml_id, time) storing the received signal level (dBm) by each SML at each time stamp  
time  = The time stamp (min). from the start date of the period, i.e. 2022-08-01 00:00 (UTC coordinates)
sml_id  =  alphanumeric ID of each SML
site_0_lat, site_0_lon, site_0_alt = latitude (deg), longitude (deg) and altitude amsl (m) of each SML ground terminal
satellite_azimuth,  satellite_elevation = azimuth (deg) and elevation angle (deg) of each ground-satellite link
satellite = string descriptor of the satellite transmitting the signal received by each SML ground receiver
hardware =  string identifier of the hardware version installed by the ground receiver
deg0l =  2D array (sml_id, time), storing the 0 degree altitude  above ground' (m) as taken from ERA5 hourly reanalysis data.

Note: frequency and polarization of each SML are NOT provided  in the dataset.    The rsl values of each SML are 1-min averages  of the 
signal received in horizontal polarization and over the bandwidth 11.7 - 12.75 GHz.

------
rg_data_2022.nc

The file has two dimensions:  weather station identifier (id) and time stamp  (time) 

Variables:

rainfall_amount = 2D array (id, time) storing the  accumulated rainfall_amount (mm)  in 6-min intervals by the rain gauge of each weather statuon
time  = The time stamp (min). from the start date of the period, i.e. 2022-08-01 00:00 (UTC coordinates)
id  =  alphanumeric ID of each weather station
lat, lon, elev = latitude (deg), longitude (deg) and altitude amsl (m) of each weather station
location = string descriptor of the location of every  weather station

------
radar_along_sml_data_2022.nc

The file has two dimensions:  SML dentifier (sml_id) and time stamp  (time) 

Variables:

rainfall_amount  2D array (sml_id, time) storing the  accumulated rainfall_amount (mm)  in 5-min intervals as measured by the radar over the radar pixels intersecting the projection at ground of the SML path.
time  = The time stamp (min). from the start date of the period, i.e. 2022-08-01 00:00 (UTC coordinates)
sml_id  =  alphanumeric ID of each SML



------
last updated: 01/08/2025 by Roberto Nebuloni, IEIIT-CNR,  roberto.nebuloni@cnr.it

