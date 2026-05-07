
import xarray as xr 
from glob import glob 
import re
import xesmf as xe


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
## make function for easy opening of files - of nc files declan has made
def open_declan_nc(date, 
                   experiment,
                   model_run_loc = '/gws/ssde/j25a/dcmex/users/dfinney/data/CASIM/dd455_4apr2024_ncs_27feb2025v2',
                   max_time = 21):
    
    """
    open up nc files declan has already made - of density + hydrometeors, and assign true height 

    Args:
        date (str): date of model run - in format 'YYYYMMDD'
        experiment (str): which experiment - e.g. 'expt1'
        model_run_loc (str, optional): location of model runs. Defaults to '/gws/ssde/j25a/dcmex/users/dfinney/data/CASIM/dd455_4apr2024_ncs_27feb2025v2'.
        max_time (int, optional): max time to include in dataset - in hours. Defaults to 21
    
    Returns:
        xarray dataset: dataset of model run, with hydrometeors and density, and true height as coordinate
    """


    day_str = f'{date}T0000Z'
    file_pattern = glob(f'{model_run_loc}/{day_str}/{experiment}/{day_str}_LMagda_km1p5set1_{experiment}_pz*.nc')
    
    # removing files after max time (if max time is not None)
    if max_time is not None:
        filtered_files = [f for f in file_pattern if int(re.search(r'_pz(\d+)', f).group(1)) < max_time]
    else:
        filtered_files = file_pattern

    # gather files of interest 
    hydro_files = sorted(f for f in filtered_files if "_density" not in f and "_w" not in f)
    updraft_files = sorted(f for f in filtered_files if "_w" in f)
    density_files = sorted(f for f in filtered_files if "_density" in f)

    # orography bits - so can assign true height
    orog_loc = '/gws/ssde/j25a/dcmex/users/ezriab/modelling/20220716T0000Z_LMagda_km1p5set1_expt1_pa000_model_orog_rgd.nc'
    plain_model = xr.open_mfdataset(density_files[0], decode_timedelta=True)
    orog_model = xr.open_mfdataset(orog_loc, decode_timedelta=True)
    true_level_height_asl = plain_model['level_height'] + plain_model['sigma'] *  orog_model['surface_altitude']


    # open up the files and combine together
    all_hydro = xr.open_mfdataset(hydro_files, decode_timedelta=True, combine='nested', concat_dim='time')
    all_density = xr.open_mfdataset(density_files, decode_timedelta=True, combine='nested',concat_dim='time')
    all_updraft = xr.open_mfdataset(updraft_files, decode_timedelta=True, combine='nested', concat_dim='time')

    ## !! need to make some alterations !!
    # density is output by the model on slightly different times, we can assume the closest times of each variable match. 
    all_density = all_density.assign_coords(time=all_hydro.time)

    # again updraft output is different times, so just use hydro time to keep simple
    all_updraft = all_updraft.assign_coords(time=all_hydro.time)

    ## updraft is also larger than hydro and density - need to slice down - and also gridded slightly differently - so fix this too
    hydro_grid_lat_min = all_hydro['grid_latitude'].min().values
    hydro_grid_lat_max = all_hydro['grid_latitude'].max().values
    hydro_grid_lon_min = all_hydro['grid_longitude'].min().values
    hydro_grid_lon_max = all_hydro['grid_longitude'].max().values
    sliced_updraft = all_updraft.sel(grid_latitude=slice(hydro_grid_lat_min, hydro_grid_lat_max), grid_longitude=slice(hydro_grid_lon_min, hydro_grid_lon_max))

    # re-grid too
    regridder = xe.Regridder(sliced_updraft, all_hydro, "bilinear")
    corrected_updraft = regridder(sliced_updraft)

    combined = xr.merge([all_hydro, all_density, corrected_updraft], compat='no_conflicts')

    # add in asl height
    combined = combined.assign_coords(true_level_height_asl=true_level_height_asl)

    return combined
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

def slice_to_domain(xr_ds, 
                    min_gridlat = -0.15, 
                    max_gridlat = 0.14,
                    min_gridlon = 359.93,
                    max_gridlon = 360.11):
    
    """ 
    slice xr dataset to main domain of interest

    Args:
        xr_ds (xarray dataset): dataset to slice
        min_gridlat (float, optional): minimum grid latitude. Defaults to -0.15.
        max_gridlat (float, optional): maximum grid latitude. Defaults to 0.14.
        min_gridlon (float, optional): minimum grid longitude. Defaults to 359.93.
        max_gridlon (float, optional): maximum grid longitude. Defaults to 360.11.
    
    Returns:
        xarray dataset: sliced dataset
    """
    
    sliced_domain = xr_ds.sel(grid_longitude=slice(min_gridlon, max_gridlon), grid_latitude=slice(min_gridlat, max_gridlat))
    return sliced_domain

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
def make_variable(big_xr,
                  var_name,
                  number_or_mass = 'number',
                  m3 = True,
                  L = True,
                  density_name = 'Density used for rad layers (kg m-3)'):
    """
    function to make new variable for a given hydrometeor type, e.g. graupel, cloud droplets, ice, snow, rain
    This also converts units

    Args:
        big_xr (xarray dataset): the dataset to add the new variable to
        var_name (str): the hydrometeor type, e.g. 'graupel', 'cloud', 'ice', 'snow', 'rain'
        number_or_mass (str): whether to calculate number concentration or mass concentration, options are 'number' or 'mass'
        m3 (bool): whether to calculate concentration in m-3 (default True)
        L (bool): whether to calculate concentration in L-1 (default True)
        density_name (str): the name of the density variable in big_xr to use for conversion (default 'Density used for rad layers (kg m-3)')

    Returns:
        big_xr (xarray dataset): the input dataset with the new variable(s) added
    """

    # 
    if number_or_mass == 'number':
        var_dict = {'graupel': 'number_of_graupel_particles_per_kg_of_air',
                    'cloud' : 'number_of_cloud_droplets_per_kg_of_air',
                    'ice' : 'number_of_ice_particles_per_kg_of_air',
                    'snow' : 'number_of_snow_aggregates_per_kg_of_air',
                    'rain' : 'number_of_rain_drops_per_kg_of_air'}
    else:
        # mass values
        var_dict = {'graupel': 'mass_fraction_of_graupel_in_air',
                    'cloud' : 'mass_fraction_of_cloud_liquid_water_in_air',
                    'ice' : 'mass_fraction_of_cloud_ice_in_air',
                    'snow' : 'mass_fraction_of_cloud_ice_crystals_in_air',
                    'rain' : 'mass_fraction_of_rain_in_air'}
    
    var_casim_name = var_dict[var_name]

    if m3:
        units = {'number': 'm-3', 'mass': 'kg m-3'}
        new_var_name = f'{var_name}_{number_or_mass}_m3'
        big_xr[new_var_name] = big_xr[var_casim_name] * big_xr[density_name]
        big_xr[new_var_name].attrs["units"] = units[number_or_mass]
        big_xr[new_var_name].attrs["long_name"] = f"{var_name} {number_or_mass} concentration ({units[number_or_mass]})"

    if L:
        units = {'number': 'L-1', 'mass': 'kg L-1'}
        new_var_name_l = f'{var_name}_{number_or_mass}_L'
        big_xr[new_var_name_l] = big_xr[f'{var_name}_{number_or_mass}_m3'] * 1e-3
        big_xr[new_var_name_l].attrs["units"] = units[number_or_mass]
        big_xr[new_var_name_l].attrs["long_name"] = f"{var_name} {number_or_mass} concentration ({units[number_or_mass]})"
    
    return big_xr
