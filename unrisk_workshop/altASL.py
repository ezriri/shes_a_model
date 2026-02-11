### get ASL height ###
import xesmf as xe
import xarray as xr

#e.g. hydro file (1700)
hydro = '/gws/nopw/j04/dcmex/users/dfinney/data/CASIM/dd455_4apr2024_ncs_27feb2025v2/20220730T0000Z/expt1/20220730T0000Z_LMagda_km1p5set1_expt1_pz017.nc'
hydro_nc = xr.load_dataset(hydro)

orog = '/gws/nopw/j04/dcmex/users/dfinney/data/CASIM/dd455_4apr2024_ncs_27feb2025v2/20220716T0000Z_LMagda_km1p5set1_expt1_pa000_model_orog.nc'
orography = xr.load_dataset(orog) ## orography over mountain

## silly output is sometimes on different grids
def re_grid_stuff(file_to_change, file_correct_grid):
    regridder = xe.Regridder(file_to_change, file_correct_grid, "bilinear")
    corrected_file = regridder(file_to_change)
    return corrected_file
##########

orography_regrid = re_grid_stuff(orography, hydro_nc)

altASL = hydro_nc['level_height'] + hydro_nc['sigma'] *  orography_regrid['surface_altitude']
