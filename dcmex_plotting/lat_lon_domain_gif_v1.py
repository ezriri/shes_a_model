## (kind of slightly rough code, for making domain max gifs - and can layer up variables)

from matplotlib.colors import PowerNorm
import matplotlib as mpl
import xarray as xr 
import pandas as pd
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 
from scipy.special import gamma, gammainc
import re
import xesmf as xe
import os
import imageio
from matplotlib.lines import Line2D


import sys
# needed so I can call the functions from another folder.
sys.path.append('/home/users/esree/shes_a_model/model_functions/')
# loads of homemade functions <3 <3
from casim_boring_functions import open_declan_nc, make_variable, slice_to_domain

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
### inputs
a_date = '20220730'
a_experiment = 'expt1'
max_time = 22 # can specify a max time to load up to, or just load whole thing

variable_1 = 'ice_number_L' # this will be a heatmap
# think use a different colour if using specific vmax
variable_1_colour = ['Purples_r']#None#['Greens_r']#None #['coolwarm']
variable_1_vmax = 100 #1000
log_colourbar = False

# contours added on - max 2 different variables atm
variable_2 = 'air_temperature_c' # can pass None if dont want any contours
variable_2_contour_levels = [-7.5, -2.5, -38]
variable_2_colours = ['red', 'yellow', 'black']

variable_3 = 'upward_air_velocity'
variable_3_contour_levels = [2, 5]
variable_3_colours = ['cyan', 'lime']

delete_pngs = True # delete pngs after gif made

save_folder = '/home/users/esree/shes_a_model/dcmex_plotting/plot_saving/lat_lon_birdseye/'

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
new_combined = open_declan_nc(a_date, a_experiment, max_time = max_time)
sliced_domain = slice_to_domain(new_combined)

## make new units for some hydrometeors
new_hydro_list = ['graupel', 'cloud', 'ice', 'snow', 'rain']
for hydro in new_hydro_list:
    sliced_domain = make_variable(sliced_domain, hydro)

for hydro in new_hydro_list:
    sliced_domain = make_variable(sliced_domain, hydro, 'mass')


## also make temperature - ºC
sliced_domain['air_temperature_c'] = sliced_domain['air_temperature'] - 273.15
sliced_domain['air_temperature_c'].attrs["units"] = 'Celsius'


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
def easy_plotting_lat_lon_map(a_time_step, 
                              variable_name = 'orography', 
                              colourmesh_contour = 'colourmesh',
                              contour_level = [2250],
                              a_vmax = None,
                              a_vmin = None,
                              max_alt = 15000,
                              specific_colourmap = None,
                              multi_contour = None):
    ### this needs to be placed after the defined plot, just kind of slots into place
    ## can layer up plots

    """
    easy stacking plotting function for lat / lon maps, can overlay different things?

    Args:
        a_time_step (xarray dataset): a single time step of the model output, soelected using .sel(time=...)
        variable_name (str, optional): the variable you want to plot. Defaults to 'orography'.
        colourmesh_contour (str, optional): whether you want to plot the variable as a colourmesh or a contour. Defaults to 'colourmesh'.
        contour_level (list, optional): if plotting as a contour, the level(s) to plot. Defaults to [2250].
        vmax (float, optional): max value for colourmesh. Defaults to None.
        vmin (float, optional): min value for colourmesh. Defaults to None.
        var_units (str, optional): units of the variable being plotted, for colourbar labelling. Defaults to None.
        max_alt (int, optional): max altitude to plot. Defaults to 15000.
        specific_colourmap (list, optional): colourmap to use for colourmesh. Defaults to None, which will use the default colourmap.
        multi_contour (list, optional): .
    """

    min_lat = 34.2
    max_lat = 33.8
    min_lon = -107.4
    max_lon = -107.0

    # then for each grid lat / lon, we choose the max height
    lon_1d = a_time_step["lon"].mean(dim="grid_latitude") #(grid_longitude,)
    lat_1d = a_time_step["lat"].mean(dim="grid_longitude") #(grid_latitude,)

    if variable_name == 'orography':
        orography = a_time_step["true_level_height_asl"].isel(model_level_number=0)
        max_height_lat = orography.max(dim="grid_latitude") # function of longitude
        max_height_lon = orography.max(dim="grid_longitude") # fuction of latitude
        
        ax1.plot(lon_1d, max_height_lat, color="grey")
        ax4.plot(max_height_lon, lat_1d, color="grey")
        ax3.contour(
            a_time_step["lon"].values,
            a_time_step["lat"].values,
            orography.values,
            levels=contour_level,   # only this level
            colors="grey",
            linewidths=2)
    
    elif variable_name != 'orography': 
        lat_max = a_time_step[variable_name].max(dim="grid_latitude") # (model_level_number, grid_longitude)
        z_lat_max = a_time_step["true_level_height_asl"].max(dim="grid_latitude") # (model_level_number, grid_longitude)
        lon_max = a_time_step[variable_name].max(dim="grid_longitude") # (model_level_number, grid_latitude)
        z_lon_max = a_time_step["true_level_height_asl"].max(dim="grid_longitude") # (model_level_number, grid_latitude)
        z_max = a_time_step[variable_name].max(dim="model_level_number") # (grid_latitude, grid_longitude)
        
        lon_2d = np.tile(lon_1d.values,(lat_max.shape[0], 1))
        lat_2d = np.tile(lat_1d.values,(lon_max.shape[0], 1)).T

        if variable_name == 'upward_air_velocity':
            var_units = 'm/s'
        else:
            var_units = a_time_step[variable_name].units

        
        if colourmesh_contour == 'colourmesh':
            cmap = specific_colourmap[0] if specific_colourmap is not None else "Blues_r"
            
            if log_colourbar:
                norm = PowerNorm(gamma=0.5, vmin=a_vmin, vmax=a_vmax)

                common_kwargs = dict(
                    shading="auto",
                    cmap=cmap,
                    norm=norm)
            else:
                norm = None

                common_kwargs = dict(
                    shading="auto",
                    cmap=cmap,
                    vmin=a_vmin,
                    vmax=a_vmax)# Choose normalization
            
            ax1.pcolormesh(
                lon_1d.values,
                z_lat_max.values,
                lat_max.values,
                **common_kwargs)

            pcm = ax3.pcolormesh(
                a_time_step["lon"].values,
                a_time_step["lat"].values,
                z_max.values,
                **common_kwargs)

            ax4.pcolormesh(
                z_lon_max.T,
                lat_1d,
                lon_max.T,
                **common_kwargs)

            # Colorbar source
            cbar_source = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) if norm else pcm

            cbar = fig.colorbar(
                cbar_source,
                ax=[ax1, ax3, ax4],
                orientation="vertical",
                location="left",
                pad=-0.0001,
                fraction=0.03
            )

            cbar.set_label(f"max {variable_name} ({var_units})")

        
        elif colourmesh_contour != 'colourmesh':
            number_of_contours = len(contour_level)
            # contour plot
            if specific_colourmap is None or len(specific_colourmap) < number_of_contours:
                specific_colourmap = ['red', 'yellow', 'blue', 'green', 'orange']
            
            if multi_contour is None:
                legend_lines = []
            else:
                legend_lines = list(multi_contour)
            
            if variable_name == 'air_temperature_c':
                plt_linestyle = 'dashed'
            else:
                plt_linestyle = 'solid'

            for i in range(len(contour_level)):
                a_contour_level = contour_level[i]
                a_colour = specific_colourmap[i]
            
                ax1.contour(
                    lon_2d,
                    z_lat_max.values,
                    lat_max.values,
                    levels=[a_contour_level],
                    colors=a_colour,
                    alpha=0.6,
                    linestyles=plt_linestyle,
                    linewidths=2,)

                # so get ledgend
                ax3.contour(
                    z_max["lon"].values,
                    z_max["lat"].values,
                    z_max.values,
                    levels=[a_contour_level],
                    colors=a_colour,
                    alpha=0.6,
                    linestyles=plt_linestyle,
                    linewidths=2)
            
                ax4.contour(
                    z_lon_max.T,
                    lat_2d,
                    lon_max.T,
                    levels=[a_contour_level],
                    colors=a_colour,
                    alpha=0.6,
                    linestyles=plt_linestyle,
                    linewidths=2)
            
                legend_lines.append(Line2D(
                    [0], [0],
                    color=a_colour,
                    lw=2,
                    label=f"{variable_name} = {a_contour_level} {var_units}"))
            
            ax2.legend(handles=legend_lines, loc= "lower center")
            
    
    ax1.set_ylabel("alt (m)")
    ax1.set_ylim(0, max_alt)
    ax1.set_xlim(min_lon, max_lon)
    ax1.tick_params(axis="x", bottom=False, labelbottom=False) # turn off x-axis ticks and labels for ax1

    ax3.set_xlim(min_lon, max_lon)
    ax3.set_ylim(max_lat, min_lat)
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    ax4.set_xlabel("alt (m)")
    ax4.set_xlim(0, max_alt)
    ax4.set_ylim(max_lat, min_lat)
    ax4.tick_params(axis="y", left=False, labelleft=False) # turn off y-axis ticks and labels for ax4

    if colourmesh_contour != 'colourmesh':
        return legend_lines # this can be fed into next contour plot - build up 


def lat_lon_map_temp_contour(a_time_step,
                              contour_level = [-3, -10],
                              max_alt = 15000,
                              specific_colourmap = None):
    ### this needs to be placed after the defined plot, just kind of slots into place
    ## can layer up plots

    """
    **** FOR TEMPERATURE VAR - using minimum ****
    easy stacking plotting function for lat / lon maps, can overlay different things?

    Args:
        a_time_step (xarray dataset): a single time step of the model output, soelected using .sel(time=...)
        contour_level (list, optional): if plotting as a contour, the level(s) to plot. Defaults to [2250].
        max_alt (int, optional): max altitude to plot. Defaults to 15000.
        specific_colourmap (list, optional): colourmap to use for colourmesh. Defaults to None, which will use the default colourmap.
    """

    min_lat = 34.2
    max_lat = 33.8
    min_lon = -107.4
    max_lon = -107.0

    # then for each grid lat / lon, we choose the max height
    lon_1d = a_time_step["lon"].mean(dim="grid_latitude") #(grid_longitude,)
    lat_1d = a_time_step["lat"].mean(dim="grid_longitude") #(grid_latitude,)

    lat_max = a_time_step['air_temperature_c'].min(dim="grid_latitude") # (model_level_number, grid_longitude)
    z_lat_max = a_time_step["true_level_height_asl"].min(dim="grid_latitude") # (model_level_number, grid_longitude)
    lon_max = a_time_step['air_temperature_c'].min(dim="grid_longitude") # (model_level_number, grid_latitude)
    z_lon_max = a_time_step["true_level_height_asl"].min(dim="grid_longitude") # (model_level_number, grid_latitude)
    z_max = a_time_step['air_temperature_c'].min(dim="model_level_number") # (grid_latitude, grid_longitude)
    
    lon_2d = np.tile(lon_1d.values,(lat_max.shape[0], 1))
    lat_2d = np.tile(lat_1d.values,(lon_max.shape[0], 1)).T

    var_units = a_time_step['air_temperature_c'].units

    number_of_contours = len(contour_level)
    # contour plot
    if specific_colourmap is None or len(specific_colourmap) < number_of_contours:
        specific_colourmap = ['red', 'yellow', 'blue', 'green', 'orange']
    legend_lines = []

    for i in range(len(contour_level)):
        a_contour_level = contour_level[i]
        a_colour = specific_colourmap[i]
    
        ax1.contour(
            lon_2d,
            z_lat_max.values,
            lat_max.values,
            levels=[a_contour_level],
            colors=a_colour,
            linewidths=2,)

        # so get ledgend
        ax3.contour(
            z_max["lon"].values,
            z_max["lat"].values,
            z_max.values,
            levels=[a_contour_level],
            colors=a_colour,
            linewidths=2)
    
        ax4.contour(
            z_lon_max.T,
            lat_2d,
            lon_max.T,
            levels=[a_contour_level],
            colors=a_colour,
            linewidths=2)
    
        legend_lines.append(Line2D(
            [0], [0],
            color=a_colour,
            lw=2,
            label=f"temp = {a_contour_level} {var_units}"))
        
    ax2.legend(handles=legend_lines, loc= "lower center")
        
    
    ax1.set_ylabel("alt (m)")
    ax1.set_ylim(0, max_alt)
    ax1.set_xlim(min_lon, max_lon)
    ax1.tick_params(axis="x", bottom=False, labelbottom=False) # turn off x-axis ticks and labels for ax1

    ax3.set_xlim(min_lon, max_lon)
    ax3.set_ylim(max_lat, min_lat)
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    ax4.set_xlabel("alt (m)")
    ax4.set_xlim(0, max_alt)
    ax4.set_ylim(max_lat, min_lat)
    ax4.tick_params(axis="y", left=False, labelleft=False) # turn off y-axis ticks and labels for ax4

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
specif_save_folder = f'{save_folder}{a_date}_{a_experiment}/{variable_1}'
if not os.path.exists(specif_save_folder):
    os.makedirs(specif_save_folder)
saving_location = specif_save_folder+'/'

# global things defined for the dataset
time_list = sliced_domain["time"].values
if variable_1_vmax is None:
    vmax_n = sliced_domain[variable_1].max().values
else:
    vmax_n = variable_1_vmax
vmin_n = sliced_domain[variable_1].min().values


for t in time_list:
    a_time = sliced_domain.sel(time=t)
    fig = plt.figure(figsize=(10, 8))

    # Create a 3x3 logical grid
    gs = fig.add_gridspec(3, 3)

    # Subplots with custom spans:
    ax1 = fig.add_subplot(gs[0, 0:2])   # top-left → 2 cols × 1 row
    ax2 = fig.add_subplot(gs[0, 2])     # top-right → 1 col × 1 row
    ax2.axis("off")
    ax3 = fig.add_subplot(gs[1:3, 0:2]) # bottom-left → 2 cols × 2 rows
    ax4 = fig.add_subplot(gs[1:3, 2])   # bottom-right → 1 col × 2 rows

    save_name = f"{pd.to_datetime(str(t)).strftime('%y%m%d_%H%M%S')}_{variable_1}"
    non_time_save_name = f'{variable_1}'
    
    easy_plotting_lat_lon_map(a_time) # orography
    easy_plotting_lat_lon_map(a_time, variable_name=variable_1, a_vmin=vmin_n, a_vmax=vmax_n, specific_colourmap=variable_1_colour)
    
    if variable_2 is not None:
        ledg_handles = easy_plotting_lat_lon_map(a_time, variable_name=variable_2, colourmesh_contour='contour', contour_level= variable_2_contour_levels , specific_colourmap=variable_2_colours)
        save_name += f"_{variable_2}_contour"
        non_time_save_name += f"_{variable_2}_contour"

   
    if variable_3 is not None:
        ledg_handles = easy_plotting_lat_lon_map(a_time, variable_name=variable_3, colourmesh_contour='contour', contour_level= variable_3_contour_levels , specific_colourmap=variable_3_colours, multi_contour = ledg_handles)
        save_name += f"_{variable_3}_contour"
        non_time_save_name += f"_{variable_3}_contour"
    
    ## set time 
    time_name = pd.to_datetime(str(t)).strftime("%H:%M:%S")
    title_str = f'{variable_1} \n {time_name} UTC'
    ax2.set_title(title_str, loc="center", y=0.8, fontsize=16)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(saving_location+save_name, bbox_inches='tight')
    plt.close()

images_loc = saving_location+f'*{non_time_save_name}.png'

## making gif
image_list = glob(images_loc)
image_list.sort()
frames = []
for i in range(len(image_list)):
    image = imageio.v2.imread(image_list[i])
    frames.append(image)

gif_save_name = f'{a_date}_{a_experiment}_{variable_1}'

if variable_1_vmax is not None:
    gif_save_name += f'_{variable_1_vmax}' # upper limit for colourmesh - to distinguish between different gifs if using same variable but different limits

if variable_2 is not None:
    gif_save_name += f'_{variable_2}_contour'

if variable_3 is not None:
    gif_save_name += f'_{variable_3}_contour'

imageio.mimsave(saving_location+f'{gif_save_name}.gif', frames, fps=5)
if delete_pngs:
    for image in image_list:
        os.remove(image)

print('done')