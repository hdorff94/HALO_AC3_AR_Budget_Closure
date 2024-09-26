# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:09:04 2024

@author: Henning Dorff
"""
#Base modules
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import itertools
# Plotting modules
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature
from cartopy.mpl.patch import geos_to_path

from cmcrameri import cm as cmaeri
from matplotlib import cm
from matplotlib.collections import LineCollection, PolyCollection

from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

start_path=os.getcwd()
plot_path=start_path+"/../plots/"
ac3_scripts_path=start_path+"/../scripts/"
my_git_path=start_path+"/../../"
major_work_path=my_git_path+"/../Work/GIT_Repository/"
synth_ar_path=my_git_path+"/Synthetic_Airborne_Arctic_ARs/"
hamp_processing_path=my_git_path+"/hamp_processing_python/"
config_path=synth_ar_path+"config/"
sys.path.insert(1,config_path)
sys.path.insert(2,ac3_scripts_path)
sys.path.insert(3,start_path+"/../src/")
sys.path.insert(4,synth_ar_path+"/src/")
sys.path.insert(5,synth_ar_path+"/plotting/")
sys.path.insert(6,hamp_processing_path)
sys.path.insert(7,hamp_processing_path+"/src/")
sys.path.insert(8,hamp_processing_path+"/plotting/")
sys.path.insert(9,start_path+"/../plotting/")
import data_config
###########################################################################
import flightcampaign

###########################################################################
#Grid Data
import reanalysis as Reanalysis
from reanalysis import ERA5,CARRA
from ICON import ICON_NWP as ICON
import gridonhalo as Grid_on_HALO
from atmospheric_rivers import Atmospheric_Rivers
###########################################################################
#Radar data
import data_config
import measurement_instruments_ql
import quicklook_dicts
import halodataplot
###########################################################################
# Collocated data
from simplified_flight_leg_handling import simplified_run_grid_main
###########################################################################
# Moisture Budget Packages
import moisturebudget as Budgets
# Plot Packages
import Airborne_Budget_Plots as Airplots
import matplotlib
        
def map_AR_sea_ice(cfg_dict, radar_ds):
    orig_map = plt.cm.get_cmap('Blues') # getting the original colormap using cm.get_cmap() function
    reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function
                                            # normally the actual bahamas file is used from HALO-(AC)3. However,
                                            # this is not feasible now for testing

    add_quiver=True
    #sea_ice_file=cfg_dict["device_data_path"]+"sea_ice/"+\
    #                    "asi-AMSR2-n6250-"+cfg_dict["date"]+"-v5.4.nc"

    #seaice = xr.open_dataset(sea_ice_file)
    #seaice = seaice.seaice
    # Create a Stamen terrain background instance.
    #stamen_terrain = StadiaStamen('terrain-background')
    class StadiaStamen(cimgt.Stamen):
        def _image_url(self, tile):
            x,y,z = tile
            url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.jpg?api_key=0963bb5f-6e8c-4978-9af0-4cd3a2627df9"
            return url
    stamen_terrain = StadiaStamen('terrain-background')

    with_sondes=True    
    llcrnlat = 68
    llcrnlon = -10
    urcrnlat = 82
    urcrnlon = 20

    extent = [llcrnlon-5, urcrnlon+5, llcrnlat, urcrnlat]
    coordinates=dict(EDMO=(11.28, 48.08), 
                         Keflavik=(-22.6307, 63.976),
                         Kiruna=(20.336, 67.821),
                         Bergen=(5.218, 60.293),
                         Longyearbyen=(15.46, 78.25),
                         Lerwick=(-1.18, 60.13),
                         Ittoqqortoormiit=(-21.95, 70.48),
                         Tasiilaq=(-37.63, 65.60))
    # get plot properties
    #matplotlib.rcParams.update({"font.size":12})
    # start plotting
    fig, ax_2d = plt.subplots(figsize=(10,8), 
                subplot_kw={"projection": ccrs.NorthPolarStereo()})
    ax_2d.add_image(stamen_terrain, 4)
    
    sector_colors = {"warm_sector": "orange", 
                     "cold_sector": "purple",
                     "internal": "grey"}

    orig_map = plt.cm.get_cmap('Blues')
    reversed_map = orig_map.reversed()

    seaice_file = cfg_dict["device_data_path"] + "sea_ice/asi-AMSR2-n6250-" + cfg_dict["date"] + "-v5.4.nc"
    seaice = xr.open_dataset(seaice_file).seaice

    #llcrnlat = 74
    #llcrnlon = -10
    #urcrnlat = 80
    #urcrnlon = 20
    extent = [llcrnlon - 5, urcrnlon + 5, llcrnlat, urcrnlat]

    ax_2d.add_image(stamen_terrain, 6)
    ax_2d.coastlines(resolution="50m")
    ax_2d.add_feature(cartopy.feature.BORDERS)
    
    ax_2d.set_extent(extent, crs=ccrs.PlateCarree())
    pmesh = ax_2d.pcolormesh(seaice.lon, seaice.lat, seaice, transform=ccrs.PlateCarree(), cmap=reversed_map, alpha=0.9)
    ax_2d.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax_2d.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                          x_inline=False, y_inline=False)
    gl.bottom_labels = True
    gl.right_labels  = False
    gl.top_labels    = False
    # add sea ice extent
    
    x1,y1 = coordinates["Kiruna"]   
    x2, y2 =coordinates["Longyearbyen"]
    ax_2d.plot(x1, y1, '.r', markersize=15, markeredgecolor="k",
            transform=ccrs.PlateCarree(),zorder=10)

    
    ax_2d.text(x1 - 2, y1 + 0.9, "Kiruna (KRN)", fontsize=9,
             transform=ccrs.PlateCarree(),color="red",
             bbox=dict(facecolor='lightgrey',edgecolor="black"),zorder=12)

    return fig,ax_2d  # Return pcolormesh to control the display later if needed

# Function to create the 3D radar view
def map_3d_radar_view(ax, processed_radar, Dropsondes, radar_ds_mean_z,
                      relevant_sondes_dict,inflow_times,
                      internal_times,outflow_times,):
    # Radar period
    curved_radar=pd.DataFrame(data=np.array(processed_radar["dBZg"].values[:]),
            index=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:])),
            columns=np.array(processed_radar["height"].values[:]))

    curved_radar = curved_radar[~curved_radar.index.duplicated(keep='first')]
    # This import registers the 3D projection, but is otherwise unused.
    x, y, z = processed_radar["lon"], \
        processed_radar["lat"], processed_radar["alt"]
    z+=200
    x_contour=np.tile(x,(len(processed_radar["height"]),1)).T
    y_contour=np.tile(y,(len(processed_radar["height"]),1)).T

    z_contour_1d=np.array(processed_radar["height"][:])
    z_contour=np.tile(z_contour_1d,(len(processed_radar["time"]),1))
    
    # Set up plot
    inflow_st_ind=curved_radar.index.get_loc(inflow_times[0]).start
    inflow_end_ind=curved_radar.index.get_loc(inflow_times[-1]).start
    internal_st_ind=curved_radar.index.get_loc(internal_times[0]).start
    internal_end_ind=curved_radar.index.get_loc(internal_times[-1]).start
    outflow_st_ind=curved_radar.index.get_loc(outflow_times[0]).start
    outflow_end_ind=curved_radar.index.get_loc(outflow_times[-1]).start

    # Plot dropsonde profiles
    sector_colors={"warm_sector":"orange",
                  "cold_sector":"blue",
                  "internal":"grey"}
    
    ax.scatter(x_contour[inflow_st_ind:inflow_end_ind,:].flatten(),
        y_contour[inflow_st_ind:inflow_end_ind,:].flatten(),
        z_contour[inflow_st_ind:inflow_end_ind,:].flatten()/1000,s=0.1,
        c=np.array(curved_radar.iloc[inflow_st_ind:inflow_end_ind,:].values[:]),
        cmap=cmaeri.roma_r,vmin=-30,vmax=30,zorder=2)
    
    ax.scatter(x_contour[internal_st_ind:internal_end_ind,:].flatten(),
        y_contour[internal_st_ind:internal_end_ind,:].flatten(),
        z_contour[internal_st_ind:internal_end_ind,:].flatten()/1000,s=0.1,
        c=np.array(curved_radar.iloc[internal_st_ind:internal_end_ind,:].values[:]),
        cmap=cmaeri.roma_r,vmin=-30,vmax=30,
               zorder=1)

    # Often Unused for optic reasons
    ax.scatter(x_contour[outflow_st_ind:outflow_end_ind,:].flatten(),y_contour[outflow_st_ind:outflow_end_ind,:].flatten(),
           z_contour[outflow_st_ind:outflow_end_ind,:].flatten()/1000,s=0.1,
           c=np.array(curved_radar.iloc[outflow_st_ind:outflow_end_ind,:].values[:]),
           cmap=cmaeri.roma_r,vmin=-30,vmax=30,zorder=0)

    #ax.plot(processed_radar["lon"],processed_radar["lat"],
    #        z.mean()/1000,color="grey",linewidth=2,ls="-",zorder=4)
    ax.plot(x, y, z.mean()/1000+0.4,linewidth=1,ls="--",color="grey",zorder=4)
    ax.plot(processed_radar["lon"][inflow_st_ind:outflow_end_ind],
            processed_radar["lat"][inflow_st_ind:outflow_end_ind],
            z.mean()/1000+0.4,color="w",lw=6,zorder=5)
    ax.plot(processed_radar["lon"][inflow_st_ind:inflow_end_ind],
            processed_radar["lat"][inflow_st_ind:inflow_end_ind],
            z.mean()/1000+0.4,color="purple",lw=3,zorder=6)
    ax.plot(processed_radar["lon"][outflow_st_ind:outflow_end_ind],
            processed_radar["lat"][outflow_st_ind:outflow_end_ind],
            z.mean()/1000+0.4,color="purple",lw=3,zorder=6)
    ax.plot(processed_radar["lon"][inflow_end_ind:outflow_st_ind],
            processed_radar["lat"][inflow_end_ind:outflow_st_ind],
            z.mean()/1000+0.4,color="grey",lw=3,zorder=6)    
    for sector in ["warm_sector","cold_sector","internal"]:
            if not sector=="internal":
                for in_out in ["in","out"]:
                    sonde_no=relevant_sondes_dict[sector][in_out]
                    relevant_Sondes=Dropsondes["Lon"].iloc[sonde_no]

                    ax.scatter(Dropsondes["Lon"].iloc[sonde_no],
                        Dropsondes["Lat"].iloc[sonde_no],z.mean()/1000+0.4,
                        color=sector_colors[sector],marker="v",
                              s=250,edgecolor="k",lw=3,zorder=10)
            else:
                sonde_no=relevant_sondes_dict[sector],
                ax.scatter(Dropsondes["Lon"].iloc[sonde_no],
                        Dropsondes["Lat"].iloc[sonde_no],
                        z.mean()/1000+0.4,
                        color=sector_colors[sector],marker="v",
                        s=250,edgecolor="k",lw=3,zorder=10)
                
    #    # Set view angle
    #ax.xaxis._axinfo['juggled'] = (1,2,0) #---> to be uncommented
    #ax.view_init(25,280)
    ax.set_zlim([0,12])
    ax.set_xlim([processed_radar["lon"].values.min(),
                 processed_radar["lon"].values.max()])
    ax.set_ylim([processed_radar["lat"].values.min(),processed_radar["lat"].values.max()])
    # changing grid lines thickness of x axis to 1
    # Get rid of colored axes planes
    # First remove fill#
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.xaxis._axinfo["grid"].update({"linewidth":1,"color":"w"})
    ax.yaxis._axinfo["grid"].update({"linewidth":2,"color":"w"})
    ax.zaxis._axinfo["grid"].update({"linewidth":2,"color":"w"})
    ax.xaxis.set_tick_params(width=10,color="grey")
    ax.yaxis.set_tick_params(width=0,color="grey")
    ax.zaxis.set_tick_params(width=10,color="grey")

    ax.set_zlabel("Height (km)",linespacing=3.1)

    
    ax.set_xlabel("Longitude (°E)",color="grey")
    ax.set_ylabel("Latitude (°N)",color="grey")
    ax.set_zlabel("Height (km)",color="grey")
    ax.view_init(40, 290)  # Set view angle
    ax.set_xlim([-15,10])#[radar_ds["lon"].min(), radar_ds["lon"].max()])
    ax.set_yticks([72,74,76])
    ax.set_ylim([71,78])
    
    #ax_3d.set_ylim([radar_ds["lat"].min(), radar_ds["lat"].max()])
    ax.set_zlim([0, 12])  # Adjust this based on your radar height
    
# Main function to create the combined figure
def combined_plot(cfg_dict, radar_ds, Dropsondes,relevant_sondes_dict,
                  inflow_times,internal_times,outflow_times,
                  add_2d=False,add_3d=True):
    fig = plt.figure(figsize=(24, 6))
    # Get z measurements
    z_mean_tmp = radar_ds["alt"].values[:].mean()
    z_mean = pd.Series(data=z_mean_tmp,index=np.arange(radar_ds["alt"].shape[0]))

    if add_2d:
        matplotlib.rcParams.update({"font.size":18})
        fig,ax_2d=map_AR_sea_ice(cfg_dict, radar_ds)

        # Create map boundaries and grid lines
        ax_2d.tick_params(labelcolor="none")  # Don't draw tick labels
        ax_2d.set_xticks([])  # Remove x ticks
        ax_2d.set_yticks([])  # Remove y ticks
        ax_2d.spines['top'].set_color('none')  # Remove box top spine
        ax_2d.spines['right'].set_color('none')  # Remove box right spine
        ax_2d.spines['left'].set_color('none')  # Remove box left spine
        ax_2d.spines['bottom'].set_color('none')  # Remove box bottom spine
        fig_name=os.getcwd()+"/../plots/Fig03_sea_ice_background.png"
        fig.savefig(fig_name, dpi=300, bbox_inches="tight")
        print("Figure saved as:", fig_name)
        # Map only the z values onto the 3D plot
    if add_3d:
            # Create 3D plot
        ax_3d = fig.add_subplot(111, projection='3d')

        map_3d_radar_view(ax_3d, radar_ds, Dropsondes, z_mean, 
            relevant_sondes_dict,inflow_times,internal_times,outflow_times)
        ax_3d.grid(False)
        ax_3d.spines['top'].set_color('none')  # Remove box top spine
        ax_3d.spines['right'].set_color('none')  # Remove box right spine
        ax_3d.spines['left'].set_color('none')  # Remove box left spine
        ax_3d.spines['bottom'].set_color('none')  # Remove box bottom spine
        
        fig_name=os.getcwd()+"/../plots/fig03_radar.png"
        fig.savefig(fig_name,dpi=300,transparent=True)
        print("Figure saved as:",fig_name)
    plt.tight_layout()
    #plt.show()
def main(ar_of_day="AR_entire_1",flight=["RF05"]):
    # Airborne configuration
    campaign="HALO_AC3"
    flight_dates={"RF05":"20220315","RF16":"20220410"}
    #-------------------------------------------------------------------------------------------------#
    flight_sequence=["RF05__AR_entire_1","RF05__AR_entire_2",
                     "RF06__AR_entire_1","RF06__AR_entire_2"]
    radar_sequence={}
    reflectivity_for_snow="Z_g"
    sector_to_plot="warm"
    take_arbitary=True
    do_plotting=False
    calibrated_radar=True
    print("Load and process data")
    with HiddenPrints():
        halo_era5,halo_df,cmpgn_cls,ERA5_on_HALO,radar,Dropsondes=\
            simplified_run_grid_main(flight=flight,config_file_path=\
                                 major_work_path,ar_of_day=ar_of_day)
        for rf in flight_sequence:
            ar_rf=rf.split("__")[1]
            flight_rf=[rf.split("__")[0]]
            tpm_era5,tmp_halo,tmp_cls,tmp_era5_cls,radar_sequence[rf],_=\
                simplified_run_grid_main(flight=flight_rf,
                    config_file_path=major_work_path,ar_of_day=ar_rf)
    
    inflow_times,internal_times,outflow_times=\
        cmpgn_cls.define_budget_legs(flight,ar_of_day)
    new_halo_dict={flight[0]:{"inflow":halo_df.loc[inflow_times[0]:inflow_times[-1]],
                          "internal":halo_df.loc[internal_times[0]:internal_times[-1]],
                          "outflow":halo_df.loc[outflow_times[0]:outflow_times[-1]]}}

    AR_inflow,AR_outflow=Atmospheric_Rivers.locate_AR_cross_section_sectors(
                                    new_halo_dict,ERA5_on_HALO.halo_era5,
                                    flight[0])
    for sector in ["warm_sector","core","cold_sector"]:
        AR_sector_in                       = AR_inflow["AR_inflow_"+sector]
        AR_sector_out                      = AR_outflow["AR_outflow_"+sector]
    relevant_sondes_dict={}
    sonde_times_series=pd.Series(index=Dropsondes["IWV"].index.values,
                                 data=range(Dropsondes["IWV"].shape[0]))

    relevant_warm_sector_sondes=[0,1,2,3,9,10,11,12]
    relevant_cold_sector_sondes=[4,5,6]
    relevant_internal_sondes=[7,8,13,14]
    relevant_sondes_dict["warm_sector"]        = {}
    relevant_sondes_dict["warm_sector"]["in"]  = \
        sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
    relevant_sondes_dict["warm_sector"]["out"] = \
        sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
    relevant_sondes_dict["cold_sector"]        = {}
    relevant_sondes_dict["cold_sector"]["in"]  = \
        sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
    relevant_sondes_dict["cold_sector"]["out"] = \
        sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
    relevant_sondes_dict["internal"]           = \
        sonde_times_series.iloc[relevant_internal_sondes]
    #-------------------------------------------------------------------------#
    # Get airborne data
    radar_dict={}
    bahamas_dict={}  
    campaign=cmpgn_cls.name
    airborne_data_importer_path=major_work_path+\
        "hamp_processing_py/"+"hamp_processing_python/Flight_Data/"+campaign+"/"

    date=flight_dates[flight[0]]
    ###########################################################################
    inflow=False

    with HiddenPrints():
        # Radar reflectivity
        cfg_dict=quicklook_dicts.get_prcs_cfg_dict(flight,date,
            campaign,cmpgn_cls.campaign_path)
        cfg_dict["device_data_path"]=airborne_data_importer_path
        # Data Handling 
        datasets_dict, data_reader_dict=\
            quicklook_dicts.get_data_handling_attr_dicts()
        # Get Plotting Handling
        plot_handler_dict, plot_cls_args_dict,plot_fct_args_dict=\
                            quicklook_dicts.get_plotting_handling_attrs_dict()

        HALO_Devices_cls=measurement_instruments_ql.HALO_Devices(cfg_dict)
        HALO_Devices_cls.update_major_data_path(cmpgn_cls.campaign_path)
        Bahamas_cls=measurement_instruments_ql.BAHAMAS(HALO_Devices_cls)
        Radar_cls=measurement_instruments_ql.RADAR(HALO_Devices_cls)

        Radar_cls.radar_ds=Radar_cls.open_version_specific_processed_radar_data(
            for_calibrated_file=calibrated_radar)
        Radar_cls.correct_for_gaseous_attenuation()
        radar_ds=Radar_cls.radar_ds            
    Dropsondes["Lon"] = pd.Series(data=np.array(
            Dropsondes["reference_lon"].values()),index=Dropsondes["IWV"].index)
    Dropsondes["Lat"] = pd.Series(data=np.array(
            Dropsondes["reference_lat"].values()),index=Dropsondes["IWV"].index)
    
    #-------------------------------------------------------------------------#
    # Run Plot 
    print("Plotting")
    matplotlib.rcParams.update({"font.size":14})

    combined_plot(cfg_dict, radar_ds, Dropsondes,relevant_sondes_dict,
                  inflow_times,internal_times,outflow_times)
if __name__=="__main__":
    main()