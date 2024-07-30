# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:46:49 2024

@author: Henning Dorff
"""

import glob
import os
import sys
import warnings

current_path=os.getcwd()
git_path=current_path+"/../../"
synth_path=git_path+"Synthetic_Airborne_Arctic_ARs//"

sys.path.insert(1,synth_path+"/src/")

import numpy as np
import pandas as pd
import xarray as xr

#import flightmaps
#from moisturebudget import Moisture_Budget, Moisture_Convergence, Moisture_Budget_Plots        


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.patheffects as PathEffects
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker

from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.image import imread
    
import seaborn as sns

import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

set_font=16
matplotlib.rcParams.update({'font.size':set_font})
plt.rcParams.update({'hatch.color': 'k'})  
plt.rcParams.update({'hatch.linewidth':1.5})


matplotlib.rcParams.update({"font.size":24})
# getting the original colormap using cm.get_cmap() function
orig_map = plt.cm.get_cmap('Blues')
# reversing the original colormap using reversed() function
reversed_map = orig_map.reversed() 
# normally the actual bahamas file is used from HALO-(AC)3. However,
# this is not feasible now for testingb
def importer():
    paths_dict={}
    paths_dict["current_path"]=os.getcwd()
    paths_dict["actual_working_path"]=os.getcwd()+\
        "/../../Synthetic_Airborne_Arctic_ARs/"
    sys.path.insert(1,paths_dict["actual_working_path"]+"/config/")
    import init_paths
    import data_config

    paths_dict["working_path"]=init_paths.main()
        
    paths_dict["airborne_data_importer_path"]       =\
            paths_dict["working_path"]+"/Work/GIT_Repository/"
    paths_dict["airborne_script_module_path"]       =\
            paths_dict["actual_working_path"]+"/scripts/"
    paths_dict["airborne_processing_module_path"]   =\
        paths_dict["actual_working_path"]+"/src/"
    paths_dict["hamp_processing_path"]              =\
        paths_dict["actual_working_path"]+"/../hamp_processing_python/"
    paths_dict["src_hamp_processing_path"]          =\
        paths_dict["hamp_processing_path"]+"/src/"
    paths_dict["airborne_plotting_module_path"]     =\
        paths_dict["actual_working_path"]+"/plotting/"
    paths_dict["manuscript_path"]                   =\
        paths_dict["working_path"]+"Work/Synthetic_AR_Paper/Manuscript/Paper_Plots/"
    paths_dict["scripts_path"]                      =\
        paths_dict["actual_working_path"]+"/major_scripts/"
                            
    sys.path.insert(2,paths_dict["airborne_script_module_path"])
    sys.path.insert(3,paths_dict["airborne_processing_module_path"])
    sys.path.insert(4,paths_dict["airborne_plotting_module_path"])
    sys.path.insert(5,paths_dict["airborne_data_importer_path"])
    sys.path.insert(6,paths_dict["scripts_path"])
    sys.path.insert(7,paths_dict["hamp_processing_path"])
    sys.path.insert(8,paths_dict["src_hamp_processing_path"])
    return paths_dict


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class StadiaStamen(cimgt.Stamen):
    def _image_url(self, tile):
         x,y,z = tile
         url =  f"https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.jpg?api_key=0963bb5f-6e8c-4978-9af0-4cd3a2627df9"
        #https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.png?api_key={API_KEY}"
         return url

#import flightmapping
class AR_mapper():
    """
    This is the major plotting handler for all HALO-(AC)3 AR maps. 
    It is mainly designed for the second manuscript of the PhD of Henning Dorff.
    This study determines all moisture budget components for an AR event and 
    assesses the budget equation closure over the AR dissipation.
    
    """
    def __init__(self,cmpgn_cls,flight,config_file,sea_ice,halo_df,
                 ar_of_day):
                 #grid_name="ERA5",do_instantan=False,sonde_no=3,
                 #scalar_based_div=True):
        
        #super().__init__(cmpgn_cls,flight,config_file)
        self.cmpgn_cls   = cmpgn_cls
        self.flight      = flight
        self.ar_of_day   = ar_of_day
        self.config_file = config_file
        self.plot_path   = os.getcwd()+"/../plots/" # ----> to be filled
        self.seaice      = sea_ice
        self.halo_df     = halo_df
    
    def add_fancy_patch_around(self,ax, bb, **kwargs):
        self.fancy = FancyBboxPatch(bb.p0, bb.width, bb.height,
                           fc=(1, 0.8, 1, 0.5), ec=(1, 0.5, 1, 0.5),
                           **kwargs)
        ax.add_patch(self.fancy)
        return self.fancy

    def plot_AR_map_with_budget_HALO_ICON_comparison(self,era5,icon_ivt,
            Dropsondes,relevant_sondes_dict,internal_sondes_dict,
            warm_icon_rain,merged_precipitation,
            halo_sonde_values,icon_sonde_values,ivt_threshold=100):
        matplotlib.rcParams.update({"font.size":16})
        i=9 # hour index
        print("Hour of the day:",i)
        #x1, y1 = coordinates["Kiruna"]  
        calc_time=era5.hours[i]
        #70N - 85N 20W - 30E
        big_extent=[-25,25,70,85]#[-40,30,55,90]
        # Background terrain
        self.stamen_terrain = StadiaStamen('terrain-background')
    
        coordinates= dict(EDMO=(11.28, 48.08), 
                          Kiruna=(20.336, 67.821),
                          Longyearbyen=(15.46, 78.25),
                          Meiningen=(10.38, 50.56),
                          Lerwick=(-1.18, 60.13),
                          Ittoqqortoormiit=(-21.95, 70.48),
                          Tasiilaq=(-37.63, 65.60))
        coordinates["Ny-Alesund"]=(11.909895,78.923538)
        x1, y1 = coordinates["Kiruna"]   
        #Create a GeoAxes in the tile's projection.
        x=np.linspace(-90,90,41)
        y=np.linspace(55,90,93)
        x_grid,y_grid=np.meshgrid(x,y)
        white_overlay= np.zeros((41,93))
        plt.rcdefaults()
        icon_box_lon=[np.linspace(30,-20, 1000), np.array([-20, -20]),
                        np.linspace(-20, 30, 1000), np.array([30.0, 30.0])]
        icon_box_lat=[np.linspace(70.5, 70.5,1000),np.array([70.5,85.5]),
                      np.linspace(85.5, 85.5, 1000), np.array([85.5, 70.5])]             
    
        fig = plt.figure(figsize=(18,9))
        ax1=fig.add_subplot(121,projection=ccrs.NorthPolarStereo(central_longitude=0))
        # Terrain    
        ax1.text(-0.05,1.05,"(a)",fontsize=18,transform=ax1.transAxes)
        ax1.add_image(self.stamen_terrain, 3)
        ax1.set_extent([big_extent[0]+1,big_extent[1]+3,big_extent[2]-5,big_extent[3]+3],
                       crs=ccrs.Geodetic())
        ax1.contourf(x_grid,y_grid,white_overlay.T,cmap="Greys",vmin=0,vmax=1,
                     transform=ccrs.PlateCarree(),alpha=0.4)
        # Sea ice
        ax1.pcolormesh(self.seaice.lon,self.seaice.lat,np.array(self.seaice[:]), 
                    transform=ccrs.PlateCarree(), cmap=reversed_map)
        # ICON IVT
        C1=ax1.scatter(np.rad2deg(icon_ivt["lon"]),np.rad2deg(icon_ivt["lat"]),
         c=icon_ivt["IVT"],s=0.01,cmap='magma_r',transform=ccrs.PlateCarree(),
         vmin=ivt_threshold,vmax=500,alpha=0.7)
        cbaxes = fig.add_axes([0.145, 0.135, 0.2, 0.02]) 
        cbar=plt.colorbar(C1,cax=cbaxes, 
                          ticks=[ivt_threshold,(500+ivt_threshold)/2,500],
                          orientation='horizontal',extend="max")# ICON BOX
        
        cbar.ax.set_xlabel('IVT (kg $\mathrm{m}^{-1}\,\mathrm{s}^{-1}$)',
                           color="k",fontsize=16)
        
        cbar.ax.tick_params(labelcolor="k",labelsize=16)
        for edge_lon, edge_lat in  \
                            zip(icon_box_lon, icon_box_lat):
            ax1.plot(edge_lon, edge_lat,
                 color="grey", lw=4,ls="-", 
                 transform=ccrs.PlateCarree(),
                    zorder=3)
            ax1.plot(edge_lon, edge_lat,
                 color="orange", lw=2,ls="-.", 
                 transform=ccrs.PlateCarree(),
                    zorder=4)
        icon_legend_patch_name="ICON-2km domain"
        #------------------------------------------------------------------
        # HALO
        # entire flight track
        ax1.plot(self.halo_df["longitude"],self.halo_df["latitude"],
                 color="lightgrey",lw=3,ls="--",transform=ccrs.PlateCarree())
        # Sector legs
        inflow_times=["2022-03-15 10:11","2022-03-15 11:08"]
        internal_times=["2022-03-15 11:17","2022-03-15 12:13"]
        outflow_times=["2022-03-15 12:20","2022-03-15 13:14"]
    
        ax1.text(x1 - 5.5, y1 - 0.5, "Kiruna (KRN)", fontsize=12,
                     transform=ccrs.PlateCarree(),color="red",
                     bbox=dict(facecolor='lightgrey',edgecolor="black"))
        ax1.plot(self.halo_df["longitude"].loc[inflow_times[0]:inflow_times[-1]],
                 self.halo_df["latitude"].loc[inflow_times[0]:inflow_times[-1]],
                 lw=8,color="lightgrey",transform=ccrs.PlateCarree(),zorder=5)
        ax1.plot(self.halo_df["longitude"].loc[internal_times[0]:internal_times[-1]],
                 self.halo_df["latitude"].loc[internal_times[0]:internal_times[-1]],
                 lw=8,color="lightgrey",transform=ccrs.PlateCarree(),zorder=5)
        ax1.plot(self.halo_df["longitude"].loc[outflow_times[0]:outflow_times[-1]],
                 self.halo_df["latitude"].loc[outflow_times[0]:outflow_times[-1]],
                 lw=8,color="lightgrey",transform=ccrs.PlateCarree(),zorder=5)
        ax1.plot(self.halo_df["longitude"].loc[inflow_times[0]:inflow_times[-1]],
                 self.halo_df["latitude"].loc[inflow_times[0]:inflow_times[-1]],
                 lw=4,color="lightgreen",label="cross-section leg",
                 transform=ccrs.PlateCarree(),zorder=5)
        
        ax1.plot(self.halo_df["longitude"].loc[internal_times[0]:internal_times[-1]],
                 self.halo_df["latitude"].loc[internal_times[0]:internal_times[-1]],
                 lw=4,color="teal",label="internal leg",
                 transform=ccrs.PlateCarree(),zorder=5)
        ax1.plot(self.halo_df["longitude"].loc[outflow_times[0]:outflow_times[-1]],
                 self.halo_df["latitude"].loc[outflow_times[0]:outflow_times[-1]],
                 lw=4,color="lightgreen",transform=ccrs.PlateCarree(),zorder=5)
    
        ax1.scatter(x1,y1,s=100,transform=ccrs.PlateCarree(),
                    color="red",edgecolor="k")
        ax1.coastlines(resolution="50m")
        ax1.add_feature(cartopy.feature.BORDERS)
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                                      x_inline=False, y_inline=False)
    
        # Dropsondes
        ax1.scatter(Dropsondes["Lon"].values,Dropsondes["Lat"].values,
                        marker="v",s=100,color="whitesmoke",edgecolor="dimgrey",
                        transform=ccrs.PlateCarree(),zorder=10)
        ax1.scatter(Dropsondes["Lon"].iloc[relevant_sondes_dict["warm_sector"]["in"]],
                    Dropsondes["Lat"].iloc[relevant_sondes_dict["warm_sector"]["in"]],
                    marker="v",s=200,color="mintcream",edgecolor="k",
                    transform=ccrs.PlateCarree(),zorder=10)
        ax1.scatter(Dropsondes["Lon"].iloc[internal_sondes_dict["warm"]],
                    Dropsondes["Lat"].iloc[internal_sondes_dict["warm"]],
                   marker="v",s=100,color="whitesmoke",edgecolor="dimgrey",transform=ccrs.PlateCarree(),zorder=10)
    
        ax1.scatter(Dropsondes["Lon"].iloc[relevant_sondes_dict["warm_sector"]["out"]],
                    Dropsondes["Lat"].iloc[relevant_sondes_dict["warm_sector"]["out"]],
                   marker="v",s=200,color="mintcream",edgecolor="k",transform=ccrs.PlateCarree(),zorder=10)
    
        #ax1.scatter(Dropsondes["Lon"].iloc[relevant_sondes_dict["cold_sector"]["in"]],
        #            Dropsondes["Lat"].iloc[relevant_sondes_dict["cold_sector"]["in"]],
        #           marker="v",s=100,color="blue",edgecolor="k",transform=ccrs.PlateCarree())
    
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 18}
        gl.ylabel_style = {'size': 18}
        ax1.legend(loc="lower left",fontsize=14,facecolor="whitesmoke",
                   framealpha=0.8,edgecolor="k",title="Flight legs",
                  title_fontsize=14,alignment='left')
    
        markersize=300
        ms=20
        ax2=fig.add_subplot(122)
        ax2.text(-0.15,0.975,"(b)",fontsize=18,transform=ax2.transAxes)
        ax2.text(0.35,0.9,"HALO obs \n\nICON along HALO",fontsize=16,
                 horizontalalignment='left',transform=ax2.transAxes)
        #----------------------------------------------------------------------------
        # dIWV/dt
        # HALO
        d_IWV_dt_hamp=pd.Series(data=[-0.72,-0.19,0.66,0.56],
                                index=["S1","S2","S3","S4"]) 
        d_IWV_dt_uncertainty=pd.Series(data=[0.21,0.23872126958677942,0.30,0.1],
                                       index=["S1","S2","S3","S4"])
        ax2.errorbar([1.375],[d_IWV_dt_hamp[0]],yerr=[d_IWV_dt_uncertainty[0]],
            marker="v",ms=ms,markeredgecolor="k",mfc="teal",elinewidth=2,
            ecolor="teal",ls="",markeredgewidth=2,zorder=2)
        # ICON
        d_IWV_dt_icon=-0.4273540676372614 
        d_IWV_dt_icon_unc=0.3055211824129308
        ax2.scatter(1.625,-0.42,marker="o",s=markersize,edgecolor="k",
                    linewidth=2,color="teal",zorder=2)
        ax2.plot([1.625,1.625],[d_IWV_dt_icon-d_IWV_dt_icon_unc,
                                d_IWV_dt_icon+d_IWV_dt_icon_unc],
                 color="teal",lw=2,ls="-",zorder=2)
        #-----------------------------------------------------------------------------------------------------------------#
        # Mass div
        ax2.errorbar([3.375],-1*halo_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div"],
            yerr=halo_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div_unc"],
            marker="v",ms=ms,markeredgecolor="k",markeredgewidth=2,mfc="lightgreen",
            elinewidth=2,ecolor="lightgreen",ls="",label="mass CONV",zorder=2)
    
        ax2.scatter([3.625],-1*icon_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div"],
            marker="o",s=markersize,edgecolor="k",facecolor="lightgreen",
            linewidth=2,linestyle="-",zorder=2)
    
        ax2.plot([3.625,3.625],[-1*icon_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div"]-\
            icon_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div_unc"],
            -1*icon_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div"]+\
            icon_sonde_values[self.flight+"_"+self.ar_of_day]["mass_div_unc"]],
                 color="lightgreen",ls="-",lw=2,zorder=2)
        #----------------------------------------------------------------------------------------------------------------#
        #Q ADV
        ax2.errorbar([2.375],-1*halo_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV"],
                    yerr=halo_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV_unc"],
                    marker="v",ms=ms,markeredgecolor="k",markeredgewidth=2,
                    mfc="lightgreen",elinewidth=2,ecolor="lightgreen",
                    ls="",label="Q ADV",zorder=2)
        ax2.scatter([2.625],-1*icon_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV"],
                    #yerr=halo_sonde_values[flight+"_"+ar_of_day]["q_ADV_unc"],
                    lw=2,marker="o",s=markersize,edgecolor="k",
                    facecolor="lightgreen",linestyle="-",zorder=2)
        ax2.plot([2.625,2.625],[-1*icon_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV"]-\
                    icon_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV_unc"],
                    -1*icon_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV"]+\
                    icon_sonde_values[self.flight+"_"+self.ar_of_day]["q_ADV_unc"]],
                    color="lightgreen",linestyle="-",lw=2,zorder=2)
        #-----------------------------------------------------------------------------------------------------#
        # Precipitation
        #precip_ymin=warm_icon_rain.quantile([0.1,0.9])["Interp_Precip"].iloc[0]
        #precip_ymax=warm_icon_rain.quantile([0.1,0.9])["Interp_Precip"].iloc[-1]
        ax2.plot([4.625,4.625],[warm_icon_rain.quantile([0.1,0.9])["Interp_Precip"].iloc[-0],
                warm_icon_rain.quantile([0.1,0.9])["Interp_Precip"].iloc[-1]],
                color="teal",lw=2,ls="-",zorder=2)
        ax2.scatter(4.625,warm_icon_rain["Interp_Precip"].mean(),marker="o",
                    s=markersize,color="teal",edgecolor="k",linestyle="-",
                   zorder=2,lw=2)
        ###HALO
        ax2.plot([4.375,4.375],[0,merged_precipitation.quantile([0.1,0.9])["rate"].iloc[-1]],color="teal",lw=2,
                zorder=2,)
        ax2.scatter(4.375,merged_precipitation["rate"].mean(),marker="v",s=markersize,color="teal",edgecolor="k",
                   zorder=2,lw=2)
        #-----------------------------------------------------------------------------------------------------#
        # Evaporation
        ICON_Evap_Sectors={}
        ICON_Evap_Sectors["mean"]=[-0.0013212743604341494] # ICON values from internal_evaporation notebook
        ICON_Evap_Sectors["std"]=[0.008429681186788331]
        ax2.plot([5.625,5.625],[ICON_Evap_Sectors["mean"][0]-\
                                ICON_Evap_Sectors["std"][0],
                                ICON_Evap_Sectors["mean"][0]+\
                                    ICON_Evap_Sectors["std"][0]],
                 color="teal",lw=2,ls="-",zorder=2)
        ax2.scatter(5.625,ICON_Evap_Sectors["mean"][0],marker="o",
                    s=markersize,color="teal",
                    edgecolor="k",lw=2,linestyle="-",zorder=2)
    
    
        ### HALO
        ax2.plot([5.375,5.375],[0,0.005], color="teal",lw=2,ls="-",zorder=2) 
        ax2.scatter(5.375,0.002,marker="v",s=markersize,
                    color="teal",edgecolor="k",lw=2,zorder=2)
        #-----------------------------------------------------------------------------------------------------------------------------
        # Cosmetics
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.xaxis.set_tick_params(width=2,length=4)
        ax2.yaxis.set_tick_params(width=2,length=4)
        for spines in ["right","top"]:
            ax2.spines[spines].set_visible(False)
        ax2.set_xlim([1,6])
        ax2.set_xticks([1.5,2.5,3.5,4.5,5.5])
        ax2.set_xticklabels(
            ["dIWV/dt","$ADV$","$DIV_{\mathrm{mass}}$","Precip","Evap"],
            fontsize=16)
        ax2.set_ylabel(
            "Moisture budget \n contribution ($\mathrm{mm}\,\mathrm{h}^{-1}$)",
            fontsize=16)
        ax2.set_ylim([-1,1.02])
        ax2.set_yticks([-1,-0.5,0,0.5,1])
        ax2.set_yticklabels(["-1","-0.5","0","0.5","1"],fontsize=16)
        ax2.grid(axis='x', color="grey",ls=":",lw=1)
        ax2.axhline(y=0,ls="--",color="darkgrey",lw=3,zorder=0)
        ax2.scatter(1.4,0.975,s=markersize,marker="v",color="grey",edgecolor="k")
        ax2.scatter(1.4,0.825,s=markersize,marker="o",color="grey",edgecolor="k")
        plt.subplots_adjust(wspace=0.25)
        fig_name="S1_AR_Budget_components_intercomparison_HALO_ICON.png"
        fig.savefig(self.plot_path+fig_name,dpi=600,bbox_inches="tight")
        print("Figure saved as:", self.plot_path+fig_name)
    
    def plot_AR_synoptics(self):
        # -*- coding: utf-8 -*-
        base_path=os.getcwd()+"/../../../"
        work_path=base_path+"/Work/GIT_Repository/"
        synth_git_path=base_path+"/my_GIT/Synthetic_Airborne_Arctic_ARs/"
        budget_script_path=base_path+"/my_GIT/HALO_AC3_AR_Budget_Closure/scripts/"
        src_path=synth_git_path+"/src/"
        src_plot_path=synth_git_path+"/plotting/"
        #sys.path.insert(1,synth_git_path)
        #sys.path.insert(2,src_path)
        #sys.path.insert(3,src_plot_path)
        #sys.path.insert(4,work_path)
        #sys.path.insert(5,budget_script_path)
        
        airborne_data_path=work_path+"hamp_processing_py\\hamp_processing_python\\"
        airborne_importer_path=airborne_data_path
        
        # Plotting modules
        # Data and processing modules
        import data_config
        import Performance
        
        import flightcampaign
        from reanalysis import ERA5
        from ICON import ICON_NWP as ICON
        import atmospheric_rivers as AR
        from simplified_flight_leg_handling import simplified_run_grid_main
            
        
        #---------------------------------------------------------------#
        # Definitions
        
        warnings.filterwarnings("ignore")
        
        performance=Performance.performance()
        name="data_config_file"
        config_file_exists=False
        campaign_name="HALO_AC3"  
        flights=["RF05"]
        met_variable="IVT"
        should_plot_era_map=True
        # Check if config-File exists and if not create the relevant first one
        if data_config.check_if_config_file_exists(name):
            config_file=data_config.load_config_file(work_path,name)
        else:
            data_config.create_new_config_file(file_name=name+".ini")
        system_is_windows=True
        
        if system_is_windows:
                if not config_file["Data_Paths"]["system"]=="windows":
                    windows_paths={
                        "system":"windows",
                        "campaign_path":os.getcwd()+"/"    
                            }
                    windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
                    data_config.add_entries_to_config_object(name,windows_paths)
                
        
        
        is_flight_campaign=True
        cmpgn_cls=flightcampaign.HALO_AC3(is_flight_campaign=True,
                major_path=config_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",interested_flights=[flights[0]],
                instruments=["radar","radiometer","sonde"])
        cmpgn_cls.specify_flights_of_interest(flights)
        use_era5_ARs=True
        
        flight="RF05"
        ar_of_day="AR_entire_1"
        hour=11
        hour_str=str(hour).zfill(2)
        
        #---------------------------------------------------------------------#
        # Plot configurations
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
        plt.rcParams.update({'hatch.color': 'k'})  
        plt.rcParams.update({'hatch.linewidth':1.5})
        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                       "IVT_u":"IVT_u","IVT_v":"IVT_v"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"ocean_r",
                                       "IVT_v":"speed",
                                       "IVT_u":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(10,25,101),
                                       "IVT":np.linspace(50,500,101),
                                       "IVT_v":np.linspace(0,500,101),
                                       "IVT_u":np.linspace(0,500,101)}
        met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                       "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        # Load HALO aircraft
        halo_dict={}
        cmpgn_cls.load_AC3_bahamas_ds(flight)
        halo_dict=cmpgn_cls.bahamas_ds
        
        if isinstance(halo_dict,pd.DataFrame):
            halo_df=halo_dict.copy() 
        elif isinstance(halo_dict,xr.Dataset):
            halo_df=pd.DataFrame(data=np.nan,columns=["alt","Lon","Lat"],
                            index=pd.DatetimeIndex(np.array(halo_dict["TIME"][:])))
            halo_df["Lon"]=halo_dict["IRS_LON"].data
            halo_df["Lat"]=halo_dict["IRS_LAT"].data
            if len(halo_dict.keys())==1:
                halo_df=halo_dict.values()[0]
        halo_df["Hour"]=halo_df.index.hour
        halo_df=halo_df.rename(columns={"Lon":"longitude",
                                "Lat":"latitude"})
        #---------------------------------------------------------------------#
        # Data Query
        era5=ERA5(for_flight_campaign=True,campaign=cmpgn_cls.name,
                  research_flights=flight,
                  era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
        plot_path=cmpgn_cls.campaign_path+"/plots/"+flight+"/"
        hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
        
        file_name="total_columns_"+cmpgn_cls.year+"_"+\
                                    cmpgn_cls.flight_month[flight]+"_"+\
                                    cmpgn_cls.flight_day[flight]+".nc"    
        
        coordinates= dict(EDMO=(11.28, 48.08), 
                          Kiruna=(20.336, 67.821),
                          Longyearbyen=(15.46, 78.25),
                          Meiningen=(10.38, 50.56),
                          Lerwick=(-1.18, 60.13),
                          Ittoqqortoormiit=(-21.95, 70.48),
                          Tasiilaq=(-37.63, 65.60))
        coordinates["Ny-Alesund"]=(11.909895,78.923538)
        x1, y1 = coordinates["Kiruna"]   
        
        ds,era_path=era5.load_era5_data(file_name)
        
        #if meteo_var.startswith("IVT"):
        ds["IVT_v"]=ds["p72.162"]
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        
        flight_date=cmpgn_cls.year+"-"+cmpgn_cls.flight_month[flight]
        flight_date=flight_date+"-"+cmpgn_cls.flight_day[flight]
        
        
        # theta 850hpa
        era5_850hpa_file_name="temp850hPa_"+"".join(flight_date.split("-"))+".nc"
        theta_ds,_=era5.load_era5_850hPa(era5_850hpa_file_name)
        theta_ds=era5.calculate_theta_e(theta_ds)
        meteo_var="IVT"
        ar_IVT=ds[meteo_var][hour,:,:]
        ar_theta=theta_ds["theta_e"][hour,:,:]
        
        map_fig=plt.figure(figsize=(12,9))
        ax1 = map_fig.add_subplot(121,
                projection=ccrs.AzimuthalEquidistant(central_longitude=-10.0,
                                                     central_latitude=70))
        ax2 = map_fig.add_subplot(122,
                projection=ccrs.AzimuthalEquidistant(central_longitude=-10.0,
                                                     central_latitude=70))
        
        ax1.coastlines(resolution="50m")
        ax2.coastlines(resolution="50m")
        ax1.gridlines()
        ax2.gridlines()
        
        ax1.set_extent([-40,25,55,85]) 
        ax2.set_extent([-40,25,55,85]) 
        
        ax1.text(-0.07,0.95,"a)",fontsize=18,transform=ax1.transAxes)
        #calc_time=era5.hours[i]
        
        #-----------------------------------------------------------------#
        # Meteorological Data plotting
        # Plot Water Vapour Quantity    
        C1=ax1.contourf(ds["longitude"],ds["latitude"],
                        ds[met_var_dict["ERA_name"][meteo_var]][hour,:,:],
                        levels=met_var_dict["levels"][meteo_var],
                        extend="max",transform=ccrs.PlateCarree(),
                        cmap=met_var_dict["colormap"][meteo_var],alpha=0.95)
        
        cb=map_fig.colorbar(C1,ax=ax1,orientation="horizontal",shrink=0.7,pad=0.03)
        cb.set_label(meteo_var+" "+met_var_dict["units"][meteo_var])
        cb.set_ticks([50,250,500])
        
        # Mean surface level pressure
        pressure_color="purple"##"royalblue"
        sea_ice_colors=["darkorange","saddlebrown"]
        #
        ax1.scatter(x1,y1,s=100,transform=ccrs.PlateCarree(),color="whitesmoke",edgecolor="k")
        ax1.text(x1 + 3.5, y1+0.75, "Kiruna", fontsize=11,
                     transform=ccrs.PlateCarree(),color="k",
                     bbox=dict(boxstyle="round",facecolor='lightgrey',
                               edgecolor="black",pad=0.35))
        ax2.text(x1 + 3.5, y1+0.75, "Kiruna", fontsize=11,
                     transform=ccrs.PlateCarree(),color="k",
                     bbox=dict(boxstyle="round",facecolor='lightgrey',
                               edgecolor="black",pad=0.35), zorder=10)
        
        C_p=ax1.contour(ds["longitude"],ds["latitude"],ds["msl"][hour,:,:]/100,
                        levels=np.linspace(950,1050,11),linestyles="-.",
                        linewidths=1.5,colors=pressure_color,
                        transform=ccrs.PlateCarree())
        
        plt.clabel(C_p, inline=1, fmt='%03d hPa',fontsize=12)
        # mean sea ice cover
        C_i=ax1.contour(ds["longitude"],ds["latitude"],
                        ds["siconc"][hour,:,:]*100,levels=[15,85],
                        linestyles="-",linewidths=[1,1.5],colors=sea_ice_colors,
                        transform=ccrs.PlateCarree())
        plt.clabel(C_i, inline=1, fmt='%02d %%',fontsize=10)
        
        #-----------------------------------------------------------------#
        # Quiver-Plot
        step=15
        quiver_lon=np.array(ds["longitude"][::step])
        quiver_lat=np.array(ds["latitude"][::step])
        u=ds["IVT_u"][hour,::step,::step]
        v=ds["IVT_v"][hour,::step,::step]
        v=v.where(v>200)
        v=np.array(v)
        u=np.array(u)
        quiver=plt.quiver(quiver_lon,quiver_lat,
                              u,v,color="lightgrey",edgecolor="k",lw=1,
                              scale=800,scale_units="inches",
                              pivot="mid",width=0.008,
                              transform=ccrs.PlateCarree())
        plt.rcParams.update({'hatch.color': 'lightgrey'})
        #-----------------------------------------------------------------#
        # AR detection
        AR=AR.Atmospheric_Rivers("ERA",use_era5=use_era5_ARs)
        AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019,
                                       year=cmpgn_cls.year,
                                       month=cmpgn_cls.flight_month[flight])
        AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)
        
        hatches=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                         AR_era_ds.shape[0,AR_era_data["model_runs"].start+hour,
                                         0,:,:],
                         hatches=["//"],cmap="bone_r",
                         alpha=0.1,transform=ccrs.PlateCarree())
        for c,collection in enumerate(hatches.collections):
            collection.set_edgecolor("k")
        #-----------------------------------------------------------------#
        # HALO flight track
        plot_halo_df=halo_df[halo_df.index.hour<hour]
        ax1.plot(plot_halo_df["longitude"],
                plot_halo_df["latitude"],
                lw=4,color="whitesmoke",transform=ccrs.PlateCarree(),zorder=11)
        
        ax1.plot(plot_halo_df["longitude"],
                plot_halo_df["latitude"],
                lw=2,ls="-",color="k",transform=ccrs.PlateCarree(),zorder=11)
        #-----------------------------------------------------------------#
        ax2.scatter(x1,y1,s=100,transform=ccrs.PlateCarree(),color="whitesmoke",
                    edgecolor="k",zorder=11)
        
        
        ax2.text(-0.07,0.95,"b)",fontsize=18,transform=ax2.transAxes) 
        sns_colour=sns.color_palette("Spectral",31)
        C2=ax2.contourf(theta_ds["longitude"],theta_ds["latitude"],
                        theta_ds["theta_e"][hour,:,:],
                        levels=np.linspace(255,285,31),
                        extend="both",transform=ccrs.PlateCarree(),
                        cmap="jet",alpha=0.95)
        
        cb=map_fig.colorbar(C2,ax=ax2,orientation="horizontal",pad=0.03,shrink=0.7)
        cb.set_label("$\Theta_{e}$ (K)")
        cb.set_ticks([255,265,275,285])
        
        C_p2=ax2.contour(ds["longitude"],ds["latitude"],ds["msl"][hour,:,:]/100,
                        levels=np.linspace(950,1050,11),linestyles="-.",
                        linewidths=1.5,colors="white",
                        transform=ccrs.PlateCarree())
        
        plt.clabel(C_p2, inline=1, fmt='%03d hPa',fontsize=12)
        # mean sea ice cover
        C_i2=ax2.contour(ds["longitude"],ds["latitude"],
                        ds["siconc"][hour,:,:]*100,levels=[15,85],
                        linestyles="-",linewidths=[1,1.5],colors=["grey","k"],
                        transform=ccrs.PlateCarree())
        hatches2=ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                         AR_era_ds.shape[0,AR_era_data["model_runs"].start+hour,
                                         0,:,:],
                         hatches=["//"],cmap="bone_r",
                         alpha=0.2,transform=ccrs.PlateCarree())
        for c,collection in enumerate(hatches2.collections):
            collection.set_edgecolor("k")
        ax2.plot(plot_halo_df["longitude"],
                plot_halo_df["latitude"],
                lw=4,color="whitesmoke",transform=ccrs.PlateCarree(),zorder=11)
        
        ax2.plot(plot_halo_df["longitude"],
                plot_halo_df["latitude"],
                lw=2,ls="-",color="k",transform=ccrs.PlateCarree(),zorder=11)
        ### handle gridlines
        #-------------------------------------------------------------------------#
        import matplotlib.ticker as mticker
        gls1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                          x_inline=False, y_inline=False)
        gls1.ylocator = mticker.FixedLocator([60,65,70,75,80,85])
        gls1.xlocator = mticker.FixedLocator([-60,60])
        gls2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                          x_inline=False, y_inline=False)
        gls2.ylocator = mticker.FixedLocator([60,65,])
        gls2.xlocator = mticker.FixedLocator([-60,60])
        gls1.bottom_labels      = False
        gls1.right_labels       = False
        gls1.top_labels         = True
        gls2.right_labels       = True
        gls2.left_labels        = False
        gls2.bottom_labels      = False
        #gls2.ylocator = mticker.FixedLocator([80,85])
        gls1.xlabel_style = {'size': 16}
        gls1.ylabel_style = {'size': 16}
        gls2.xlabel_style = {'size': 16}
        gls2.ylabel_style = {'size': 16}
        
        #-------------------------------------------------------------------------#    
        
        plt.subplots_adjust(wspace=0.15,hspace=-0.3)
        fig_name="Fig01_AR_RF05_synoptic_conditions_"+hour_str+"_UTC.png"
        map_fig.savefig(self.plot_path+fig_name,dpi=600,bbox_inches="tight")
        print("Figure saved as:", self.plot_path+fig_name)
    
class AR_IVT_tendency_plotter():
    def __init__(self,config_dict,do_plot_tendency=True,
                  do_plot_all_sectors=False,
                  only_both_flights=True):
        #import quicklook_dicts
        #import measurement_instruments_ql
        #import flightcampaign
        
        #booleans
        self.do_plot_tendency   = do_plot_tendency
        self.do_plot_all_sectors   = do_plot_all_sectors
        self.only_both_flights  = only_both_flights
        self.save_in_manuscript_path = False
        # sea ice maps
        self.orig_map           = plt.cm.get_cmap('Blues')
        # reversing the original colormap using reversed() function
        self.reversed_map       = self.orig_map.reversed() 
        
        
        self.campaign           = config_dict["campaign_name"]
        self.campaign_path      = config_dict["campaign_path"]
        self.met_var            = config_dict["meteo_var"] 
        self.met_var_dict       = config_dict["met_var_dict"]
        self.leg_dict           = config_dict["leg_dict"]
        self.flight_dict        = config_dict["flight_dict"]
        self.paths_dict         = config_dict["paths_dict"]
        
        self.projection         = ccrs.AzimuthalEquidistant(
                                    central_longitude=-2.0,
                                    central_latitude=72)
        self.flight_dates       = config_dict["flight_dates"]
        self.haloac3            = config_dict["haloac3"]
        self.era_index_dict     = config_dict["era_index_dict"]
        self.amsr2_sea_ice_path = config_dict["amsr2_sea_ice_path"]
        self.plot_path          = self.paths_dict["current_path"]+"/../plots/"
        self.ar_label           = config_dict["ar_label"]
        self.relevant_sondes    = config_dict["relevant_sondes"]
        
        self.stamen_terrain = StadiaStamen('terrain-background')
        
    def plot_sector_tendency(self):
        import quicklook_dicts
        import measurement_instruments_ql
        import flightcampaign
        from reanalysis import ERA5        
        matplotlib.rcParams.update({"font.size":24})
        gls=[None,None,None,None]
        row_no=2
        col_no=2
        fig,axs=plt.subplots(row_no,col_no,sharex=True,sharey=True,
                figsize=(16,12),subplot_kw={'projection': self.projection})
        fig_labels=["(a) S1, RF05","(b) S2, RF05","(c) S3, RF06","(d) S4, RF06"]
        axis=axs.flatten()
            
        for k in range(4):
            axis[k].set_extent([-20,27,67.5,80])
        
            if k < 2:
                key=0
                row=0
            else:
                key=1
                row=1
            col=0
            flight_date= [*self.flight_dict.keys()][key]
            if (k+1)%2==0:
                col=1
                ar_of_day="AR_entire_2"
            else:
                ar_of_day="AR_entire_1"
                
            leg_dict_key=[*self.flight_dates[self.campaign].keys()][key]+"_"+ar_of_day
            print("Leg dict key:",leg_dict_key)
               
            cmpgn_cls=[*self.flight_dict.values()][key][0]
            flight=[*self.flight_dict.values()][key][1]
            cfg_dict=quicklook_dicts.get_prcs_cfg_dict(
                            flight, self.flight_dates[self.campaign][flight], 
                            self.campaign,self.campaign_path)
            HALO_Devices_cls=measurement_instruments_ql.HALO_Devices(cfg_dict)
            Sondes_cls=measurement_instruments_ql.Dropsondes(HALO_Devices_cls)
            Sondes_cls.open_all_sondes_as_dict()
            sonde_data=Sondes_cls.sonde_dict
    
    
            # Aircraft   
            self.haloac3.load_AC3_bahamas_ds(flight)
            halo_dict=self.haloac3.bahamas_ds
            if isinstance(halo_dict,pd.DataFrame):
                halo_df=halo_dict.copy() 
            elif isinstance(halo_dict,xr.Dataset):
                halo_df=pd.DataFrame(data=np.nan,columns=["alt","Lon","Lat"],
                        index=pd.DatetimeIndex(np.array(halo_dict["TIME"][:])))
                halo_df["longitude"]=halo_dict["IRS_LON"].data
                halo_df["latitude"]=halo_dict["IRS_LAT"].data
    
            ##### Load ERA5-data
            self.era5=ERA5(for_flight_campaign=True,campaign=cmpgn_cls.name,
                  research_flights=flight,
                  era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
           
            hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
            file_name="total_columns_"+flight_date[0:4]+"_"+\
                       flight_date[4:6]+"_"+\
                       flight_date[6:8]+".nc"    
            
            era_ds,era_path=self.era5.load_era5_data(file_name)
            era_index=self.era_index_dict[flight_date] 
            era_ds["IVT_v"]=era_ds["p72.162"]
            era_ds["IVT_u"]=era_ds["p71.162"]
            era_ds["IVT"]=np.sqrt(era_ds["IVT_u"]**2+era_ds["IVT_v"]**2)
            era_ds["IVT"]=era_ds["IVT"].where(era_ds.IVT>100)
            # Make Grid but adapt it for specific subplot
            gls[key] = axis[k].gridlines(crs=ccrs.PlateCarree(),
                        draw_labels=True, x_inline=False, y_inline=False,
                        zorder=15)
            gls[key].bottom_labels = False
            gls[key].ylocator = mticker.FixedLocator([70,75,80,85])
            gls[key].xlocator = mticker.FixedLocator([-30,0,30])
            gls[key].top_labels    = True
            gls[key].bottom_labels = True
            gls[key].bottom_labels = True
            gls[key].bottom_labels = True
            if col==0: 
                gls[key].right_labels  = False
            if row==0:
                gls[key].bottom_labels = False
            if row==1:    
                gls[key].top_labels    = False
            if col==1:
                gls[key].left_labels   = False
                
    
                
            gls[key].xlabel_style = {'size': 24}
            gls[key].ylabel_style = {'size': 24}
            #-----------------------------------------------------------------#
            # Plot Geomap
            axis[k].add_image(self.stamen_terrain, 3)
            # Plot sea ice
            sea_ice_file_list=glob.glob(self.amsr2_sea_ice_path+"*"+\
                                        flight_date+"*.nc")
            sea_ice_ds=xr.open_dataset(sea_ice_file_list[0])
            seaice=sea_ice_ds["seaice"]                                       
            # Sea ice
            axis[k].pcolormesh(seaice.lon, seaice.lat,np.array(seaice[:]), 
                               transform=ccrs.PlateCarree(), 
                               cmap=self.reversed_map)
            # Create white overlay for less strong colors
            x=np.linspace(-90,90,41)
            y=np.linspace(55,90,93)
            x_grid,y_grid=np.meshgrid(x,y)
            white_overlay= np.zeros((41,93))+0.3
            axis[k].contourf(x_grid,y_grid,white_overlay.T,
                          cmap="Greys",vmin=0,vmax=1,
                          transform=ccrs.PlateCarree(),alpha=0.6)
        
            # Plot IVT
            C1=axis[k].contourf(era_ds["longitude"],era_ds["latitude"],
                era_ds[self.met_var_dict["ERA_name"][self.met_var]][era_index[col],:,:],
                levels=self.met_var_dict["levels"][self.met_var],extend="max",
                transform=ccrs.PlateCarree(),
                cmap=self.met_var_dict["colormap"][self.met_var],alpha=0.6,
                zorder=6)
        
            axis[k].coastlines(resolution="50m",zorder=9)
            # Date and Timestep
            axis[k].text(-10, 69, self.ar_label[flight_date]+\
                      " "+str(era_index[col])+" UTC",
                      fontsize=20,transform=ccrs.PlateCarree(),
                      color="k",bbox=dict(
                          facecolor='whitesmoke',edgecolor="black"),zorder=10)
            axis[k].plot(halo_df["longitude"],halo_df["latitude"],
                         color="white",lw=2,transform=ccrs.PlateCarree(),
                         zorder=10)
            axis[k].plot(halo_df["longitude"],halo_df["latitude"],lw=1,
                        color="k",transform=ccrs.PlateCarree(),zorder=10)
            
            if not leg_dict_key=="RF05_AR_entire_2":
                start=self.leg_dict[leg_dict_key]["inflow_times"][0]
                end=self.leg_dict[leg_dict_key]["outflow_times"][-1]
            else:
                start=self.leg_dict[leg_dict_key]["outflow_times"][0]
                end=self.leg_dict[leg_dict_key]["inflow_times"][-1]
            axis[k].plot(halo_df["longitude"].loc[start:end],
                         halo_df["latitude"].loc[start:end],
                         color="white",lw=6,transform=ccrs.PlateCarree(),
                         zorder=11)
            axis[k].plot(halo_df["longitude"].loc[start:end],
                         halo_df["latitude"].loc[start:end],
                         color="mediumorchid",lw=4,transform=ccrs.PlateCarree(),
                         zorder=12)
            
            if k==0:
                axis[k].annotate('HALO',
                             xy=(0, -0.275), xycoords='axes fraction', 
                             xytext=(0.2, -0.285),fontsize=24,
                             arrowprops=dict(arrowstyle="-",lw=2, color='k'))
                #axis[k].annotate("AR corridor",xy=(0,-))
    
            # AR label (AR1)
            axis[k].text(-35.5,78.1,fig_labels[k],fontsize=24,
                transform=ccrs.PlateCarree(),color="k",
                bbox=dict(facecolor="whitesmoke",edgecolor="black"),
                zorder=20)
            # Add dropsondes
            sonde_shapes="v"
            for s,sonde in enumerate(sonde_data["launch_time"].keys()):
                release_lat=sonde_data["reference_lat"][sonde].data[0]
                release_lon=sonde_data["reference_lon"][sonde].data[0]
                scat=axis[k].scatter(release_lon,release_lat,
                        marker=sonde_shapes,s=50,edgecolors="k",
                        color="mintcream",transform=ccrs.PlateCarree(), zorder=13)
                if s in self.relevant_sondes[leg_dict_key]["sondes_no"]:
                    scat=axis[k].scatter(release_lon,release_lat,
                        marker=sonde_shapes,s=300,edgecolors="k",
                            color="orange",transform=ccrs.PlateCarree(),
                            zorder=14)
                if s in self.relevant_sondes[leg_dict_key]["internal_sondes_no"]:
                    scat=axis[k].scatter(release_lon,release_lat,
                        marker=sonde_shapes,s=300,edgecolors="k",
                            color="whitesmoke",transform=ccrs.PlateCarree(),
                            zorder=14)
            
            sonde_list=[*sonde_data["launch_time"].keys()]
            lat_sonde_series=pd.Series(data=[*sonde_data["reference_lat"].values()])
            lon_sonde_series=pd.Series(data=[*sonde_data["reference_lon"].values()])
            left_edge_int=self.relevant_sondes[leg_dict_key]["left_edge"]
            left_edge_lat=lat_sonde_series.iloc[left_edge_int]
            right_edge_int=self.relevant_sondes[leg_dict_key]["right_edge"]
            right_edge_lat=lat_sonde_series.iloc[right_edge_int]
            
            left_edge_lon=lon_sonde_series.iloc[left_edge_int]
            right_edge_lon=lon_sonde_series.iloc[right_edge_int]
               
            axis[k].plot(left_edge_lon,left_edge_lat,lw=5,
                         ls="--",color="orange",transform=ccrs.PlateCarree(),
                         zorder=16)
            axis[k].plot(right_edge_lon,right_edge_lat,lw=5,
                         ls="--",color="orange",transform=ccrs.PlateCarree(),
                         zorder=16)
        
        cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.02])
        cbar=fig.colorbar(C1, cax=cbar_ax,
                          extend="max",orientation="horizontal")
        cbar.set_ticks([100,250,500])
        cbar_ax.text(0.5,-3.5,self.met_var+" "+\
                     self.met_var_dict["units"][self.met_var],
                     fontsize=24,transform=cbar_ax.transAxes)   
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig_name="Fig09_budget_corridors_tendency.png"
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
    def plot_tendencies(self):
        if self.only_both_flights:
            self.plot_both_flights()
        if self.do_plot_all_sectors:
            self.plot_sector_tendency()                
        
    
    def plot_both_flights(self,):
        import quicklook_dicts
        import measurement_instruments_ql
        from reanalysis import ERA5
        flights=[*self.flight_dict.keys()]
        col_no=len(flights)
        row_no=1
        fig,axs=plt.subplots(row_no,col_no,sharex=True,sharey=True,figsize=(12,16),
                                 subplot_kw={'projection': self.projection})
        key=0
        # Aircraft data
        fig_labels=["(a)","(b)","(c)"]
        gls=[None,None,None]
        
        
        for col in range(col_no):
            flight_date= [*self.flight_dict.keys()][key]
            print("Flight date",flight_date)
                   
            cmpgn_cls=[*self.flight_dict.values()][key][0]
            flight=[*self.flight_dict.values()][key][1]
            cfg_dict=quicklook_dicts.get_prcs_cfg_dict(
                        flight, self.flight_dates[self.campaign][flight], 
                        self.campaign,self.campaign_path)
            HALO_Devices_cls=measurement_instruments_ql.HALO_Devices(cfg_dict)
            Sondes_cls=measurement_instruments_ql.Dropsondes(HALO_Devices_cls)
            Sondes_cls.open_all_sondes_as_dict()
            Sondes_cls.calc_integral_variables(integral_var_list=["IWV","IVT"])
            sonde_data=Sondes_cls.sonde_dict
        
        
            # Aircraft   
            self.haloac3.load_AC3_bahamas_ds(flight)
            halo_dict=self.haloac3.bahamas_ds
            if isinstance(halo_dict,pd.DataFrame):
                halo_df=halo_dict.copy() 
            elif isinstance(halo_dict,xr.Dataset):
                halo_df=pd.DataFrame(data=np.nan,columns=["alt","Lon","Lat"],
                            index=pd.DatetimeIndex(np.array(halo_dict["TIME"][:])))
            halo_df["longitude"]=halo_dict["IRS_LON"].data
            halo_df["latitude"]=halo_dict["IRS_LAT"].data
    
            ##### Load ERA5-data
            self.era5=ERA5(for_flight_campaign=True,campaign=cmpgn_cls.name,
                      research_flights=flight,
                      era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
               
            hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
            file_name="total_columns_"+flight_date[0:4]+"_"+\
                           flight_date[4:6]+"_"+\
                           flight_date[6:8]+".nc"    
                
            era_ds,era_path=self.era5.load_era5_data(file_name)
            era_index=self.era_index_dict[flight_date] 
            era_ds["IVT_v"]=era_ds["p72.162"]
            era_ds["IVT_u"]=era_ds["p71.162"]
            era_ds["IVT"]=np.sqrt(era_ds["IVT_u"]**2+era_ds["IVT_v"]**2)
            era_ds["IVT"]=era_ds["IVT"].where(era_ds.IVT>100)
            # Make Grid but adapt it for specific subplot
            gls[key] = axs[col].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                              x_inline=False, y_inline=False)
            gls[key].bottom_labels = False
            gls[key].ylocator = mticker.FixedLocator([70,75,80,85])
            gls[key].xlocator = mticker.FixedLocator([-60,0,60])
            if key==0:
               gls[key].right_labels = False
               gls[key].top_labels = True
            elif key==1:
                gls[key].right_labels = True
                gls[key].left_labels = False
                gls[key].ylocator = mticker.FixedLocator([80,85])
            elif key==2:
                gls[key].left_labels = False
                gls[key].top_labels = False
                gls[key].right_labels=False
                
            gls[key].xlabel_style = {'size': 18}
            gls[key].ylabel_style = {'size': 18}
            #ICON box-------------------------------------------------------------#
            icon_box_lon=[np.linspace(30,-20, 1000), np.array([-20, -20]),
                        np.linspace(-20, 30, 1000), np.array([30.0, 30.0])]
            icon_box_lat=[np.linspace(70.5, 70.5,1000),np.array([70.5,85.5]),
                      np.linspace(85.5, 85.5, 1000), np.array([85.5, 70.5])]          
            for edge_lon, edge_lat in  \
                            zip(icon_box_lon, icon_box_lat):
                axs[col].plot(edge_lon, edge_lat,
                         color="grey", lw=4,ls="-", 
                         transform=ccrs.PlateCarree(),
                         zorder=11)
                axs[col].plot(edge_lon, edge_lat,
                         color="orange", lw=2,ls="-.", 
                         transform=ccrs.PlateCarree(),
                         zorder=11)
                icon_legend_patch_name="ICON-2km domain"
            #----------------------------------------------------------------------#
            # Plot Geomap
            axs[col].add_image(self.stamen_terrain, 1)
            # Plot sea ice
            sea_ice_file_list=glob.glob(self.amsr2_sea_ice_path+\
                                        "*"+flight_date+"*.nc")
            sea_ice_ds=xr.open_dataset(sea_ice_file_list[0])
            seaice=sea_ice_ds["seaice"]                                       
            # Sea ice
            axs[col].pcolormesh(seaice.lon, seaice.lat,np.array(seaice[:]), 
                    transform=ccrs.PlateCarree(), cmap=self.reversed_map)
        
            # Create white overlay for less strong colors
            x=np.linspace(-90,90,41)
            y=np.linspace(55,90,93)
            x_grid,y_grid=np.meshgrid(x,y)
            white_overlay= np.zeros((41,93))+0.3
            axs[col].contourf(x_grid,y_grid,white_overlay.T,cmap="Greys",vmin=0,vmax=1,
                     transform=ccrs.PlateCarree(),alpha=0.6)
            
            # Plot IVT
            C1=axs[col].contourf(era_ds["longitude"],era_ds["latitude"],
                era_ds[self.met_var_dict["ERA_name"][self.met_var]][era_index[0],:,:],
                levels=self.met_var_dict["levels"][self.met_var],extend="max",
                transform=ccrs.PlateCarree(),alpha=0.6,zorder=6,
                cmap=self.met_var_dict["colormap"][self.met_var])
            
            # Plot surface presure
            C_p=axs[col].contour(era_ds["longitude"],era_ds["latitude"],
                                    era_ds["msl"][era_index[0],:,:]/100,
                                    levels=np.linspace(950,1050,11),
                                    linestyles="-.",linewidths=2,
                                    colors="grey",transform=ccrs.PlateCarree(),
                                    zorder=8)
            axs[col].clabel(C_p, inline=True,inline_spacing=2,
                            fmt='%03d',fontsize=10)
            # mean sea ice cover
            #C_i=axs[col].contour(era_ds["longitude"],era_ds["latitude"],
            #            era_ds["siconc"][era_index,:,:]*100,levels=[15,85],
            #            linestyles="-",linewidths=[1.5,3],
            #            colors=sea_ice_colors,transform=ccrs.PlateCarree())
            #
            #axs[col].clabel(C_i, inline=True, inline_spacing=2,
            #                fmt='%02d %%',fontsize=12)
                    
                    
            axs[col].coastlines(resolution="50m",zorder=9)
            axs[col].set_extent([-20,27,67.5,87.5])
            # Date and Timestep
            axs[col].text(-18.35, 66.5, self.ar_label[flight_date]+\
                          " "+str(era_index[0])+" UTC",
                          fontsize=14,transform=ccrs.PlateCarree(),
                          color="k",bbox=dict(
                              facecolor='lightgrey',edgecolor="black"),zorder=10)
            axs[col].plot(halo_df["longitude"],
                                 halo_df["latitude"],
                                 color="white",lw=5,
                                 transform=ccrs.PlateCarree(),zorder=10)
            axs[col].plot(halo_df["longitude"],
                                 halo_df["latitude"],lw=2,
                                 color="k",
                                 transform=ccrs.PlateCarree(),zorder=10)
            
            if col==0:
                axs[col].annotate('HALO track',
                                 xy=(0, -0.275), xycoords='axes fraction', 
                                 xytext=(0.2, -0.285),fontsize=18,
                                 arrowprops=dict(arrowstyle="-",lw=2, color='k'))    
    
            # AR label (AR1)
            axs[col].text(-78.5,84,fig_labels[key],fontsize=18,
                          transform=ccrs.PlateCarree(),
                          color="k",bbox=dict(facecolor="lightgrey",
                                              edgecolor="black"),
                          zorder=10)
            step=20   
            quiver_lon=np.array(era_ds["longitude"][::step])
            quiver_lat=np.array(era_ds["latitude"][::step])
            u=era_ds["IVT_u"][era_index[0],::step,::step]
            v=era_ds["IVT_v"][era_index[0],::step,::step]
            v=v.where(v>200)
            v=np.array(v)
            u=np.array(u)
            quiver=axs[col].quiver(quiver_lon,quiver_lat,u,v,color="white",
                                 edgecolor="k",lw=1,
                                 scale=600,scale_units="inches",
                                 pivot="mid",width=0.015,
                                 transform=ccrs.PlateCarree(),zorder=10)
            # Add dropsondes
            sonde_shapes=["v","s"]
            for sonde in sonde_data["launch_time"].keys():
                release_lat=sonde_data["reference_lat"][sonde].data[0]
                release_lon=sonde_data["reference_lon"][sonde].data[0]
                scat=axs[col].scatter(release_lon,release_lat,
                        marker=sonde_shapes[1],s=300,edgecolors="k",
                        c=sonde_data["IVT"].loc[sonde],cmap="BuGn",
                        vmin=0, vmax=500,transform=ccrs.PlateCarree(), zorder=9)
                scat2=axs[col].scatter(release_lon,release_lat,
                    marker=sonde_shapes[0],
                           s=100,edgecolors="k",c=sonde_data["IWV"].loc[sonde],
                           cmap="BuPu",vmin=0, vmax=20,
                           transform=ccrs.PlateCarree(), zorder=12)
        
            if key==2:
                q_typ=600.0
                axs[col].quiverkey(quiver,0.36,0.875,q_typ,
                        label=str(q_typ)+' $\mathrm{kgm}^{-1}\mathrm{s}^{-1}$',
                        coordinates="axes",labelpos="E",fontproperties={"size":18})
            key+=1
            # Adjust the location of the subplots on the page to make room for the colorbar
        
        fig.subplots_adjust(#bottom=0.15, top=0.9, left=0.15, right=0.9,
                                wspace=0.05)#, hspace=0.05)
            # Add a colorbar axis at the bottom of the graph
        cbar_ax = fig.add_axes([0.15, 0.3, 0.7, 0.01])
        cbar=fig.colorbar(C1, cax=cbar_ax,extend="max",orientation="horizontal")
        cbar.set_ticks([100,250,500])
        cbar_ax.text(0.4,-3.5,self.met_var+" "+\
                     self.met_var_dict["units"][self.met_var],
                     fontsize=18,transform=cbar_ax.transAxes)
        cbar2_ax=fig.add_axes([0.6,0.28,0.1,0.01])
        cbar2 = fig.colorbar(scat2,cax=cbar2_ax,
                             extend="max",orientation="horizontal")
        cbar2.set_label("IWV ($\mathrm{kg}\mathrm{m}^{-2}$)",fontsize=16)
            
        if not self.save_in_manuscript_path:
            fig_path=self.plot_path#paths_dict["current_path"]+"/../plots/"
        else:
            pass
            #    fig_path=paths_dict["manuscript_path"]
        fig_name="Fig02_AR_RF05_RF06_tendency.png"
        fig.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",fig_path+fig_name)
    def plot_ar_cases(self,merged_bahamas,ar_surface_mask,lon_sondes,
                      lat_sondes,ar_geo_sondes):
        merged_bahamas_df=merged_bahamas.to_dataframe()
        date_list=list(np.unique(merged_bahamas_df.index.date.astype(str)))
    
def run_plot_IVT_tendency(do_plot_tendency=True,
                          plot_all_sectors=True,
                          plot_overview=False,
                          only_both_flights=True):
    
    paths_dict=importer()
    #-------------------------------------------------------------------------#
    # Define the flight campaign classes
    try:
        from typhon.plots import styles
    except:
        print("Typhon module cannot be imported")
        
    
    if "data_config" in sys.modules:
        import data_config
    
    save_in_manuscript_path=False
    config_file=data_config.load_config_file(
                    paths_dict["airborne_data_importer_path"],
                        "data_config_file")
    
    flights=["RF05","RF06"]
    # Initiate HALO-AC3 campaign
    is_flight_campaign=True
    
    import flightcampaign
    haloac3=flightcampaign.HALO_AC3(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",interested_flights=[flights[0]],
                          instruments=["radar","radiometer","sonde"])
    haloac3.specify_flights_of_interest(flights)
    
    #haloac3.create_directory(directory_types=["data"])
    
    #-------------------------------------------------------------------------#
    # Get the flight data    
    if do_plot_tendency:
        flight_dict={#"20220313":[haloac3,"RF03"],
                 "20220315":[haloac3,"RF05"],
                 "20220316":[haloac3,"RF06"]
                 # "20220410":[haloac3,"RF16"],
                 }
    
        flight_dates={"HALO_AC3":
              {#"RF03":"20220313",
               "RF05":"20220315",
               "RF06":"20220316"}
               #"RF16":"2020410"},
              }
    if plot_overview:
        flight_dict={
            "20220312":[haloac3,"RF02"],
            "20220313":[haloac3,"RF03"],
            "20220314":[haloac3,"RF04"],
            "20220315":[haloac3,"RF05"],
            "20220316":[haloac3,"RF06"],
            "20220320":[haloac3,"RF07"],
            "20220321":[haloac3,"RF08"],
            "20220410":[haloac3,"RF16"],
                 }
    
        flight_dates={"HALO_AC3":
              {"RF02":"20220312",
               "RF03":"20220313",
               "RF04":"20220314",
               "RF05":"20220315",
               "RF06":"20220316",
               "RF07":"20220320",
               "RF08":"20220321",
               "RF16":"2020410"},
              }
    
    leg_dict={}
    # RF05
    leg_dict["RF05_AR_entire_1"]={}
    leg_dict["RF05_AR_entire_1"]["inflow_times"]=\
        ["2022-03-15 10:11","2022-03-15 11:13"]
    leg_dict["RF05_AR_entire_1"]["internal_times"]=\
            ["2022-03-15 11:18","2022-03-15 12:14"]
    leg_dict["RF05_AR_entire_1"]["outflow_times"]=\
            ["2022-03-15 12:20","2022-03-15 13:15"]
    leg_dict["RF05_AR_entire_2"]={}
    leg_dict["RF05_AR_entire_2"]["inflow_times"]=\
    ["2022-03-15 14:30","2022-03-15 15:25"]
    leg_dict["RF05_AR_entire_2"]["internal_times"]=\
    ["2022-03-15 13:20","2022-03-15 14:25"]
    leg_dict["RF05_AR_entire_2"]["outflow_times"]=\
    ["2022-03-15 12:20","2022-03-15 13:15"]
    #"RF06":
    leg_dict["RF06_AR_entire_1"]={}

    leg_dict["RF06_AR_entire_1"]["inflow_times"]=\
    ["2022-03-16 10:45","2022-03-16 11:21"]
    leg_dict["RF06_AR_entire_1"]["internal_times"]=\
        ["2022-03-16 11:25","2022-03-16 12:10"]
    leg_dict["RF06_AR_entire_1"]["outflow_times"]=\
        ["2022-03-16 12:15","2022-03-16 12:50"]
    
    leg_dict["RF06_AR_entire_2"]={}
    leg_dict["RF06_AR_entire_2"]["inflow_times"]=\
        ["2022-03-16 12:12","2022-03-16 12:55"]
    leg_dict["RF06_AR_entire_2"]["internal_times"]=\
        ["2022-03-16 12:58","2022-03-16 13:40"]
    leg_dict["RF06_AR_entire_2"]["outflow_times"]=\
        ["2022-03-16 13:45","2022-03-16 14:18"]
    # Dropsondes of sector
    relevant_sondes={}
    relevant_sondes["RF05_AR_entire_1"]={}
    relevant_sondes["RF05_AR_entire_1"]["sondes_no"]=[0,1,2,3,9,10,11,12]
    relevant_sondes["RF05_AR_entire_1"]["left_edge"]=[3,12]
    relevant_sondes["RF05_AR_entire_1"]["right_edge"]=[0,9]
    relevant_sondes["RF05_AR_entire_1"]["internal_sondes_no"]=[7,13]
    
    relevant_sondes["RF05_AR_entire_2"]={}
    relevant_sondes["RF05_AR_entire_2"]["sondes_no"]=[9,10,11,12,15,16,17,18]
    relevant_sondes["RF05_AR_entire_2"]["internal_sondes_no"]= [13,22]
    relevant_sondes["RF05_AR_entire_2"]["left_edge"]         = [12,18]
    relevant_sondes["RF05_AR_entire_2"]["right_edge"]        = [9,15]
    
    relevant_sondes["RF06_AR_entire_1"]={}
    relevant_sondes["RF06_AR_entire_1"]["sondes_no"]          = [0,1,2,8,9,10]
    relevant_sondes["RF06_AR_entire_1"]["internal_sondes_no"] = [7,22]
    relevant_sondes["RF06_AR_entire_1"]["left_edge"]          = [2,10]
    relevant_sondes["RF06_AR_entire_1"]["right_edge"]         = [0,8]
    
    relevant_sondes["RF06_AR_entire_2"]={}
    relevant_sondes["RF06_AR_entire_2"]["sondes_no"]          = [8,9,16,17]
    relevant_sondes["RF06_AR_entire_2"]["internal_sondes_no"] = [14,22]
    relevant_sondes["RF06_AR_entire_2"]["left_edge"]          = [9,17]
    relevant_sondes["RF06_AR_entire_2"]["right_edge"]         = [8,16]
    col_no=len(flights)
    row_no=1
    
    key=0
    era_index_dict={"20220313":16,
                    "20220315":[12,14],
                    "20220316":[11,13],
                    "20220410":14,
                    }
    ar_label={"20220313":"13 March 2022",
              "20220315":"15 March 2022",
              "20220316":"16 March 2022",
              "20220410":"10 April 2022"}
    meteo_var="IVT"
    pressure_color="purple"##"royalblue"
    sea_ice_colors=["orange",#"gold",
                    "saddlebrown"]#["mediumslateblue", "indigo"]
    
    campaign=[*flight_dates.keys()][0]
    campaign_path="C://Users/u300737/Desktop/Desktop_alter_Rechner/"+\
                    "PhD_UHH_WIMI/Work/GIT_Repository/"+"hamp_processing_py/"+\
                        "hamp_processing_python/Flight_Data/"+campaign+"/"
    
    amsr2_sea_ice_path=campaign_path+\
        "\\sea_ice\\"
    
    #-------------------------------------------------------------------------#
    # Plot the map
    # Define the plot specifications for the given variables
    met_var_dict={}
    met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                   "IVT_u":"IVT_u","IVT_v":"IVT_v"}
    met_var_dict["colormap"]    = {"IWV":"density","IVT":"ocean_r",
                                   "IVT_v":"speed","IVT_u":"speed"}
    met_var_dict["levels"]      = {"IWV":np.linspace(10,25,101),
                                   "IVT":np.linspace(100,500,301),
                                   "IVT_v":np.linspace(0,500,101),
                                   "IVT_u":np.linspace(0,500,101)}
    met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                   "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                   "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                   "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
    
    config_dict={}
    config_dict["campaign_name"]      = "HALO_AC3"
    config_dict["campaign_path"]      = campaign_path
    config_dict["haloac3"]            = haloac3
    config_dict["meteo_var"]          = meteo_var
    config_dict["met_var_dict"]       = met_var_dict
    config_dict["paths_dict"]         = paths_dict
    config_dict["leg_dict"]           = leg_dict
    config_dict["flight_dict"]        = flight_dict
    config_dict["flight_dates"]       = flight_dates
    config_dict["relevant_sondes"]    = relevant_sondes
    config_dict["era_index_dict"]     = era_index_dict
    config_dict["amsr2_sea_ice_path"] = amsr2_sea_ice_path
    config_dict["ar_label"]           = ar_label 
    IVT_tendency_plotter=AR_IVT_tendency_plotter(config_dict,
                        do_plot_all_sectors=True,only_both_flights=True)
    
    if IVT_tendency_plotter.do_plot_tendency:
        if IVT_tendency_plotter.do_plot_all_sectors:
            IVT_tendency_plotter.plot_tendencies()
        else:        
            IVT_tendency_plotter.plot_both_flights()
    else:
        # PLOT HALO-(AC)3 AR overviews
        pass
