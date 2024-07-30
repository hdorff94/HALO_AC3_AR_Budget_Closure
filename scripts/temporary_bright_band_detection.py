# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 08:44:18 2023

@author: Henning Dorff
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:35:51 2020

@author: u300737
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

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
sys.path.insert(3,synth_ar_path+"/src/")
sys.path.insert(4,synth_ar_path+"/plotting/")
sys.path.insert(5,hamp_processing_path)
sys.path.insert(6,hamp_processing_path+"/plotting/")
import data_config
###############################################################################
import flightcampaign
import moisturebudget as Budgets
###############################################################################
#Grid Data
from reanalysis import ERA5,CARRA
from ICON import ICON_NWP as ICON
import gridonhalo as Grid_on_HALO
###############################################################################
#-----------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")

def plot_radar_with_melting_layer(processed_radar,bb_height,precip_type_series):
    # Now raw_uni_radar and ds (processed uni radar) can be compared
    # via plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    from matplotlib import colors
    from cmcrameri import cm as cmaeri

    fig,axs=plt.subplots(4,1,figsize=(18,12),
                         gridspec_kw=dict(height_ratios=(1,1,1,0.1)),sharex=True)
    y=np.array(processed_radar["height"][:])
    statement="Plotting HAMP Cloud Radar (processed"
    if not calibrated_radar: statement+=")"
    else: statement+=" and calibrated)"
    print(statement)
    #######################################################################
    #######################################################################
    ### Processed radar
    print("flag nans")
    processed_radar["dBZg"]=processed_radar["dBZg"].where(
        processed_radar["radar_flag"].isnull(), drop=True)
    processed_radar["Zg"]=processed_radar["Zg"].where(
        processed_radar["radar_flag"].isnull(), drop=True)
    processed_radar["LDRg"]=processed_radar["LDRg"].where(
        processed_radar["radar_flag"].isnull(), drop=True)
    
    print("flagging done")
    
    surface_Zg=processed_radar["Zg"][:,4]
    surface_Zg=surface_Zg.where(surface_Zg!=-888.)
    
    
    ##### so far rain rate is ignored
    #rain_rate=get_rain_rate(surface_Zg)
    
    #processed_radar
    time=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:]))
    #Plotting
    C1=axs[0].pcolormesh(time,y,np.array(processed_radar["dBZg"][:]).T,
                    cmap=cmaeri.roma_r,vmin=-30,vmax=30)
    
    print("dBZ plotted")
    if inflow_times[0]<outflow_times[-1]:
        axs[0].axvspan(pd.Timestamp(inflow_times[-1]),
                   pd.Timestamp(internal_times[0]),
                   alpha=0.5, color='grey')
        axs[0].axvspan(pd.Timestamp(internal_times[-1]),
                   pd.Timestamp(outflow_times[0]),
                   alpha=0.5, color='grey')   
    else:
        axs[0].axvspan(pd.Timestamp(outflow_times[-1]),
                   pd.Timestamp(internal_times[0]),
                   alpha=0.5, color='grey')
        axs[0].axvspan(pd.Timestamp(internal_times[-1]),
                   pd.Timestamp(inflow_times[0]),
                   alpha=0.5, color='grey')   
    
    axs[0].plot(bb_height.index,bb_height,color="purple",
                lw=3,label="Bright Band")
    cax1=fig.add_axes([0.9, 0.675, 0.01, 0.15])
    cb = plt.colorbar(C1,cax=cax1,orientation='vertical',extend="both")
    cb.set_label('Reflectivity (dBZ)')
    title_str="Processed radar"
    if calibrated_radar: title_str+=" and calibrated"
    title_str+=" "+flight[0]+" "+ar_of_day
    #axs[0].set_title()
    axs[0].set_title(title_str)
    axs[0].set_xlabel('')
    axs[0].set_yticks([0,1000,2000,4000,6000,8000,10000,12000])
    axs[0].set_ylim([0,2000])
    axs[0].set_yticklabels(["0","1","2","4","6","8","10","12"])
    axs[0].set_xticklabels([])
    axs[0].set_ylabel("Altitude (km)")
    axs[0].text(pd.Timestamp(inflow_times[0]),1500,"Inflow")
    axs[0].text(pd.Timestamp(internal_times[0]),1500,"Internal")
    axs[0].text(pd.Timestamp(outflow_times[0]),1500,"Outflow")
    axs[0].set_ylabel("Height (km)")
    # Radar LDR
    C2=axs[1].pcolormesh(time,y,np.array(processed_radar["LDRg"][:].T),
                         cmap=cmaeri.batlowK,vmin=-25, vmax=-10)        
    axs[1].plot(bb_height.index,bb_height,color="purple",
                lw=2,label="Bright Band")
    
    axs[1].set_yticks([0,1000,2000,4000,6000,8000,10000,12000])
    axs[1].set_yticklabels(["0","1","2","4","6","8","10","12"])
    axs[1].set_ylim([0,2000])
    
    print("LDR plotted")
    if inflow_times[0]<outflow_times[-1]:
        axs[1].axvspan(pd.Timestamp(inflow_times[-1]),
                   pd.Timestamp(internal_times[0]),
                   alpha=0.5, color='grey')
        axs[1].axvspan(pd.Timestamp(internal_times[-1]),
                   pd.Timestamp(outflow_times[0]),
                   alpha=0.5, color='grey')   
    else:
        axs[1].axvspan(pd.Timestamp(outflow_times[-1]),
                   pd.Timestamp(internal_times[0]),
                   alpha=0.5, color='grey')
        axs[1].axvspan(pd.Timestamp(internal_times[-1]),
                   pd.Timestamp(inflow_times[0]),
                   alpha=0.5, color='grey')   
    
        #axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
    axs[1].text(pd.Timestamp(inflow_times[0]),1500,"Inflow")
    axs[1].text(pd.Timestamp(internal_times[0]),1500,"Internal")
    axs[1].text(pd.Timestamp(outflow_times[0]),1500,"Outflow")
    axs[1].set_xticklabels([])
    axs[1].set_ylabel("Height (km)")
    # Orecipitation rate
    #axs[2].plot(halo_era5["Interp_Precip"],lw=2,ls="--",color="k",label="ERA5:"+\
            str(round(float(halo_era5["Interp_Precip"].mean()),2)),zorder=5)
    axs[2].plot(halo_icon_hmp["Interp_Precip"],lw=3,ls="-",color="k",label="ICON-NWP:"+\
            str(round(float(halo_icon_hmp["Interp_Precip"].mean()),2)),zorder=6)
    axs[2].plot(halo_icon_hmp["Interp_Precip"],lw=2,ls="--",color="w",zorder=7)
    axs[2].plot(rain_rate["mean"],lw=3,label="avg:"+str(round(float(rain_rate["mean"].mean()),2)))
    axs[2].plot(rain_rate["norris"],lw=1,label="nor:"+str(round(float(rain_rate["norris"].mean()),2)))
    axs[2].plot(rain_rate["palmer"],lw=1,label="pal:"+str(round(float(rain_rate["palmer"].mean()),2)))
    axs[2].plot(rain_rate["chandra"],lw=1,label="cha:"+str(round(float(rain_rate["chandra"].mean()),2)))
    
    if inflow_times[0]<outflow_times[-1]:
        axs[2].axvspan(pd.Timestamp(inflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
        axs[2].axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(outflow_times[0]),
               alpha=0.5, color='grey')   
else:
    axs[2].axvspan(pd.Timestamp(outflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
    axs[2].axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(inflow_times[0]),
               alpha=0.5, color='grey')   

axs[2].set_ylim([0,1.0])
axs[2].legend(loc="top left",ncol=6)
axs[2].set_xticks=axs[1].get_xticks()
axs[2].set_xticklabels([])
axs[2].set_ylabel("Precipitation\nrate ($\mathrm{mmh}^{-1}$)")

    
    # add subplot 3 Precipitation Phase
    #axs[2]
    import matplotlib.cm as cm
    from matplotlib import colors
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    precip_colorbar     = cm.get_cmap('BuPu_r', 5)
    blue_cb             = precip_colorbar(np.linspace(0, 1, 5))
    brown_rgb           = np.array(colors.hex2color(colors.cnames['brown']))
    grey_rgb            = np.array(colors.hex2color(colors.cnames["grey"]))
    blue_cb[:1, :] = [*brown_rgb,1]
    blue_cb[4:,:]   = [*grey_rgb,1]
    newcmp = ListedColormap(blue_cb)
    im = axs[3].pcolormesh(np.array([
        pd.DatetimeIndex(precip_type_series.index),
        pd.DatetimeIndex(precip_type_series.index)]),
        np.array([0, 1]),
        np.array([precip_type_series.values]),
        cmap=newcmp, vmin=-1.5, vmax=3.5,
        shading='auto')
    cax = fig.add_axes([0.7, 0.06, 0.1, axs[3].get_position().height])
    C1=fig.colorbar(im, cax=cax, orientation='horizontal')
    #C1.set_label(label='Precip \ntype')
    C1.ax.set_xticks([-1,-0.5,0,0.5,1,2,3])
    #C1.ax.set_xticklabels(["land","dry","snow","rain","uncertain"],rotation=90)

    #C1.ax.tick_params(labelsize=fs_small)
    #axs[2].tick_params(axis='x', labelleft=False, 
    #                      left=False,labelsize=fs_small)
    #axs[2].tick_params(axis='y', labelleft=False, left=False)
    #axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

#%% Melting layer
def find_melting_layer(radar_dict,vertical_value_to_use="max"):
    ldr_threshold=-17
    lower_height_thres=5
    maximum_height=2000
    maximum_gradient=60
    # based on Austen et al. 2023    
    height=np.array(radar_dict["height"][:])
    ldr_df=pd.DataFrame(data=np.array(radar_dict["LDRg"][:]),
                 columns=height,
                 index=pd.DatetimeIndex(
                     np.array(radar_dict["time"])))

    ldr_cutted_df=ldr_df.copy()#.reindex(cutted_aircraft_df.index)

    low_ldr_df=ldr_cutted_df.iloc[:,lower_height_thres:70]
    #-------------------------------------------------------------------------#
    # LDR threshold
    low_ldr_df[low_ldr_df<ldr_threshold]=np.nan

    # which value to use if a vertical column is above ldr_threshold
    # my method was the maximum value which always shifts the bright band above
    # version auf Austen is 
    if vertical_value_to_use=="max":    
        ldr_mlayer_height=low_ldr_df.idxmax(axis=1)
    elif vertical_value_to_use=="lowest": # Austen et al. 2023
        #this is a bad method but performs well
        mask_of_ldr_values=low_ldr_df/low_ldr_df
        ldr_mlayer_height=mask_of_ldr_values.idxmin(axis=1)
        #for idx in low_ldr_df.index:
        #    vertical_profile=pd.Series(data=low_ldr_df.iloc[idx,:],
        #                               index=)
    
    #-------------------------------------------------------------------------#
    # LDR should always lie below maximum height defined above
    ldr_mlayer_height[ldr_mlayer_height>maximum_height]=np.nan
    
    #-------------------------------------------------------------------------#
    # Gradient criteria (continuity), it is less strong than in Austin et al.
    ldr_ml_height_gradient=ldr_mlayer_height.diff()
    strong_gradient=ldr_ml_height_gradient[\
                        abs(ldr_ml_height_gradient)>maximum_gradient]
    # set value to nan for too strong gradients
    ldr_mlayer_height.loc[strong_gradient.index]=np.nan
    #-------------------------------------------------------------------------#
    # 5s rolling mean
    ldr_mlayer_height=ldr_mlayer_height.rolling("5s",min_periods=5).mean()   
    #-------------------------------------------------------------------------#
    # Melting layer mask
    mlayer_mask=pd.Series(data=np.zeros(ldr_mlayer_height.shape[0]),
                          index=ldr_mlayer_height.index)
    mlayer_mask[~ldr_mlayer_height.isnull()]+=1
    #-------------------------------------------------------------------------#
    # max 10 s gap filling via interpolation
    ldr_mlayer_height=ldr_mlayer_height.interpolate(method="polynomial",order=5,
                                                    limit=10,limit_area="inside",
                                                    limit_direction="both")
    # Extrapolate
    ldr_mlayer_height=ldr_mlayer_height.interpolate(method="polynomial",order=5,
                                                    limit_area="outside",limit=10,
                                                    fill_value="extrapolate")
    condition_1=mlayer_mask==0
    condition_2=~ldr_mlayer_height.isnull()
    both_conditions= condition_1 & condition_2
    mlayer_mask[both_conditions]=2
    #
    mlayer_mask[ldr_mlayer_height.between(0,270,inclusive="right")]=2
    #------------------------------------------------------------------------- #
    #
    return ldr_mlayer_height,low_ldr_df,ldr_cutted_df,mlayer_mask

def classify_precipitation_type(radar_dict, bb_height,bb_mask):
    surface_Zg=radar_dict["Zg"][:,4]
    surface_Zg=surface_Zg.where(surface_Zg!=-888.)
    sfc_zg_series=pd.Series(data=np.array(surface_Zg[:]),
                            index=pd.DatetimeIndex(
                               np.array(surface_Zg.time[:])))
    surface_type=pd.Series(data=np.array(radar_dict["radar_flag"].values[:,0]),
                           index=sfc_zg_series.index)
    
    precip_type_series=pd.Series(data=np.nan,
                                 index=pd.DatetimeIndex(
                                     np.array(surface_Zg.time[:])))
    precip_type_series[sfc_zg_series.isnull()]=0
    precip_type_series[~sfc_zg_series.isnull()]=1.0 # snow
    precip_type_series.loc[bb_mask==1.0]=2.0 # rain
    precip_type_series.loc[bb_mask==2.0]=3.0 # uncertain    
    precip_type_series[surface_type==-0.1]=-1.0
    # take as last conditions do not anymore look for clear rain defined signals
    # but maybe it is not important once applied to the rain reflectivities
    return precip_type_series
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


###############################################################################
# Preallocations
campaign="HALO_AC3"
ar_of_day="AR_entire_1"
flight=["RF05"]
sector_to_plot="warm"
take_arbitary=True
do_plotting=True
calibrated_radar=False
from simplified_flight_leg_handling import simplified_run_grid_main
with HiddenPrints():
    halo_era5,halo_df,cmpgn_cls,ERA5_on_HALO,radar,Dropsondes=\
        simplified_run_grid_main(flight=flight,config_file_path=major_work_path,
                                 ar_of_day=ar_of_day)
flight=cmpgn_cls.flight

flight_dates={
    "RF03":"20220313",
    "RF05":"20220315",
    "RF06":"20220316",
    "RF16":"20220410"}
#halo_era5
###############################################################################
#%% Flight leg definitions
if flight[0]=="RF03":
    if ar_of_day=="AR_entire_1":
            inflow_times=["2022-03-13 10:00","2022-03-13 10:35"]
            internal_times=["2022-03-13 10:37","2022-03-13 11:10"]
            outflow_times=["2022-03-13 11:16","2022-03-13 11:40"]
    
if flight[0]=="RF05":
    if ar_of_day=="AR_entire_1":
            inflow_times=["2022-03-15 10:11","2022-03-15 11:13"]
            internal_times=["2022-03-15 11:18","2022-03-15 12:14"]
            outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
    elif ar_of_day=="AR_entire_2":
            inflow_times=["2022-03-15 14:30 ","2022-03-15 15:25"]
            internal_times=["2022-03-15 13:20 ","2022-03-15 14:25"]
            outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
if flight[0]=="RF06":
    if ar_of_day=="AR_entire_1":
            inflow_times=["2022-03-16 10:45","2022-03-16 11:21"]
            internal_times=["2022-03-16 11:25","2022-03-16 12:10"]
            outflow_times=["2022-03-16 12:15","2022-03-16 12:50"]
    elif ar_of_day=="AR_entire_2":
            inflow_times=["2022-03-16 12:12","2022-03-16 12:55"]
            internal_times=["2022-03-16 12:58","2022-03-16 13:40"]
            outflow_times=["2022-03-16 13:45","2022-03-16 14:18"]

if flight[0]=="RF16":
    if ar_of_day=="AR_entire_1":
            inflow_times=["2022-04-10 10:40","2022-04-10 11:08"]
            internal_times=["2022-04-10 11:10","2022-04-10 11:36"]
            outflow_times=["2022-04-10 11:57","2022-04-10 12:15"]
    elif ar_of_day=="AR_entire_2":
            inflow_times=["2022-03-16 12:12","2022-03-16 12:55"]
            internal_times=["2022-03-16 12:58","2022-03-16 13:40"]
            outflow_times=["2022-03-16 13:45","2022-03-16 14:18"]

new_halo_dict={flight[0]:{
                "inflow":halo_df.loc[inflow_times[0]:inflow_times[-1]],
                "internal":halo_df.loc[internal_times[0]:internal_times[-1]],
                "outflow":halo_df.loc[outflow_times[0]:outflow_times[-1]]}}

from atmospheric_rivers import Atmospheric_Rivers
AR_inflow,AR_outflow=Atmospheric_Rivers.locate_AR_cross_section_sectors(
                                    new_halo_dict,ERA5_on_HALO.halo_era5,
                                    flight[0])

print(AR_inflow["AR_inflow"].keys())
relevant_sondes_dict={}
if take_arbitary:
    for sector in ["warm_sector","core","cold_sector"]:
        print("Analyse frontal sector ",sector)
        #add_sonde=1
        #if sector=="core":
        #    add_sonde=1
        AR_sector_in                       = AR_inflow["AR_inflow_"+sector]
        AR_sector_out                      = AR_outflow["AR_outflow_"+sector]

###############################################################################
#%% Radar data and class
# processing_path packages
import data_config

import measurement_instruments_ql
#import halodataplot as Data_Plotter
import quicklook_dicts

#BAHAMAS.bahamas_ds
radar_dict={}
bahamas_dict={}  
campaign=cmpgn_cls.name

airborne_data_importer_path=major_work_path+\
                                "hamp_processing_py/"+\
                                    "hamp_processing_python/Flight_Data/"+campaign+"/"
print(airborne_data_importer_path)

date=flight_dates[flight[0]]
###############################################################################
inflow=False
# Radar reflectivity
cfg_dict=quicklook_dicts.get_prcs_cfg_dict(flight,date,campaign,cmpgn_cls.campaign_path)
cfg_dict["device_data_path"]=airborne_data_importer_path

# Data Handling 
datasets_dict, data_reader_dict=quicklook_dicts.get_data_handling_attr_dicts()
# Get Plotting Handling
plot_handler_dict, plot_cls_args_dict,plot_fct_args_dict=\
                            quicklook_dicts.get_plotting_handling_attrs_dict()

HALO_Devices_cls=measurement_instruments_ql.HALO_Devices(cfg_dict)
HALO_Devices_cls.update_major_data_path(cmpgn_cls.campaign_path)
Bahamas_cls=measurement_instruments_ql.BAHAMAS(HALO_Devices_cls)
Radar_cls=measurement_instruments_ql.RADAR(HALO_Devices_cls)

radar_ds=Radar_cls.open_version_specific_processed_radar_data(
                                        for_calibrated_file=calibrated_radar)

# cut radar data
# Radar data
radar_dict["dBZg"]=pd.DataFrame(data=np.array(radar_ds["dBZg"][:]),
                       index=pd.DatetimeIndex(np.array(radar_ds.time[:])),
                       columns=np.array(radar_ds.height[:]))

radar_dict["LDRg"]=pd.DataFrame(data=np.array(radar_ds["LDRg"][:]),
                       index=pd.DatetimeIndex(np.array(radar_ds.time[:])),
                       columns=np.array(radar_ds.height[:]))
if inflow_times[0]<outflow_times[-1]:
    processed_radar=radar_ds.sel({"time":slice(inflow_times[0],
                                               outflow_times[-1])})
else:
    processed_radar=radar_ds.sel({"time":slice(outflow_times[0],
                                               inflow_times[-1])})
###############################################################################

import halodataplot
# Radar reflectivity
processed_radar=halodataplot.replace_fill_and_missing_values_to_nan(
                                processed_radar,["dBZg","Zg","LDRg","VELg",
                                                      "radar_flag"])        
       
ldr_bb_height,low_ldr_df,ldr_cutted_df,mlayer_mask=find_melting_layer(processed_radar)
precip_type_series=classify_precipitation_type(processed_radar, ldr_bb_height,mlayer_mask)
plot_radar_with_melting_layer(processed_radar,ldr_bb_height,precip_type_series)