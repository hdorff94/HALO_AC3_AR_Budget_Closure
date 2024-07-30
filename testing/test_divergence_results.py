# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:12:42 2023

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
import scipy.interpolate as scint

start_path=os.getcwd()
ac3_scripts_path=start_path+"/../scripts/"
my_git_path=start_path+"/../../"
major_work_path=my_git_path+"/../Work/GIT_Repository/"
hamp_processing_path=my_git_path+"/hamp_processing_python/"

synth_ar_path=my_git_path+"/Synthetic_Airborne_Arctic_ARs/"
retrieval_src_path=my_git_path+"/hamp_retrieval_haloac3/"
config_path=synth_ar_path+"config/"
sys.path.insert(1,config_path)
sys.path.insert(2,ac3_scripts_path)
sys.path.insert(3,synth_ar_path+"src/")
sys.path.insert(4,synth_ar_path+"plotting/")
sys.path.insert(5,hamp_processing_path)
sys.path.insert(6,hamp_processing_path+"plotting/")
sys.path.insert(7,retrieval_src_path+"src/")
sys.path.insert(8,start_path+"/../plotting/")
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

import matplotlib

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

test_divergence_results=True
test_divergence_calculation=True
#%% Recalculate divergence for given leg
#---> to be filled
if test_divergence_calculation:
    raise Exception("So far not provided!")

#%% Plot sonde results         
if test_divergence_results:
    relevant_warm_sector_sondes={}
    relevant_warm_sector_sondes["RF05_AR_entire_1_in"]=[0,1,2,3]
    relevant_warm_sector_sondes["RF05_AR_entire_1_out"]=[9,10,11,12]
    
    relevant_warm_sector_sondes["RF05_AR_entire_2_in"] = [9,10,11,12]
    relevant_warm_sector_sondes["RF05_AR_entire_2_out"]= [15,16,17,18]
    
    relevant_warm_sector_sondes["RF06_AR_entire_1_in"]=[0,1,2]
    relevant_warm_sector_sondes["RF06_AR_entire_1_out"]=[8,9,10]
    
    relevant_warm_sector_sondes["RF06_AR_entire_2_in"]=[8,9]
    relevant_warm_sector_sondes["RF06_AR_entire_2_out"]=[16,17]
    
    from simplified_flight_leg_handling import simplified_run_grid_main
    
    import matplotlib.pyplot as plt
    sonde_budget_path=major_work_path+"//HALO_AC3/data/budgets/"#RF06_AR_entire_2_warm_Real_Sondes_adv_q.csv
    tendency_nabla_ivt={}
    tendency_nabla_ivt["adv_q"]={}
    tendency_nabla_ivt["mass_conv"]={}
    tendency_nabla_ivt["sum"]={}
    sector_colors=["darkgreen","forestgreen","olivedrab","darkseagreen"]
    #unc_colors=[,"lightcyan","mintcream","palegoldenrod"]
    div_trend_fig=plt.figure(figsize=(18,12))
    ax1=div_trend_fig.add_subplot(121)
    ax2=div_trend_fig.add_subplot(122)
    Dropsondes_dict={}
    
    for l,leg in enumerate(["RF05_AR_entire_1","RF05_AR_entire_2","RF06_AR_entire_1","RF06_AR_entire_2"]):
        rf=leg.split("_")[0]
        ar_of_day=leg.split("_")[-3]+"_"+leg.split("_")[-2]+"_"+leg.split("_")[-1]
        print(ar_of_day)
        #with HiddenPrints():
        halo_era5,halo_df,cmpgn_cls,ERA5_on_HALO,radar,Dropsondes_dict[leg]=\
                simplified_run_grid_main(flight=[rf],config_file_path=major_work_path,ar_of_day=ar_of_day)
        adv_file_name=rf+"_"+ar_of_day+"_warm_Real_Sondes_"+"adv_q"+".csv"
        mass_file_name=rf+"_"+ar_of_day+"_warm_Real_Sondes_"+"mass_convergence"+".csv"
        #print(file_name)
        tendency_nabla_ivt["adv_q"][leg]=pd.read_csv(sonde_budget_path+adv_file_name,index_col=0)
        tendency_nabla_ivt["mass_conv"][leg]=pd.read_csv(sonde_budget_path+mass_file_name,index_col=0)
        tendency_nabla_ivt["sum"][leg]=tendency_nabla_ivt["adv_q"][leg]+\
                                        tendency_nabla_ivt["mass_conv"][leg]
    
    #matplotlib.rcParams.update({"font.size":16})
    ##sonde_fig,axs=plt.subplots(nrows=4, ncols=4,figsize=(12,18))
    new_index=np.arange(0,12000,30)
                                    
    for l,leg in enumerate(["RF05_AR_entire_1","RF05_AR_entire_2","RF06_AR_entire_1","RF06_AR_entire_2"]):
        relevant_times=[*Dropsondes_dict[leg]["reference_time"].keys()]
        uninterp_vars={}
        interp_vars={}
        interp_vars_df_in={}
        interp_vars_df_in["q"]     = pd.DataFrame()
        interp_vars_df_in["v"]     = pd.DataFrame()
        interp_vars_df_in["tra"] = pd.DataFrame()
        interp_vars_df_out={}
        interp_vars_df_out["q"]     = pd.DataFrame()
        interp_vars_df_out["v"]     = pd.DataFrame()
        interp_vars_df_out["tra"] = pd.DataFrame()
        for s,sonde in enumerate([*Dropsondes_dict[leg]["q"].keys()]):
            key=[*Dropsondes_dict[leg]["q"].keys()][s]
            Dropsondes_dict[leg]["transport"]={}
            Dropsondes_dict[leg]["transport"][key]=Dropsondes_dict[leg]["q"][key]*\
                    Dropsondes_dict[leg]["wspd"][str(relevant_times[s])]
            if s in relevant_warm_sector_sondes[leg+"_in"]:
                q_values  = Dropsondes_dict[leg]["q"][key].values
                z_values  = Dropsondes_dict[leg]["alt"][str(relevant_times[s])]
                v_values  = Dropsondes_dict[leg]["wspd"][str(relevant_times[s])].values
                tra_values= Dropsondes_dict[leg]["transport"][key].values
    
                q_series=pd.Series(data=q_values,index=z_values)
                q_series.dropna(inplace=True)
                
                v_series=pd.Series(data=v_values,index=z_values)
                v_series.dropna(inplace=True)
                v_series=v_series[v_series.index.notnull()]
                
                tra_series=pd.Series(data=tra_values,index=z_values)
                tra_series.dropna(inplace=True)
                uninterp_vars["q"]=q_series
                uninterp_vars["v"]=v_series
                uninterp_vars["tra"]=tra_series
                for var in ["q","v","tra"]:
                    interp_func=scint.interp1d(uninterp_vars[var].index,
                                uninterp_vars[var],kind="nearest",bounds_error=False,fill_value=np.nan)
    
                    interp_vars[var]=pd.Series(data=interp_func(new_index),
                                        index=new_index)
                    if var=="tra":
                        print(interp_vars[var])
                    interp_vars_df_in[var][sonde]=interp_vars[var]
                leg_type="in"
            elif s in relevant_warm_sector_sondes[leg+"_out"]:
                q_values  = Dropsondes_dict[leg]["q"][key].values
                z_values  = Dropsondes_dict[leg]["alt"][str(relevant_times[s])]
                v_values  = Dropsondes_dict[leg]["wspd"][str(relevant_times[s])].values
                tra_values= Dropsondes_dict[leg]["transport"][key].values
                q_series=pd.Series(data=q_values,index=z_values)
                q_series.dropna(inplace=True)
                q_series=q_series[q_series.index.notnull()]
                v_series=pd.Series(data=v_values,index=z_values)
                v_series.dropna(inplace=True)
                v_series=v_series[v_series.index.notnull()]
                tra_series=pd.Series(data=tra_values,index=z_values)
                tra_series.dropna(inplace=True)
                leg_type="out"
                
                uninterp_vars["q"]=q_series
                uninterp_vars["v"]=v_series
                uninterp_vars["tra"]=tra_series
                for var in ["q","v","tra"]:
                    interp_func=scint.interp1d(uninterp_vars[var].index,
                                uninterp_vars[var],kind="nearest",bounds_error=False)
                    
                    interp_vars[var]=pd.Series(data=interp_func(new_index),
                                        index=new_index)
                    if var=="tra":
                        print(interp_vars[var])
                    interp_vars_df_out[var][sonde]=interp_vars[var]
    
            else: 
                continue
            if leg_type=="in":
                line_style="-"
                
            else:
                line_style="--"
            # Plotting
            #axs[l,0].plot(q_series.values*1000,
            #                q_series.index/1000,color="lightblue",lw=1,ls=line_style)
            #axs[l,1].plot(v_series.values,
            #                v_series.index/1000,color="thistle",lw=1,ls=line_style)
            #axs[l,2].plot(tra_series.values,tra_series.index/1000,color="grey",lw=1,ls=line_style)
    #    axs[l,0].plot(interp_vars_df_in["q"].mean(axis=1).values*1000,new_index/1000,color="darkblue",lw=2,ls="-")
    #    axs[l,0].plot(interp_vars_df_out["q"].mean(axis=1).values*1000,new_index/1000,color="darkblue",lw=2,ls="--")
        
            