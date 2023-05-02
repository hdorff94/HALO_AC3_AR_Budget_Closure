# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:00:45 2023

@author: u300737
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

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

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

campaign="HALO_AC3"
ar_of_day="AR_entire_1"
flight=["RF05"]
flight_dates={"RF05":"20220315",
              "RF06":"20220316"}
sector_to_plot="warm"
take_arbitary=False
do_plotting=True
plot_path=start_path+"/../plots/"
print(plot_path)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    
from simplified_flight_leg_handling import simplified_run_grid_main
#-----------------------------------------------------------------------------#        
"""
###############################################################################
    Main Script for running interpolation of griddata on flight path
###############################################################################
"""    
with HiddenPrints():
    halo_era5,halo_df,cmpgn_cls,ERA5_on_HALO,radar,Dropsondes=\
        simplified_run_grid_main(flight=flight,
                                 config_file_path=major_work_path,
                                 ar_of_day=ar_of_day)

if not "Lat" in [*Dropsondes.keys()]:
    sondes_lon=[[*Dropsondes["reference_lon"].values()][sonde].data[0] \
                    for sonde in range(Dropsondes["IWV"].shape[0])]
                    
    sondes_lat=[[*Dropsondes["reference_lat"].values()][sonde].data[0]\
                    for sonde in range(Dropsondes["IWV"].shape[0])]
    Dropsondes["Lat"]=pd.Series(data=np.array(sondes_lat),
                                                index=Dropsondes["IWV"].index)
    Dropsondes["Lon"]=pd.Series(data=np.array(sondes_lon),
                                                index=Dropsondes["IWV"].index)
sonde_times_series=pd.Series(index=Dropsondes["IWV"].index.values,
                             data=range(Dropsondes["IWV"].shape[0]))

import reanalysis as Reanalysis
file_name="total_columns_"+cmpgn_cls.years[flight[0]]+"_"+\
                    cmpgn_cls.flight_month[flight[0]]+"_"+\
                    cmpgn_cls.flight_day[flight[0]]+".nc"    
        
era5=Reanalysis.ERA5(for_flight_campaign=True,campaign="HALO_AC3",
                  research_flights=flight,
                  era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
        
ds,era_path=era5.load_era5_data(file_name)
        
        #IVT Processing
ds["IVT_v"]=ds["p72.162"]
ds["IVT_u"]=ds["p71.162"]
ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
ds["IVT_conv"]=ds["p84.162"]*3600 # units in seconds


###############################################################################
inflow_times=["2022-03-15 10:11","2022-03-15 11:13"]
internal_times=["2022-03-15 11:18","2022-03-15 12:14"]
outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
                    
new_halo_dict={flight[0]:{"inflow":
                          halo_df.loc[inflow_times[0]:inflow_times[-1]],
                          "internal":
                          halo_df.loc[internal_times[0]:internal_times[-1]],
                          "outflow":
                          halo_df.loc[outflow_times[0]:outflow_times[-1]]}}

from atmospheric_rivers import Atmospheric_Rivers
AR_inflow,AR_outflow=Atmospheric_Rivers.locate_AR_cross_section_sectors(
                                    new_halo_dict,ERA5_on_HALO.halo_era5,
                                    flight[0])
print(AR_inflow["AR_inflow"].keys())
relevant_sondes_dict={}
internal_sondes_dict={}
relevant_warm_sector_sondes=[0,1,2,3,9,10,11,12]
relevant_cold_sector_sondes=[4,5,6]
relevant_warm_internal_sondes=[7,13]
relevant_sondes_dict["warm_sector"]        = {}
relevant_sondes_dict["warm_sector"]["in"]  = sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
relevant_sondes_dict["warm_sector"]["out"] = sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
relevant_sondes_dict["cold_sector"]        = {}
relevant_sondes_dict["cold_sector"]["in"]  = sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
#relevant_sondes_dict["cold_sector"]["out"] = sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
synthetic_sonde_times_series=pd.Series(data=["7synth","8synth","9synth"],
                        index=pd.DatetimeIndex(["2022-03-15 12:55",
                                                "2022-03-15 13:05",
                                                "2022-03-15 13:15"]))
relevant_sondes_dict["cold_sector"]["out"] = synthetic_sonde_times_series
internal_sondes_dict["warm"]               = sonde_times_series.iloc[relevant_warm_internal_sondes]
internal_sondes_dict["cold"]               = ["2022-03-15 11:30:00","2022-03-15 13:35"]   
#internal_sondes_dict["warm"]       = {}

###############################################################################
relevant_sector_sondes={}
relevant_sector_sondes["warm"]=relevant_warm_sector_sondes
relevant_sector_sondes["cold"]=relevant_cold_sector_sondes
print(relevant_sector_sondes)
## --> add relevant sondes with "cold" and "warm" key.    
inflow=False
# Load config file
config_file=data_config.load_config_file(major_work_path,"data_config_file")
cmpgn_cls=flightcampaign.HALO_AC3(is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])               

Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,flight,config_file,
                 grid_name="Real_Sondes",sector_types=[sector_to_plot],
                 do_instantan=False)
###############################################################################
relevant_sector_sondes
Moisture_CONV.perform_entire_sonde_ac3_divergence_calcs(Dropsondes,
                                                        relevant_sector_sondes)
Moisture_CONV.vertically_integrated_divergence()
print("Divergence:",Moisture_CONV.integrated_divergence)
###############################################################################
relevant_sondes=relevant_sector_sondes["warm"]

sonde_sector_times=sonde_times_series.iloc[relevant_sondes].index
