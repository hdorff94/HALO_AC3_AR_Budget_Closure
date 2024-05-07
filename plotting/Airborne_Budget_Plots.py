# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:45:59 2024

@author: u300737
"""

import os
import sys


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

try: 
    from moisturebudget import Moisture_Budgets, Moisture_Convergence, Moisture_Budget_Plots
except:
    current_path=os.getcwd()
    git_path=current_path+"/../../"
    synth_path=git_path+"Synthetic_Airborne_Arctic_ARs//"
    if not synth_path in sys.path:
        sys.path.insert(10,synth_path+"src/")
    from moisturebudget import Moisture_Budget,Moisture_Convergence, Moisture_Budget_Plots
class HALO_AC3_Budget_Plots(Moisture_Budget_Plots):
    """
    This is the major plotting class for all HALO-(AC)3 moisture budget 
    components. It is mainly designed for the second manuscript of the PhD of 
    Henning Dorff. This study determines all moisture budget components for
    an AR event and assesses the budget equation closure.
    
    """
    def __init__(self,cmpgn_cls,flight,ar_of_day,
                 grid_name="ERA5",do_instantan=False,sonde_no=3,
                 scalar_based_div=True):
        
        super().__init__(cmpgn_cls,flight,#config_file,
                         grid_name,do_instantan)
        self.ar_of_day=ar_of_day
        self.plot_path=os.getcwd()+"/../plots/" # ----> to be filled
        self.grid_name=grid_name
        self.sonde_no=sonde_no
        self.scalar_based_div=scalar_based_div
        self.cmpgn_cls=cmpgn_cls
        self.budget_index={"RF05_AR_entire_1":"S1",
                           "RF05_AR_entire_2":"S2",
                           "RF06_AR_entire_1":"S3",
                           "RF06_AR_entire_2":"S4"}
        self.budget_sector_path=os.getcwd()+"/../../"+"/../Work/GIT_Repository/"+\
            self.cmpgn_cls.name+"/data/budgets/"
        print(self.budget_sector_path)
        self.budget_sector_file="Warm_Sectors_budget_components.csv"
        
    def save_sector_budget_components(self):
        self.budget_df.to_csv(self.budget_sector_path+self.budget_sector_file,
                              index=True)
        print("budget df saved as:",self.budget_sector_path+self.budget_sector_file)
    
    def load_sector_budget_components(self):
        if not os.path.exists(self.budget_sector_path+self.budget_sector_file):
            #create dataframe
            self.budget_df=pd.DataFrame(data=np.nan,index=["S1","S2","S3","S4"],
                                   columns=["IWV_dt","IWV_dt_unc",
                                            "ADV_q","ADV_q_unc",
                                            "DIV_mass","DIV_mass_unc",
                                            "Precip","Precip_unc",
                                            "Evap","Evap_unc"])
            #print(self.budget_df)
            self.save_sector_budget_components()
        else:
            self.budget_df=pd.read_csv(self.budget_sector_path+\
                                       self.budget_sector_file,index_col=0)
            try:
                self.budget_df.index=self.budget_df["Unnamed: 0"]
                del self.budget_df["Unnamed: 0"]
            except:
                pass
    def change_values_in_budget_df(self,budget_comps):
        self.load_sector_budget_components()
        idx=self.budget_index[self.flight[0]+"_"+self.ar_of_day]
        print(budget_comps)
        for comp in budget_comps.index:
            self.budget_df[comp].loc[idx]=float(budget_comps.loc[comp])
        print(self.budget_df)
        self.save_sector_budget_components()    
        

class HALO_AC3_evaporation(HALO_AC3_Budget_Plots):
    def __init__(self,cmpgn_cls,flight,ar_of_day,#config_file,
                 halo_df,grid_name="ERA5",do_instantan=False,sonde_no=3,
                 scalar_based_div=True,is_flight_campaign=True,
                 major_path=os.getcwd(),sector="warm",
                 aircraft=None,instruments=[],flights=[],
                 interested_flights="all"):
        
        super().__init__(cmpgn_cls,flight,ar_of_day)
        self.major_path=major_path+"/../"
        self.plot_path=self.major_path+"/plots/"
        self.grid_name=grid_name
        
        self.halo_df=halo_df
        self.is_flight_campaign=is_flight_campaign
        self.aircraft=aircraft
        self.instruments=instruments
        self.flight_day={}
        self.flight_month={}
        self.interested_flights="all"
        self.major_path=major_path
        self.is_synthetic_campaign=True
        self.sector=sector
        matplotlib.rcParams.update({"font.size":24})
        
    def describe_model_evap(self):
        if hasattr(self,"halo_era5"):
            print("halo_era5")
            self.halo_era5.describe()
        if hasattr(self,"halo_icon_hmp"):
            self.halo_icon_hmp.describe()
    def plot_ICON_ERA5_evaporation_comparison(self):
        self.halo_icon_hmp["Interp_EV"]=-1*self.halo_icon_hmp["Interp_EV"]
        self.halo_icon_hmp["Interp_EV"].plot(label="ICON")
        self.halo_era5["Interp_E"].plot(label="ERA5")
        plt.plot(self.halo_era5.index,np.zeros(self.halo_era5.shape[0]),
                 ls="--",lw=3)
        plt.legend()
        fig_name=self.flight[0]+"_"+self.ar_of_day+"_ICON_ERA5_comparison.png"
        plt.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Evaporation fig saved as:",self.plot_path+fig_name)        
    def run_ICON_ERA5_evaporation_comparison(self,halo_era5):
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
        #######################################################################
        import flightcampaign
        import moisturebudget as Budgets
        #######################################################################
        #Grid Data
        from reanalysis import ERA5,CARRA
        from ICON import ICON_NWP as ICON
        import gridonhalo as Grid_on_HALO
        #######################################################################
        take_arbitary=True
        
        self.halo_era5=halo_era5

        self.flight_dates={
            "RF03":"20220313",
            "RF05":"20220315",
            "RF06":"20220316",
            "RF16":"20220410"}
        if not hasattr(self,"halo_icon_hmp"):
            # Load ICON File
            icon_major_path=self.cmpgn_cls.campaign_path+"/data/ICON_LEM_2KM/"
            self.hydrometeor_icon_path=self.cmpgn_cls.campaign_path+"/data/ICON_LEM_2KM/"
            icon_resolution=2000 # units m
            upsample_time="20min"
            date=self.flight_dates[self.flight[0]]
            interp_icon_hmp_file=self.flight[0]+"_"+self.ar_of_day+"_"+"interpolated_HMP.csv"
        
            icon_var_list=ICON.lookup_ICON_AR_period_data(self.cmpgn_cls.name,
                            self.flight,self.ar_of_day,icon_resolution,
                            self.hydrometeor_icon_path,synthetic=False)

            ICON_on_HALO=Grid_on_HALO.ICON_on_HALO(
                self.cmpgn_cls,icon_var_list,self.halo_df,self.flight,date,
                interpolated_hmp_file=interp_icon_hmp_file,
                interpolated_hmc_file=None,ar_of_day=self.ar_of_day,
                upsample_time=upsample_time,
                synthetic_icon=False,
                synthetic_flight=False)

            if self.cmpgn_cls.name=="HALO_AC3":
                hydrometeor_icon_path=self.hydrometeor_icon_path+self.flight[0]+"/"
                ICON_on_HALO.update_ICON_hydrometeor_data_path(hydrometeor_icon_path)

            self.halo_icon_hmp=ICON_on_HALO.load_interpolated_hmp()
        self.describe_model_evap()
        self.plot_ICON_ERA5_evaporation_comparison()
    def plot_quantities_quicklook(self,Evap_cls):
        self.Evap_cls=Evap_cls
        surface_data=self.Evap_cls.surface_data
        sst=self.Evap_cls.halo_era5["Interp_SST"]
        import matplotlib.pyplot as plt

        evap_fig=plt.figure(figsize=(18,12))
        ax1=evap_fig.add_subplot(311)
        ax2=evap_fig.add_subplot(312,sharex=ax1)
        ax3=evap_fig.add_subplot(313,sharex=ax1)

        ax1.plot(sst-273.15,lw=2,ls="-",color="darkred")
        ax1.errorbar(surface_data.index,
                     surface_data["ERA5_SST"].values-273.15,
                     ls="",yerr=1,elinewidth=2,markersize=10,
                     marker="s",color="grey",markeredgecolor="darkred")
        
        ax1.axhline(y=0,ls="--",lw=3,color="grey")
        ax1.set_ylim([-2,8])
        ax1.set_ylabel("SST / degC")
        ax2.errorbar(surface_data.index,surface_data["Shum"]*1000,
                     yerr=surface_data["Shum_unc"]*1000,
                     markersize=10, color="lightgreen", marker="o",
                     markeredgecolor="k",elinewidth=2,label="q")

        ax2.errorbar(surface_data.index,surface_data["Qsat"]*1000,
                     yerr=surface_data["Qsat_unc"]*1000,
                     markersize=10, color="darkgreen", marker="o",
                     markeredgecolor="k",elinewidth=2,label="$q_{sat}$")
        
        ax2.set_ylabel("Specific \nhumidity / g kg$^{-1}$")
        ax2.set_ylim([0,6])
        ax2.legend()
        ax3.plot(-Evap_cls.halo_era5["Interp_E"],color="darkblue",lw=2,ls="-")
        ax3.errorbar(surface_data.index,self.Evap_cls.evap_mm_h, 
             yerr=abs(self.Evap_cls.evap_mm_h_unc),
             markersize=10, color="lightblue", marker="s",
             markeredgecolor="k",elinewidth=2,label="Evap_${\text{Sonde}}$")
        ax3.axhline(y=0,ls="--",lw=3,color="grey")
        ax3.set_ylabel("Evaporation / mm h$^{-1}$")
        ax3.set_xlim([sst.index[0],sst.index[-1]])
        fig_name="Sonde_Evap_"+self.flight[0]+"_"+self.ar_of_day+".png"
        plot_path=self.plot_path+"/supplements/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        evap_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
    def plot_evap_internal(self,Evap_cls,warm_internal_halo):
        self.warm_internal_halo = warm_internal_halo
        self.Evap_cls           = Evap_cls
        evap_mm_h               = self.Evap_cls.evap_mm_h
        evap_mm_h_unc           = self.Evap_cls.evap_mm_h_unc
        surface_data=self.Evap_cls.surface_data
        sst=self.Evap_cls.halo_era5["Interp_SST"]
        matplotlib.rcParams.update({"font.size":24})
        evap_halo_fig=plt.figure(figsize=(18,9))
        ax1=evap_halo_fig.add_subplot(211)
        ax2=evap_halo_fig.add_subplot(212)
        # Specific humidity
        
        ax1.errorbar(surface_data.index,surface_data["Shum"]*1000, 
                     yerr=0.4, markersize=15, color="lightgreen", 
                     elinewidth=2,marker="v",markeredgecolor="k",
                     linestyle="",label="q")
        ax1.errorbar(surface_data.index,surface_data["Qsat"]*1000, yerr=0.4,
                     markersize=15, color="darkgreen",
                     elinewidth=2,marker="v",markeredgecolor="k",
                     linestyle="",label="$q_{sat}$")
        ax1.axvspan(sst.index[0],
                    warm_internal_halo.index[0],
                    color="lightgrey",alpha=0.3,zorder=3)
        ax1.axvspan(warm_internal_halo.index[-1],
                    sst.index[-1],color="lightgrey",alpha=0.3,zorder=3)
        if self.flight[0]=="RF05" and self.ar_of_day=="AR_entire_1":
            ax1.text(pd.Timestamp("2022-03-15 11:45"),2,
                     "Pre-frontal \ninternal leg",color="k")
        ax1.set_ylabel("Specific \nhumidity (g kg$^{-1}$)")
        ax1.set_ylim([0,6])
        ax1.set_yticks([0,2,4,6])
        ax1.legend()
        # Evaporation
        ax2.errorbar(surface_data.index,evap_mm_h, yerr=abs(evap_mm_h_unc),
                     color="lightblue", marker='v',markeredgecolor="k",
                     linestyle="",elinewidth=2,
                     markersize=15,label="${E}_{\mathrm{Sonde}}$")
        ax2.plot(-Evap_cls.halo_era5["Interp_E"],color="darkblue",lw=2,ls="-",
                 label="${E}_{\mathrm{ERA5}}$")
        ax2.axvspan(sst.index[0],
                    warm_internal_halo.index[0],
                    color="lightgrey",alpha=0.4,zorder=3)
        ax2.axvspan(warm_internal_halo.index[-1],
                    sst.index[-1],color="lightgrey",alpha=0.4,zorder=3)
        ax2.axhline(y=0,ls="--",lw=3,color="grey")
        ax2.set_ylabel("Evaporation (mm h$^{-1}$)")
        ax2.legend()
        ax1.set_xlim([sst.index[0],sst.index[-1]])
        ax2.set_xlim([sst.index[0],sst.index[-1]])
        ax1.spines["left"].set_linewidth(3)
        ax1.spines["bottom"].set_linewidth(3)
        ax2.spines["left"].set_linewidth(3)
        ax2.spines["bottom"].set_linewidth(3)
        ax1.tick_params(axis="x",width=3,length=8)
        ax2.tick_params(axis="x",width=3,length=8)
        ax1.tick_params(axis="y",width=3,length=8)
        ax2.tick_params(axis="y",width=3,length=8)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.set_xlabel("Time (UTC)")
        sector_label=self.budget_index[self.flight[0]+"_"+self.ar_of_day]
        fig_name="Sonde_Evap_"+self.flight[0]+"_"+sector_label+".png"
        sns.despine(offset=10)
        if self.flight[0]=="RF05" and self.ar_of_day=="AR_entire_1":
            fig_name="Fig08_"+fig_name
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+"/supplements/"
        evap_halo_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name) 