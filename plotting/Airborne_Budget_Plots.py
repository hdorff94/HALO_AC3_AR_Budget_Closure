# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:45:59 2024

@author: Henning Dorff
"""

import os
import sys


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import cartopy
import cartopy.crs as ccrs

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from cmcrameri import cm as cmaeri

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
                                            "Precip","Precip_min","Precip_max",
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
    def plot_AR_intensity_boxplot_Sonde_ICON_single(self,
            merged_halo_icon_ivt,sector_sondes):
        """
        This is the old version directly comparing sondes and ICON 
        in terms of IWV and IVT. It created the initial Fig. 10 of 
        HALO-AC3 AR moisture budget paper.

        Parameters
        ----------
        merged_halo_icon_ivt : TYPE
            DESCRIPTION.
        sector_sondes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        AR_intensity_fig=plt.figure(figsize=(12,5.5))
        ax1=AR_intensity_fig.add_subplot(111)
        ax2=ax1.twinx()
        ivt_props=dict(widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="mediumseagreen",color="k",lw=2),
            medianprops=dict(color="whitesmoke",lw=3,),
            whiskerprops=dict(lw=2),capprops=dict(lw=2))
    
        iwv_props=dict(widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="purple",color="k",lw=2),
            medianprops=dict(color="gold",lw=3,),
            whiskerprops=dict(lw=2),capprops=dict(lw=2))
    
        ax1.boxplot([merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S1"]["IVT"],
                    merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S2"]["IVT"],
                    merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S3"]["IVT"],
                    merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S4"]["IVT"]],
                    positions=[0.8,1.8,2.8,3.8],**ivt_props)
        ax1.scatter(np.repeat([0.8],sector_sondes["S1"]["IVT"].shape[0]),sector_sondes["S1"]["IVT"],marker="v",s=100,zorder=3,
                   color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([1.8],sector_sondes["S2"]["IVT"].shape[0]),sector_sondes["S2"]["IVT"],marker="v",s=100,zorder=3,
                   color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([2.8],sector_sondes["S3"]["IVT"].shape[0]),sector_sondes["S3"]["IVT"],marker="v",s=100,zorder=3,
                   color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([3.8],sector_sondes["S4"]["IVT"].shape[0]),sector_sondes["S4"]["IVT"],marker="v",s=100,zorder=3,
                   color="mintcream",edgecolor="k",lw=1.5)
    
        ax2.boxplot([merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S1"]["IWV_calc"],
                    merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S2"]["IWV_calc"],
                    merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S3"]["IWV_calc"],
                    merged_halo_icon_ivt[merged_halo_icon_ivt["Sector"]=="S4"]["IWV_calc"]],
                    positions=[1.2,2.2,3.2,4.2],**iwv_props)
    
        ax2.scatter(np.repeat([1.2],sector_sondes["S1"]["IWV"].shape[0]),sector_sondes["S1"]["IWV"],marker="v",s=100,zorder=3,
                   color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([2.2],sector_sondes["S2"]["IWV"].shape[0]),sector_sondes["S2"]["IWV"],marker="v",s=100,zorder=3,
                   color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([3.2],sector_sondes["S3"]["IWV"].shape[0]),sector_sondes["S3"]["IWV"],marker="v",s=100,zorder=3,
                   color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([4.2],sector_sondes["S4"]["IWV"].shape[0]),sector_sondes["S4"]["IWV"],marker="v",s=100,zorder=3,
                   color="lavender",edgecolor="k",lw=1.5)
    
        ax1.set_xlabel("AR Corridor")
        ax1.set_xticks([1,2,3,4])
        ax1.set_xticklabels(["S1","S2","S3","S4"])
        ax1.set_yticks([0,200,400])
        ax1.set_ylim([0,500])
        ax2.set_yticks([0,8,16])
        ax2.set_ylim([0,20])
        ax1.set_ylabel("IVT ($\mathrm{kg\,m}^{-1}{\mathrm{s}}^{-1})$",color="darkgreen")
        ax2.set_ylabel("IWV ($\mathrm{kg\,m}^{-2})$",color="purple")
        ax1.spines["bottom"].set_linewidth(2)
        ax1.spines["left"].set_linewidth(2)
        ax1.tick_params("x",width=2,length=8)
        ax1.tick_params("y",width=2,length=8)
        ax2.tick_params("y",width=2,length=8)
    
        sns.despine(offset=10,ax=ax1)
        sns.despine(offset=10,ax=ax2)
        fig_name="Fig10_HALO_RF05_RF06_IVT_tendency.pdf"
        plot_path=os.getcwd()+"/../plots/"
        AR_intensity_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:", plot_path+fig_name)
        
    def plot_sonde_based_AR_intensity_boxplot(self,sector_sondes):
        matplotlib.rcParams.update({"font.size":20})
        AR_intensity_fig=plt.figure(figsize=(12,5.5))
        ax1=AR_intensity_fig.add_subplot(111)
        ax2=ax1.twinx()
        ivt_props=dict(widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="mediumseagreen",color="k",lw=2),
            medianprops=dict(color="whitesmoke",lw=3,),
            whiskerprops=dict(lw=2),capprops=dict(lw=2))

        iwv_props=dict(widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="purple",color="k",lw=2),
            medianprops=dict(color="gold",lw=3,),
            whiskerprops=dict(lw=2),capprops=dict(lw=2))

        ax1.boxplot([sector_sondes["S1"]["IVT"],
                sector_sondes["S2"]["IVT"],
                sector_sondes["S3"]["IVT"],
                sector_sondes["S4"]["IVT"]],
                positions=[0.8,1.8,2.8,3.8],**ivt_props)
        ax1.scatter(np.repeat([0.8],sector_sondes["S1"]["IVT"].shape[0]),
                    sector_sondes["S1"]["IVT"],marker="v",s=100,zorder=3,
                    color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([1.8],sector_sondes["S2"]["IVT"].shape[0]),
                    sector_sondes["S2"]["IVT"],marker="v",s=100,zorder=3,
               color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([2.8],sector_sondes["S3"]["IVT"].shape[0]),
                    sector_sondes["S3"]["IVT"],marker="v",s=100,zorder=3,
               color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([3.8],sector_sondes["S4"]["IVT"].shape[0]),
                    sector_sondes["S4"]["IVT"],marker="v",s=100,zorder=3,
               color="mintcream",edgecolor="k",lw=1.5)

        ax2.boxplot([sector_sondes["S1"]["IWV"],
            sector_sondes["S2"]["IWV"],
            sector_sondes["S3"]["IWV"],
            sector_sondes["S4"]["IWV"]],
            positions=[1.2,2.2,3.2,4.2],
            **iwv_props)

        ax2.scatter(np.repeat([1.2],sector_sondes["S1"]["IWV"].shape[0]),
                    sector_sondes["S1"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([2.2],sector_sondes["S2"]["IWV"].shape[0]),
                    sector_sondes["S2"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([3.2],sector_sondes["S3"]["IWV"].shape[0]),
                    sector_sondes["S3"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)
        
        ax2.scatter(np.repeat([4.2],sector_sondes["S4"]["IWV"].shape[0]),
                    sector_sondes["S4"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)

        ax1.set_xlabel("AR Corridor")
        ax1.set_xticks([1,2,3,4])
        ax1.set_xticklabels(["S1","S2","S3","S4"])
        ax1.set_yticks([0,200,400])
        ax1.set_ylim([0,500])
        ax2.set_yticks([0,8,16])
        ax2.set_ylim([0,20])
        ax1.set_ylabel("IVT ($\mathrm{kg\,m}^{-1}{\mathrm{s}}^{-1})$",
                       color="darkgreen")
        ax2.set_ylabel("IWV ($\mathrm{kg\,m}^{-2})$",
                       color="purple")
        ax1.spines["bottom"].set_linewidth(2)
        ax1.spines["left"].set_linewidth(2)
        ax1.tick_params("x",width=2,length=8)
        ax1.tick_params("y",width=2,length=8)
        ax2.tick_params("y",width=2,length=8)
    
        sns.despine(offset=10,ax=ax1)
        sns.despine(offset=10,ax=ax2)
        fig_name="Fig10_HALO_RF05_RF06_sonde_IWV_IVT_tendency.pdf"
        AR_intensity_fig.savefig(
            self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:", self.plot_path+fig_name)
    def plot_AR_intensity_sonde_boxplots_with_icon(
            self,merged_halo_icon_ivt,sector_sondes):
        
        print("Not yet available. Under construction.")
        sys.exit()
        matplotlib.rcParams.update({"font.size":20})
        AR_intensity_fig=plt.figure(figsize=(12,5.5))
        ax1=AR_intensity_fig.add_subplot(111)
        ax2=ax1.twinx()
        ivt_props=dict(widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="mediumseagreen",color="k",lw=2),
            medianprops=dict(color="whitesmoke",lw=3,),
            whiskerprops=dict(lw=2),capprops=dict(lw=2))

        iwv_props=dict(widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="purple",color="k",lw=2),
            medianprops=dict(color="gold",lw=3,),
            whiskerprops=dict(lw=2),capprops=dict(lw=2))

        ax1.boxplot([sector_sondes["S1"]["IVT"],
                sector_sondes["S2"]["IVT"],
                sector_sondes["S3"]["IVT"],
                sector_sondes["S4"]["IVT"]],
                positions=[0.8,1.8,2.8,3.8],**ivt_props)
        ax1.scatter(np.repeat([0.8],sector_sondes["S1"]["IVT"].shape[0]),
                    sector_sondes["S1"]["IVT"],marker="v",s=100,zorder=3,
                    color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([1.8],sector_sondes["S2"]["IVT"].shape[0]),
                    sector_sondes["S2"]["IVT"],marker="v",s=100,zorder=3,
               color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([2.8],sector_sondes["S3"]["IVT"].shape[0]),
                    sector_sondes["S3"]["IVT"],marker="v",s=100,zorder=3,
               color="mintcream",edgecolor="k",lw=1.5)
        ax1.scatter(np.repeat([3.8],sector_sondes["S4"]["IVT"].shape[0]),
                    sector_sondes["S4"]["IVT"],marker="v",s=100,zorder=3,
               color="mintcream",edgecolor="k",lw=1.5)

        ax2.boxplot([sector_sondes["S1"]["IWV"],
            sector_sondes["S2"]["IWV"],
            sector_sondes["S3"]["IWV"],
            sector_sondes["S4"]["IWV"]],
            positions=[1.2,2.2,3.2,4.2],
            **iwv_props)

        ax2.scatter(np.repeat([1.2],sector_sondes["S1"]["IWV"].shape[0]),
                    sector_sondes["S1"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([2.2],sector_sondes["S2"]["IWV"].shape[0]),
                    sector_sondes["S2"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)
        ax2.scatter(np.repeat([3.2],sector_sondes["S3"]["IWV"].shape[0]),
                    sector_sondes["S3"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)
        
        ax2.scatter(np.repeat([4.2],sector_sondes["S4"]["IWV"].shape[0]),
                    sector_sondes["S4"]["IWV"],marker="v",s=100,zorder=3,
                    color="lavender",edgecolor="k",lw=1.5)

        ax1.set_xlabel("AR Corridor")
        ax1.set_xticks([1,2,3,4])
        ax1.set_xticklabels(["S1","S2","S3","S4"])
        ax1.set_yticks([0,200,400])
        ax1.set_ylim([0,500])
        ax2.set_yticks([0,8,16])
        ax2.set_ylim([0,20])
        ax1.set_ylabel("IVT ($\mathrm{kg\,m}^{-1}{\mathrm{s}}^{-1})$",
                       color="darkgreen")
        ax2.set_ylabel("IWV ($\mathrm{kg\,m}^{-2})$",
                       color="purple")
        ax1.spines["bottom"].set_linewidth(2)
        ax1.spines["left"].set_linewidth(2)
        ax1.tick_params("x",width=2,length=8)
        ax1.tick_params("y",width=2,length=8)
        ax2.tick_params("y",width=2,length=8)
    
        sns.despine(offset=10,ax=ax1)
        sns.despine(offset=10,ax=ax2)
        fig_name="Fig10_HALO_RF05_RF06_sonde_IWV_IVT_tendency.pdf"
        AR_intensity_fig.savefig(
            self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:", self.plot_path+fig_name)
        
    def updt_plot_component_tendency(self,with_residuals=False,
                                     save_as_manuscript_fig=True):
        matplotlib.rcParams.update({"font.size":20})
        budget_components_fig=plt.figure(figsize=(12,6))
        ax1=budget_components_fig.add_subplot(111)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.xaxis.set_tick_params(width=2,length=4)
        ax1.yaxis.set_tick_params(width=2,length=4)
        marker_size=18
        # Local change IWV
        ax1.errorbar([0.7,1.7,2.7,3.7],self.budget_df["IWV_dt"].values,
            yerr=self.budget_df["IWV_dt_unc"], marker='v', mfc='grey',
            mec='black', ecolor="black",ms=marker_size+2,
            lw=2,mew=1,ls="",label="$\delta IWV/ \delta t$",barsabove=True)
        # Evaporation
        ax1.errorbar([0.9,1.9,2.9,3.9], self.budget_df["Evap"].values,
            yerr=self.budget_df["Evap_unc"], marker='s',
            mfc='red',mec='black', ecolor="black",lw=2,
            ms=marker_size, mew=1,ls="",label="$E$",barsabove=True)
        # Precipitation
        ax1.errorbar([1,2,3,4],-1*self.budget_df["Precip"].values,
            yerr=[self.budget_df["Precip_max"].values-\
                  self.budget_df["Precip"].values,
                  self.budget_df["Precip"].values-\
                  self.budget_df["Precip_min"].values],
            marker="s",ecolor="black",mfc="lightblue",
            mec="black",ms=marker_size,mew=1,lw=2,
            ls="",label="-$P$",barsabove=True)
                     # ---> think about direction and sign of errorbars
            
        # Mass divergence
        ax1.errorbar(np.array([1.1,2.1,3.1,4.1]),
            self.budget_df["DIV_mass"].values,
            yerr=self.budget_df["DIV_mass_unc"].values,marker="s",
            mfc="teal",ls="",ms=marker_size,mew=1,markeredgecolor="k",
            lw=2,ecolor="k",label="$-IDIV_{\mathrm{mass}}$",
            barsabove=True)
        
            #yerr=[mass_div_series_max-mass_div_series,mass_div_series-mass_div_series_min],
            
        # Moisture Advection
        ax1.errorbar(np.array([1.2,2.2,3.2,4.2]),
                     self.budget_df["ADV_q"].values,
            yerr=self.budget_df["ADV_q_unc"],
            #yerr=[adv_q_series_max-adv_q_series,adv_q_series-adv_q_series_min],
            marker="s",ms=marker_size,mew=1,mfc="darkgreen",ls="",
            markeredgecolor="k",ecolor="k",lw=2,
            label="$IADV_{\mathrm{q}}$",barsabove=True)
        #Residuals
        if with_residuals:
            ax1.errorbar(np.array([1.3,2.3,3.3,4.3]),
                self.budget_df["residual"],
                yerr=self.budget_df["residual_unc"],
                marker="x",mfc="k",mec="k",ecolor="k",lw=2,
                ms=marker_size,mew=1,ls="",label="residual",
                barsabove=True)
            
        ##
        ax1.axhline(y=0,ls="--",lw=2,color="k")
        # create vertical border lines
        ax1.axvline(x=1.5,ls="--",lw=1,color="lightgrey")
        ax1.axvline(x=2.5,ls="--",lw=1,color="lightgrey")
        ax1.axvline(x=3.5,ls="--",lw=1,color="lightgrey")
        ax1.set_ylabel(
            "Moisture Budget \nContribution ($\mathrm{mm\,h}^{-1}$)")
        ax1.set_xticks([1,2,3,4])
        ax1.set_xticklabels(["S1","S2","S3","S4"])#
        
        col_number=5
        if with_residuals:
            col_number+=1
        legend=ax1.legend(loc="lower right",
            ncol=col_number,frameon=True,
            fontsize=16)
        frame = legend.get_frame()
        frame.set_color('lightgrey')
        frame.set_edgecolor('black')
        ax1.set_ylim([-1,1])
        ax1.set_yticks([-1,-.5,0,.5,1])
        #import matplotlib.patches as mpatches
        sns.despine(offset=10)
        file_end=".pdf"
        fig_name="HALO_Budget_components_tendency"
        if save_as_manuscript_fig:
            fig_name="Fig11_"+fig_name
        if with_residuals:
            fig_name+="_with_residuals"
        fig_name+=file_end
        budget_components_fig.savefig(self.plot_path+fig_name,
                                      dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
    
    def plot_budget_residuals(self,save_as_manuscript_plot=True):
        """
        This routine plots the resiudals emerging from
        the moisture budget closure for all sectors
        """
    
        matplotlib.rcParams.update({"font.size":20})
        residuals_fig, ax = plt.subplots(1,1,figsize=(12,6))
        ax.errorbar(np.array([1,2,3,4]),self.budget_df["residual"],
                yerr=self.budget_df["residual_unc"],marker="X",mfc="red",
                mec="k",ecolor="darkred",lw=4,ms=20,
                mew=2,ls="")
    
        ax.axhline(y=0,ls="--",lw=1,color="k")
        ax.set_ylabel("Moisture Budget \nContribution ($\mathrm{mm\,h}^{-1}$)")
        ax.set_xticks([1,2,3,4])
        ax.set_xticklabels(["S1","S2","S3","S4"])#
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2,length=4)
        ax.yaxis.set_tick_params(width=2,length=4)
        
        ax.bar([0.75,1.75,2.75,3.75],self.budget_df["IWV_dt"],
               width=0.1,edgecolor="white",
               yerr=self.budget_df["IWV_dt_unc"],
               label="$\delta IWV/ \delta t$",
               color="grey",alpha=0.5)
        ax.bar([0.85,1.85,2.85,3.85],
               self.budget_df["Evap"],width=0.1,edgecolor="white",
                   yerr=self.budget_df["Evap_unc"],
               label="$E$",color="red",alpha=0.5)
        ax.bar([0.95,1.95,2.95,3.95],-self.budget_df["Precip"],
               width=0.1,edgecolor="white",
               yerr=(self.budget_df["Precip_max"]-\
                    self.budget_df["Precip_min"])/2,
               label="$-P$",color="lightblue",alpha=0.5)
        
        ax.bar([1.05,2.05,3.05,4.05],self.budget_df["DIV_mass"],
               width=0.1,edgecolor="white",
               yerr=self.budget_df["DIV_mass_unc"],
               label="$-IDIV_{\mathrm{mass}}$",
               color="teal",alpha=0.5)
            
        ax.bar([1.15,2.15,3.15,4.15],self.budget_df["ADV_q"],
               width=0.1,edgecolor="white",
               yerr=self.budget_df["ADV_q_unc"],
               label="$IADV_{\mathrm{q}}$",
               color="lightgreen",alpha=0.5)
        
        
        ax.set_ylim([-1,1.5])
        ax.set_yticks([-1,-.5,0,.5,1,1.5])
        ax.legend(loc="lower right",ncol=5,
                  frameon=True,fontsize=16)
        
        sns.despine(offset=10)
        file_end=".pdf"
        fig_name="HALO_Budget_residual_tendency"
        fig_name+=file_end
        if save_as_manuscript_plot:
            fig_name="Fig13_"+fig_name
        residuals_fig.savefig(self.plot_path+fig_name,
                              dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
    
        

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
        font_size=24
        matplotlib.rcParams.update({"font.size":font_size})
        evap_halo_fig=plt.figure(figsize=(18,9))
        ax1=evap_halo_fig.add_subplot(211)
        ax2=evap_halo_fig.add_subplot(212)
        # Specific humidity
        
        ax1.errorbar(surface_data.index,
            surface_data["Shum"]*1000, yerr=0.4, 
            markersize=15, color="lightgreen", 
            elinewidth=2,marker="v",markeredgecolor="k",
            linestyle="",label="q")
        ax1.errorbar(surface_data.index,
            surface_data["Qsat"]*1000, yerr=0.4,
            markersize=15, color="darkgreen",
            elinewidth=2,marker="v",markeredgecolor="k",
            linestyle="",label="$q_{sat}$")
        ax1.axvspan(sst.index[0],
                    warm_internal_halo.index[0],
                    color="lightgrey",alpha=0.3,zorder=3)
        ax1.axvspan(warm_internal_halo.index[-1],
            sst.index[-1],color="lightgrey",
            alpha=0.3,zorder=3)
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
        ax1.text(0.005,0.9,"(a)",color="k",fontsize=font_size-4,
            bbox=dict(facecolor='whitesmoke',
            edgecolor="k",boxstyle="round",
            alpha=0.8),transform=ax1.transAxes,zorder=10)
        ax2.text(0.005,0.9,"(b)",color="k",fontsize=font_size-4,
            bbox=dict(facecolor='whitesmoke',edgecolor="k",
                      boxstyle="round",alpha=0.8),
            transform=ax2.transAxes,zorder=10)
        sector_label=self.budget_index[self.flight[0]+\
                        "_"+self.ar_of_day]
        fig_name="Sonde_Evap_"+self.flight[0]+"_"+\
            sector_label+".png"
        sns.despine(offset=10)
        if self.flight[0]=="RF05" and \
            self.ar_of_day=="AR_entire_1":
            fig_name="Fig08_"+fig_name
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+"/supplements/"
        evap_halo_fig.savefig(plot_path+fig_name,
            dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
        
class HALO_AC3_precipitation(HALO_AC3_Budget_Plots,):
    def __init__(self,cmpgn_cls,halo_df,halo_era5,halo_icon_hmp,hydro_ds,
        processed_radar,warm_radar_rain,warm_icon_rain,
        cold_radar_rain,cold_icon_rain,precipitation_rate,
        strong_precip_rate,radar_str,flight,ar_of_day,
        grid_name="ERA5",do_instantan=False,is_flight_campaign=True,
        major_path=os.getcwd(),sector="warm",calibrated_radar=True,
        attenuation_value=4,aircraft=None,instruments=[],flights=[],
        interested_flights="all"):
        
        super().__init__(cmpgn_cls,flight,ar_of_day)
        self.major_path=major_path+"/../"
        self.airborne_data_importer_path=\
            self.major_path+"/../../Work/GIT_Repository/hamp_processing_py/"+\
                "hamp_processing_python/Flight_Data/"+self.cmpgn_cls.name+"/"
        self.plot_path=self.major_path+"/plots/"
        self.grid_name=grid_name
        
        self.is_flight_campaign    = is_flight_campaign
        self.aircraft              = aircraft
        self.instruments           = instruments
        self.flight_day            = {}
        self.flight_month          = {}
        self.interested_flights    = "all"
        self.major_path            = major_path
        self.is_synthetic_campaign = True
        self.sector                = sector
        self.radar_str             = radar_str
        self.calibrated_radar      = calibrated_radar
        
        self.halo_df               = halo_df
        self.hydro_ds              = hydro_ds
        self.halo_era5             = halo_era5
        self.halo_icon_hmp         = halo_icon_hmp
        self.processed_radar       = processed_radar
        self.precipitation_rate    = precipitation_rate
        self.strong_precip_rate    = strong_precip_rate
        self.attenuation_value     = attenuation_value
        self.warm_radar_rain       = warm_radar_rain
        self.warm_icon_rain        = warm_icon_rain
        self.cold_radar_rain       = cold_radar_rain
        self.cold_icon_rain        = cold_icon_rain
        
        matplotlib.rcParams.update({"font.size":24})
        ### Times dict for all sectors
        self.times_dict={}
        
        self.times_dict["RF05__AR_entire_1"]={}
        self.times_dict["RF05__AR_entire_2"]={}
        self.times_dict["RF06__AR_entire_1"]={}
        self.times_dict["RF06__AR_entire_2"]={}
        
        self.times_dict["RF05__AR_entire_1"]["inflow"]=\
            ["2022-03-15 10:11","2022-03-15 11:13"]
        self.times_dict["RF05__AR_entire_1"]["internal"]=\
            ["2022-03-15 11:18","2022-03-15 12:14"]
        self.times_dict["RF05__AR_entire_1"]["outflow"]=\
            ["2022-03-15 12:20","2022-03-15 13:15"]
        self.times_dict["RF05__AR_entire_2"]["inflow"]=\
            ["2022-03-15 14:30","2022-03-15 15:25"]
        self.times_dict["RF05__AR_entire_2"]["internal"]=\
            ["2022-03-15 13:20","2022-03-15 14:25"]
        self.times_dict["RF05__AR_entire_2"]["outflow"]=\
            ["2022-03-15 12:20","2022-03-15 13:15"]
        self.times_dict["RF06__AR_entire_1"]["inflow"]=\
            ["2022-03-16 10:45","2022-03-16 11:21"]
        self.times_dict["RF06__AR_entire_1"]["internal"]=\
            ["2022-03-16 11:25","2022-03-16 12:10"]
        self.times_dict["RF06__AR_entire_1"]["outflow"]=\
            ["2022-03-16 12:15","2022-03-16 12:50"]
        self.times_dict["RF06__AR_entire_2"]["inflow"]=\
            ["2022-03-16 12:12","2022-03-16 12:55"]
        self.times_dict["RF06__AR_entire_2"]["internal"]=\
            ["2022-03-16 12:58","2022-03-16 13:40"]
        self.times_dict["RF06__AR_entire_2"]["outflow"]=\
            ["2022-03-16 13:45","2022-03-16 14:18"]
#    def save_sector_radar_precip(self):
#        #describe warm rain rate
#        sector_precip_file_name=self.sector+"_precip_"+\
#            self.flight[0]+"_"+self.ar_of_day+".csv"
#        self.warm_radar_rain.to_csv(self.precip_rate_path+warm_precip_file_name)
#print("Warm precipitation saved as:",precip_rate_path+warm_precip_file_name)
#try:
#    cold_precip_file_name="cold_precip_"+flight[0]+"_"+ar_of_day+".csv"
#    cold_radar_rain.to_csv(precip_rate_path+cold_precip_file_name)
#    print("Cold rain saved as:",precip_rate_path+cold_precip_file_name)
#except:
#    "No cold sector available"
    def map_precipitation_region(self,sector_radar_rain,sector="warm"):
        matplotlib.rcParams.update({"font.size":18})
        precip_map=plt.figure(figsize=(12,12))

        if self.flight[0]=="RF05":
            central_lat=70
            central_lon=-10
        elif self.flight[0]=="RF06":
            central_lat=74
            central_lon=15
        time_step=0
        #print(self.hydro_ds.time[time_step])
        ax1 = plt.subplot(2,2,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=central_lon,
                                central_latitude=central_lat))
        # IVT convergence as background based on ERA5
        C1=ax1.scatter(np.rad2deg(self.hydro_ds["clon"]),
            np.rad2deg(self.hydro_ds.clat),
            c=self.hydro_ds["hourly_prec"][time_step,:],
            cmap="GnBu",s=0.5,vmin=0.0,vmax=0.8,
            transform=ccrs.PlateCarree())
        cbar=plt.colorbar(C1,ax=ax1,shrink=0.5)
        cbar.set_label("Precipitation ($\mathrm{mm\,h}^{-1}$)")
        ax1.set_extent([self.halo_df["longitude"].min()-4,
                        self.halo_df["longitude"].max()+2,
                        self.halo_df["latitude"].min()-1.75,
                        self.halo_df["latitude"].max()+2])
        #%print(sector_radar_rain.loc["2022-03-15 11:25":"2022-03-15 11:39"])
        sector_radar_rain.loc["2022-03-15 11:25":"2022-03-15 11:43"]=np.nan
        #print(sector_radar_rain.loc["2022-03-15 11:25":"2022-03-15 11:39"]["rate"])
        sector_radar_rain=sector_radar_rain.dropna(subset="rate",how="all")
        if sector!="":
            # cut the halo for the rectangle just to the specific sector
            halo_df=self.halo_df.loc[sector_radar_rain.index]
            if sector=="warm":
                halo_color="orange"
            else:
                halo_color="purple"
    
        #ax1.plot([halo_df["longitude"].min(),halo_df["longitude"].min()+2,
        #          halo_df["longitude"].max()+1,halo_df["longitude"].max()-1,
        #          halo_df["longitude"].min()],
        #        [halo_df["latitude"].min()+.5,halo_df["latitude"].max(),
        #          halo_df["latitude"].max()-.5,halo_df["latitude"].min()-.5,
        #          halo_df["latitude"].min()+.5],
        ax1.plot([-7.5,-1.5,
                  13,5,-7.5],
                 [72,77,75.5,70.5,72],
                 lw=3,ls="-",color="white",
                 transform=ccrs.PlateCarree(),zorder=3)
    
        ax1.plot([-7.5,-1.5,
                  13,5,-7.5],
                 [72,77,75.5,70.5,72],
                 lw=2,ls="--",color=halo_color,
                 transform=ccrs.PlateCarree(),zorder=3)
    
        ax1.scatter(sector_radar_rain["lon"],
                    sector_radar_rain["lat"],
                    color="darkgrey",s=40,
                    transform=ccrs.PlateCarree(),zorder=5)
        ax1.scatter(sector_radar_rain["lon"],sector_radar_rain["lat"],
                c=sector_radar_rain["rate"],cmap="GnBu",s=15,vmin=0.0,
                vmax=0.8,transform=ccrs.PlateCarree(),zorder=6)
        ax1.coastlines(resolution="50m")
        gl1=ax1.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        gl1.xlabels_top = False
        gl1.ylabels_right = False
        ax1.plot(self.processed_radar["lon"],
                 self.processed_radar["lat"],color="k",ls="--",lw=1,
                 transform=ccrs.PlateCarree())
        fig_name="Map_box_precipitation"
        fig_name=self.flight[0]+"_"+self.ar_of_day+"_"+sector+\
            "_"+fig_name
        fig_path=self.plot_path
        #sns.despine(offset=1)
        plt.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as ",fig_path+fig_name)

    def quicklook_sector(self,radar_rain,sector="warm"):
        if sector=="warm":
            halo_color="orange"
        else:
            halo_color="purple"
            plt.scatter(radar_rain["lon"],
                        radar_rain["lat"],color=halo_color)
        #cold_icon_rain=halo_icon_hmp.loc[cold_radar_rain.index]
        #cold_icon_rain["rate"]=cold_icon_rain["Interp_Precip"]
    
        #plt.scatter(cold_radar_rain["lon"],
        #cold_radar_rain["lat"],color="purple")
        #warm_icon_rain
    
    def boxplot_precip_comparison(self,precip_icon_series,
                                  halo_icon_hmp,halo_era5,
                                  radar_precip_rate):
        # statistics only refer to cases when it precipitates:
        #    conditional statistics
        # Organise data
        x1 = precip_icon_series.copy()
        x1 = x1[x1>0]
        x1 = x1.dropna()
        x1.index=range(len(x1))
        x2 = halo_icon_hmp["Interp_Precip"]
        x2 = x2[x2>0]
        x2.index=range(len(x2))
        x3 = halo_era5["Interp_Precip"]
        x3 = x3[x3>0]
        x3.index = range(len(x3))
        x4 = radar_precip_rate["rate"]
        x4 = x4[x4>0]
        x4.index = range(len(x4))
        
        x_appended=pd.DataFrame(data=np.nan,columns=["Rates","Data","No"],
                                index=x1.index)
        x_appended["Rates"]=x1.values.astype(float)
        x_appended["Data"] ="ICON_AR_Region"
        x_appended["No"]   = 1

        x2_append=pd.DataFrame(data=np.nan,columns=["Rates","Data"],
                               index=x2.index)
        x2_append["Rates"]=x2.values
        x2_append["Data"] ="ICON_HALO_Track"
        x2_append["No"]  =2

        x3_append=pd.DataFrame(data=np.nan,columns=["Rates","Data"],
                               index=x3.index)
        x3_append["Rates"]=x3.values
        x3_append["Data"] ="ERA5_HALO_Track"
        x3_append["No"]   =3

        x4_append=pd.DataFrame(data=np.nan,columns=["Rates","Data"],
                               index=x4.index)
        x4_append["Rates"]=x4.values
        x4_append["Data"] ="Radar_HALO_Track"
        x4_append["No"]   =4

        x_appended=x_appended.append(x2_append,ignore_index=True)
        x_appended=x_appended.append(x3_append,ignore_index=True)
        x_appended=x_appended.append(x4_append,ignore_index=True)
        x_to_plot=x_appended.copy()#iloc[0:1000,:]
        del x_to_plot["Data"]
        print("Create boxplot")

        boxpl=sns.boxplot(x=np.log10(x_to_plot["Rates"]),y=x_to_plot["No"],
            palette={1:"dodgerblue",2:"orange",3:"deeppink",4:"k"},orient="h")
        boxpl.set_xlim([-5,1])
        boxpl.set_yticks([0,1,2,3],["ICON AR","ICON \nHALO Track",
                                    "ERA5 \nHALO Track","Radar \n HALO Track"])
        fig_name="Boxplot_Precip_representativeness.png"
        plt.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
        ### KDE Plot of precip distribution

    def plot_kde_precip(self,sector_precip_icon_field,
                        regridded_warm_radar_rain,
                        do_sensitivity_study=False,do_conditional_dist=False,
                        radar_era5=pd.DataFrame()):
        matplotlib.rcParams.update({"font.size":24})
        sns.set_style("white")
        
        x1 = sector_precip_icon_field.values
        x2 = self.warm_icon_rain["rate"].values
        if not radar_era5.shape[0]==0:
            x3 = radar_era5["Interp_Precip"]
        x4 = regridded_warm_radar_rain["rate"]
    
        if do_conditional_dist:
            x1=x1[x1>0]
            x2=x2[x2>0]
        if not radar_era5.shape[0]==0:
            x3=x3[x3>0]
        x4=x4[x4>0]
    
        kwargs = dict(hist_kws={'alpha':.3}, kde_kws={'linewidth':2})
        bins=np.linspace(0,.8,81)
        # Plot
        plt.figure(figsize=(10,7), dpi= 300)#
        if not radar_era5.shape[0]==0:
            sns.distplot(x3,bins=bins ,color="deeppink", 
                     label="ERA5 HALO Track", **kwargs)
    
        sns.distplot(x1, bins=bins,color="orange",
            label="ICON AR region, mean="+str(round(x1.mean(),2)), **kwargs)
    
        sns.distplot(x2, bins=bins,color="dodgerblue",
            label="ICON HALO Track, mean="+str(round(x2.mean(),2)), **kwargs)
        sns.distplot(x4, bins=bins,color="k", 
            label="Radar HALO, mean="+str(round(x4.mean(),2)), **kwargs)
    
        if do_sensitivity_study:
            warm_strong_precip=self.strong_precip_rate.loc[\
                                self.warm_radar_rain.index]
            warm_strong_precip=self.warm_strong_precip.loc[\
                                self.warm_radar_rain.index] 
            x5=warm_strong_precip.resample("10s").mean().dropna(
                    subset=["rate"])["rate"]
            sns.distplot(x5, bins=bins,color="indigo", 
                     label="Radar HALO +4 dBZ, mean="+\
                         str(round(x5.mean(),2)),**kwargs)

        plt.ylim(0,20)
        plt.xlim(bins[0],bins[-1])
        if not do_conditional_dist:
            plt.xlabel("Precipitation rate ($\mathrm{mmh}^{-1}$)")
        else:
            plt.xlabel("Conditional Precipitation rate ($\mathrm{mmh}^{-1}$)")
        plt.legend(fontsize=18)
        file_end=".png"
        fig_name="Warm_Sector_Rain_Rate_Representativeness"
        if do_sensitivity_study:
            fig_name+="_plus_4dbz"
        fig_name+=file_end
        if do_conditional_dist:
            fig_name="Conditional_"+fig_name
        fig_name=self.flight[0]+"_"+self.ar_of_day+"_"+fig_name
        fig_path=self.plot_path
        sns.despine(offset=1)
        plt.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as ",fig_path+fig_name)
        return None
    
    def plot_frontal_sector_radar_icon_rain_comparison(self,
                                                    do_conditional=False):
        matplotlib.rcParams.update({"font.size":18})
        # Preprocess the data
        radar_str="processed_radar"
        x1=self.warm_radar_rain["rate"]
        x2=self.warm_icon_rain["rate"]
        x3=self.cold_radar_rain["rate"]
        x4=self.cold_icon_rain["rate"]
        mean_term="mean:"
        if do_conditional:
            x1=x1.loc[x1>0.001]
            x2=x2.loc[x2>0.0]
        x3=x3.loc[x3>0.001]
        x4=x4.loc[x4>0.0]
        mean_term="cond. "+mean_term
        x1.index=range(len(x1))
        x2.index=range(len(x2))
        x3.index = range(len(x3))
        x4.index = range(len(x4))

        x_appended=pd.DataFrame(data=np.nan,
                columns=["Rates","Data","No"],index=x1.index)
        x_appended["Rates"]=x1.values.astype(float)
        x_appended["Data"] ="Pre-frontal Radar"
        x_appended["No"]   = 1

        x2_append=pd.DataFrame(data=np.nan,columns=["Rates","Data"],
                               index=x2.index)
        x2_append["Rates"]=x2.values
        x2_append["Data"] ="Pre-frontal ICON"
        x2_append["No"]  =3
        
        x3_append=pd.DataFrame(data=np.nan,columns=["Rates","Data"],
                               index=x3.index)
        x3_append["Rates"]=x3.values
        x3_append["Data"] ="Post-frontal Radar"
        x3_append["No"]   =2
        
        x4_append=pd.DataFrame(data=np.nan,columns=["Rates","Data"],
                               index=x4.index)
        x4_append["Rates"]=x4.values
        x4_append["Data"] ="Post-frontal ICON"
        x4_append["No"]   =4

        x_appended=x_appended.append(x2_append,ignore_index=True)
        x_appended=x_appended.append(x3_append,ignore_index=True)
        x_appended=x_appended.append(x4_append,ignore_index=True)
        
        x_to_plot=x_appended.copy()

        del x_to_plot["Data"]
        print("Create boxplot")

        fig=plt.figure(figsize=(16,7))
        boxpl=sns.boxplot(x=x_to_plot["Rates"],
                          y=x_to_plot["No"],linewidth=3,
            orient="h",palette=["darkorange","darkviolet","bisque","plum"])
        ##,colours=["darkorange","orange","darkviolet","mediumorchid"])
        boxpl.spines['left'].set_linewidth(3)
        boxpl.spines['bottom'].set_linewidth(3)
        boxpl.text(0.5,-.25,s="cond. mean:"+str(np.round(x1.mean(),2))+\
                   "$\mathrm{mmh}^{-1}$")
        boxpl.text(0.5,.75,s="cond. mean:"+str(np.round(x3.mean(),2))+\
                   "$\mathrm{mmh}^{-1}$")
        boxpl.text(0.5,1.75,s="cond. mean:"+str(np.round(x2.mean(),2))+\
                   "$\mathrm{mmh}^{-1}$")
        boxpl.text(0.5,2.75,s="cond. mean:"+str(np.round(x4.mean(),2))+\
                   "$\mathrm{mmh}^{-1}$")

        boxpl.set_xlim([0,1])
        boxpl.set_xticks([0,.25,.5,.75,1])
        boxpl.set_yticklabels(["Radar \npre-frontal","Radar \npost-frontal",
                               "ICON \npre-frontal","ICON \npost-frontal"])
        #boxpl.xaxis.xtick_params({"length":2,"width":3})
        boxpl.set_ylabel("")
        if not do_conditional:
            boxpl.set_xlabel("Precipitation Rates ($\mathrm{mm h}^{-1}$)")
        else:
            boxpl.set_xlabel(
                "Conditional Precipitation Rates ($\mathrm{mm h}^{-1}$)")
        sns.despine(ax=boxpl,offset=10)
        if self.calibrated_radar: radar_str="calibrated_and_"+radar_str
        fig_name="Rain_rate_Statistics_"+self.flight[0]+"_"+self.sector+"_"+\
            radar_str+"_"+"Zg"+".png"
        if do_conditional:
            fig_name="Conditional_"+fig_name
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)

    def plot_radar_rain_rates(self,sector_times,sub_sector_precip_rates,
            mlayer_height,sector="internal",add_surface_mask=False,
            add_mlayer_height=False,
            save_as_manuscript_plot=False):
        """
        This plotting module creates the manuscript Figure 07 if we consider
        RF05 and AR entire 1. The multiplots contain dBZ, LDR and precip rates.
        They show the data along the entire leg. The post-frontal half is 
        hatched in grey to clearly that it is not considered into the mean 
        values (as it was before?!).

        Parameters
        ----------
        sector_times : list
            start and end timestamp.
        sub_sector_precip_rates : pd.DataFrame()
            Sector segment of the internal leg to be depicted in mean rates.
        sector : str, optional
            part of sector to be used. The default is "internal".
        add_surface_mask : boolean, optional
            If the surface mask should be plotted in addition.
            The default is False.
        save_as_manuscript_plot : boolean, optional
            if the figure is saved. The default is False.

        Returns
        -------
        None.

        """
        sub_sector_precip_rates=sub_sector_precip_rates.replace(0,value=np.nan)
        # Now raw_uni_radar and ds (processed uni radar) can be compared
        # via plotting
        font_size=16
        matplotlib.rcParams.update({"font.size":font_size})
    
        if not add_surface_mask:
            fig,axs=plt.subplots(3,1,figsize=(12,12),
                    gridspec_kw=dict(height_ratios=(1,0.7,0.8)),sharex=True)
        else:
            fig,axs=plt.subplots(4,1,figsize=(12,12),
                    gridspec_kw=dict(height_ratios=(1,0.7,0.8,0.1)),sharex=True)
        
        sector_radar=self.processed_radar.sel(
            {"time":slice(sector_times[0],sector_times[-1])})
    
        y=np.array(sector_radar["height"][:])
        #######################################################################
        #######################################################################
        ### Processed radar
        sector_radar["dBZg"]=sector_radar["dBZg"].where(
            sector_radar["radar_flag"].isnull(), drop=True)
        sector_radar["Zg"]=sector_radar["Zg"].where(
            sector_radar["radar_flag"].isnull(), drop=True)
        sector_radar["LDRg"]=sector_radar["LDRg"].where(
                    sector_radar["radar_flag"].isnull(), drop=True)
    
        surface_Zg=sector_radar["Zg"][:,4]
        surface_Zg=surface_Zg.where(surface_Zg!=-888.)
    
        #processed_radar
        time=pd.DatetimeIndex(np.array(sector_radar["dBZg"].time[:]))
        #Plotting
        C1=axs[0].pcolor(time,y,np.array(sector_radar["dBZg"][:]).T,
                        cmap=cmaeri.roma_r,vmin=-30,vmax=30)
    
        cax1=fig.add_axes([0.925, 0.7, 0.01, 0.15])
        cb = plt.colorbar(C1,cax=cax1,orientation='vertical',extend="both")
        cb.set_label('Reflectivity (dBZ)')
        print("dBZ plotted")
        if pd.Timestamp(sector_times[0])<\
            pd.Timestamp(sub_sector_precip_rates.index[0]):
            axs[0].axvspan(pd.Timestamp(sector_times[0]),
                       pd.Timestamp(sub_sector_precip_rates.index[0]),
                       alpha=0.5,color="lightgrey",zorder=3)
            axs[0].text(0.16,0.7,"Post-frontal",bbox=dict(
               facecolor='lightgrey',edgecolor="k",boxstyle="round",alpha=0.95),
               ha='center',color="slateblue", va='center',fontsize=18,
               transform=axs[0].transAxes,zorder=5)
        else:
            axs[0].axvspan(pd.Timestamp(sub_sector_precip_rates.index[-1]),
                           pd.Timestamp(sector_times[-1]),
                           alpha=0.5,color="lightgrey",zorder=3)
            axs[0].text(0.16,0.7,"Post-frontal",bbox=dict(
               facecolor='lightgrey',edgecolor="k",boxstyle="round",alpha=0.95),
               ha='center',color="slateblue", va='center',fontsize=18,
               transform=axs[0].transAxes,zorder=5)
            
        #        if inflow_times[0]<outflow_times[-1]:
        #            axs[0].axvspan(pd.Timestamp(inflow_times[-1]),
        #               pd.Timestamp(internal_times[0]),
        #               alpha=0.5, color='grey')
        #            axs[0].axvspan(pd.Timestamp(internal_times[-1]),
        #               pd.Timestamp(outflow_times[0]),
        #               alpha=0.5, color='grey')   
        #        else:
        #            axs[0].axvspan(pd.Timestamp(outflow_times[-1]),
        #               pd.Timestamp(internal_times[0]),
        #               alpha=0.5, color='grey')
        #            axs[0].axvspan(pd.Timestamp(internal_times[-1]),
        #               pd.Timestamp(inflow_times[0]),
        #               alpha=0.5, color='grey')   
        axs[0].set_xlabel('')
        axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0].set_ylim([0,12000])
        axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0].set_xticklabels([])
        axs[0].set_ylabel("Height (km)")
        axs[0].axhline(xmin=0.005,xmax=0.995,y=1000,
                       color="w",ls="-",lw=4,zorder=4)
        axs[0].axhline(xmin=0.005,xmax=0.995,y=100,
                       color="w",ls="-",lw=4,zorder=4)
        axs[0].axvline(x=pd.Timestamp(time[10]),ymin=0,ymax=1/12,
                       color="w",ls="-",lw=4,zorder=4)
        axs[0].axvline(x=pd.Timestamp(time[-10]),ymin=0,ymax=1/12,
                       color="w",ls="-",lw=4,zorder=4)
        axs[0].axhline(xmin=0.005,xmax=0.995,y=1000,
                       color="grey",ls="--",lw=2,zorder=5)
        axs[0].axhline(xmin=0.005,xmax=0.995,y=100,
                       color="grey",ls="--",lw=2,zorder=5)
        axs[0].axvline(x=pd.Timestamp(time[10]),ymin=0,ymax=1/12,
                       color="grey",ls="--",lw=2,zorder=5)
        axs[0].axvline(x=pd.Timestamp(time[-10]),ymin=0,ymax=1/12,
                       color="grey",ls="--",lw=2,zorder=5)
        
        axs[0].set_ylabel("Height (km)")
    
        #axs[0].text(pd.Timestamp(inflow_times[0]),10000,"Inflow")
        #axs[0].text(pd.Timestamp(internal_times[0]),10000,"Internal")
        #axs[0].text(pd.Timestamp(outflow_times[0]),10000,"Outflow")
        #--------------------------------------------------------------------------------------------#
        # LDR subplot
        #--------------------------------------------------------------------------------------------#
        axs[1].spines['left'].set_color('grey')
        axs[1].spines['right'].set_color('grey')
        axs[1].spines['bottom'].set_color('grey')
        axs[1].spines['top'].set_color('grey')
        axs[1].spines['left'].set_linestyle("--")
        axs[1].spines['right'].set_linestyle("--")
        axs[1].spines['bottom'].set_linestyle("--")
        axs[1].spines['top'].set_linestyle("--")
        axs[1].spines['left'].set_linewidth(3)
        axs[1].spines['right'].set_linewidth(3)
        axs[1].spines['bottom'].set_linewidth(3)
        axs[1].spines['top'].set_linewidth(3)
    
        # Radar LDR
        C2=axs[1].pcolor(time,y,np.array(sector_radar["LDRg"][:]).T,
                                 cmap=cmaeri.batlowK,vmin=-25, vmax=-10)        
        if add_mlayer_height:
            # above ML
            axs[1].plot(mlayer_height+150,ls="-",lw=4,color="white")
            axs[1].plot(mlayer_height+150,ls="--",lw=1,color="lightblue")
            # Melting layer
            #axs[1].plot(mlayer_height,lw=3,color="white")
            #axs[1].plot(mlayer_height,lw=1,color="dimgrey")
        axs[1].set_yticks([0,500,1000])
        axs[1].set_ylim([0,1000])
        axs[1].set_yticklabels(["0","0.5","1"])
        axs[1].set_ylabel("Height (km)")
        if pd.Timestamp(sector_times[0])<\
            pd.Timestamp(sub_sector_precip_rates.index[0]):
            axs[1].axvspan(pd.Timestamp(sector_times[0]),
                       pd.Timestamp(sub_sector_precip_rates.index[0]),
                       alpha=0.5,color="lightgrey",zorder=3)
        print("LDR plotted")
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        
        cax2=fig.add_axes([0.925, 0.4, 0.01, 0.15])
        cb = plt.colorbar(C2,cax=cax2,orientation='vertical',extend="both")
        cb.set_label('LDR (dB)')
        ##########################################################################
        # Precipitation rates
        ##########################################################################
        axs[2].plot(self.precipitation_rate["mean_rain"],
            lw=3,color="darkgreen",label="Avg$_{\mathrm{R}}$: "+str(round(float(\
                sub_sector_precip_rates["mean_rain"].mean()),2)))
        axs[2].plot(self.precipitation_rate["r_norris"],
            lw=1,color="lightgreen",label="Nor2020: "+str(round(float(\
                sub_sector_precip_rates["r_norris"].mean()),2)))
        mixed_precip=self.precipitation_rate["mean_mixed"].copy()
        mixed_precip[mixed_precip==0]=np.nan
        axs[2].plot(mixed_precip,
                    lw=1,color="grey")
        axs[2].plot(self.precipitation_rate["r_palmer"],
            lw=1,color="mediumseagreen",label="MP1948: "+str(round(float(\
                sub_sector_precip_rates["r_palmer"].mean()),2)))
        axs[2].plot(self.precipitation_rate["r_chandra"],
            lw=1,color="green",label="Cha2015: "+str(round(float(\
            sub_sector_precip_rates["r_chandra"].mean()),2)))
    
        # Snow 
        axs[2].plot(self.precipitation_rate["mean_snow"],lw=3,color="darkblue",
            label="Avg$_{\mathrm{S}}$: "+str(round(float(\
                sub_sector_precip_rates["mean_snow"].mean()),2)))
        axs[2].plot(self.precipitation_rate["s_schoger"],
            lw=0.5,color="lightblue",label="Sch2021: "+str(round(float(\
                sub_sector_precip_rates["s_schoger"].mean()),2)))
        axs[2].plot(self.precipitation_rate["s_matrosov"],
            lw=0.5,color="blue",label="Mat2007: "+str(round(float(\
                sub_sector_precip_rates["s_matrosov"].mean()),2)))
        axs[2].plot(self.precipitation_rate["s_heymsfield"],
            lw=0.5,color="cadetblue",label="Hey2018: "+str(round(float(\
                sub_sector_precip_rates["s_heymsfield"].mean()),2)))
    
        axs[2].set_ylim([0,1.4])
        axs[2].legend(title="Pre-frontal precipitation, cond. mean (mm/h)",
                      loc="upper left",ncol=4,fontsize=font_size-4)
        axs[2].set_xticks=axs[1].get_xticks()
        axs[2].set_ylabel("Precipitation\nrate ($\mathrm{mmh}^{-1}$)")
        if pd.Timestamp(sector_times[0])<\
            pd.Timestamp(sub_sector_precip_rates.index[0]):
            axs[2].axvspan(pd.Timestamp(sector_times[0]),
                       pd.Timestamp(sub_sector_precip_rates.index[0]),
                       alpha=0.5,color="lightgrey",zorder=3)
            
        sns.despine(ax=axs[0],offset=10)
        sns.despine(offset=10,ax=axs[2])
        if not add_surface_mask:
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
            for a, axis in enumerate(axs[0:3]):
                if not a==1:
                    axis.spines["left"].set_linewidth(2)
                    axis.spines["bottom"].set_linewidth(2)
                axis.tick_params(axis="x",length=6,width=2)
                axis.tick_params(axis="y",length=6,width=2)
                
            axs[2].tick_params(axis="x",length=6,width=2)
            axs[2].set_xlabel("Time (UTC)")
    
        else: 
            axs[2].set_xticklabels([])
            # Add surface mask
            # plot AMSR2 sea ice concentration
            fs = 14
            fs_small = fs - 2
            fs_dwarf = fs - 4
            marker_size = 15
    
            bah_df=pd.DataFrame()
            bah_df["sea_ice"]=pd.Series(data=np.array(\
                self.internal_radar["radar_flag"][:,0]),
                index=pd.DatetimeIndex(np.array(self.internal_radar.time[:])))
            
            blue_colorbar=cm.get_cmap('Blues_r', 22)
            blue_cb=blue_colorbar(np.linspace(0, 1, 22))
            brown_rgb = np.array(colors.hex2color(colors.cnames['brown']))
            blue_cb[:2, :] = [*brown_rgb,1]
            newcmp = ListedColormap(blue_cb)
            im = axs[3].pcolormesh(np.array([\
                    pd.DatetimeIndex(bah_df["sea_ice"].index),
                    pd.DatetimeIndex(bah_df["sea_ice"].index)]),
                    np.array([0, 1]),
                    np.tile(np.array(bah_df["sea_ice"].values),(2,1)),
                    cmap=newcmp, vmin=-0.1, vmax=1,
                    shading='auto')
    
            cax = fig.add_axes([0.7, 0.08, 0.1, axs[3].get_position().height])
            C1=fig.colorbar(im, cax=cax, orientation='horizontal')
            C1.set_label(label='Sea ice [%]',fontsize=fs_small)
            C1.ax.tick_params(labelsize=fs_small)
            axs[3].tick_params(axis='x', labelleft=False, 
                               left=False,labelsize=fs_small)
            axs[3].tick_params(axis='y', labelleft=False, left=False)
            axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
            for a,axis in enumerate(axs[0:3]):
                if not a==1:
                    axis.spines["left"].set_linewidth(3)
                    axis.spines["bottom"].set_linewidth(3)
                axis.tick_params(axis="x",length=6,width=2)
                axis.tick_params(axis="y",length=6,width=2)
    
            axs[3].tick_params(axis="x",length=6,width=2)
            axs[3].set_xlabel("Time (UTC)")
        # Limit axis spacing:
        plt.subplots_adjust(hspace=0.35)    # removes space between subplots
        axs[0].text(0.02,0.9,"(a)",color="k",fontsize=font_size-2,
            bbox=dict(facecolor='whitesmoke',edgecolor="k",boxstyle="round",
                      alpha=0.8),transform=axs[0].transAxes,zorder=10)
        axs[1].text(0.02,0.9,"(b)",color="k",fontsize=font_size-2,
            bbox=dict(facecolor='whitesmoke',edgecolor="k",boxstyle="round",
                      alpha=0.8),transform=axs[1].transAxes,zorder=10)
        axs[2].text(0.02,0.9,"(c)",color="k",fontsize=font_size-2,
            bbox=dict(facecolor='whitesmoke',edgecolor="k",boxstyle="round",
                      alpha=0.8),transform=axs[2].transAxes,zorder=10)
        #box = axs[3].get_position()        
        #box.y0 = box.y0 + 0.025
        #box.y1 = box.y1 + 0.025        
        #axs[3]=axs[3].set_position(box)
        if self.flight[0]=="RF05" and self.ar_of_day=="AR_entire_1":
            Ws="WS1"
        else:
            Ws=""
        fig_name=self.flight[0]+"_"+self.ar_of_day+"_"+Ws+"_Precip_"+\
            self.sector+"_"+\
            "Processed_radar.png"
        if save_as_manuscript_plot:
            fig_name="Fig07_"+fig_name
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
    
    def radar_model_precip_timeseries_comparison(self, 
                                    internal_times,sub_sector_precip_rates):
    
        internal_times=self.times_dict[self.flight[0]+"__"+\
                                       self.ar_of_day]["internal"]
        
        # cut to internal leg
        internal_precipitation_rate=\
            self.precipitation_rate.loc[internal_times[0]:internal_times[-1]]
        internal_precipitation_rate[["mean_snow","mean_rain","mean_mixed"]]=\
            internal_precipitation_rate[["mean_snow","mean_rain","mean_mixed"]].fillna(0)
        internal_precipitation_rate["rate"]=\
            internal_precipitation_rate["mean_snow"]+\
                internal_precipitation_rate["mean_rain"]+\
                    internal_precipitation_rate["mean_mixed"]
                
        strong_internal_precip_rate=\
            self.strong_precip_rate.loc[internal_times[0]:internal_times[-1]]
        internal_era5 = self.halo_era5.loc[internal_times[0]:internal_times[-1]]
        internal_icon = self.halo_icon_hmp.loc[internal_times[0]:internal_times[-1]]
    
        internal_rain_rate=internal_precipitation_rate[["r_norris","r_palmer",
                                                "r_chandra"]]
        internal_snow_rate=internal_precipitation_rate[["s_schoger","s_matrosov",
                                                    "s_heymsfield"]]
        #unc_period=internal_precipitation_rate["precip_phase"]=="uncertain"
        matplotlib.rcParams.update({"font.size":22})
        precip_rate_fig=plt.figure(figsize=(16,9))
        ax1=precip_rate_fig.add_subplot(111)
        ax1.fill_between(internal_rain_rate.index,
            internal_rain_rate.min(axis=1).values,
            y2=internal_rain_rate.max(axis=1).values,
            color="mediumseagreen",label="HALO rain")
        ax1.fill_between(internal_snow_rate.index,
            internal_snow_rate.min(axis=1).values,
            y2=internal_snow_rate.max(axis=1).values,
            color="lightblue",label="HALO snow")
        # uncertain period
        ax1.fill_between(internal_precipitation_rate.index,
            internal_precipitation_rate["min_mixed"].values,
            y2=internal_precipitation_rate["max_mixed"].values,
            color="lightgrey",label="HALO mixed")
        ax1.plot(internal_precipitation_rate.index,
            internal_precipitation_rate["rate"].values,color="white",lw=4)
        ax1.plot(internal_precipitation_rate.index,
            internal_precipitation_rate["rate"].values,color="orchid",lw=2,
            label="HALO mean: "+str(round(float(
            sub_sector_precip_rates["rate"].mean()),2)))
        ###
        ax1.plot(strong_internal_precip_rate.index,
            strong_internal_precip_rate["rate"].values,
            color="purple",lw=2,
            label="HALO (+"+str(int(self.attenuation_value))+" dBZ): "+\
                str(round(float(strong_internal_precip_rate["rate"].loc[\
                sub_sector_precip_rates.index].mean()),2)))
        ###
        ax1.plot(internal_era5["Interp_Precip"],
            lw=2,ls="--",color="k",label="ERA5: "+str(round(float(
            internal_era5["Interp_Precip"].loc[\
                sub_sector_precip_rates.index].mean()),2)),
            zorder=5)
        ax1.plot(internal_icon["Interp_Precip"],
            lw=3,ls="-",color="k",label="ICON-2km: "+str(round(float(
            internal_icon["Interp_Precip"].loc[\
                sub_sector_precip_rates.index].mean()),2)),
            zorder=6)
        if pd.Timestamp(internal_rain_rate.index[0])<\
            pd.Timestamp(sub_sector_precip_rates.index[0]):
            ax1.axvspan(pd.Timestamp(internal_rain_rate.index[0]),
                       pd.Timestamp(sub_sector_precip_rates.index[0]),
                       alpha=0.5,color="lightgrey",zorder=3)
        
        #ax1.axvspan(pd.Timestamp(cold_sector[0]),
        #               pd.Timestamp(cold_sector[-1]),
        #               alpha=0.3, color='lightgrey')
        ax1.text(0.2,0.8,"Post-frontal",
            bbox=dict(facecolor='lightgrey',edgecolor="k",
                      boxstyle="round",alpha=0.8),
            ha='center',color="slateblue", va='center',fontsize=22,
            transform=ax1.transAxes)
    
    
        ax1.legend(title="Rates ($\mathrm{mm h}^{-1}$)",
                   loc="upper right",fontsize=18)
        sns.despine(offset=10)
        ax1.spines["left"].set_linewidth(2)
        ax1.spines["bottom"].set_linewidth(2)
        ax1.set_xlabel("Time (UTC)")
        ax1.set_yticks([0,0.5,1.0,1.5])
        ax1.set_ylim([0,1.5])
        ax1.set_ylabel("Precipitation rate ($\mathrm{mm\,h}^{-1}$)")
        ax1.tick_params("x",length=8,width=2)
        ax1.tick_params("y",length=8,width=2)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig_path=self.plot_path
        fig_name=self.flight[0]+"_"+self.ar_of_day+\
            "_internal_leg_precipitation_rate_comparison.pdf"
        if self.flight[0]=="RF05" and self.ar_of_day=="AR_entire_1":
            fig_name="Fig16_"+fig_name
        precip_rate_fig.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",fig_path+fig_name)

    def precip_sector_trends(self,flight_sequence,
                             save_as_essay_fig=True):
        import quicklook_dicts
        import measurement_instruments_ql
        import halodataplot
        #######################################################################
        # Pre-definitions
        font_size=16
        matplotlib.rcParams.update({"font.size":font_size})
        
        gs_kw = dict(height_ratios=[1,0.5,1,0.5,1,0.5,1,0.5])
        precip_trend_fig, axs = plt.subplots(ncols=1, nrows=8,
            constrained_layout=True,gridspec_kw=gs_kw,figsize=(12,18))
        #######################################################################
        
        fig_labels=["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)"]
        for f,seq in enumerate(flight_sequence):
            flight_rf=seq.split("__")[0]
            ar_rf=seq.split("__")[1]
            import moisturebudget as Budgets
            temporary_Precip_cls   = Budgets.Surface_Precipitation("HALO_AC3",
                        self.cmpgn_cls,[flight_rf],os.getcwd(),
                        flight_dates={},sector_types=["warm","core","cold"],
                        ar_of_day=ar_rf,grid_name="ERA5",do_instantan=False)
            #inflow_times,internal_times,outflow_times=\
            #    self.cmpgn_cls.define_budget_legs(flight_rf,ar_rf)

            #warm_halo_df_field=halo_df.loc[warm_radar_precip_field.index]
            if f<2:
                date="20220315"
            else:
                date="20220316"
            cfg_dict=quicklook_dicts.get_prcs_cfg_dict(
                flight_rf,date,"HALO_AC3",self.cmpgn_cls.campaign_path)
            cfg_dict["device_data_path"]=self.airborne_data_importer_path
            # Data Handling 
            datasets_dict, data_reader_dict=\
                quicklook_dicts.get_data_handling_attr_dicts()
            # Get Plotting Handling
            plot_handler_dict, plot_cls_args_dict,plot_fct_args_dict=\
                                    quicklook_dicts.get_plotting_handling_attrs_dict()
        
            HALO_Devices_cls=measurement_instruments_ql.HALO_Devices(cfg_dict)
            HALO_Devices_cls.update_major_data_path(self.cmpgn_cls.campaign_path)
            Bahamas_cls=measurement_instruments_ql.BAHAMAS(HALO_Devices_cls)
            Radar_cls=measurement_instruments_ql.RADAR(HALO_Devices_cls)
            Radar_cls.radar_ds=Radar_cls.open_version_specific_processed_radar_data(
                for_calibrated_file=self.calibrated_radar)
            Radar_cls.correct_for_gaseous_attenuation()
            #if f>1:
            #    Radar_cls.cfg_dict["flight"]="RF06"
            prc_radar=Radar_cls.radar_ds
            halo_df=prc_radar[["lat","lon"]].to_dataframe()#Bahamas_cls.op
            halo_df=halo_df.rename({"lat":"latitude","lon":"longitude"},
                                           axis=1)
            # Replace nans
            prc_radar=halodataplot.replace_fill_and_missing_values_to_nan(
                                        prc_radar,["dBZg","Zg","LDRg",
                                                   "VELg","radar_flag"])        
            # find melting layer   
            mlayer_height,low_ldr_df,ldr_cutted_df,bb_mask=\
                    Radar_cls.find_melting_layer(prc_radar)
            # classify precipitation types
            prec_type_series,zg_s=Radar_cls.classify_precipitation_type(
                prc_radar,mlayer_height,bb_mask)
            prec_type_series.plot(ylim=[0,3])
            surface_mask=pd.Series(data=np.array(prc_radar["radar_flag"][:,0]),
                index=pd.DatetimeIndex(np.array(prc_radar.time[:])))
            # apply z-r relationships
            z_s_dict={}
            z_s_dict["zg"]=zg_s
            prec_rate=Radar_cls.take_correct_precipitation_rates(
                z_s_dict,surface_mask,prec_type_series,
                z_for_snow="Zg")#)
            prec_rate["lat"]=halo_df["latitude"].loc[\
                                            prec_rate.index]
            prec_rate["lon"]=halo_df["longitude"].loc[\
                                                prec_rate.index]
            prec_rate.index=pd.DatetimeIndex(prec_rate.index)    
            temp_warm_radar_precip_field,temp_warm_icon_precip_track=\
                temporary_Precip_cls.select_warm_precip(prec_rate,
                        self.halo_icon_hmp,include_icon=False)
            
            
            inflow_times=self.times_dict[seq]["inflow"]
            internal_times=self.times_dict[seq]["internal"]
            outflow_times=self.times_dict[seq]["outflow"]
            #print("internal_times,",internal_times)
            if inflow_times[0]<outflow_times[-1]:
                prc_radar=prc_radar.sel(
                    {"time":slice(inflow_times[0],outflow_times[-1])})
            else:
                prc_radar=prc_radar.sel(
                    {"time":slice(outflow_times[0],inflow_times[-1])})
            internal_warm=temp_warm_radar_precip_field.loc[\
                        internal_times[0]:internal_times[-1]]
            internal_warm[["mean_snow","mean_mixed","mean_rain"]]=\
                internal_warm[\
                ["mean_snow","mean_mixed","mean_rain"]].fillna(0)
            internal_warm["rate"]=internal_warm["mean_snow"]+\
                internal_warm["mean_rain"]+\
                    internal_warm["mean_mixed"]
            
            sector_radar=prc_radar.sel(
                {"time":slice(internal_times[0],internal_times[-1])})
            
            y=np.array(sector_radar["height"][:])
            ###################################################################
            ###################################################################
            ### Processed radar
            sector_radar["dBZg"]=sector_radar["dBZg"].where(
                sector_radar["radar_flag"].isnull(), drop=True)
            sector_radar["Zg"]=sector_radar["Zg"].where(
                sector_radar["radar_flag"].isnull(), drop=True)
            sector_radar["LDRg"]=sector_radar["LDRg"].where(
                        sector_radar["radar_flag"].isnull(), drop=True)
            surface_Zg=sector_radar["Zg"][:,4]
            surface_Zg=surface_Zg.where(surface_Zg!=-888.)
            time=pd.DatetimeIndex(np.array(sector_radar["dBZg"].time[:]))
            C1=axs[f*2].pcolor(time,y,np.array(sector_radar["dBZg"][:]).T,
                            cmap=cmaeri.roma_r,vmin=-30,vmax=30)
            if pd.DatetimeIndex(sector_radar.time[0])<internal_warm.index[0]:
                #print(sector_radar.time[0],internal_warm.index[0])
                axs[f*2].axvspan(
                    pd.Timestamp(sector_radar.time.values[0]),
                    pd.Timestamp(internal_warm.index[0]),
                           alpha=0.4, color='grey',zorder=2)
            else:
                axs[f*2].axvspan(pd.Timestamp(internal_warm.index[-1]),
                               pd.Timestamp(sector_radar.time.values[-1]),
                               alpha=0.4,color="grey",zorder=2)
            #            axs[0].axvspan(pd.Timestamp(internal_times[-1]),
            #               pd.Timestamp(outflow_times[0]),
            #               alpha=0.5, color='grey')   
            #        else:
            #            axs[0].axvspan(pd.Timestamp(outflow_times[-1]),
            #               pd.Timestamp(internal_times[0]),
            #               alpha=0.5, color='grey')
            #            axs[0].axvspan(pd.Timestamp(internal_times[-1]),
            #               pd.Timestamp(inflow_times[0]),
            #               alpha=0.5, color='grey')   
            axs[f*2+1].set_xlabel('')
            axs[f*2].set_xlim([pd.Timestamp(internal_times[0]),
                               pd.Timestamp(internal_times[-1])])
            
            axs[f*2].set_yticks([0,2000,4000,6000,8000,10000,12000])
            axs[f*2].set_ylim([0,10000])
            axs[f*2].set_yticklabels(["0","2","4","6","8","10","12"])
            axs[f*2].text(0.185,0.55,"Post-frontal",bbox=dict(
               facecolor='whitesmoke',edgecolor="k",boxstyle="round",alpha=0.9),
               ha='center',color="slateblue", va='center',fontsize=font_size,
               transform=axs[f*2].transAxes,zorder=5)
            
            if f==0:
                axs[f*2].set_ylabel("Height (km)")
                axs[f*2+1].set_ylabel("Rate ($\mathrm{mm\,h}^{-1}$)")
            
            # Add rain rates
            prec_rate=prec_rate.fillna(0)
            prec_rate["rate"]=prec_rate["mean_snow"]+\
                prec_rate["mean_rain"]+prec_rate["mean_mixed"]
            internal_precipitation_rate=prec_rate.loc[\
                        internal_times[0]:internal_times[-1]] 
            internal_rain_rate=internal_precipitation_rate[\
                        ["r_norris","r_palmer","r_chandra"]]
            internal_snow_rate=internal_precipitation_rate[
                ["s_schoger","s_matrosov","s_heymsfield"]]
            internal_unc_rate=internal_precipitation_rate[[\
                                    "min_mixed","max_mixed"]]
            axs[f*2+1].fill_between(internal_rain_rate.index,
                internal_rain_rate.min(axis=1).values,
                y2=internal_rain_rate.max(axis=1).values,
                color="mediumseagreen",label="HALO rain")
            axs[f*2+1].fill_between(internal_snow_rate.index,
                internal_snow_rate.min(axis=1).values,
                y2=internal_snow_rate.max(axis=1).values,
                color="lightblue",label="HALO snow")
            axs[f*2+1].fill_between(internal_unc_rate.index,
                internal_unc_rate.min(axis=1).values,
                y2=internal_unc_rate.max(axis=1).values,
                color="grey",label="HALO mixed")
            axs[f*2+1].text(0.7,0.7,"Mean rate:"+\
                str(round(internal_warm["rate"].mean(),2))+\
                "$\mathrm{mm\,h}^{-1}$",
                fontsize=14,transform=\
                    axs[f*2+1].transAxes)
            
                
            sns.despine(ax=axs[f*2+1],offset=1)
            axs[f*2+1].plot(internal_precipitation_rate.index,
                internal_precipitation_rate["rate"].values,
                color="white",lw=3)
            axs[f*2+1].plot(internal_precipitation_rate.index,
                internal_precipitation_rate["rate"].values,
                color="orchid",lw=1,label="HALO mean: "+str(round(float(
                    internal_warm["rate"].mean()),2)))
            axs[f*2+1].set_xlim([pd.Timestamp(internal_times[0]),
                                 pd.Timestamp(internal_times[-1])])
            if f==3:
                axs[f*2+1].set_xlabel("Time (UTC)")
            axs[f*2].xaxis.set_major_formatter(
                mdates.DateFormatter('%H:%M'))
            axs[f*2+1].xaxis.set_major_formatter(
                mdates.DateFormatter('%H:%M'))###
            
            if pd.DatetimeIndex(sector_radar.time[0])<internal_warm.index[0]:
                #print(sector_radar.time[0],internal_warm.index[0])
                axs[f*2+1].axvspan(pd.Timestamp(sector_radar.time.values[0]),
                           pd.Timestamp(internal_warm.index[0]),
                           alpha=0.4, color='grey',zorder=2)
            else:
                axs[f*2+1].axvspan(pd.Timestamp(internal_warm.index[-1]),
                               pd.Timestamp(sector_radar.time.values[-1]),
                               alpha=0.4,color="grey",zorder=2)
            #ax1.axvspan(pd.Timestamp(cold_sector[0]),
            #               pd.Timestamp(cold_sector[-1]),
            #               alpha=0.3, color='lightgrey')
            #ax1.text(0.2,0.8,"Post-frontal",bbox=dict(facecolor='lightgrey',
            #edgecolor="k",boxstyle="round",alpha=0.8),
            #         ha='center',color="slateblue", va='center',
            #         fontsize=22,transform=ax1.transAxes)
        
            #axs[f*2].axhline(xmin=0.005,xmax=0.995,y=1000,
            # color="w",ls="-",lw=4,zorder=2)
            #axs[f*2].axhline(xmin=0.005,xmax=0.995,y=100,
            #color="w",ls="-",lw=4,zorder=2)
            #axs[f*2].axvline(x=pd.Timestamp(time[10]),
            #ymin=0,ymax=1/12,color="w",ls="-",lw=4,zorder=2)
            #axs[f*2].axvline(x=pd.Timestamp(time[-10]),#
            #ymin=0,ymax=1/12,color="w",ls="-",lw=4,zorder=2)
            #axs[f*2].axhline(xmin=0.005,xmax=0.995,y=1000,
            #color="grey",ls="--",lw=2,zorder=3)
            #axs[f*2].axhline(xmin=0.005,xmax=0.995,y=100,
            #color="grey",ls="--",lw=2,zorder=3)
            #axs[f*2].axvline(x=pd.Timestamp(time[10]),
            #ymin=0,ymax=1/12,color="grey",ls="--",lw=2,zorder=3)
            #axs[f*2].axvline(x=pd.Timestamp(time[-10]),
            #ymin=0,ymax=1/12,color="grey",ls="--",lw=2,zorder=3)
            axs[f*2+1].set_ylim([0,1.5])
            axs[f*2+1].set_xticklabels([])
            
            axs[f*2].spines["top"].set_visible(False)
            axs[f*2].spines["right"].set_visible(False)
            axs[f*2+1].spines["top"].set_visible(False)
            axs[f*2+1].spines["right"].set_visible(False)
            axs[f*2].spines["left"].set_linewidth(2)
            axs[f*2].spines["bottom"].set_linewidth(2)
            axs[f*2+1].spines["left"].set_linewidth(2)
            axs[f*2+1].spines["bottom"].set_linewidth(2)
            axs[f*2].tick_params("x",length=6,width=2)
            axs[f*2+1].tick_params("x",length=6,width=2)
            axs[f*2].tick_params("y",length=6,width=2)
            axs[f*2+1].tick_params("y",length=6,width=2)
            
            axs[f*2].text(0.01,0.875,fig_labels[f*2]+" S"+str(f+1),
                          fontsize=font_size,color="k",
                          bbox=dict(facecolor='whitesmoke',edgecolor="k",
                                    boxstyle="round",alpha=0.95),
                          #   ha='center',color="k", va='center',fontsize=font_size,
                          #   transform=axs[f*2].transAxes,zorder=5
                          transform=axs[f*2].transAxes,zorder=5)
            #axs[f*2].text(0.95,0.91,"S"+str(f+1),bbox=dict(
            #   facecolor='lightgrey',edgecolor="k",boxstyle="round",alpha=0.95),
            #   ha='center',color="k", va='center',fontsize=font_size,
            #   transform=axs[f*2].transAxes,zorder=5)
            axs[f*2+1].text(0.01,0.75,fig_labels[f*2+1]+" S"+str(f+1),
                fontsize=font_size,color="k",bbox=dict(facecolor='whitesmoke',
                        edgecolor="k",boxstyle="round",alpha=0.95),
                transform=axs[f*2+1].transAxes,zorder=5)
            
        #axs[f*2+1].
        caxis=precip_trend_fig.add_axes([0.6,-0.03,0.3,0.02])
        cb = plt.colorbar(C1,cax=caxis,orientation='horizontal',extend="both")
        cb.set_label('Reflectivity (dBZ)')
            
        #plt.subplots_adjust(hspace=-0.1,wspace=-0.1)
        fig_path=self.plot_path
        fig_name="Precip_internal_tendency.png"
        if save_as_essay_fig:
            fig_name="Fig3_4_"+fig_name
        
        precip_trend_fig.savefig(fig_path+fig_name,dpi=600,
                                 bbox_inches="tight")
        print("Figure saved as:", fig_path+fig_name)        

class HALO_AC3_IWV_tendency(HALO_AC3_Budget_Plots):
    def __init__(self,cmpgn_cls,flight,ar_of_day,
                 mwr,halo_df,internal_iwv_sonde,IWV_retrieved,
                 halo_icon_hmc,snd_halo_icon_hmc,
                 halo_era5,snd_halo_era5,
                 internal_sondes_dict,
                 sonde_times_series,Dropsondes,
                 grid_name="ERA5",do_instantan=False,
                 is_flight_campaign=True,
                 major_path=os.getcwd(),sector="warm",
                 aircraft=None,instruments=[],flights=[],
                 interested_flights="all"):
        
        super().__init__(cmpgn_cls,flight,ar_of_day)
        self.major_path=major_path+"/../"
        self.plot_path=self.major_path+"/plots/"
        self.grid_name=grid_name
        
        self.sonde_times_series  = sonde_times_series
        self.internal_iwv_sonde  = internal_iwv_sonde
        self.Dropsondes          = Dropsondes
        self.internal_sondes_dict= internal_sondes_dict
        self.halo_icon_hmc       = halo_icon_hmc
        self.snd_halo_icon_hmc   = snd_halo_icon_hmc
        self.halo_era5           = halo_era5
        self.snd_halo_era5       = snd_halo_era5
        self.IWV_retrieved       = IWV_retrieved
        
        self.halo_df             = halo_df
        self.mwr                 = mwr
        self.is_flight_campaign=is_flight_campaign
        self.aircraft=aircraft
        self.instruments=instruments
        self.flight_day={}
        self.flight_month={}
        self.interested_flights="all"
        self.major_path=major_path
        self.is_synthetic_campaign=True
        self.sector=sector
        #matplotlib.rcParams.update({"font.size":24})
        matplotlib.rcParams.update({"font.size":28})
        if self.flight[0]=="RF05":
            if self.ar_of_day=="AR_entire_1":
                self.inflow_times=["2022-03-15 10:11","2022-03-15 11:13"]
                self.internal_times=["2022-03-15 11:18","2022-03-15 12:14"]
                self.outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
            elif self.ar_of_day=="AR_entire_2":
                self.inflow_times=["2022-03-15 14:30","2022-03-15 15:25"]
                self.internal_times=["2022-03-15 13:20","2022-03-15 14:25"]
                self.outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
        if self.flight[0]=="RF06":
            if self.ar_of_day=="AR_entire_1":
                self.inflow_times=["2022-03-16 10:45","2022-03-16 11:21"]
                self.internal_times=["2022-03-16 11:25","2022-03-16 12:10"]
                self.outflow_times=["2022-03-16 12:15","2022-03-16 12:50"]
            elif self.ar_of_day=="AR_entire_2":
                self.inflow_times=["2022-03-16 12:12","2022-03-16 12:55"]
                self.internal_times=["2022-03-16 12:58","2022-03-16 13:40"]
                self.outflow_times=["2022-03-16 13:45","2022-03-16 14:18"]

        self.new_halo_dict={self.flight[0]:\
        {"inflow":self.halo_df.loc[self.inflow_times[0]:self.inflow_times[-1]],
         "internal":self.halo_df.loc[self.internal_times[0]:self.internal_times[-1]],
         "outflow":self.halo_df.loc[self.outflow_times[0]:self.outflow_times[-1]]}}

        #from atmospheric_rivers import Atmospheric_Rivers
        #AR_inflow,AR_outflow=Atmospheric_Rivers.locate_AR_cross_section_sectors(
        #                            new_halo_dict,ERA5_on_HALO.halo_era5,
        #                            flight[0])

        self.relevant_sondes_dict={}
        self.internal_sondes_dict={}
            #pd.Timestamp("2023-03-16 16:00"))
            
        

    def get_hamp_IWV_for_periods_around_all_sonde_locations(
            self,half_period="30s"):
        #hamp_gradients using 1min mean
        hamp_iwv_internal=pd.DataFrame()
        hamp_iwv_internal_mean=pd.Series(data=np.nan,
                        index=self.internal_iwv_sonde.index)
        for t,sonde_time in enumerate(self.internal_iwv_sonde.index):
            if self.flight[0]=="RF06" and \
                self.ar_of_day=="AR_entire_1" and t==0:
                sonde_period_start= pd.Timestamp("2022-03-16 12:01")-\
                    pd.Timedelta(half_period)
                sonde_period_end  = pd.Timestamp("2022-03-16 12:01")+\
                    pd.Timedelta(half_period)
            else:    
                sonde_period_start= pd.Timestamp(sonde_time)-\
                                        pd.Timedelta(half_period)
                sonde_period_end  = pd.Timestamp(sonde_time)+\
                                        pd.Timedelta(half_period)
            temporary_series=self.IWV_retrieved.loc[\
                                    str(sonde_period_start):str(sonde_period_end)]
            
            temporary_series.index=range(temporary_series.shape[0])
            hamp_iwv_internal[sonde_time]=temporary_series
            mean_iwv_hamp_values=self.IWV_retrieved.loc[
                            str(sonde_period_start):str(sonde_period_end)].mean()
            hamp_iwv_internal_mean.loc[sonde_time]=mean_iwv_hamp_values.mean()
            self.hamp_iwv_internal     = hamp_iwv_internal
            self.hamp_iwv_internal_mean= hamp_iwv_internal_mean
        return hamp_iwv_internal,hamp_iwv_internal_mean

    def prepare_ICON(self,half_period="1min"):
        icon_q_sondes={}
        icon_z_sondes={}
        for c,sonde_time in enumerate(self.internal_sondes_dict[self.sector]):
            # Check if time difference lies out of budget corridor 
            # in consecutive one
            if self.relevant_timestamp>self.halo_df.index[-1]:
                icon_q_sondes["Sonde"+str(c)]=\
                    self.snd_halo_icon_hmc["q"].loc[\
                    self.relevant_timestamp-pd.Timedelta(half_period):\
                    self.relevant_timestamp+pd.Timedelta(half_period)].mean(axis=0)
                icon_z_sondes["Sonde"+str(c)]=\
                    self.snd_halo_icon_hmc["Z_Height"].loc[\
                    self.relevant_timestamp-pd.Timedelta(half_period):\
                    self.relevant_timestamp+pd.Timedelta(half_period)].mean(axis=0)
                self.icon_iwv_diff.iloc[c]=self.snd_halo_icon_hmp["Interp_IWV"].loc[\
                    self.relevant_timestamp-pd.Timedelta(half_period):\
                    self.relevant_timestamp+pd.Timedelta(half_period)].mean(axis=0)
            else:
                icon_q_sondes["Sonde"+str(c)]=self.halo_icon_hmc["q"].loc[\
                    self.relevant_timestamp-pd.Timedelta(half_period):\
                    self.relevant_timestamp+pd.Timedelta(half_period)].mean(axis=0)
            icon_z_sondes["Sonde"+str(c)]=self.halo_icon_hmc["Z_Height"].loc[\
                    self.relevant_timestamp-pd.Timedelta(half_period):\
                    self.relevant_timestamp+pd.Timedelta(half_period)].mean(axis=0)
            self.icon_iwv_diff.iloc[c]=self.halo_icon_hmp["Interp_IWV"].loc[\
                    self.relevant_timestamp-pd.Timedelta(half_period):\
                    self.relevant_timestamp+pd.Timedelta(half_period)].mean(axis=0)   
            
        self.icon_q_sondes   = icon_q_sondes
        self.icon_z_sondes   = icon_z_sondes
        #self.icon_iwv_sondes = icon_iwv_sondes

    def plot_internal_iwv_tendency(self,half_period="1min",include_ICON=True):
        # Internal sondes iwv
        time_list=[str(time) for time in self.sonde_times_series.index]
        self.Dropsondes["new_alt"]=self.Dropsondes["alt"].copy()
        new_key_list=[]
        # IWV from internal sondes
        for key in [*self.Dropsondes["alt"].keys()]:
            new_key=str(pd.Timestamp(key))
            self.Dropsondes["alt"][new_key] = self.Dropsondes["alt"].pop(key)
        if not hasattr(self,"internal_iwv_sonde"):
            if self.flight[0]=="RF06" and self.ar_of_day=="AR_entire_2":
                self.internal_iwv_sonde=pd.Series(data=
                [self.Dropsondes["IWV"].iloc[\
                    self.internal_sondes_dict[self.sector][0]],np.nan],
                            index=self.internal_sondes_dict[self.sector].index)
            else:
                self.internal_iwv_sonde=self.Dropsondes["IWV"].iloc[\
                        self.internal_sondes_dict[self.sector]]
        
        minutes_to_divide=\
            self.internal_iwv_sonde.index.to_series().diff().astype(
                'timedelta64[m]')[-1]
        self.iwv_sonde_diff_h=self.internal_iwv_sonde.diff()[-1]/\
            minutes_to_divide*60
    
        #hamp_gradients using 5min mean (default)
        hamp_iwv_internal,hamp_iwv_internal_mean=\
            self.get_hamp_IWV_for_periods_around_all_sonde_locations(
            half_period)
        
        self.hamp_iwv_diff_h=hamp_iwv_internal_mean.diff()/minutes_to_divide*60
        # Include ICON
        # 
        self.icon_iwv_diff_h=pd.Series(data=np.nan, 
            index=self.internal_iwv_sonde.index)
        
        if include_ICON:
            self.prepare_ICON()
        #if self.sector=="warm":
        self.iwv_icon_diff_h=self.icon_iwv_diff.diff()[-1]/minutes_to_divide*60
        if not self.flight[0]=="RF06" and self.ar_of_day=="AR_entire_2":
            iwv_sonde_fig=plt.figure(figsize=(8,14))
            ax1=iwv_sonde_fig.add_subplot(111)
            matplotlib.rcParams['axes.linewidth'] = 3
            ax1.patch.set_facecolor('lightgrey')
            ax1.patch.set_alpha(0.5)
            time_key_list=[*self.Dropsondes["q"].keys()]
            if self.sector=="warm":
                colors=["darkorange","moccasin"]
            else:
                colors=["darkviolet","plum"]
    
            for c,sonde_time in enumerate(self.internal_sondes_dict[self.sector]):
                if self.sector=="warm":
                    relevant_timestamp=time_key_list[sonde_time]
                    ax1.scatter(self.Dropsondes["q"][relevant_timestamp]*1000,
                                self.Dropsondes["alt"][str(relevant_timestamp)]/1000,
                    color="k",s=140)
                    ax1.scatter(self.Dropsondes["q"][relevant_timestamp]*1000,
                                self.Dropsondes["alt"][str(relevant_timestamp)]/1000,
                        label="Sonde "+str(relevant_timestamp),color=colors[c],s=80)
                    iwv_sonde=self.Dropsondes["q"][relevant_timestamp]
                else:
                    relevant_timestamp=pd.Timestamp(sonde_time)
    
                ax1.scatter(self.icon_q_sondes["Sonde"+str(c)]*1000,
                        self.icon_z_sondes["Sonde"+str(c)]/1000,
                        marker="v", s=150, color=colors[c],edgecolor="k",
                        label="ICON "+str(relevant_timestamp))
            
            ax1.set_xlabel("Specific humidity (${\mathrm{gkg}}^{-1}$)")
            ax1.set_ylabel("Height (km)")
            ax1.set_xlim([0,5])
            ax1.set_ylim([0,12])
            if self.sector=="warm":
                ax1.text(1.4,7,
                    "$\dfrac{\delta {IWV}_{\mathrm{Sonde}}}{\delta t}=$"+\
                        str(round(self.iwv_sonde_diff_h,2))+" $\mathrm{mmh}^{-1}$",
                     fontsize=25)
            ax1.text(1.4,6,"$\dfrac{\delta {IWV}_{\mathrm{ICON}}}{\delta t}=$"+\
                         str(round(self.iwv_icon_diff_h,2))+" $\mathrm{mmh}^{-1}$",
                         fontsize=25)
            ax1.text(1.4,5,"$\dfrac{\delta {IWV}_{\mathrm{HAMP}}}{\delta t}=$"+\
                         str(round(self.hamp_iwv_diff[-1],2))+" $\mathrm{mmh}^{-1}$",
                         fontsize=25)
            ax1.xaxis.set_tick_params(width=3,length=10)
            ax1.yaxis.set_tick_params(width=3,length=10)
            ax1.legend(fontsize=24)
            sns.despine(offset=10)
            fig_name=self.flight[0]+"_"+self.ar_of_day+"_"+self.sector+\
                "_sonde_iwv_tendency_"+half_period
            fig_name+=".png"
            iwv_sonde_fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
            print("Figure saved as:",self.plot_path+fig_name)
            
        else:
            print("For ",self.flight[0]," ",self.ar_of_day, "no plotting")
        
    #return hamp_iwv_diff

    def plot_IWV_data_intercomparison(self,half_period):
        if not self.flight[0]=="RF06" and self.ar_of_day=="AR_entire_2":
            iwv_sonde_fig=plt.figure(figsize=(8,14))
            ax1=iwv_sonde_fig.add_subplot(111)
            matplotlib.rcParams['axes.linewidth'] = 3
            ax1.patch.set_facecolor('lightgrey')
            ax1.patch.set_alpha(0.5)
            time_key_list=[*self.Dropsondes["q"].keys()]
            if self.sector=="warm":
                colors=["darkorange","moccasin"]
            else:
                colors=["darkviolet","plum"]
    
            for c,sonde_time in enumerate(
                    self.internal_sondes_dict[self.sector]):
                if self.sector=="warm":
                    relevant_timestamp=time_key_list[sonde_time]
                    ax1.scatter(self.Dropsondes["q"][relevant_timestamp]*1000,
                        self.Dropsondes["alt"][str(relevant_timestamp)]/1000,
                        color="k",s=140)
                    ax1.scatter(self.Dropsondes["q"][relevant_timestamp]*1000,
                                self.Dropsondes["alt"][str(relevant_timestamp)]/1000,
                        label="Sonde "+str(relevant_timestamp),color=colors[c],s=80)
                    iwv_sonde=self.Dropsondes["q"][relevant_timestamp]
                else:
                    relevant_timestamp=pd.Timestamp(sonde_time)
        
                    ax1.scatter(self.icon_q_sondes["Sonde"+str(c)]*1000,
                        self.icon_z_sondes["Sonde"+str(c)]/1000,
                        marker="v", s=150, color=colors[c],edgecolor="k",
                        label="ICON "+str(relevant_timestamp))
                
                ax1.set_xlabel("Specific humidity (${\mathrm{gkg}}^{-1}$)")
                ax1.set_ylabel("Height (km)")
                ax1.set_xlim([0,5])
                ax1.set_ylim([0,12])
                if self.sector=="warm":
                    ax1.text(1.4,7,
                        "$\dfrac{\delta {IWV}_{\mathrm{Sonde}}}{\delta t}=$"+\
                            str(round(self.iwv_sonde_diff_h,2))+" $\mathrm{mm\,h}^{-1}$",
                            fontsize=25)
                    ax1.text(1.4,6,
                        "$\dfrac{\delta {IWV}_{\mathrm{ICON}}}{\delta t}=$"+\
                            str(round(self.iwv_icon_diff_h,2))+\
                            " $\mathrm{mmh}^{-1}$",fontsize=25)
                    ax1.text(1.4,5,
                        "$\dfrac{\delta {IWV}_{\mathrm{HAMP}}}{\delta t}=$"+\
                            str(round(self.hamp_iwv_diff_h[-1],2))+" $\mathrm{mm\,h}^{-1}$",
                            fontsize=25)
                ax1.xaxis.set_tick_params(width=3,length=10)
                ax1.yaxis.set_tick_params(width=3,length=10)
                ax1.legend(fontsize=24)
                sns.despine(offset=10)
                fig_name=self.flight[0]+"_"+self.ar_of_day+"_"+self.sector+\
                    "_sonde_iwv_tendency_"+half_period
                fig_name+=".png"
                iwv_sonde_fig.savefig(self.plot_path+fig_name,
                                      dpi=300,bbox_inches="tight")
                print("Figure saved as:",self.plot_path+fig_name)
            else:
                print("For ",self.flight[0]," ",self.ar_of_day, "no plotting")
            #return hamp_iwv_diff

    def plot_iwv_comparison_sonde_hamp(self,
            #iwv_sonde,mwr,halo_df,
            half_period):
        matplotlib.rcParams.update({"font.size":24})
        internal_map=plt.figure(figsize=(12,8))
        ax1=internal_map.add_subplot(111)
        self.hamp_iwv_internal.plot.hist(bins=np.linspace(5,20,21),
                                    alpha=0.3,xlim=[10,20],ax=ax1)
    
        ins = ax1.inset_axes([0.15,0.25,0.3,0.35])
        ins.plot(self.mwr["lon"],self.mwr["lat"],color="grey",ls="--",lw=1)
        ins.scatter(self.halo_df["longitude"],self.halo_df["latitude"],s=3,color="k")
        colors=["blue","orange"]
        for i,sonde_time in enumerate(self.internal_iwv_sonde.index):
            ## Mean values
            #  Dropsondes
            if not self.internal_iwv_sonde.iloc[i]==np.nan:
                ax1.axvline(x=self.internal_iwv_sonde.iloc[i],
                            lw=3,ls="--",color="k")
                ax1.axvline(x=self.internal_iwv_sonde.iloc[i],
                            lw=2,ls="--",color=colors[i])
            # HAMP
            ax1.axvline(x=self.hamp_iwv_internal_mean.iloc[i],
                        lw=3,color=colors[i])
    
            sonde_period_start= pd.Timestamp(sonde_time)-pd.Timedelta(half_period)
            sonde_period_end  = pd.Timestamp(sonde_time)+pd.Timedelta(half_period)
            rlv_geoloc=self.mwr.sel({"time":slice(sonde_period_start,
                                                  sonde_period_end)})
            sonde_lon=self.mwr["lon"].sel({"time":sonde_time})
            sonde_lat=self.mwr["lat"].sel({"time":sonde_time})
            ins.scatter(rlv_geoloc["lon"],rlv_geoloc["lat"],color=colors[i])
            ins.scatter(sonde_lon,sonde_lat, s=100, marker="v",
                        color=colors[i],lw=2,edgecolor="k")
        ins.set_xlim([self.halo_df["longitude"].min()-1,
                          self.halo_df["longitude"].max()+1])
        ins.set_xlabel("Lon (°E)")
        ins.set_ylabel("Lat (°N)")
        
        ins.set_ylim([self.halo_df["latitude"].min()-1,
                          self.halo_df["latitude"].max()+1])
        ins.set_yticks([72,74,76,78])
        for spine in ["bottom","left","right","top"]:
            ins.spines[spine].set_linewidth(2)
        ax1.set_xlabel("HAMP-based IWV ($\mathrm{kg}\,\mathrm{m}^{-2})$")
        
        ax1.set_yticks([0,250,500])
        ax1.set_ylim([0,500])
        sns.despine(ax=ax1,offset=5)
        for spine in ["bottom","left"]:
            ax1.spines[spine].set_linewidth(2)
        ax1.tick_params("x",width=2,length=8)
        ax1.tick_params("y",width=2,length=8)
            
        fig_name=self.flight[0]+"_"+self.ar_of_day+"_"+self.sector+\
                "_hamp_sonde_iwv_comparison"+"_twice_"+half_period+".png"
        if self.flight[0]=="RF05" and self.ar_of_day=="AR_entire_1":
                fig_name="Fig05_"+fig_name
        internal_map.savefig(self.plot_path+fig_name,dpi=600,
                             bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)

# Compare sondes and HAMP IWV with ERA5
    def plot_iwv_flight_trend(self,surface_index,half_period):
        iwv_trend=plt.figure(figsize=(16,9))
        plt.scatter(self.sonde_times_series.index,
                    self.Dropsondes["IWV"],
                    s=100,marker="v",lw=3,edgecolor="k",
                    color="purple",label="Sondes")
        plt.plot(self.halo_era5.index,self.halo_era5["Interp_IWV"],
                 color="grey",lw=3,label="ERA5",ls="--")
        if not self.snd_halo_era5.shape[0]==0:
                plt.plot(self.snd_halo_era5.index,
                     self.snd_halo_era5["Interp_IWV"],
                     color="grey",lw=3,label="ERA5",ls="--")
        sea_IWV_retrieved=self.IWV_retrieved.copy()
        sea_IWV_retrieved[~surface_index]=np.nan
        plt.plot(sea_IWV_retrieved.index,
                 sea_IWV_retrieved,
                 color="purple",lw=1,label="HAMP")
        for i,sonde_time in enumerate(self.internal_iwv_sonde.index):
            if i > 0 and self.flight[0]!="RF06" and \
                self.ar_of_day!="AR_entire_2":
                plt.scatter(sonde_time,self.Dropsondes["IWV"].loc[sonde_time],
                            marker="o",facecolors="none",edgecolor="k",lw=3,s=300)
            sonde_period_start= pd.Timestamp(sonde_time)-pd.Timedelta(half_period)
            sonde_period_end  = pd.Timestamp(sonde_time)+pd.Timedelta(half_period)
            rlv_geoloc=self.mwr.sel({"time":slice(sonde_period_start,sonde_period_end)})
            plt.plot(rlv_geoloc.time,self.IWV_retrieved.loc[rlv_geoloc.time],
                     color="grey",lw=8)
            plt.plot(rlv_geoloc.time,self.IWV_retrieved.loc[rlv_geoloc.time],
                     color="purple",lw=2)
        fig_name="IWV_trend_comparison_"+self.flight[0]+"_half_period_"+\
            half_period
        fig_name+=".png"
        plot_path=self.plot_path+"/supplements/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        iwv_trend.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)

# Old
#def rain_distribution_comparison(precip_icon_series,halo_icon_hmp,halo_era5, 
#radar_precip_rate,
#                                 flight,ar_of_day,sector_to_plot,
#                                conditional_dist=False):
#    import seaborn as sns
#    sns.set_style("white")#
#
#    x1 = precip_icon_series.copy()
#    x2 = halo_icon_hmp["Interp_Precip"]
#    x3 = halo_era5["Interp_Precip"]
#    x4 = radar_precip_rate["rate"]
#    
#    if conditional_dist:
#        x1=x1[x1>0]
#        x2=x2[x2>0]
#        x3=x3[x3>0]
#        x4=x4[x4>0]
#    # Plot
#    kwargs = dict(hist_kws={'alpha':.3}, kde_kws={'linewidth':2})

#    plt.figure(figsize=(10,7), dpi= 300)#
#    sns.distplot(x3, bins=np.linspace(0,2,91),color="deeppink", label="ERA5 HALO Track", **kwargs)
#    sns.distplot(x1, bins=np.linspace(0,2,91),color="dodgerblue", label="ICON AR region", **kwargs)
#    sns.distplot(x2, bins=np.linspace(0,2,91),color="orange", label="ICON HALO Track", **kwargs)
#    sns.distplot(x4, bins=np.linspace(0,2,91),color="k", label="Radar HALO Track", **kwargs)
#    plt.ylim(0,50)
#    plt.xlim(0,1)
#    if not conditional_dist:
#       plt.xlabel("Precipitation rate / $\mathrm{mmh}^{-1}$")
#   else:
#        plt.xlabel("Conditional Precipitation rate / $\mathrm{mmh}^{-1}$")
#    
#    plt.legend()
#    fig_name="Rain_rate_representativeness.png"
#    if conditional_dist:
#        fig_name="Conditional_"+fig_name
#    fig_name=flight[0]+"_"+ar_of_day+"_"+sector_to_plot+"_"+fig_name
#    fig_path=plot_path
#    sns.despine(offset=1)
#    plt.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
#    print("Figure saved as ",fig_path+fig_name)

#def sector_rain_halo_icon_comparison(warm_radar_rain,cold_radar_rain,
#                                    warm_icon_rain,cold_icon_rain,
#                                    flight,ar_of_day,sector_to_plot,
#                                     plot_path):
#    x1=warm_radar_rain["rate"]#.loc[warm_radar_rain["rate"]>0.001]
#    x2=warm_icon_rain["rate"]#.loc[warm_icon_rain["rate"]>0.0]
#    x3=cold_radar_rain["rate"]#.loc[cold_radar_rain["rate"]>0.001]
#    x4=cold_icon_rain["rate"]#.loc[cold_icon_rain["rate"]>0.0]
#
#    x1.index=range(len(x1))
#    x2.index=range(len(x2))
#    x3.index = range(len(x3))
#    x4.index = range(len(x4))
#    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
#    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(12,12))#
#
#    import seaborn as sns
#    sns.set_style("white")
#    import matplotlib
#    matplotlib.rcParams.update({"font.size":22})
#    # Plot
#    kwargs = dict(hist_kws={'alpha':.3}, kde_kws={'linewidth':2})
#    sns.distplot(x3, bins=np.linspace(0,2,91),color="darkorange", label="Pre-frontal radar",ax=ax_hist, **kwargs)
#    sns.distplot(x1, bins=np.linspace(0,2,91),color="orange", label="Pre-frontal ICON",ax=ax_hist, **kwargs)
#    sns.distplot(x2, bins=np.linspace(0,2,91),color="darkviolet", label="Post-frontal radar", ax=ax_hist, **kwargs)
#    sns.distplot(x4, bins=np.linspace(0,2,91),color="mediumorchid", label="Post-frontal ICON", ax=ax_hist,**kwargs)
#    plt.ylim(0,50)
#    plt.xlim(0,1)
#    plt.xlabel("Conditional Precipitation rate / $\mathrm{mmh}^{-1}$")
#    plt.legend()
#    fig_name=flight[0]+"_"+ar_of_day+"_"+sector_to_plot+"_Conditional_Rain_rate_representativeness.png"
#    fig_path=plot_path
#    sns.despine(offset=1)
#    plt.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
#    print("Figure saved as ",fig_path+fig_name)
            