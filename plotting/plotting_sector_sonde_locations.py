# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:39:03 2023

@author: Henning Dorff
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Overview map
import cartopy
import matplotlib
import cartopy.crs as ccrs
            
import atmospheric_rivers as AR
import reanalysis as Reanalysis

def main(flight,ar_of_day,ds,halo_df,Dropsondes,relevant_sondes_dict,
         internal_sondes_dict,snd_halo_icon_hmp,plot_path,
         add_other_sectors=False,add_other_sondes=False):
    """
    

    Parameters
    ----------
    flight : list
        list of flights, mostly it is just a list of length=1, e.g. ["RF00"].
    ar_of_day : str
        AR corridor label to analyse
    ds : xr.Dataset 
        ERA5 dataset containing IVT background values.
    halo_df : pd.DataFrame
        HALO-BAHAMAS geolocation
    Dropsondes : dict
        Dropsonde data for given flight (all sondes)
    relevant_sondes_dict: dict
        Sector specific dict indicating the sondes to use per frontal-sector
    internal_sondes_dict: dict
        Analogeously, but purely for the sondes in the internal section
    snd_halo_icon_hmp: pd.DataFrame
        Collocated ICON HMP data such as IWV and IVT
    plot_path: str
        Plot path to store figure in.
    Returns
    -------
    None.

    """
    #############################
    # Predefinitions
    # Define the plot specifications for the given variables
    import matplotlib.ticker as mticker
    met_var_dict={}
    met_var_dict["ERA_name"]    = {"EV":"e","TP":"tp",
                                           "IWV":"tcwv","IVT":"IVT",
                                           "IVT_conv":"IVT_conv"}
    met_var_dict["colormap"]    = {"EV":"Blues","IVT_conv":"BrBG_r",
                                           "TP":"Blues","IVT":"speed"}
    met_var_dict["levels"]      = {"IWV":np.linspace(10,50,51),
                                           "EV":np.linspace(0,1.5,51),
                                           "TP":np.linspace(0,1.5,51),
                                           "IVT_conv":np.linspace(-2,2,101),
                                           "IVT":np.linspace(50,600,61)}
    
    met_var_dict["units"]       = {"EV":"(kg$\mathrm{m}^{-2}$)",
                            "TP":"(kg$\mathrm{m}^{-2}$)",
                            "IVT_conv":"(\mathrm{mm\,h}^{-1}$)",
                            "IWV":"(kg$\mathrm{m}^{-2}\mathrm{h}^{-1}$)",
                            "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}     
    #############################

    
    fig_name=flight[0]+"_"+ar_of_day+"_ERA5_IVTdiv.png"
    if flight[0]=="RF05":
        central_lat=70
        central_lon=-10
    elif flight[0]=="RF06":
        central_lat=74
        central_lon=15
    last_hour=12
    #ERA5_on_HALO=era_on_halo_cls
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
    map_fig=plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,2,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=central_lon,
                                central_latitude=central_lat))
    
    # IVT convergence as background based on ERA5
    C1=ax1.contourf(ds["longitude"],ds["latitude"],
            ds[met_var_dict["ERA_name"]["IVT_conv"]][last_hour,:,:]/997*1000,
            levels=met_var_dict["levels"]["IVT_conv"],extend="both",
            transform=ccrs.PlateCarree(),
            cmap=met_var_dict["colormap"]["IVT_conv"],alpha=0.95)
    print("IVT conv. mapped")

    ax1.coastlines(resolution="50m")
    gl1=ax1.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
    # Add flight track and dropsonde locations
    ax1.plot(halo_df["longitude"],halo_df["latitude"],color="white",
             ls="-",lw=4,transform=ccrs.PlateCarree())
    ax1.plot(halo_df["longitude"],halo_df["latitude"],color="purple",ls="-",
             lw=2,transform=ccrs.PlateCarree())
    
    ax1.set_extent([halo_df["longitude"].min()-2,halo_df["longitude"].max()+2,
                halo_df["latitude"].min()-2,halo_df["latitude"].max()+2])
    # all sondes
    if add_other_sondes:
        ax1.scatter(Dropsondes["Lon"].values,Dropsondes["Lat"].values,
                marker="v",s=8,color="lightgrey",edgecolor="darkgrey",
                transform=ccrs.PlateCarree(),zorder=2)
        
        ax1.scatter(Dropsondes["Lon"].iloc[internal_sondes_dict["warm"][0]],
                Dropsondes["Lat"].iloc[internal_sondes_dict["warm"][0]],
                marker="o",s=100,color="grey",edgecolor="k",
                transform=ccrs.PlateCarree(),zorder=3)
    # warm sectors
    ax1.scatter(Dropsondes["Lon"].iloc[relevant_sondes_dict["warm_sector"]["in"]],
        Dropsondes["Lat"].iloc[relevant_sondes_dict["warm_sector"]["in"]],
        marker="v",s=100,color="orange",edgecolor="k",
        transform=ccrs.PlateCarree(),zorder=3)
    
    
        
    ax1.scatter(Dropsondes["Lon"].iloc[relevant_sondes_dict["warm_sector"]["out"]],
            Dropsondes["Lat"].iloc[relevant_sondes_dict["warm_sector"]["out"]],
            marker="v",s=100,color="orange",edgecolor="k",
           transform=ccrs.PlateCarree(),zorder=3)
    
    ax1.scatter(Dropsondes["Lon"].iloc[relevant_sondes_dict["cold_sector"]["in"]],
            Dropsondes["Lat"].iloc[relevant_sondes_dict["cold_sector"]["in"]],
           marker="v",s=100,color="blue",edgecolor="k",
           transform=ccrs.PlateCarree(),zorder=3)
    
    if add_other_sectors:
        if relevant_sondes_dict["cold_sector"]["out"].shape[0]>0:
            if flight[0]=="RF05" and ar_of_day=="AR_entire_1":
                print("synthetic sondes for cold sector are included")
                ax1.scatter(halo_df["longitude"].loc[\
                            relevant_sondes_dict["cold_sector"]["out"].index],
                        halo_df["latitude"].loc[\
                            relevant_sondes_dict["cold_sector"]["out"].index],
                       marker="*",s=100,color="blue",edgecolor="k",
                       transform=ccrs.PlateCarree())
            
                print("synthetic sondes for internal sector are included")
                ax1.scatter(halo_df["longitude"].loc[pd.DatetimeIndex([\
                                        internal_sondes_dict["cold"][0]])],
                        halo_df["latitude"].loc[pd.DatetimeIndex([\
                                        internal_sondes_dict["cold"][0]])],
                       marker="*",s=100,color="grey",edgecolor="k",
                       transform=ccrs.PlateCarree())
                ax1.scatter(
                        snd_halo_icon_hmp["Halo_Lon"].loc[pd.DatetimeIndex([\
                                internal_sondes_dict["cold"][1]])],
                        snd_halo_icon_hmp["Halo_Lat"].loc[pd.DatetimeIndex([\
                                internal_sondes_dict["cold"][1]])],
                        marker="*",s=100,color="grey",edgecolor="k",
                        transform=ccrs.PlateCarree())
        else:
            ax1.scatter(Dropsondes["Lon"].iloc[\
                        relevant_sondes_dict["cold_sector"]["out"]],
                        Dropsondes["Lat"].iloc[\
                        relevant_sondes_dict["cold_sector"]["out"]],
                        marker="v",s=100,color="blue",edgecolor="k",
                        transform=ccrs.PlateCarree())

    gl1.xlocator = mticker.FixedLocator([-30,-15,0,15,30])
    
    gl1.top_labels=True
    gl1.bottom_labels=False
    gl1.left_labels=True
    gl1.right_labels=False
    
    
    axins1=inset_axes(ax1,width="3%",
                              height="80%",
                              loc="center",
                              bbox_to_anchor=(0.55,0,1,1),
                              bbox_transform=ax1.transAxes,
                              borderpad=0)       
    cb=map_fig.colorbar(C1,cax=axins1)
    cb.set_label("$ div\,IVT\,"+" "+met_var_dict["units"]["IVT_conv"])
    cb.set_ticks([-1.0,0,1.0])
    fig_name="Fig15_"+fig_name
    fig_plot_path=plot_path+fig_name
    map_fig.savefig(fig_plot_path,dpi=300,bbox_inches="tight")
    print("Figure saved as:",fig_plot_path)
#if __name__=="__main__":
#    main()
    