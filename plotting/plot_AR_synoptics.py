# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:16:37 2024

@author: u300737
"""
def main():
    #%% Import
    import warnings
    
    import os
    import sys
    
    import numpy as np
    import pandas as pd
    import xarray as xr
    
    base_path=os.getcwd()+"/../../../"
    work_path=base_path+"/Work/GIT_Repository/"
    synth_git_path=base_path+"/my_GIT/Synthetic_Airborne_Arctic_ARs/"
    budget_script_path=base_path+"/my_GIT/HALO_AC3_AR_Budget_Closure/scripts/"
    src_path=synth_git_path+"/src/"
    src_plot_path=synth_git_path+"/plotting/"
    sys.path.insert(1,synth_git_path)
    sys.path.insert(2,src_path)
    sys.path.insert(3,src_plot_path)
    sys.path.insert(4,work_path)
    sys.path.insert(5,budget_script_path)
    
    airborne_data_path=work_path+"hamp_processing_py\\hamp_processing_python\\"#\\Flight_Data\\HALO_AC3\\sea_ice\\"
    airborne_importer_path=airborne_data_path#actual_working_path+"/../../"+desired_path_str
    
    # Plotting modules
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatch
    import matplotlib.patheffects as PathEffects
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.image import imread
    
    import seaborn as sns
    
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    
    # Data and processing modules
    import data_config
    import Performance
    
    import flightcampaign
    from reanalysis import ERA5
    from ICON import ICON_NWP as ICON
    import atmospheric_rivers as AR
    from simplified_flight_leg_handling import simplified_run_grid_main
        
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
             url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.jpg?api_key=0963bb5f-6e8c-4978-9af0-4cd3a2627df9"
            #https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.png?api_key={API_KEY}"
             return url
    #-----------------------------------------------------------------------------#
    #%% Definitions
    
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
    
    #%% Plot configurations
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
    
    #%% Load HALO aircraft
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
    #%% Data Query
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
    
    ax1.set_extent([-40,25,55,90]) 
    ax2.set_extent([-40,25,55,90]) 
    
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
    plt.subplots_adjust(wspace=0.15,hspace=-0.3)
    fig_name="AR_RF05_synoptic_conditions_"+hour_str+"_UTC.png"
    plot_path=os.getcwd()+"/../plots/"
    map_fig.savefig(plot_path+fig_name,dpi=600,bbox_inches="tight")
    print("Figure saved as:", plot_path+fig_name)

if __name__=="__main__":
    main()