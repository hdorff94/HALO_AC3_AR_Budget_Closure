# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:16:43 2023

@author: u300737
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:09:33 2023

@author: u300737
"""
import numpy as np
import pandas as pd
import xarray as xr

import glob
import os
import sys

def importer():
    paths_dict={}
    paths_dict["current_path"]=os.getcwd()
    paths_dict["actual_working_path"]=os.getcwd()+\
        "/../../Synthetic_Airborne_Arctic_ARs/"
    os.chdir(paths_dict["actual_working_path"]+"/config/")
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
                            
    os.chdir(paths_dict["airborne_processing_module_path"])
    sys.path.insert(1,paths_dict["airborne_script_module_path"])
    sys.path.insert(2,paths_dict["airborne_processing_module_path"])
    sys.path.insert(3,paths_dict["airborne_plotting_module_path"])
    sys.path.insert(4,paths_dict["airborne_data_importer_path"])
    sys.path.insert(5,paths_dict["scripts_path"])
    sys.path.insert(6,paths_dict["hamp_processing_path"])
    sys.path.insert(7,paths_dict["src_hamp_processing_path"])
    return paths_dict

def main():
    
    paths_dict=importer()
    #%% Define the flight campaign classes
    import quicklook_dicts
    import measurement_instruments_ql
    import flightcampaign
    if "data_config" in sys.modules:
        import data_config
    
    save_in_manuscript_path=False
    config_file=data_config.load_config_file(
                    paths_dict["airborne_data_importer_path"],
                        "data_config_file")
    
    flights=["RF05","RF06"]
    # Initiate HALO-AC3 campaign
    is_flight_campaign=True
    haloac3=flightcampaign.HALO_AC3(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",interested_flights=[flights[0]],
                          instruments=["radar","radiometer","sonde"])
    haloac3.specify_flights_of_interest(flights)
    #haloac3.create_directory(directory_types=["data"])
    
    #%% Get the flight data    
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
    
    #%% Plot the map
    import matplotlib
    matplotlib.rcParams.update({"font.size":20})
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy
    import cartopy.crs as ccrs
    try:
        from typhon.plots import styles
    except:
        print("Typhon module cannot be imported")
        
    from reanalysis import ERA5        
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
                
        
    col_no=len(flights)
    row_no=1
    projection=ccrs.AzimuthalEquidistant(central_longitude=-2.0,central_latitude=72)
    fig,axs=plt.subplots(row_no,col_no,sharex=True,sharey=True,figsize=(12,16),
                             subplot_kw={'projection': projection})
    key=0
    era_index_dict={"20220313":16,
                    "20220315":16,
                    "20220316":14,
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
    # Aircraft data
    fig_labels=["(a)","(b)","(c)"]
    gls=[None,None,None]
    campaign=[*flight_dates.keys()][0]
    campaign_path="C://Users/u300737/Desktop/Desktop_alter_Rechner/"+\
                    "PhD_UHH_WIMI/Work/GIT_Repository/"+"hamp_processing_py/"+\
                        "hamp_processing_python/Flight_Data/"+campaign+"/"
    
    import cartopy.io.img_tiles as cimgt

    class StadiaStamen(cimgt.Stamen):
        def _image_url(self, tile):
            x,y,z = tile
            url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.jpg?api_key=0963bb5f-6e8c-4978-9af0-4cd3a2627df9"
            return url
    stamen_terrain = StadiaStamen('terrain-background')
    
    amsr2_sea_ice_path=campaign_path+\
        "\\sea_ice\\"
    # sea ice maps
    orig_map = plt.cm.get_cmap('Blues')
    # reversing the original colormap using reversed() function
    reversed_map = orig_map.reversed() 

    for col in range(col_no):
        flight_date= [*flight_dict.keys()][key]
        print("Flight date",flight_date)
               
        cmpgn_cls=[*flight_dict.values()][key][0]
        flight=[*flight_dict.values()][key][1]
        cfg_dict=quicklook_dicts.get_prcs_cfg_dict(
                            flight, flight_dates[campaign][flight], 
                            campaign,campaign_path)
        HALO_Devices_cls=measurement_instruments_ql.HALO_Devices(cfg_dict)
        Sondes_cls=measurement_instruments_ql.Dropsondes(HALO_Devices_cls)
        Sondes_cls.open_all_sondes_as_dict()
        Sondes_cls.calc_integral_variables(integral_var_list=["IWV","IVT"])
        sonde_data=Sondes_cls.sonde_dict
    
    
        # Aircraft   
        haloac3.load_AC3_bahamas_ds(flight)
        halo_dict=haloac3.bahamas_ds
        if isinstance(halo_dict,pd.DataFrame):
            halo_df=halo_dict.copy() 
        elif isinstance(halo_dict,xr.Dataset):
            halo_df=pd.DataFrame(data=np.nan,columns=["alt","Lon","Lat"],
                        index=pd.DatetimeIndex(np.array(halo_dict["TIME"][:])))
        halo_df["longitude"]=halo_dict["IRS_LON"].data
        halo_df["latitude"]=halo_dict["IRS_LAT"].data

        ##### Load ERA5-data
        era5=ERA5(for_flight_campaign=True,campaign=cmpgn_cls.name,
                  research_flights=flight,
                  era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
           
        hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
        file_name="total_columns_"+flight_date[0:4]+"_"+\
                       flight_date[4:6]+"_"+\
                       flight_date[6:8]+".nc"    
            
        era_ds,era_path=era5.load_era5_data(file_name)
        era_index=era_index_dict[flight_date] 
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
        axs[col].add_image(stamen_terrain, 1)
        # Plot sea ice
        sea_ice_file_list=glob.glob(amsr2_sea_ice_path+"*"+flight_date+"*.nc")
        sea_ice_ds=xr.open_dataset(sea_ice_file_list[0])
        seaice=sea_ice_ds["seaice"]                                       
        # Sea ice
        axs[col].pcolormesh(seaice.lon, seaice.lat,np.array(seaice[:]), 
                transform=ccrs.PlateCarree(), cmap=reversed_map)
    
        # Create white overlay for less strong colors
        x=np.linspace(-90,90,41)
        y=np.linspace(55,90,93)
        x_grid,y_grid=np.meshgrid(x,y)
        white_overlay= np.zeros((41,93))+0.3
        axs[col].contourf(x_grid,y_grid,white_overlay.T,cmap="Greys",vmin=0,vmax=1,
                 transform=ccrs.PlateCarree(),alpha=0.6)
        
        # Plot IVT
        C1=axs[col].contourf(era_ds["longitude"],era_ds["latitude"],
                era_ds[met_var_dict["ERA_name"][meteo_var]][era_index,:,:],
                levels=met_var_dict["levels"][meteo_var],extend="max",
                transform=ccrs.PlateCarree(),
                cmap=met_var_dict["colormap"][meteo_var],alpha=0.6,
                zorder=6)
        
        # Plot surface presure
        C_p=axs[col].contour(era_ds["longitude"],era_ds["latitude"],
                                era_ds["msl"][era_index,:,:]/100,
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
        axs[col].text(-18.35, 66.5, ar_label[flight_date]+\
                      " "+str(era_index)+" UTC",
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
        u=era_ds["IVT_u"][era_index,::step,::step]
        v=era_ds["IVT_v"][era_index,::step,::step]
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
            scat=axs[col].scatter(release_lon,release_lat,marker=sonde_shapes[1],
                       s=300,edgecolors="k",c=sonde_data["IVT"].loc[sonde],
                       cmap="BuGn",vmin=0, vmax=500,
                       transform=ccrs.PlateCarree(), zorder=9)
            scat2=axs[col].scatter(release_lon,release_lat,marker=sonde_shapes[0],
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
    cbar_ax.text(0.4,-3.5,meteo_var+" "+met_var_dict["units"][meteo_var],
                 fontsize=18,transform=cbar_ax.transAxes)
    cbar2_ax=fig.add_axes([0.6,0.28,0.1,0.01])
    cbar2 = fig.colorbar(scat2,cax=cbar2_ax,
                         extend="max",orientation="horizontal")
    cbar2.set_label("IWV ($\mathrm{kg}\mathrm{m}^{-2}$)",fontsize=16)
        
    if not save_in_manuscript_path:
        fig_path=paths_dict["current_path"]+"/../plots/"#paths_dict["airborne_plotting_module_path"]
    else:
        pass
        #    fig_path=paths_dict["manuscript_path"]
    fig_name="Fig02_AR_RF05_RF06_tendency.png"
    fig.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
    print("Figure saved as:",fig_path+fig_name)
    
if __name__=="__main__":
    main()