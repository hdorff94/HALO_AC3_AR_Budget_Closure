# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:08:19 2023

@author: Henning Dorff
"""

def simplified_run_grid_main(config_file_path="",
         campaign="HALO_AC3",hmp_plotting_desired=True,
         hmc_plotting_desired=False,
         plot_data=True,ar_of_day="AR_entire_1",
         flight=["RF05"],
         era_is_desired=True,carra_is_desired=False,
         icon_is_desired=False,synthetic_flight=False,
         ######################################################################
         # USEFUL values
         upsample_time="20min", # ---> very important for computational ressources
         ######################################################################
         track_type="internal",
         merge_all_legs=False,
         pick_legs=["inflow","internal","outflow"],
         open_calibrated=True):
    
    import flightcampaign
    ###############################################################################
    #Grid Data
    from reanalysis import ERA5,CARRA 
    from ICON import ICON_NWP as ICON
    import gridonhalo as Grid_on_HALO
    ###############################################################################

    import data_config
    import os
    import sys
    
    import numpy as np
    import pandas as pd
    import xarray as xr
    # real campaigns
    years={"RF02":"2022","RF03":"2022","RF04":"2022","RF05":"2022",
               "RF06":"2022","RF07":"2022","RF08":"2022","RF16":"2022"}
    months={"RF02":"03","RF03":"03","RF04":"03","RF05":"03",
               "RF06":"03","RF07":"03","RF08":"03","RF16":"04"}
    days={"RF02":"12","RF03":"13","RF04":"14","RF05":"15",
               "RF06":"16","RF07":"20","RF08":"21","RF16":"10"}
    hours_time=['00:00', '01:00', '02:00','03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00','09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00','15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00','21:00', '22:00', '23:00',]
    analysing_campaign=True
    
    airborne_data_importer_path=config_file_path+\
                                "hamp_processing_py/"+\
                                    "hamp_processing_python/"
    print("Analyse given flight: ",flight[0])
    config_file=data_config.load_config_file(config_file_path,
                                             "data_config_file")
    
    date=years[flight[0]]+months[flight[0]]+days[flight[0]]
    
    plot_cfad=False
    #-------------------------------------------------------------------------#
    # Boolean Definition of Task to do in Analysis
    # Define the hydrometeor parameters to analyze and to plot         
    include_retrieval=False
    do_orographic_masking=False
    do_moisture_budget=False
    #-------------------------------------------------------------------------#
    if flight[0]=="RF12":
        do_orographic_masking=True
    print("Analyse AR:",ar_of_day)
    
    if plot_data:
        if not any("plotting" in path for path in sys.path):
            # add plot_path to import things
            current_path=os.getcwd()
            plot_path=current_path+"/plotting/"
            print(sys.path)
        # Plot modules
        import matplotlib.pyplot as plt
        try:
            from typhon.plots import styles
        except:
            print("Typhon module cannot be imported")
        
        from flightmapping import FlightMaps
        import interpdata_plotting 

    else:
        print("No data is plotted.")
       
    print("Main path:",config_file["Data_Paths"]["campaign_path"])
    
    ac3=flightcampaign.HALO_AC3(is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",instruments=["radar","dropsondes","sonde"])
    cmpgn_cls=ac3
    working_path=os.getcwd()+"/../../../Work/"
    airborne_data_importer_path=working_path+"/GIT_Repository/"+\
                                "hamp_processing_py/"+\
                                    "hamp_processing_python/"
    measurement_processing_path=os.getcwd()+\
                "/../../hamp_processing_python/src/"
    
    sys.path.insert(4,measurement_processing_path)
    import campaign_time
    import config_handler
    import measurement_instruments_ql as Instruments
                
    cfg=config_handler.Configuration(
                    major_path=airborne_data_importer_path)
                
    processing_cfg_name="unified_grid_cfg"    
    cfg.add_entries_to_config_object(processing_cfg_name,
                            {"t1":date,"t2":date,
                             "date":date,"flight_date_used":date})
    processing_config_file=cfg.load_config_file(processing_cfg_name)
    processing_config_file["Input"]["data_path"]=\
                    processing_config_file["Input"]["campaign_path"]+\
                        "Flight_Data/"
    processing_config_file["Input"]["device_data_path"]=\
                    processing_config_file["Input"]["data_path"]+campaign+"/"
                
    prcs_cfg_dict=dict(processing_config_file["Input"])    
    prcs_cfg_dict["date"]=date
    Campaign_Time_cls=campaign_time.Campaign_Time(
                    campaign,date)
    prcs_cfg_dict["Flight_Dates_used"] =\
                    Campaign_Time_cls.specify_dates_to_use(prcs_cfg_dict)
    
    HALO_cls=Instruments.HALO_Devices(prcs_cfg_dict)
    Bahamas_cls=Instruments.BAHAMAS(HALO_cls)
    Bahamas_cls.open_bahamas_data(raw_or_processed="processed")
    bahamas_ds=Bahamas_cls.bahamas_ds[["alt","lat","lon","speed_gnd"]]
    bahamas_ds=bahamas_ds.rename_vars({"lat":"latitude","lon":"longitude",
                                    "speed_gnd":"groundspeed"})
    halo_df=bahamas_ds.to_dataframe()
    
    halo_df["Hour"]=pd.DatetimeIndex(halo_df.index).hour
    halo_df["Minutes"]=pd.DatetimeIndex(halo_df.index).minute
    halo_df["Minutesofday"]=halo_df["Hour"]*60+halo_df["Minutes"]
    
    if "distance" in halo_df.columns:
        del halo_df["distance"]
    
    #Define the file names of hydrometeor data and paths
    flight_name=flight[0]
    
    if ar_of_day:
        interpolated_hmp_file=flight_name+"_"+ar_of_day+\
                                "_HMP_ERA_HALO_"+date+".csv"
    else:
        interpolated_hmp_file="HMP_ERA_HALO_"+date+".csv"
    
    hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
    hydrometeor_lvls_file="hydrometeors_pressure_levels_"+date+".nc"
        
    if synthetic_flight:
        interpolated_hmp_file="Synthetic_"+interpolated_hmp_file
        # Until now ERA5 is not desired
    if ar_of_day is not None:
        interpolated_iwc_file=flight_name+"_"+ar_of_day+"_IWC_"+date+".csv"        
    else:
        interpolated_iwc_file=flight_name+"_IWC_"+date+".csv"
    if synthetic_flight:
        interpolated_iwc_file="Synthetic_"+interpolated_iwc_file
    #if icon_is_desired:
    #    if synthetic_icon:
    #        hydrometeor_lvls_path=hydrometeor_lvls_path+"Latitude_"+\
    #        str(synthetic_icon_lat)+"/"
       
    #else:
    #    print("This none dataset of the flight campaign.")
    #    print("No airborne datasets will be integrated.")
    #-------------------------------------------------------------------------#
    #%% ERA5 class & ERA5 on HALO Class
    era5=ERA5(for_flight_campaign=True,campaign=campaign,research_flights=None,
                     era_path=hydrometeor_lvls_path)

    ERA5_on_HALO=Grid_on_HALO.ERA_on_HALO(
                                halo_df,hydrometeor_lvls_path,
                                hydrometeor_lvls_file,interpolated_iwc_file,
                                analysing_campaign,campaign,
                                config_file["Data_Paths"]["campaign_path"],
                                flight,date,config_file,ar_of_day=ar_of_day,
                                synthetic_flight=synthetic_flight,
                                do_instantaneous=False)
    #%% CARRA class & CARRA on HALO class
    if carra_is_desired:
        interpolated_carra_file=""
        carra_lvls_path=cmpgn_cls.campaign_path+"/data/CARRA/"
    
        carra=CARRA(for_flight_campaign=True,
                    campaign=campaign,research_flights=None,
                    carra_path=carra_lvls_path) 
        
        CARRA_on_HALO=Grid_on_HALO.CARRA_on_HALO(
                                halo_df,carra_lvls_path,
                                analysing_campaign,campaign,
                                config_file["Data_Paths"]["campaign_path"],
                                flight,date,config_file,ar_of_day=ar_of_day,
                                upsample_time=upsample_time,
                                synthetic_flight=synthetic_flight,
                                do_instantaneous=False)
   
    # Measurement instruments if needed
    HAMP_cls=Instruments.HAMP(HALO_cls)
    HAMP_cls.open_processed_hamp_data(open_calibrated=open_calibrated,
                            newest_version=True)
    
    RADAR_cls=Instruments.RADAR(HALO_cls)
    RADAR_cls.open_processed_radar_data(reflectivity_is_calibrated=open_calibrated)
            
    if not open_calibrated:
        radar_ds=RADAR_cls.processed_radar_ds
        mwr=HAMP_cls.processed_hamp_ds
    else:
        radar_ds=RADAR_cls.calib_processed_radar_ds
        mwr=HAMP_cls.calib_processed_hamp_ds
    mwr=mwr.rename({"TB":"T_b"})
    radar={}
    radar["Reflectivity"]=pd.DataFrame(data=np.array(
                                    radar_ds["dBZg"].values[:]),
                                       index=pd.DatetimeIndex(
                                           np.array(radar_ds.time[:])),
                                       columns=np.array(radar_ds["height"][:]))
    radar["LDR"]=pd.DataFrame(data=np.array(radar_ds["LDRg"].values[:]),
                            index=pd.DatetimeIndex(
                            np.array(radar_ds.time[:])),
                            columns=np.array(radar_ds["height"][:]))
    
    print(radar["Reflectivity"].index)
    radar["Position"]=halo_df.copy()
    del radar_ds
    # Cut dataset to AR core cross-section
    if ar_of_day:
            #radar
            halo_df,radar,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(
                                                ar_of_day, flight[0], 
                                                halo_df,radar,
                                                device="radar")
           
            #radiometer
            halo_df,mwr,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(
                                                ar_of_day, flight[0], 
                                                halo_df,mwr,
                                                device="radiometer")
        
            # Update halo_df in ERA5_on_HALO class with cutted dataset
            ERA5_on_HALO.update_halo_df(halo_df,change_last_index=True)
            if carra_is_desired:
                CARRA_on_HALO.update_halo_df(halo_df,change_last_index=True)
            if cmpgn_cls.name=="HALO_AC3":
                pos_path=hydrometeor_lvls_path+"/../BAHAMAS/"
            else:
                pos_path=cmpgn_cls.campaign_data_path
            radar["Position"].to_csv(path_or_buf=pos_path+\
                             "HALO_Aircraft_"+flight[0]+".csv")
        
        # Load Dropsonde datasets
            Sondes_cls=Instruments.Dropsondes(HALO_cls)
            Sondes_cls.calc_integral_variables(integral_var_list=["IWV","IVT"])
            Dropsondes=Sondes_cls.sonde_dict

                #Dropsondes={}
    else:
        Dropsondes={}
        radar={}
        mwr={}
    
    last_index=len(halo_df.index)
    lat_changed=False
    
    #%% Gridded data (Simulations and Reanalysis)
    #hydrometeor_icon_path=hydrometeor_icon_path+flight[0]+"/"
    #ICON_on_HALO.update_ICON_hydrometeor_data_path(hydrometeor_icon_path)
    #%% Processing, Interpolation onto Flight Path
    # If interpolated data does not exist, load ERA-5 Dataset
    
    # Create HALO interpolated total column data  if not existent, 
    # if HMPs already interpolated onto HALO for given flight, load csv-file.
    if hmp_plotting_desired:
        if not os.path.exists(hydrometeor_lvls_path):
                os.makedirs(hydrometeor_lvls_path)
        print("Path to open: ", hydrometeor_lvls_path)
        print("open hydrometeor_levels")
        
        #----------------- ERA-5 ---------------------------------------------#
        ERA5_on_HALO.update_interpolated_hmp_file(interpolated_hmp_file)
        halo_era5=ERA5_on_HALO.load_hmp(cmpgn_cls)
        halo_era5=halo_era5.groupby(level=0).first()#drop_duplicates(keep="first")
        halo_df=halo_df.groupby(level=0).first()
        if "Interp_IVT" in halo_era5.columns:
            if not "groundspeed" in halo_df.columns:
                if radar!={}:
                    halo_df.index=pd.DatetimeIndex(halo_df.index)
                    halo_df["groundspeed"]=radar["Position"]["groundspeed"].\
                                            loc[halo_df.index]
        halo_era5=cmpgn_cls.calc_distance_to_IVT_max(
                        halo_df,
                        halo_era5)
        ERA5_on_HALO.halo_era5=halo_era5.copy()
        #---------------------------------------------------------------------#
        #if carra_is_desired:
        #    CARRA_on_HALO.load_or_calc_interpolated_hmp_data()
        #    #update_interpolated_hmp_file(self,interpolated_hmp_file):
        #    high_res_hmp=CARRA_on_HALO.halo_carra_hmp.copy()
        #----------------- ICON ----------------------------------------------#
        #if icon_is_desired:
        #    ICON_on_HALO.update_ICON_hydrometeor_data_path(hydrometeor_icon_path)
        #    halo_icon_hmp=ICON_on_HALO.load_interpolated_hmp()
            #high_res_hmp=halo_icon_hmp.copy()
       #----------------------------------------------------------------------# 
    cmpgn_cls.flight=flight
    return halo_era5,halo_df,cmpgn_cls,ERA5_on_HALO,radar,Dropsondes