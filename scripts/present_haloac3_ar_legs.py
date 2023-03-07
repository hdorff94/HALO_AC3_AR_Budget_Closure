# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:54:08 2023

@author: u300737
"""

import os
import glob
import sys

def main(campaign="HALO_AC3",flights=["RF05","RF06"],
         ar_of_days=["AR_internal"],
         do_daily_plots=True,calc_hmp=True,calc_hmc=True,
         era_is_desired=True,carra_is_desired=False,
         icon_is_desired=False,do_instantaneous=False):
    
    #%% Predefining all paths to take scripts and data from and where to store
    actual_working_path=os.getcwd()+"/../../Synthetic_Airborne_Arctic_ARs/"
    os.chdir(actual_working_path+"/config/")

    import init_paths
    import data_config
    working_path=init_paths.main()
    
    airborne_data_importer_path=working_path+"/Work/GIT_Repository/"
    airborne_script_module_path=actual_working_path+"/scripts/"
    airborne_processing_module_path=actual_working_path+"/src/"
    airborne_plotting_module_path=actual_working_path+"/plotting/"
    os.chdir(airborne_processing_module_path)
    sys.path.insert(1,airborne_script_module_path)
    sys.path.insert(2,airborne_processing_module_path)
    sys.path.insert(3,airborne_plotting_module_path)
    sys.path.insert(4,airborne_data_importer_path)
    # %% Load relevant modules from Synthetic_Airborne_Arctic_ARs repository
    import flightcampaign
    import run_grid_data_on_halo
    # Load config file
    config_file=data_config.load_config_file(airborne_data_importer_path,
                                             "data_config_file")
    
    do_plots=do_daily_plots
    if campaign=="HALO_AC3":
        synthetic_campaign=False
        synthetic_flight=False
        cpgn_cls_name="HALO_AC3"
        ac3_run=flightcampaign.HALO_AC3(is_flight_campaign=True,
            major_path=config_file["Data_Paths"]["campaign_path"],
            aircraft="HALO",interested_flights=flights,
            instruments=["radar","radiometer","sonde"])
        cmpgn_cls=ac3_run
    
    HMCs={}
    HMPs={}
    HALO_dict_dict={}
    i=0
    for flight in flights:
        
        HMCs[flight]={}
        HMPs[flight]={}
    
        for ar_of_day in [ar_of_days]:
            ar_of_day=ar_of_day[0]
            print(ar_of_day)
            #sys.exit()
            if not flight.startswith("S"):
                synthetic_campaign=False
            else:
                if cpgn_cls_name=="NAWDEX":
                    print("Campaign name has to be changed")
                    cpgn_cls_name="NA_February_Run"
                if synthetic_campaign==False:
                    synthetic_campaign=True
            if calc_hmp:
                HMPs[flight][ar_of_day],ar_rf_radar,HALO_dict_dict[flight]=\
                        run_grid_data_on_halo.main(
                            config_file_path=airborne_data_importer_path,
                            campaign=cpgn_cls_name,
                            hmp_plotting_desired=calc_hmp,
                            hmc_plotting_desired=calc_hmc,
                            plot_data=do_plots,
                            ar_of_day=ar_of_day,flight=[flight],
                            era_is_desired=era_is_desired,
                            carra_is_desired=carra_is_desired,
                            icon_is_desired=icon_is_desired,
                            synthetic_campaign=synthetic_campaign,
                            synthetic_flight=synthetic_flight,
                            do_instantaneous=do_instantaneous)
                
            if calc_hmc:
                HMCs[flight][ar_of_day],ar_rf_radar,HALO_dict_dict[flight]=\
                        run_grid_data_on_halo.main(
                            config_file_path=airborne_data_importer_path,        
                            campaign=cpgn_cls_name,
                            hmp_plotting_desired=False,
                            hmc_plotting_desired=calc_hmc,
                            plot_data=do_plots,
                            ar_of_day=ar_of_day,flight=[flight],
                            era_is_desired=era_is_desired,
                            carra_is_desired=carra_is_desired,
                            icon_is_desired=icon_is_desired,
                            synthetic_campaign=synthetic_campaign,
                            synthetic_flight=synthetic_flight,
                            do_instantaneous=do_instantaneous)    
            #if 'Reflectivity' in ar_rf_radar.keys():
            #    if i==0:
            #        AR_radar=ar_rf_radar["Reflectivity"]
            #    else:
            #        AR_radar=pd.concat([AR_radar,ar_rf_radar["Reflectivity"]],
            #                       ignore_index=True)
            i+=1
    if calc_hmp:
        print("DATASET NAME:",HMPs[flight][ar_of_day].name)
        return HMPs,HALO_dict_dict,cmpgn_cls
    if calc_hmc:
        print("DATASET NAME:",HMCs[flight][ar_of_day]["name"])
        return HMCs,HALO_dict_dict,cmpgn_cls
###############################################################################











###############################################################################
#%% Main data and plot creator
if __name__=="__main__":
    # This part runs the main funciton meaning all the stuff 
    # from run_grid_on_halo as well as plots and 
    
    # Relevant specifications for running , those are default values
    calc_hmp=True
    calc_hmc=False
    do_plotting=True
    synthetic_campaign=False
    ar_of_day=["AR_entire_2"]#["AR3"]#"AR_entire"#"#internal"#"AR_entire"
    campaign_name="HALO_AC3"#"Second_Synthetic_Study"##"HALO_AC3"
    #campaign_name="North_Atlantic_Run"#"Second_Synthetic_Study"
    
    flights_to_analyse={#"RF02":"20220312",
                            #"RF03":"20220313",
                            #"RF04":"20220314",
                            #"RF05":"20220315",
                            "RF06":"20220316",
                            #"RF07":"20220320"
                            
                            #"RF10":"20161013"
                            }        
    use_era=True
    use_carra=False
    use_icon=False
    flights=[*flights_to_analyse.keys()]
    do_instantaneous=False

    Hydrometeors,HALO_Dict,cmpgn_cls=main(campaign=campaign_name,flights=flights,
                                          ar_of_days=ar_of_day,
                                          era_is_desired=use_era, 
                                          icon_is_desired=use_icon,
                                          carra_is_desired=use_carra,
                                          do_daily_plots=do_plotting,
                                          calc_hmp=calc_hmp,calc_hmc=calc_hmc,
                                          do_instantaneous=do_instantaneous)
    if do_instantaneous:
        import sys
        sys.exit()
