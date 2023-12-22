# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:52:57 2023

@author: u300737
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

class ICON_NWP_HALO_AC3():
    
    def __init__(self,flight,campaign,ar_sector,add_hydrometeors=False,do_open=False):
        self.flight=flight
        self.campaign=campaign
        self.ar_sector=ar_sector
        self.add_hydrometeors=add_hydrometeors
        self.project_path="/work/bb1086/"
        mistral_path=self.project_path+"from_Mistral/"
        self.height_limit=40
        self.flight_dates={"HALO_AC3":{"RF02":"2022-03-12",
                                     "RF03":"2022-03-13",
                                     "RF04":"2022-03-14",
                                     "RF05":"2022-03-15",
                                     "RF06":"2022-03-16",
                                     "RF07":"2022-03-20",
                                     "RF08":"2022-03-21", 
                                     "RF16":"2022-04-10"},
                         "NAWDEX"   :{} }
        self.hydrometeor_paths=["tqi_dia","tqv_dia","tqc_dia","tot_prec","qhfl_s","hourly_prec"]
        self.vertical_vars_name={"pres":"Pressure","u":"U_wind","v":"V_Wind",
                                 "z_mc":"Z_Height","qv":"Specific_Humidity"}
        self.vertical_hydrometeors={"qc":"Cloud_Content","qg":"Graupel_Content","qi":"Ice_Content","qr":"Rain_Content","qs":"Snow_Content"}
        self.flight_path=os.getcwd()+"/"+self.flight+"/"
        if not os.path.exists(self.flight_path):
            os.makedirs(self.flight_path)
            
        if self.campaign=="NAWDEX":
            self.icon_main_path  = mistral_path+"experiment/nawdex_"+self.flight.lower()+"_2km_nwpturb/"
        elif self.campaign=="HALO_AC3":
            self.icon_file_start="cloud_DOM01_ML_"#20220312T000000Z
            
            #if int(next_hour)<12:
            #    self.icon_file_end   = "060000Z.nc"
            #elif int(next_hour)<18:
            #    self.icon_file_end   = "120000Z.nc"
            #else:
            #    self.icon_file_end   = "180000Z.nc"
                
            if (self.flight!="RF02"):# or not self.flight=="RF05":
                self.icon_main_path=self.project_path+"haloac3/icon_nwp/"+\
                        self.flight_dates[self.campaign][self.flight]+"/"#+"_2km_nwpturb/"
                if self.flight=="RF05":
                    self.icon_main_path=self.project_path+"haloac3/icon_nwp/"+\
                        self.flight_dates[self.campaign][self.flight]+"/"#+"_vs/"#+"_2km_nwpturb/"
                if self.flight=="RF06":
                    self.icon_main_path=self.project_path+"haloac3/icon_nwp/"+\
                        "2022-03-15/"
                    #icon_file_end="120000Z.nc"
        else:
            self.icon_main_path=mistral_path+"/bb1086/haloac3/icon_nwp/"+self.flight_dates[self.campaign][self.flight]+"/"
        self.icon_date   = "".join(self.flight_dates[self.campaign][self.flight].split("-"))
        self.icon_file_end   = "0000Z.nc"
        
        #if not self.flight=="RF06":
        #self.icon_file_1 = self.icon_file_start+self.icon_date+"T"+self.icon_file_end
        #else:
        #    icon_file_1 = icon_file_start+"20220315T"+icon_file_end
        #print("ICON file ", self.icon_file_1)
        self.icon_file_start+self.icon_date+"T"+self.icon_file_end
        # precipi
        relevant_hours=["00","06","12","18"]
        relevant_hours_files=[self.icon_main_path+\
                              self.icon_file_start+self.icon_date+"T"+hour+self.icon_file_end for hour in relevant_hours]
        if do_open:
            merged_icon_ds=xr.open_mfdataset(relevant_hours_files,combine="nested",concat_dim="time")
            merged_icon_ds["prec_hahourly"]=merged_icon_ds["tot_prec"].diff(dim="time")
            merged_icon_ds=merged_icon_ds.assign_coords({"time":
                        pd.to_datetime(abs(int(self.icon_date)-np.array(merged_icon_ds.time)),unit="d",
                                      origin=self.flight_dates[self.campaign][self.flight]).round("min")})
        
            half_hourly_precip=merged_icon_ds["prec_hahourly"].compute()
            self.hourly_precip=half_hourly_precip*2
        ##var*=2
        ##var.max()
        #var
        #half_hourly_precip

            
    def open_icon_file(self,next_hour):
        #icon_file_start = "3D_10m_cl_DOM01_ML_"
        #icon_file_end   = "0000Z.nc"
        mistral_path=self.project_path+"from_Mistral/"
        str_date=self.flight_dates[self.campaign][self.flight]    
        if self.campaign=="NAWDEX":
            self.icon_main_path  = mistral_path+"experiment/nawdex_"+self.flight.lower()+"_2km_nwpturb/"
        elif self.campaign=="HALO_AC3":
            icon_file_start="cloud_DOM01_ML_"#20220312T000000Z
            
            if int(next_hour)<12:
                self.icon_file_end   = "060000Z.nc"
            elif int(next_hour)<18:
                self.icon_file_end   = "120000Z.nc"
            else:
                self.icon_file_end   = "180000Z.nc"
                
            #if (self.flight!="RF02"):# or not self.flight=="RF05":
            #    icon_main_path=self.project_path+"haloac3/icon_nwp/"+\
            #            self.flight_dates[self.campaign][self.flight]+"/"#+"_2km_nwpturb/"
            #    if self.flight=="RF05":
            #        icon_main_path=self.project_path+"haloac3/icon_nwp/"+\
            #            self.flight_dates[self.campaign][self.flight]+"/"#+"_vs/"#+"_2km_nwpturb/"
            #    if self.flight=="RF06":
            #        icon_main_path=self.project_path+"haloac3/icon_nwp/"+\
            #            "2022-03-15/"
             #       #icon_file_end="120000Z.nc"
            #else:
            #    icon_main_path=mistral_path+"/bb1086/haloac3/icon_nwp/"+self.flight_dates[self.campaign][self.flight]+"/"
            self.icon_date   = "".join(self.flight_dates[self.campaign][self.flight].split("-"))
            #if not self.flight=="RF06":
            icon_file_1 = self.icon_file_start+self.icon_date+"T"+self.icon_file_end
            #else:
            #    icon_file_1 = icon_file_start+"20220315T"+icon_file_end
            #print("ICON file ", icon_file_1)
            
        elif self.campaign=="NA_February_Run":
            icon_file_start="forcing_DOM01_ML_20190319T120000Z.nc"
            self.icon_main_path=self.project_path+"aflux/aflux_nwp_20190319/"
            icon_file_1=icon_file_start#+RFs["Date"][interested_flight]+"T"+next_hour+icon_file_end
            
            if not os.path.exists(self.icon_main_path+icon_file_1):
                raise Exception("This file does not exist")
        
        self.icon_ds=xr.open_dataset(self.icon_main_path+icon_file_1)
        if not self.flight=="RF06":
            self.icon_ds=self.icon_ds.assign_coords({"time":
                        pd.to_datetime(abs(int(self.icon_date)-np.array(self.icon_ds.time)),unit="d",
                                      origin=self.icon_date).round("min")})
        else:
            self.icon_ds=self.icon_ds.assign_coords({"time":
                        pd.to_datetime(abs(int(self.icon_date)-np.array(self.icon_ds.time)),unit="d",
                                      origin=self.icon_date).round("min")})
        print(self.icon_ds.time)
        self.str_date=str_date
        self.icon_ds["hourly_prec"]=self.hourly_precip.sel({"time":slice(self.icon_ds["time"][0],
                                    self.icon_ds["time"][-1])})

        return self.icon_ds.sel(time=slice(str_date+" "+next_hour+":00",
                                                 str_date+" "+str(int(next_hour)+1)))
        
    def get_indexes_for_given_area(self,lat_range,lon_range):
        """
        Find the index to consider for desired spatial domain
    
        Input
        -----
        ds        : xr.Dataset
            Icon Simulation Dataset
        lat_range : list
            list of shape two, including lower and upper latitude boundary
        lon_range : list
            list of shape two, including lower and upper longitude boundary
        """
        self.lat_range=lat_range
        self.lon_range=lon_range
        clon_s=pd.Series(np.rad2deg(self.icon_ds.ncells.clon))
        clat_s=pd.Series(np.rad2deg(self.icon_ds.ncells.clat))

        # Cut to defined lon domain
        clon_cutted=clon_s.loc[clon_s.between(lon_range[0],lon_range[1])]
        # adapt this to the lat domain
        clat_s=clat_s.loc[clon_cutted.index]
        # Cut to defined lat domain
        clat_cutted=clat_s.loc[clat_s.between(lat_range[0],lat_range[1])]
        # Finally cut lon to this array
        clon_cutted=clon_cutted.loc[clat_cutted.index]
        print(clon_cutted.shape,
              clat_cutted.shape)
        if not clon_cutted.index.all()==clat_cutted.index.all():
            raise Exception("The indexes are not equivalent so something went wrong and",
                            "no index list can be returned ")
        self.domain_index=clon_cutted.index
    
    def open_campaign_flight_icon_data(self):      
        self.flight_ds={}
        if self.campaign=="NAWDEX":
            if self.flight=="RF10": 
                self.flight_ds["09"] = self.open_icon_file("09")
                self.flight_ds["10"] = self.open_icon_file("10")
                ### --> to be corrected
                #ds11 = open_icon_file("11",campaign)
                #ds12 = open_icon_file("12",campaign)
                #ds13 = open_icon_file("13",campaign)
                #ds14 = open_icon_file("14",campaign)
                #ds15 = open_icon_file(interested_flight,"15",campaign)
                #ds16 = open_icon_file(interested_flight,"16",campaign)

            #elif self.flight=="RF03":
                #pass
                # --to be corrected 
                #ds09 = open_icon_file(interested_flight,"09",campaign)
             #   self.flight_ds["10"] = open_icon_file(interested_flight,"10",campaign)
                #ds13 = open_icon_file(interested_flight,"13",campaign)
                #ds14 = open_icon_file(interested_flight,"14",campaign)
                #ds15 = open_icon_file(interested_flight,"15",campaign)
                #ds16 = open_icon_file(interested_flight,"16",campaign)
        elif self.campaign=="HALO_AC3":
            if self.flight=="RF02":
                #ARs["RF02"]["AR1"]={"start":"2022-03-12 10:25",
                #                "end":"2022-03-12 12:10"}
                
                #ARs["RF02"]["AR2"]={"start":"2022-03-12 11:30",
                #                "end":"2022-03-13 13:35"}
                #ARs["RF03"]["AR1"]={"start":"2022-03-13 10:00", # temporary
                #                "end":"2022-03-13 11:00"}   # temporary
                if self.ar_sector=="AR1":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                            " 10:25:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                            " 12:10:00"
                    print("sector to analyse",self.ar_sector)
                elif self.ar_sector=="AR2":
                    
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.flight_ds["14"] = self.open_icon_file("14")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                            " 11:28:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                            " 13:32:00"
                
            elif self.flight=="RF03":
                if self.ar_sector=="AR1" or self.ar_sector=="AR_entire_1":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.sector_start=\
                        self.flight_dates[self.campaign][self.flight]+" 10:00:00"
                    self.sector_end=\
                        self.flight_dates[self.campaign][self.flight]+" 11:45:00"
                #self.flight_ds["11"] = self.open_icon_file("11")
                #self.flight_ds["12"] = self.open_icon_file("12")
                #self.flight_ds["13"] = self.open_icon_file("13")
            elif self.flight=="RF04":
                self.flight_ds["16"] = self.open_icon_file("16")
                self.flight_ds["17"] = self.open_icon_file("17")
                self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 16:00:00"
                self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 16:45:00"
                    
            elif self.flight=="RF05":
                if self.ar_sector=="AR_entire_1":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 10:11:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 13:15:00"
                elif self.ar_sector=="AR_entire_2":
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.flight_ds["14"] = self.open_icon_file("14")
                    self.flight_ds["15"] = self.open_icon_file("15")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 12:20:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 15:25:00"
                elif self.ar_sector=="AR1":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    #self.flight_ds["12"] = self.open_icon_file("12")
                    #self.flight_ds["13"] = self.open_icon_file("13")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 10:11:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 11:08:00"
                elif self.ar_sector=="AR2":
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    #self.flight_ds["12"] = self.open_icon_file("12")
                    #self.flight_ds["13"] = self.open_icon_file("13")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 11:13:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 12:14:00"
                elif self.ar_sector=="AR3":
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 12:20:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 13:15:00"
                elif self.ar_sector=="AR4":
                    self.flight_ds["14"] = self.open_icon_file("14")
                    self.flight_ds["15"] = self.open_icon_file("15")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 14:24:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 15:25:00"
            elif self.flight=="RF06":
                if self.ar_sector=="AR1" or self.ar_sector=="AR_entire_1":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 10:45:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 12:52:00"
                if self.ar_sector=="AR2" or self.ar_sector=="AR_entire_2":
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.flight_ds["14"] = self.open_icon_file("14")
                    self.flight_ds["15"] = self.open_icon_file("15")
                    
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 12:12:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 14:18:00"
                    
            elif self.flight=="RF16":
                if self.ar_sector=="AR1" or self.ar_sector=="AR_entire_1":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 10:30:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 12:22:00"
                elif self.ar_sector=="AR2" or self.ar_sector=="AR_entire_2":
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.flight_ds["12"] = self.open_icon_file("12")
                    self.flight_ds["13"] = self.open_icon_file("13")
                    self.flight_ds["14"] = self.open_icon_file("14")
                    
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 11:45:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 13:45:00"
                
            elif self.flight=="RF07":
                if self.ar_sector=="AR1":
                    self.flight_ds["14"] = self.open_icon_file("14")
                    self.flight_ds["15"] = self.open_icon_file("15")
                    self.flight_ds["16"] = self.open_icon_file("16")
                    self.flight_ds["17"] = self.open_icon_file("17")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 15:22:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 16:24:00"
            elif self.flight=="RF08":
                if self.ar_sector=="AR1":
                    self.flight_ds["09"] = self.open_icon_file("09")
                    self.flight_ds["10"] = self.open_icon_file("10")
                    self.flight_ds["11"] = self.open_icon_file("11")
                    self.sector_start=self.flight_dates[self.campaign][self.flight]+\
                                    " 09:20:00"
                    self.sector_end=self.flight_dates[self.campaign][self.flight]+\
                                    " 10:25:00"
                    
    def define_flight_leg_specific_subdomain(self):
        if self.campaign=="HALO_AC3":
            if self.flight=="RF02":
                    
                if self.ar_sector=="AR1":
                    lat_range=[74,78]
                    lon_range=[-12,12]
                elif self.ar_sector=="AR2":
                    lat_range=[76.5,82.5]
                    lon_range=[-12,20]
            elif self.flight=="RF03":
                if self.ar_sector=="AR1" or self.ar_sector=="AR_entire_1":
                    lat_range=[76.5,81.5]
                    lon_range=[-12.5,12]
            elif self.flight=="RF04":
                if self.ar_sector=="AR1":
                    lat_range=[65,80]
                    lon_range=[10,40]
            elif self.flight=="RF05":
                if self.ar_sector=="AR1" or self.ar_sector=="AR2" or self.ar_sector=="AR3" \
                  or self.ar_sector=="AR4" or self.ar_sector=="AR_entire_1" or\
                    self.ar_sector=="AR_entire_2":
                    lat_range=[70,80]
                    lon_range=[-30,20]
            elif self.flight=="RF06":
                if self.ar_sector=="AR1" or self.ar_sector=="AR_entire_1":
                    lat_range=[70,75]
                    lon_range=[0,25]
                elif self.ar_sector=="AR2" or self.ar_sector=="AR_entire_2":
                    lat_range=[72,78]
                    lon_range=[0,30]
            elif self.flight=="RF07":
                    lat_range=[70,77]
                    lon_range=[-20,20]
            elif self.flight=="RF08":
                    lat_range=[70,80]
                    lon_range=[10,30]
            elif self.flight=="RF16":
                if self.ar_sector=="AR1" or self.ar_sector=="AR_entire_1":
                    lat_range=[71,77]
                    lon_range=[10,25]
                elif self.ar_sector=="AR2" or self.ar_sector=="AR_entire_2":
                    lat_range=[70,77]
                    lon_range=[2,20]
        self.get_indexes_for_given_area(lat_range,lon_range)
        
    def get_first_hour_data(self):
        first_hour=next(iter(self.flight_ds))
        print("First hour is", first_hour)
        self.ds_first_hour=self.flight_ds[first_hour]               
        #print(self.ds_first_hour["tqv_dia"].shape)
    
    def plot_subregion_iwv(self):
        if self.campaign=="NA_February_Run":
            iwv_arg="prw"
        elif self.campaign=="HALO_AC3":
            iwv_arg=["tqv_dia"]
        self.open_campaign_flight_icon_data()
        self.define_flight_leg_specific_subdomain()
        self.get_first_hour_data()
        located_first_hour=self.ds_first_hour.isel({"time":1,
                                "ncells":self.domain_index})
        iwv=located_first_hour[iwv_arg].to_array()
        plt.figure(figsize=(12,9))
        self.open_aircraft_pos()
        plt.plot(self.bahamas_df["lon"],self.bahamas_df["lat"],color="darkred")
        plt.scatter(np.rad2deg(iwv[:].clon),np.rad2deg(iwv[:].clat),
                    c=iwv[:], vmin=0, vmax=30, s=5, cmap="BuPu")
    
    def plot_ar_sector(self,):
        if self.campaign=="HALO_AC3":
            iwv_arg=["tqv_dia"]
        if not hasattr(self,"ds_first_hour"):
            self.open_campaign_flight_icon_data()
            self.define_flight_leg_specific_subdomain()
            self.get_first_hour_data()
        located_first_hour=self.ds_first_hour.isel({"time":1,
                                "ncells":self.domain_index})
        iwv=located_first_hour[iwv_arg].to_array()
        fig=plt.figure(figsize=(12,9))
        self.open_aircraft_pos()
        ax1=fig.add_subplot(111)
        ax1.plot(self.bahamas_df["lon"],self.bahamas_df["lat"],
                 color="grey")
        ax1.plot(self.bahamas_df["lon"].loc[self.sector_start:self.sector_end],
                 self.bahamas_df["lat"].loc[self.sector_start:self.sector_end],
                 color="darkred")
        
        ax1.set_xlim(self.lon_range[0],self.lon_range[1])
        ax1.set_ylim(self.lat_range[0],self.lat_range[1])
        ax1.scatter(np.rad2deg(iwv[:].clon),np.rad2deg(iwv[:].clat),
                    c=iwv[:], vmin=0, vmax=30, s=5, cmap="BuPu")
    
    def plot_ar_variable(self,var="tot_prec"):
        if not hasattr(self,"ds_first_hour"):
            self.open_campaign_flight_icon_data()
            self.define_flight_leg_specific_subdomain()
            self.get_first_hour_data()
        located_first_hour=self.ds_first_hour.isel({"time":1,
                                "ncells":self.domain_index})
        var=located_first_hour[var]
        fig=plt.figure(figsize=(12,9))
        self.open_aircraft_pos()
        ax1=fig.add_subplot(111)
        ax1.plot(self.bahamas_df["lon"],self.bahamas_df["lat"],
                 color="grey")
        ax1.plot(self.bahamas_df["lon"].loc[self.sector_start:self.sector_end],
                 self.bahamas_df["lat"].loc[self.sector_start:self.sector_end],
                 color="darkred")
        
        ax1.set_xlim(self.lon_range[0],self.lon_range[1])
        ax1.set_ylim(self.lat_range[0],self.lat_range[1])
        var_image=ax1.scatter(np.rad2deg(var[:].clon),np.rad2deg(var[:].clat),
                    c=var[:], vmin=0, vmax=1, s=5, cmap="YlGnBu")
        cax=fig.add_axes([0.9,0.5,0.1,0.15])
        C1=fig.colorbar(var_image,cax=cax,orientation="vertical")
        C1.set_label(label="Total precip / kg$\mathrm{m}^{-2}$")
    def open_aircraft_pos(self):
        aircraft_path=self.project_path+"haloac3_unified_hamp/" 
        bahamas_file="bahamas_"+self.icon_date+"_v0.6.nc"
        bahamas_ds=xr.open_dataset(aircraft_path+bahamas_file)
        self.bahamas_df=bahamas_ds[["alt","lat","lon"]].to_dataframe()
    
    def save_hydrometeor_paths(self):
        for hour in self.flight_ds.keys():
            hydrometeor_paths=self.flight_ds[hour][["tqv_dia","tqi_dia","tqc_dia","tot_prec","hourly_prec","qhfl_s"]]
            hydrometeor_paths.to_netcdf(path=self.flight_path+"/"+"Hydrometeor_ICON_"+\
                                        self.flight+"_"+self.ar_sector+"_"+\
                                        hour+"UTC.nc", mode='w', format="NETCDF3_CLASSIC")
    def save_vertical_variables(self):
        nc_compression=dict(zlib=True,complevel=9,dtype=np.float32)
        if self.add_hydrometeors:
            self.vertical_vars_name = self.vertical_vars_name | self.vertical_hydrometeors
            #self.vertical_vars_name+=self.vertical_hydrometeors
        for hour in [*self.flight_ds.keys()]:
            for vertical_var in [*self.vertical_vars_name.keys()]:
                vertical_data=self.flight_ds[hour][vertical_var]
                if not vertical_var=="z_mc":
                    vertical_data=vertical_data[:,self.height_limit:,:]
                    vertical_data=vertical_data.rolling(height=3).mean()[:,::2,:]
                else:
                    vertical_data=vertical_data[self.height_limit::2,:]
                vertical_data=vertical_data.to_dataset()
                nc_encoding={var:nc_compression for var in vertical_data.variables}
                file_name=self.vertical_vars_name[vertical_var]+\
                            "_"+self.flight+"_"+self.ar_sector+\
                                "_"+hour+"UTC.nc"
                vertical_data.to_netcdf(path=self.flight_path+"/"+\
                                        file_name,mode='w', 
                                            engine="netcdf4",
                                                format="NETCDF4_CLASSIC",
                                                  encoding=nc_encoding)
                print("Vertical var saved as:", self.flight_path+"/"+file_name) 
    
    def locate_and_save_hydrometeor_paths(self):
        if not self.flight_ds:
            self.open_campaign_flight_icon_data()
            self.define_flight_leg_specific_subdomain()
        else:
            pass
        self.save_hydrometeor_paths()