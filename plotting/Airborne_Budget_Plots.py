# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:45:59 2024

@author: u300737
"""

import os
import sys

current_path=os.getcwd()
git_path=current_path+"/../../"
synth_path=git_path+"Synthetic_Airborne_Arctic_ARs//"

sys.path.insert(1,synth_path+"/src/")
from moisturebudget import Moisture_Budget, Moisture_Convergence, Moisture_Budget_Plots

class HALO_AC3_Budget_Plots(Moisture_Budget_Plots):
    """
    This is the major plotting class for all HALO-(AC)3 moisture budget 
    components. It is mainly designed for the second manuscript of the PhD of 
    Henning Dorff. This study determines all moisture budget components for
    an AR event and assesses the budget equation closure.
    
    """
    def __init__(self,cmpgn_cls,flight,config_file,
                 grid_name="ERA5",do_instantan=False,sonde_no=3,
                 scalar_based_div=True):
        
        super().__init__(cmpgn_cls,flight,config_file,
                         grid_name,do_instantan)
        self.plot_path=self.cmpgn_cls.plot_path+"/budget/" # ----> to be filled
        self.grid_name=grid_name
        self.sonde_no=sonde_no
        self.scalar_based_div=scalar_based_div
    


