import os
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import solve_ivp as sp_solve_ivp
import xarray
from pathlib import Path
import pandas as pd

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.utils.utils_load_process import Config

# Ref: exphydro -> https://hess.copernicus.org/articles/26/5085/2022/
class ExpHydro(BaseConceptModel):
    
    def __init__(self, 
                 cfg: Config,
                 ds: xarray.Dataset,
                 odesmethod:str ='RK23'
                ):
        super().__init__(cfg, ds, odesmethod)
        
        # Interpolators
        self.interpolators = self.interpolator_dict()
        self.time_series = np.arange(len(ds['date'].values))
        
        # Parameters per basin
        self.params_dict = self.get_parameters()


    def conceptual_model(self, t, y, params):
        
        ## Unpack the parameters
        # State variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        S0 = y[0]
        S1 = y[1]
        
        # Bucket parameters                   
        # f: Rate of decline in flow from catchment bucket   
        # Smax: Maximum storage of the catchment bucket     
        # Qmax: Maximum subsurface flow at full bucket      
        # Df: Thermal degree‐day factor                   
        # Tmax: Temperature above which snow starts melting 
        # Tmin: Temperature below which precipitation is snow
        f, Smax, Qmax, Df, Tmax, Tmin = params
        
        # Interpolate the input variables
        precp = self.precp_interp(t, extrapolate='periodic')
        temp = self.temp_interp(t, extrapolate='periodic')
        lday = self.lday_interp(t, extrapolate='periodic')
        
        # Compute and substitute the 5 mechanistic processes
        Q_out = Qb(S1, f, Smax, Qmax) + Qs(S1, Smax)
        M_out = M(S0, temp, Df, Tmax)
        
        # Eq 1 - Hoge_EtAl_HESS_2022
        dS0_dt = Ps(precp, temp, Tmin) - M_out
        # Eq 2 - Hoge_EtAl_HESS_2022
        dS1_dt = Pr(precp, temp, Tmin) + M_out - ET(S1, temp, lday, Smax) - Q_out
        
        return [dS0_dt, dS1_dt]
           
    def step_function(self, x):
        return  lambda x: (np.tanh(5.0 * x) + 1.0) * 0.5
    
    def interpolator_dict(self):
        
        # Extract timepoints from the dataset
        t_values = self.ds['date'].values
        t_series = np.linspace(0, len(t_values), len(t_values))
        
        # Create a dictionary to store interpolator functions for each basin and variable
        interpolators = dict()
        
        # Loop over the basis and variables
        for basin in self.ds['basin'].values:
    
            interpolators[basin] = dict()
            for var in self.interpolator_vars:
                                
                # Get the variable values
                var_values = self.ds[var].sel(basin=basin).values
                
                # Interpolate the variable values
                interpolators[basin][var] = Akima1DInterpolator(t_series, var_values)
                
        return interpolators
    
    def get_parameters(self):
        
        print(Path(__file__))
        print(Path(__file__).resolve())
        print(Path(__file__).resolve().parent)

        params_dir = Path(__file__).resolve().parent / 'bucket_parameter_files' / f'bucket_{self.cfg.concept_model}.csv'
        try:
            params = pd.read_csv(params_dir)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{params_dir}' not found. Check the file path.")
        else:
            params_dict = dict()
            # Loop over the basins and extract the parameters
            for basin in self.ds['basin'].values:
                print(basin)
        
        
    
    
    def run(self, y_init, params):
        
        # Initial conditions
        S0 = 0
        S1 = 0
        
        # Set the initial conditions
        y0 = np.array([y[0], y[1]])
        
        # Run the model
        y = sp_solve_ivp(self.conceptual_model, t_span=(0, len(self.time_series)), y0=y0, t_eval=self.time_series, 
                         args=params, method=self.odesmethod)
        
        return y
    
    
    @property
    def interpolator_vars(self):
        return ['prcp(mm/day)', 'tmean(c)', 'dayl(s)']
        

## Auxiliary functions
# Qbucket is the runoff generated based on the available stored water in the bucket (unit: mm/day) - Patil_Stieglitz_HR_2012
Qb = lambda S1, f, Smax, Qmax, step_fct: step_fct(S1) * step_fct(S1 - Smax) * Qmax \
                                        + step_fct(S1) * step_fct(Smax - S1) * Qmax * np.exp(-f * (Smax - S1))
                                        
# Qspill is the snowmelt is available to infiltrate into the catchment bucket, but the storage S has reached full capacity Smax.
Qs = lambda S1, Smax, step_fct: step_fct(S1) * step_fct(S1 - Smax) * (S1 - Smax)

# Precipitation as snow (A1) - Hoge_EtAl_HESS_2022
Ps = lambda P, T, Tmin, step_fct: step_fct(Tmin - T) * P

# Precipitation as snow (A2) - Hoge_EtAl_HESS_2022
Pr = lambda P, T, Tmin, step_fct: step_fct(T - Tmin) * P

# Melting (A4) - Hoge_EtAl_HESS_2022
M = lambda T, Tmax, S0, Df, step_fct: step_fct(T - Tmax) * step_fct(S0) * np.minimum(S0, Df * (T - Tmax))

# Evapotranspiration (A3) - Hoge_EtAl_HESS_2022
ET = lambda S1, T, Lday, Smax, step_fct: step_fct(S1) * step_fct(S1 - Smax) * PET(T, Lday) + \
                               step_fct(S1) * step_fct(Smax - S1) * PET(T, Lday) * (S1 / Smax)
                                                 
# Potential evapotranspiration - Hamon’s formula (Hamon, 1963) - Hoge_EtAl_HESS_2022 
PET = lambda T, Lday: 29.8 * Lday * 0.611 * np.exp((17.3 * T) / (T + 237.3)) / (T + 273.2)    