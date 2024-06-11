import os
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from scipy.integrate import solve_ivp as sp_solve_ivp
from scipy.integrate import odeint
import xarray
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.utils.load_process_data import (
    Config,
    ExpHydroCommon,
)

# Ref: exphydro -> https://hess.copernicus.org/articles/26/5085/2022/
class ExpHydro(BaseConceptModel, ExpHydroCommon):
    
    def __init__(self, 
                 cfg: Config,
                 ds: xarray.Dataset,
                 scaler: dict,
                 odesmethod:str ='RK23'
                ):
        super().__init__(cfg, ds, scaler, odesmethod)
        
        # Interpolators
        self.interpolators = self.create_interpolator_dict()
        
        # Input variables
        self.precp = ds['prcp']
        self.temp = ds['tmean']
        self.lday = ds['dayl']
        
        # Parameters per basin
        self.params_dict = self.get_parameters()
        
    def conceptual_model(self, t, y, f, smax, qmax, df, tmax, tmin):
        
        # Bucket parameters                   
        # f: Rate of decline in flow from catchment bucket   
        # Smax: Maximum storage of the catchment bucket     
        # Qmax: Maximum subsurface flow at full bucket      
        # Df: Thermal degree‐day factor                   
        # Tmax: Temperature above which snow starts melting 
        # Tmin: Temperature below which precipitation is snow
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        s0 = y[0]
        s1 = y[1]

        # Interpolate the input variables
        precp = self.precp_interp(t, extrapolate='periodic')
        temp = self.temp_interp(t, extrapolate='periodic')
        lday = self.lday_interp(t, extrapolate='periodic')

        # Compute and substitute the 5 mechanistic processes
        q_out = Qb(s1, f, smax, qmax, self.step_function) + Qs(s1, smax, self.step_function)
        m_out = M(s0, temp, df, tmax, self.step_function)
        
        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = Ps(precp, temp, tmin, self.step_function) - m_out
        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = Pr(precp, temp, tmin, self.step_function) + m_out - ET(s1, temp, lday, smax, self.step_function) - q_out

        return [ds0_dt, ds1_dt]
           
    def step_function(self, x):
        '''
        Step function to be used in the model.
        
        - Args:
            x: float, input value.
            
        - Returns:
            array_like, step function applied to input value(s).
        '''
        return (np.tanh(5.0 * x) + 1.0) * 0.5
        
    def run(self, basin):
        
        basin_params = self.params_dict[basin]
        
        # Get the interpolator functions for the basin
        self.precp_interp = self.interpolators[basin]['prcp']
        self.temp_interp = self.interpolators[basin]['tmean']
        self.lday_interp = self.interpolators[basin]['dayl']
        
        # Get input variables for the basin
        self.precp_basin = self.precp.sel(basin=basin).values
        self.temp_basin = self.temp.sel(basin=basin).values
        self.lday_basin = self.lday.sel(basin=basin).values
        
        # Set the initial conditions
        y0 = np.array([basin_params[0], basin_params[1]])
        
        # Set the parameters
        params = tuple(basin_params[2:])

        # Run the model
        y = sp_solve_ivp(self.conceptual_model, t_span=(0, self.precp.shape[1] - 1), y0=y0, t_eval=self.time_series, 
                         args=params, 
                         method=self.odesmethod,
                        #  method='DOP853',
                        # rtol=1e-9, atol=1e-12,
                    )
        
        # Extract snow and water series -> two state variables representing buckets for snow and water Hoge_EtAl_HESS_2022
        s_snow = y.y[0]
        s_water = y.y[1]
        s_snow = np.maximum(s_snow, 0)
        s_water = np.maximum(s_water, 0)

        # Unpack the parameters to call the mechanistic processes
        f, smax, qmax, df, tmax, tmin = params
        
        # Calculate the mechanistic processes -> q_bucket, et_bucket, m_bucket, ps_bucket, pr_bucket
        # Discharge
        q_bucket = Qb(s_water, f, smax, qmax, self.step_function) + Qs(s_water, smax, self.step_function)
        q_bucket = np.maximum(q_bucket, 0)
        # Evapotranspiration
        et_bucket = ET(s_water, self.temp_basin, self.lday_basin, smax, self.step_function)
        et_bucket = np.maximum(et_bucket, 0)
        # Melting
        m_bucket = M(s_snow, self.temp_basin, df, tmax, self.step_function)
        m_bucket = np.maximum(m_bucket, 0)
        # Precipitation as snow
        ps_bucket = Ps(self.precp_basin, self.temp_basin, tmin, self.step_function)
        ps_bucket = np.maximum(ps_bucket, 0)
        # Precipitation as rain
        pr_bucket = Pr(self.precp_basin, self.temp_basin, tmin, self.step_function)
        pr_bucket = np.maximum(pr_bucket, 0)
        
        # Mind this order for 'save_results' method - the last element is the target variable (q_obs)
        return s_snow, s_water, et_bucket, m_bucket, ps_bucket, pr_bucket, q_bucket
    
    def save_results(self, ds, results, basin, period='train'):
        '''
        Save the model results to a CSV file.
        
        - Args:
            ds: xarray.Dataset, dataset with the input data.
            results: tuple, tuple with the model results.
            basin: str, basin name.
            period: str, period of the run ('train', 'test', 'valid').
            
        '''
        
        results_dict = {
            'date': ds['date'],
            's_snow': results[0],
            's_water': results[1],
            'et_bucket': results[2],
            'm_bucket': results[3],
            'ps_bucket': results[4],
            'pr_bucket': results[5],
            'prcp(mm/day)': ds['prcp'],
            'tmean(c)': ds['tmean'],
            'dayl(s)': ds['dayl'],     
            # the last element is the target variable (q_obs)       
            'q_bucket': results[-1],
            'q_obs': ds['obs_runoff'],
        }
        
        # Create a DataFrame from the results dictionary
        results_df = pd.DataFrame(results_dict)
        
        # Save the results to a CSV file
        results_file = os.path.join(self.cfg.results_dir, f'{basin}_results_{period}.csv')
        results_df.to_csv(results_file, index=False)
    
    def shift_initial_states(self, start_and_end_dates, basin, period):

        # Identify the previous period given the dates in 
        if period == 'test' or period == 'valid':
            
            # Find the previous period
            period_dates = start_and_end_dates[period]
            current_start_date = period_dates['start_date']
            previous_end_date = None
            
            # Loop to find the previous end date
            for name, dates in start_and_end_dates.items():
                if name != period:
                    if dates['end_date'] < current_start_date:
                        if previous_end_date is None or dates['end_date'] > previous_end_date:
                            previous_end_date = dates['end_date']
                  
            # Find the previous period name          
            if previous_end_date is not None:
                for name, dates in start_and_end_dates.items():
                    if dates['end_date'] == previous_end_date:
                        previous_period = name
                        break
                    
            # Load the last states of the model in the previous period
            previous_results_file = os.path.join(self.cfg.results_dir, f'{basin}_results_{previous_period}.csv')
            previous_results = pd.read_csv(previous_results_file)
            
            # Get the last states of the model in the previous period
            previous_states = previous_results[['s_snow', 's_water']].values[-1]
            
            # Update the initial states for the model
            self.params_dict[basin][0] = previous_states[0]
            self.params_dict[basin][1] = previous_states[1]

    @property
    def nn_outputs(self):
        return ['ps_bucket', 'pr_bucket', 'm_bucket', 'et_bucket', 'q_bucket']
    
    @property
    def model_outputs(self):
        return ['s_snow', 's_water']

        
## Auxiliary functions
# Qbucket is the runoff generated based on the available stored water in the bucket (unit: mm/day) - Patil_Stieglitz_HR_2012
# Qb = lambda s1, f, smax, qmax, step_fct: step_fct(s1) * step_fct(s1 - smax) * qmax \
#                                         + step_fct(s1) * step_fct(smax - s1) * qmax * np.exp(-f * (smax - s1))
Qb = lambda s1, f, smax, qmax, step_fct: step_fct(s1) * step_fct(s1 - smax) * qmax \
                                        + step_fct(s1) * step_fct(smax - s1) * qmax * np.exp(-f * (smax - s1))
                                        
# Qspill is the snowmelt is available to infiltrate into the catchment bucket, but the storage S has reached full capacity smax.
Qs = lambda s1, smax, step_fct: step_fct(s1) * step_fct(s1 - smax) * (s1 - smax)

# Precipitation as snow (A1) - Hoge_EtAl_HESS_2022
Ps = lambda p, temp, tmin, step_fct: step_fct(tmin - temp) * p

# Precipitation as snow (A2) - Hoge_EtAl_HESS_2022
Pr = lambda p, temp, tmin, step_fct: step_fct(temp - tmin) * p

# Melting (A4) - Hoge_EtAl_HESS_2022
# M(S0, T, Df, Tmax) = step_fct(T-Tmax)*step_fct(S0)*minimum([S0, Df*(T-Tmax)])
# M = lambda temp, tmax, s0, df, step_fct: step_fct(temp - tmax) * step_fct(s0) * np.minimum(s0, df * (temp - tmax))
def M(s0, temp, df, tmax, step_fct):
    
    if np.isscalar(temp) and np.isscalar(s0):
        if temp > tmax and s0 > 0:
            return  step_fct(np.minimum(s0, df * (temp - tmax)))
        else:
            return 0.0
    else:
        # If temp and s0 are arrays, perform element-wise calculation
        result = np.zeros_like(temp)  # Initialize result array with zeros
        
        # Check condition for each element of the arrays
        condition = np.logical_and(temp > tmax, s0 > 0)

        # Calculate result for elements satisfying the condition
        result[condition] = np.minimum(s0[condition], df * (temp[condition] - tmax))

        return result

# Evapotranspiration (A3) - Hoge_EtAl_HESS_2022
ET = lambda s1, temp, lday, smax, step_fct: step_fct(s1) * step_fct(s1 - smax) * PET(temp, lday)  \
                            + step_fct(s1) * step_fct(smax - s1) * PET(temp, lday) * (s1 / smax)
                                                 
# Potential evapotranspiration - Hamon’s formula (Hamon, 1963) - Hoge_EtAl_HESS_2022 
PET = lambda temp, lday: 29.8 * lday * 0.611 * np.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)    