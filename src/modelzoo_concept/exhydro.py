import os
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import solve_ivp as sp_solve_ivp
import xarray
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

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
        self.interpolators = self.create_interpolator_dict()
        self.time_series = np.arange(len(ds['date'].values))
        
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
    
    def create_interpolator_dict(self):
        '''
        Create interpolator functions for the input variables.
        
        - Returns:
            interpolators: dict, dictionary with the interpolator functions for each basin and variable.
        '''
        
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
        '''
        Get the parameters for the model from the parameter file.
        
        - Returns:
            params_dict: dict, dictionary with the parameters for each basin.
            
        '''
        
        params_dir = Path(__file__).resolve().parent / 'bucket_parameter_files' / f'bucket_{self.cfg.concept_model}.csv'
        try:
            params_df = pd.read_csv(params_dir)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{params_dir}' not found. Check the file path.")
        else:
            
            # Remove UNKNOWN column if it exists
            if 'UNKNOWN' in params_df.columns:
                params_df = params_df.drop(columns=['UNKNOWN'])
                
            # Make basinID to be integer if it is not
            if params_df['basinID'].dtype == 'float':
                params_df['basinID'] = params_df['basinID'].astype(int)
                
            params_dict = dict()
            # Loop over the basins and extract the parameters
            for basin in self.ds['basin'].values:
                
                # Convert basin to int to match the parameter file
                basin_int = int(basin)
                    
                try:
                    params_opt = params_df[params_df['basinID'] == basin_int].values[0]
                except IndexError:
                    # Raise warning but continue
                    # raise ValueError(f"Basin {basin} not found in the parameter file.")
                    print(f"Warning! (Data): Basin {basin} not found in the parameter file.")
                    
                # S0,S1,f,Smax,Qmax,Df,Tmax,Tmin
                params_dict[basin] = params_opt[1:]
                
                
            return params_dict
        
    def run(self, basin):
        
        basin_params = self.params_dict[basin]
        
        # Get the interpolator functions for the basin
        self.precp_interp = self.interpolators[basin]['prcp(mm/day)']
        self.temp_interp = self.interpolators[basin]['tmean(c)']
        self.lday_interp = self.interpolators[basin]['dayl(s)']

        # Set the initial conditions
        y0 = np.array([basin_params[0], basin_params[1]])
        
        # Set the parameters
        params = tuple(basin_params[2:])
        
        # print('Initial conditions:', y0)
        # print('Parameters:', params)
        # aux = input('Press any key to continue...')
        
        # print('len(precp_series)', len(self.time_series))
        # print('len(time_series)', len(self.time_series))
        # print('S_ic', y0)
        # print('self.time_series', self.time_series)
        # aux = input('Press Enter to continue...')
        
        # Run the model
        y = sp_solve_ivp(self.conceptual_model, t_span=(0, len(self.time_series) - 1), y0=y0, t_eval=self.time_series, 
                         args=params, method=self.odesmethod)
        
        # Extract snow and water series ->  two state variables representing buckets for snow and water Hoge_EtAl_HESS_2022
        s_snow = y.y[0]
        s_water = y.y[1]
        
        # Unpack the parameters to call the mechanistic processes
        f, smax, qmax, df, tmax, tmin = params
        
        # Calculate the mechanistic processes -> q_bucket, et_bucket, m_bucket, ps_bucket, pr_bucket
        q_bucket = Qb(s_water, f, smax, qmax, self.step_function) + Qs(s_water, smax, self.step_function)
        et_bucket = ET(s_water, self.temp_interp(self.time_series), self.lday_interp(self.time_series), smax, self.step_function)
        m_bucket = M(s_snow, self.temp_interp(self.time_series), df, tmax, self.step_function)
        ps_bucket = Ps(self.precp_interp(self.time_series), self.temp_interp(self.time_series), tmin, self.step_function)
        pr_bucket = Pr(self.precp_interp(self.time_series), self.temp_interp(self.time_series), tmin, self.step_function)
        
        return s_snow, s_water, q_bucket, et_bucket, m_bucket, ps_bucket, pr_bucket
    
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
            'et_bucket': results[3],
            'm_bucket': results[4],
            'ps_bucket': results[5],
            'pr_bucket': results[6],
            'prcp(mm/day)': ds['prcp(mm/day)'],
            'tmean(c)': ds['tmean(c)'],
            'dayl(s)': ds['dayl(s)'],            
            'q_bucket': results[2],
            'q_obs': ds['obs_runoff(mm/day)'],
        }
        
        # Create a DataFrame from the results dictionary
        results_df = pd.DataFrame(results_dict)
        
        # Save the results to a CSV file
        results_file = os.path.join(self.cfg.results_dir, f'{basin}_results_{period}.csv')
        results_df.to_csv(results_file, index=False)
        
    def plot_results(self, ds, q_bucket, basin, period='train', plot_prcp=False):
        '''
        Plot the model predictions and observed values.
        
        - Args:
            ds: xarray.Dataset, dataset with the input data.
            q_bucket: array_like, model predictions.
            basin: str, basin name.
            period: str, period of the run ('train', 'test', 'valid').
            plot_prcp: bool, whether to plot the precipitation rate.
            
        '''
        
        dates = ds['date']
        q_obs = ds['obs_runoff(mm/day)']            
            
        # Plot the predictions and actual values
        _, ax1 = plt.subplots(figsize=(16, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Discharge (mm/day)', color=color)
        ax1.plot(dates, q_obs, label='Observed', linewidth=3, color=color, zorder=2)
        ax1.plot(dates, q_bucket, ':', linewidth=3, label='Predicted', color='tab:red', zorder=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Set the major and minor locators and formatters for the x-axis
        years = YearLocator()   # every year
        months = MonthLocator()  # every month
        yearsFmt = DateFormatter('%Y')

        # Set the x-axis locators and formatters
        ax1.xaxis.set_major_locator(years)
        ax1.xaxis.set_major_formatter(yearsFmt)
        ax1.xaxis.set_minor_locator(months)
        
        # Set the x-axis limits
        start_date = dates.min()
        end_date = dates.max()
        ax1.set_xlim(start_date, end_date)
        # Enable autoscaling for the view
        ax1.autoscale_view()
            
        # Legend
        ax1.legend(loc='upper right')
        
        if plot_prcp:
            prcp = ds['prcp(mm/day)']
            # Create a twin Axes sharing the x-axis
            ax2 = ax1.twinx()
            color = 'lightslategray'
            ax2.set_ylabel('Precipitation Rate', color=color)
            ax2.plot(dates, prcp, label='Second Y-axis Data', color=color, zorder=1)
            ax2.tick_params(axis='y', labelcolor='darkslategray')
            ax2.invert_yaxis()  # Invert the y-axis of the second axis
            
        # print('NSE =', nse_val, nse_val.numpy(), f'{nse_val.numpy():.3f}')
        # plt.title(f'HydroNODES Model Predictions | NSE {nse_val.numpy():.3f} | {period.capitalize()} set')
        plt.title(f'Model Predictions ({self.cfg.concept_model}) | {period} period')
        
        plot_file_name = f'{basin}_{period.lower()}.png'
        
        plt.savefig(os.path.join(self.cfg.plots_dir, plot_file_name))
        plt.close()
        
    @property
    def interpolator_vars(self):
        return ['prcp(mm/day)', 'tmean(c)', 'dayl(s)']
        
## Auxiliary functions
# Qbucket is the runoff generated based on the available stored water in the bucket (unit: mm/day) - Patil_Stieglitz_HR_2012
Qb = lambda s1, f, smax, qmax, step_fct: step_fct(s1) * step_fct(s1 - smax) * qmax \
                                        + step_fct(s1) * step_fct(smax - s1) * qmax * np.exp(-f * (smax - s1))
                                        
# Qspill is the snowmelt is available to infiltrate into the catchment bucket, but the storage S has reached full capacity smax.
Qs = lambda s1, smax, step_fct: step_fct(s1) * step_fct(s1 - smax) * (s1 - smax)

# Precipitation as snow (A1) - Hoge_EtAl_HESS_2022
Ps = lambda p, t, tmin, step_fct: step_fct(tmin - t) * p

# Precipitation as snow (A2) - Hoge_EtAl_HESS_2022
Pr = lambda p, t, tmin, step_fct: step_fct(t - tmin) * p

# Melting (A4) - Hoge_EtAl_HESS_2022
M = lambda t, tmax, s0, Df, step_fct: step_fct(t - tmax) * step_fct(s0) * np.minimum(s0, Df * (t - tmax))

# Evapotranspiration (A3) - Hoge_EtAl_HESS_2022
ET = lambda s1, t, lday, smax, step_fct: step_fct(s1) * step_fct(s1 - smax) * PET(t, lday)  \
                            + step_fct(s1) * step_fct(smax - s1) * PET(t, lday) * (s1 / smax)
                                                 
# Potential evapotranspiration - Hamon’s formula (Hamon, 1963) - Hoge_EtAl_HESS_2022 
PET = lambda t, lday: 29.8 * lday * 0.611 * np.exp((17.3 * t) / (t + 237.3)) / (t + 273.2)    