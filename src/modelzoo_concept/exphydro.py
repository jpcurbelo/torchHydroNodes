import os
import numpy as np
from scipy.integrate import solve_ivp as sp_solve_ivp
import xarray
import pandas as pd
import torch
## https://github.com/rtqichen/torchdiffeq/tree/master
import torchdiffeq
import sys
from scipy.interpolate import Akima1DInterpolator

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.utils.load_process_data import (
    Config,
    ExpHydroCommon,
)

import yaml
from pathlib import Path
# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_dir)

FIXED_METHODS = ['euler', 'rk2', 'rk4', 'midpoint']

class EulerResult:
    def __init__(self, t, y):
        self.t = t
        self.y = y

# Ref: exphydro -> https://hess.copernicus.org/articles/26/5085/2022/
class ExpHydro(BaseConceptModel, ExpHydroCommon):
    
    def __init__(self, 
                cfg: Config,
                ds: xarray.Dataset,
                interpolators: dict,
                time_idx0: int,
                scaler: dict,
                odesmethod:str ='RK23'
            ):
        super().__init__(cfg, ds, interpolators, time_idx0, scaler, odesmethod)\
        
        # Input variables
        self.precp = ds['prcp']
        self.temp = ds['tmean']
        self.lday = ds['dayl']
        
        # Parameters per basin
        self.params_dict = self.get_parameters()

        self.eps = torch.finfo(cfg.precision['torch']).eps
        
    def conceptual_model(self, t, y):

        # # Flush time
        # sys.stdout.write(f"t \r{t}")
        # sys.stdout.flush()
        
        # Bucket parameters                   
        # f: Rate of decline in flow from catchment bucket   
        # Smax: Maximum storage of the catchment bucket     
        # Qmax: Maximum subsurface flow at full bucket      
        # Df: Thermal degree‐day factor                   
        # Tmax: Temperature above which snow starts melting 
        # Tmin: Temperature below which precipitation is snow
        f = self.params['f']
        smax = self.params['smax']
        qmax = self.params['qmax']
        df = self.params['df']
        tmax = self.params['tmax']
        tmin = self.params['tmin']

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

        if isinstance(s0, torch.Tensor):
            # temp to torch tensor
            temp = torch.tensor(temp, dtype=self.cfg.precision['torch'])
            # precp to torch tensor
            precp = torch.tensor(precp, dtype=self.cfg.precision['torch'])
            # lday to torch tensor
            lday = torch.tensor(lday, dtype=self.cfg.precision['torch'])
        m_out = M(s0, temp, df, tmax, self.step_function)

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = Ps(precp, temp, tmin, self.step_function) - m_out
        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = Pr(precp, temp, tmin, self.step_function) + m_out - ET(s1, temp, lday, smax, self.step_function) - q_out

        # print('Ps:', Ps(precp, temp, tmin, self.step_function))
        # print('Pr:', Pr(precp, temp, tmin, self.step_function))
        # print('m_out:', m_out)
        # print('ET:', ET(s1, temp, lday, smax, self.step_function))
        # print('q_out:', q_out)

        if self.ode_solver_lib == 'scipy':
            return [ds0_dt, ds1_dt]
        elif self.ode_solver_lib == 'torchdiffeq':
            # # print('t:', t)
            # # print('ds0_dt:', ds0_dt)
            # # print('ds1_dt:', ds1_dt)
            # return torch.stack([ds0_dt, ds1_dt])
            return torch.stack([ds0_dt, ds1_dt]).squeeze()  # Use .squeeze() to avoid extra dimensions if present
           
    def step_function(self, x):
        """
        Step function to handle both PyTorch tensors (for gradient tracking) and NumPy arrays.
        
        - Args:
            x: Input value, can be a torch.Tensor or a np.ndarray.
        
        - Returns:
            Output with step function applied.
        """
        if isinstance(x, torch.Tensor):
            # If x is a tensor, we stay within the PyTorch computation graph
            return (torch.tanh(5.0 * x) + 1.0) * 0.5
        elif isinstance(x, np.ndarray) or isinstance(x, float):
            # If x is a NumPy array, we use NumPy operations
            return (np.tanh(5.0 * x) + 1.0) * 0.5
        else:
            raise TypeError("Input must be either a torch.Tensor or a np.ndarray")
        
    def run(self, basin, basin_params=None, use_grad=False):
        
        if basin_params is None:
            basin_params = self.params_dict[basin]

        # print('basin_params:', basin_params)
        
        # Get the interpolator functions for the basin
        self.precp_interp = self.interpolators[basin]['prcp']
        self.temp_interp = self.interpolators[basin]['tmean']
        self.lday_interp = self.interpolators[basin]['dayl']

        # Get input variables for the basin
        if self.precp.ndim == 2: # When running the model - N basins
            self.precp_basin = self.precp.sel(basin=basin).values
            self.temp_basin = self.temp.sel(basin=basin).values
            self.lday_basin = self.lday.sel(basin=basin).values
        else: # When calibrating the model - 1 basin
            self.precp_basin = self.precp.values
            self.temp_basin = self.temp.values
            self.lday_basin = self.lday.values
            if use_grad:
                self.precp_basin = torch.tensor(self.precp_basin, dtype=self.cfg.precision['torch'])
                self.temp_basin = torch.tensor(self.temp_basin, dtype=self.cfg.precision['torch'])
                self.lday_basin = torch.tensor(self.lday_basin, dtype=self.cfg.precision['torch'])
        
        # Set the initial conditions
        if use_grad: # Use PyTorch tensors for initial conditions and parameters
            y0 = torch.tensor([basin_params[0], basin_params[1]], dtype=self.cfg.precision['torch'])
            y0 = y0.squeeze() if y0.dim() > 1 else y0  # Squeezes extra dimensions if they exist
        else:
            y0 = np.array([basin_params[0], basin_params[1]])
        
        # Set the parameters as class attributes to avoid passing them in `args`
        self.params = {
            'f': basin_params[2],
            'smax': basin_params[3],
            'qmax': basin_params[4],
            'df': basin_params[5],
            'tmax': basin_params[6],
            'tmin': basin_params[7]
        }

        if self.ode_solver_lib == 'scipy':

            # Run the model
            if self.odesmethod.lower() in ['euler', 'rk2', 'rk4']:           
                # Use the desired method: "euler", "rk2", or "rk4"
                y = self.solve_ivp_custom(
                    self.conceptual_model, 
                    t_span=(self.time_idx0, self.time_idx0 + self.precp.shape[-1] - 1), 
                    y0=y0, 
                    step_size=self.time_step,
                    t_eval=self.time_series, 
                    method=self.odesmethod  # Change this to "euler", "rk2", or "rk4" as needed
                )
            else:
                # Use the scipy's solve_ivp method with the desired method
                y = sp_solve_ivp(self.conceptual_model, t_span=(self.time_idx0, self.time_idx0 + self.precp.shape[-1] - 1), y0=y0, t_eval=self.time_series, 
                                method=self.odesmethod,
                                # method='LSODA',
                                #  method='DOP853',
                                # rtol=1e-3, atol=1e-3,
                                rtol=self.rtol, atol=self.atol,
                            )
        elif self.ode_solver_lib == 'torchdiffeq':
            # Run the model
            # Define rtol and atol
            # Higher rtol and atol values will make the ODE solver faster but less accurate
            if self.odesmethod in ['euler', 'rk4', 'midpoint']:
                rtol = 1e-3
                atol = 1e-3
            elif self.odesmethod in ['bosh3']:
                rtol = 1e-4
                atol = 1e-6
            elif self.odesmethod in ['dopri5', 'fehlberg2', 'dopri8', 'adaptive_heun', 'heun3']:
                rtol = 1e-3
                atol = 1e-6
            elif self.odesmethod in ['explicit_adams', 'implicit_adams', 'fixed_adams']:
                rtol = 1e-6
                atol = 1e-9
            elif self.odesmethod in ['scipy_solver']:
                rtol = 1e-4
                atol = 1e-6
                solver = self.scipy_solver

            # Set the options for the ODE solver
            if self.odesmethod.lower() in FIXED_METHODS:
                options = {"step_size": self.time_step, "interp": "cubic"}
            elif self.odesmethod == 'scipy_solver':
                options = {"solver": solver}
            else:
                options = {}

            ode_solver = torchdiffeq.odeint

            # Set the initial conditions to  torch.Tensor
            if use_grad:
                y0 = y0.clone().detach().requires_grad_(True)
            else:
                y0 = torch.tensor(y0, dtype=self.cfg.precision['torch'])
                # Ensure y0 is expanded to 2D if required, e.g., [2] -> [2, 1]
                if y0.dim() == 1:
                    # y0 = y0.unsqueeze(1)  # Adds an extra dimension
                    y0 = y0.clone().detach().requires_grad_(True).unsqueeze(1)  # Adds an extra dimension safely

            # Set time series to torch.Tensor
            self.time_series = torch.tensor(self.time_series, dtype=self.cfg.precision['torch'])

            # # # Ensure self.time_series is a PyTorch tensor, and handle it safely if it's already a tensor
            # # if isinstance(self.time_series, np.ndarray):
            # #     self.time_series = torch.tensor(self.time_series, dtype=self.cfg.precision['torch'])
            # # else:
            # #     self.time_series = self.time_series.clone().detach().requires_grad_(use_grad)

            # Run the ODE solver
            y = ode_solver(self.conceptual_model, y0=y0, t=self.time_series, method=self.odesmethod, 
                           rtol=self.rtol, atol=self.atol, options=options)
        
        # Extract snow and water series -> two state variables representing buckets for snow and water Hoge_EtAl_HESS_2022
        if self.ode_solver_lib == 'scipy':
            s_snow = y.y[0]
            s_water = y.y[1]
        elif self.ode_solver_lib == 'torchdiffeq':
            if use_grad:
                s_snow = y[:, 0]
                s_water = y[:, 1]
            else:
                s_snow = y[:, 0].detach().cpu().numpy()
                s_water = y[:, 1].detach().cpu().numpy()

        # print('s_snow:', s_snow)
        # print('s_water:', s_water)

        # Unpack the parameters to call the mechanistic processes
        f, smax, qmax, df, tmax, tmin = self.params.values()
        
        # Calculate the mechanistic processes -> q_bucket, et_bucket, m_bucket, ps_bucket, pr_bucket
        # Discharge
        # print('Qb:', Qb(s_water, f, smax, qmax, self.step_function))
        # print('Qs:', Qs(s_water, smax, self.step_function))
        q_bucket = Qb(s_water, f, smax, qmax, self.step_function) + Qs(s_water, smax, self.step_function)
        # print('before q_bucket:', q_bucket[:5], q_bucket[-5:])
        if not use_grad:
            q_bucket = np.maximum(q_bucket, self.eps)
        else:
            q_bucket = torch.clamp(q_bucket, min=self.eps)
        # print('after  q_bucket:', q_bucket[:5], q_bucket[-5:])

        # aux = input('Press any key to continue...')

        # # Plot and save q_bucket vs time
        # import matplotlib.pyplot as plt
        # plt.plot(self.time_series, q_bucket)
        # plt.title(f'basin {basin} - q_bucket')
        # plt.savefig('q_bucket.png')




        # Evapotranspiration
        et_bucket = ET(s_water, self.temp_basin, self.lday_basin, smax, self.step_function)
        if not use_grad:
            et_bucket = np.maximum(et_bucket, self.eps)
        else:
            et_bucket = torch.clamp(et_bucket, min=self.eps)

        # Melting
        m_bucket = M(s_snow, self.temp_basin, df, tmax, self.step_function)
        if not use_grad:
            m_bucket = np.maximum(m_bucket, self.eps)
        else:
            m_bucket = torch.clamp(m_bucket, min=self.eps)
        # Precipitation as snow
        ps_bucket = Ps(self.precp_basin, self.temp_basin, tmin, self.step_function)
        if not use_grad:
            ps_bucket = np.maximum(ps_bucket, self.eps)
        else:
            ps_bucket = torch.clamp(ps_bucket, min=self.eps)
        # Precipitation as rain
        pr_bucket = Pr(self.precp_basin, self.temp_basin, tmin, self.step_function)
        if not use_grad:
            pr_bucket = np.maximum(pr_bucket, self.eps)
        else:
            pr_bucket = torch.clamp(pr_bucket, min=self.eps)
        
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

        # Load the variables for the concept model
        input_vars, output_vars, _ = self.cfg._load_concept_model_vars(self.cfg.concept_model)

        # Open and read the 'variable_aliases.yml' file
        with open(Path(project_dir) / 'src' / 'utils' / 'variable_aliases.yml', 'r') as file:
            aliases = yaml.safe_load(file)

        alias_map = {}
        for col in input_vars + output_vars:
            if col in aliases:
                alias_map[col] = aliases[col]
        
        # Dynamically construct the results_dict using input_vars and the longest alias for each variable
        results_dict = {
            'date': ds['date'],
            's_snow': results[0],
            's_water': results[1],
            'et_bucket': results[2],
            'm_bucket': results[3],
            'ps_bucket': results[4],
            'pr_bucket': results[5],
        }

        # Add the dynamically generated keys and values for input variables
        for var in input_vars:
            alias = alias_map.get(var, var)  # Use the longest alias if available, otherwise fallback to var name
            # If alias is a list, then pick the longest alias
            if isinstance(alias, list):
                alias = max(alias, key=len)
            results_dict[alias] = ds[var]

        # Add the target variable (output_vars) to the results_dict
        # Assuming the target variable is 'q_obs' and is stored in the last position of the results tuple
        results_dict['q_bucket'] = results[-1]
        for var in output_vars:
            alias = alias_map.get(var, var)  # Use the longest alias if available, otherwise fallback to var name
            # If alias is a list, then pick the longest alias
            if isinstance(alias, list):
                alias = max(alias, key=len)
            results_dict[alias] = ds[var]
            # results_dict[var] = ds[var]
        
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

            # Check if the file exists
            if not os.path.exists(previous_results_file):
                print(f'File {previous_results_file} does not exist.')
                return
            
            previous_results = pd.read_csv(previous_results_file)
            
            # Get the last states of the model in the previous period
            previous_states = previous_results[['s_snow', 's_water']].values[-1]
            
            # Update the initial states for the model
            self.params_dict[basin][0] = previous_states[0]
            self.params_dict[basin][1] = previous_states[1]

    @staticmethod
    def solve_ivp_custom(fun, t_span, y0, t_eval=None, method="euler", step_size=1.0):
        """
        Solve an initial value problem (IVP) for a system of ODEs using Euler, RK2, or RK4 method.
        
        Parameters:
        - fun: callable
            Right-hand side of the system. The calling signature is `fun(t, y, *args)`.
        - t_span: 2-tuple of floats
            Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.
        - y0: array_like, shape (n,)
            Initial state.
        - t_eval: array_like or None, shape (m,), optional
            Times at which to store the computed solution. If None (default), use solver's own time steps.
        - method: str, optional
            The integration method to use: "euler", "rk2", or "rk4". Default is "euler".
        - step_size: float, optional
            Step size for time stepping. Defaults to 1.0.
        
        Returns:
        - EulerResult object with attributes:
            - t: ndarray, shape (n_points,)
            Time points.
            - y: ndarray, shape (n, n_points)
            Solution values at `t`, where each row corresponds to one of the state variables.
        """

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1] - t_span[0] + 1))
        
        # Generate internal time points based on step_size
        t_internal = np.arange(t_span[0], t_span[1] + step_size, step_size)
        
        # Initialize the solution array
        y = np.zeros((len(t_internal), len(y0)))
        y[0] = y0

        # Select the integration method
        if method == "euler":
            for i in range(1, len(t_internal)):
                dt = t_internal[i] - t_internal[i - 1]
                dydt = fun(t_internal[i - 1], y[i - 1])
                y[i] = y[i - 1] + dt * np.array(dydt)
        
        elif method == "rk2":
            for i in range(1, len(t_internal)):
                dt = t_internal[i] - t_internal[i - 1]
                k1 = np.array(fun(t_internal[i - 1], y[i - 1]))
                k2 = np.array(fun(t_internal[i - 1] + dt / 2, y[i - 1] + dt / 2 * k1))
                y[i] = y[i - 1] + dt * k2

        elif method == "rk4":
            for i in range(1, len(t_internal)):
                dt = t_internal[i] - t_internal[i - 1]
                k1 = np.array(fun(t_internal[i - 1], y[i - 1]))
                k2 = np.array(fun(t_internal[i - 1] + dt / 2, y[i - 1] + dt / 2 * k1))
                k3 = np.array(fun(t_internal[i - 1] + dt / 2, y[i - 1] + dt / 2 * k2))
                k4 = np.array(fun(t_internal[i - 1] + dt, y[i - 1] + dt * k3))
                y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods are 'euler', 'rk2', 'rk4'.")

        # Transpose the solution array to have each state variable in a row
        y = y.T  # Shape (num_time_points, num_variables), transposing to (num_variables, num_time_points)

        # Interpolate each state variable independently
        y_at_teval = np.zeros((y.shape[0], len(t_eval)))  # Initialize the output array

        # For each state variable (each row in y), perform interpolation
        for i in range(y.shape[0]):
            interpolator = Akima1DInterpolator(t_internal, y[i])
            y_at_teval[i] = interpolator(t_eval)  # Interpolate for this state variable at t_span

        return EulerResult(np.array(t_span), y_at_teval)
    
    @property
    def nn_outputs(self):
        return ['ps_bucket', 'pr_bucket', 'm_bucket', 'et_bucket', 'q_bucket']
    
    @property
    def model_outputs(self):
        return ['s_snow', 's_water']
    
# ## Auxiliary functions
# # Qbucket is the runoff generated based on the available stored water in the bucket (unit: mm/day) - Patil_Stieglitz_HR_2012
# Qb = lambda s1, f, smax, qmax, step_fct: step_fct(s1) * step_fct(s1 - smax) * qmax \
#                                         + step_fct(s1) * step_fct(smax - s1) * qmax * np.exp(-f * (smax - s1))
# # Qb = lambda s1, f, smax, qmax, step_fct: step_fct(s1) * step_fct(s1 - smax) * qmax \
# #     + step_fct(s1) * step_fct(smax - s1) * qmax * np.exp(np.clip(-f * (smax - s1), a_min=-700, a_max=700))   

# # Qspill is the snowmelt is available to infiltrate into the catchment bucket, but the storage S has reached full capacity smax.
# Qs = lambda s1, smax, step_fct: step_fct(s1) * step_fct(s1 - smax) * (s1 - smax)

def Qb(s1, f, smax, qmax, step_fct):
    """
    Calculate Qb, handling both torch.Tensor and np.ndarray inputs.
    """

    if isinstance(s1, torch.Tensor):
        exp_term = torch.exp(torch.clamp(-f * (smax - s1), min=-50, max=50))
    elif isinstance(s1, np.ndarray) or isinstance(s1, float):
        exp_term = np.exp(-f * (smax - s1))
    else:
        raise TypeError("Input must be either a torch.Tensor or a np.ndarray")
    
    return step_fct(s1) * step_fct(s1 - smax) * qmax + step_fct(s1) * step_fct(smax - s1) * qmax * exp_term

def Qs(s1, smax, step_fct):
    """
    Calculate Qs, handling both torch.Tensor and np.ndarray inputs.
    """
    return step_fct(s1) * step_fct(s1 - smax) * (s1 - smax)

# Precipitation as snow (A1) - Hoge_EtAl_HESS_2022
Ps = lambda p, temp, tmin, step_fct: step_fct(tmin - temp) * p

# Precipitation as snow (A2) - Hoge_EtAl_HESS_2022
Pr = lambda p, temp, tmin, step_fct: step_fct(temp - tmin) * p

# # Melting (A4) - Hoge_EtAl_HESS_2022
# M = lambda s0, temp, df, tmax, step_fct: step_fct(temp - tmax) * step_fct(s0) * np.minimum(s0, df * (temp - tmax))


# # Evapotranspiration (A3) - Hoge_EtAl_HESS_2022
# ET = lambda s1, temp, lday, smax, step_fct: step_fct(s1) * step_fct(s1 - smax) * PET(temp, lday)  \
#                             + step_fct(s1) * step_fct(smax - s1) * PET(temp, lday) * (s1 / smax)
                                                 
# # Potential evapotranspiration - Hamon’s formula (Hamon, 1963) - Hoge_EtAl_HESS_2022 
# PET = lambda temp, lday: 29.8 * lday * 0.611 * np.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)    

def M(s0, temp, df, tmax, step_fct):
    """Melting function, with type-checking for compatibility."""
    if isinstance(s0, torch.Tensor):
        return step_fct(temp - tmax) * step_fct(s0) * torch.minimum(s0, df * (temp - tmax))
    elif isinstance(s0, np.ndarray) or isinstance(s0, float):
        return step_fct(temp - tmax) * step_fct(s0) * np.minimum(s0, df * (temp - tmax))
    else:
        raise TypeError("Inputs must be either torch.Tensor or np.ndarray")

def ET(s1, temp, lday, smax, step_fct):
    """Evapotranspiration function, compatible with both tensor and array inputs."""
    PET_val = PET(temp, lday)
    if isinstance(s1, torch.Tensor):
        # print('s1:', type(s1), s1)
        # print('smax:', type(smax), smax)
        # print('PET_val:', type(PET_val), PET_val)
        et_val = step_fct(s1) * step_fct(s1 - smax) * PET_val + step_fct(s1) * step_fct(smax - s1) * PET_val * (s1 / smax)
    elif isinstance(s1, np.ndarray) or isinstance(s1, float):
        et_val = step_fct(s1) * step_fct(s1 - smax) * PET_val + step_fct(s1) * step_fct(smax - s1) * PET_val * (s1 / smax)
    else:
        raise TypeError("Inputs must be either torch.Tensor or np.ndarray")
    return et_val

def PET(temp, lday):
    """Potential evapotranspiration using Hamon's formula."""
    # print('In PET')
    # print('temp:', type(temp), temp)
    # print('lday:', type(lday), lday)
    if isinstance(temp, torch.Tensor):
        return 29.8 * lday * 0.611 * torch.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
    elif isinstance(temp, np.ndarray) or isinstance(temp, float):
        return 29.8 * lday * 0.611 * np.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
    else:
        raise TypeError("Temperature input must be either torch.Tensor or np.ndarray")