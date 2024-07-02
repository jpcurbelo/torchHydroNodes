
import xarray
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp as sp_solve_ivp
import torch
from tqdm import tqdm
import sys
import torch.nn as nn
## https://github.com/rtqichen/torchdiffeq/tree/master
import torchdiffeq
import time

from src.modelzoo_hybrid.basemodel import BaseHybridModel
from src.modelzoo_nn.basepretrainer import NNpretrainer
from src.utils.load_process_data import (
    Config,
    ExpHydroCommon,
    ExpHydroODEs,
)

# Ref: exphydro -> https://hess.copernicus.org/articles/26/5085/2022/
class ExpHydroM100(BaseHybridModel, ExpHydroCommon, nn.Module):

    def __init__(self,
                cfg: Config,
                pretrainer: NNpretrainer,
                ds: xarray.Dataset,
                scaler: dict,
    ):
        BaseHybridModel.__init__(self, cfg, pretrainer, ds, scaler)  # Initialize BaseHybridModel
        ExpHydroCommon.__init__(self)  # Initialize ExpHydroCommon
        nn.Module.__init__(self)  # Initialize nn.Module

        # Interpolators
        self.interpolators = self.create_interpolator_dict(is_trainer=True)

        # Parameters per basin
        self.params_dict = self.get_parameters()

        if self.cfg.scale_target_vars:
            self.scale_target_vars(is_trainer=True)
    

    def forward(self, inputs, basin):

        # print('inputs_forward', inputs.shape)

        # Extract the state variables
        # If inputs shape is 2D, then is 'mlp' model
        if len(inputs.shape) == 2:
            self.s_snow = inputs[:, 0]
            self.s_water = inputs[:, 1]
            self.precp_series = inputs[:, 2]
            self.tmean_series = inputs[:, 3]
            self.time_series = inputs[:, 4]
        # If inputs shape is 3D, then is 'lstm' model
        elif len(inputs.shape) == 3:
            self.window_size = inputs.shape[1]
            self.s_snow = inputs[:, :, 0]
            self.s_water = inputs[:, :, 1]
            self.precp_series = inputs[:, :, 2]
            self.tmean_series = inputs[:, :, 3]
            self.time_series_lstm = inputs[:, :, 4]

            # Flatten self.time_series to have a 1D vector containing the time idxs values
            #  unique_values, _ = torch.unique(flattened, sorted=True, return_inverse=True)
            self.time_series, _ = torch.unique(self.time_series_lstm.flatten(), sorted=True, return_inverse=True)

            self.s_snow_lstm = self.s_snow[0, :-1]
            self.s_water_lstm = self.s_water[0, :-1]
            # Add the first value of the sequence at the begining of the tensor
            # self.s_snow_lstm = torch.cat((self.s_snow_lstm, self.s_snow[0, -1].unsqueeze(0)), dim=0)
            # self.s_water_lstm = torch.cat((self.s_water_lstm, self.s_water[0, -1].unsqueeze(0)), dim=0)
            self.s_snow_lstm = torch.cat((self.s_snow_lstm[0].unsqueeze(0), self.s_snow_lstm), dim=0)
            self.s_water_lstm = torch.cat((self.s_water_lstm[0].unsqueeze(0), self.s_water_lstm), dim=0)
            # print('self.s_snow_lstm', self.s_snow_lstm.shape)
            # print('self.s_water_lstm', self.s_water_lstm.shape)

        # print('self.time_series', self.time_series.shape)
        # # if len(inputs.shape) == 3:
        # #     print('self.time_series_lstm', self.time_series_lstm, self.time_series_lstm.shape)

        # print('s_snow', self.s_snow.shape)
        # print('s_water', self.s_water.shape)
        # print('precp_series', self.precp_series.shape)
        # print('tmean_series', self.tmean_series.shape)
        # print('time_series', self.time_series.shape)
        # if len(inputs.shape) == 3:
        #     print('time_series_lstm', self.time_series_lstm.shape)

        # print('self.s_snow[0, -1]', self.s_snow[0, -1])
        # print('self.s_water[0, -1]', self.s_water[0, -1])

        # print(100*'*')

        # aux = input("Press Enter to continue...")


        # print('precp_series', self.precp_series, self.precp_series.device)
        # print('tmean_series', self.tmean_series, self.tmean_series.device)
        # print('time_series', self.time_series, self.time_series.device)

        # Make basin global to be used in hybrid_model
        self.basin = basin
        # basin_params = self.params_dict[self.basin]

        # Set the interpolators 
        self.precp_interp = self.interpolators[self.basin]['prcp']
        self.temp_interp = self.interpolators[self.basin]['tmean']
        self.lday_interp = self.interpolators[self.basin]['dayl']

        # # Profile the ODE solver
        # time_start = time.time()

        ode_solver = torchdiffeq.odeint
        if len(inputs.shape) == 2:
            # Set the initial conditions
            y0 = torch.stack([self.s_snow[0], self.s_water[0]], dim=0).unsqueeze(0)
            y = ode_solver(self.hybrid_model_mlp, y0=y0, t=self.time_series, method=self.odesmethod, rtol=1e-3, atol=1e-6)   # 'rk4' 'midpoint'   'euler' 'dopri5' #rtol=1e-6, atol=1e-6
            # y = ode_solver(self.hybrid_model, y0=y0, t=time_series, method='rk4', rtol=1e-3, atol=1e-6)
        elif len(inputs.shape) == 3:
            # Set the initial conditions
            y0 = torch.stack([self.s_snow[0, -1], self.s_water[0, -1]], dim=0).unsqueeze(0)
            y = ode_solver(self.hybrid_model_lstm, y0=y0, t=self.time_series[self.window_size-1:], method=self.odesmethod, rtol=1e-3, atol=1e-6)

        # # Initialize your hybrid model and ODE solver
        # # ode_solver =  torchdiffeq.odeint_adjoint
        # ode_solver =  torchdiffeq.odeint
        # hybrid_model = ExpHydroODEs(
        #     # self.precp_series,
        #     # self.tmean_series,
        #     # self.lday_series,
        #     # self.time_series,

        #     self.precp_interp,
        #     self.temp_interp,
        #     self.lday_interp,
        #     self.data_type_torch,
        #     self.device,

        #     self.scale_target_vars,
        #     self.pretrainer,
        #     self.basin,
        #     self.step_function
        # )
        # y = ode_solver(hybrid_model, y0=y0, t=self.time_series, method=self.odesmethod, rtol=1e-3, atol=1e-6)   # 'rk4' 'midpoint'   'euler' 'dopri5' #rtol=1e-6, atol=1e-6
        # # y = ode_solver(hybrid_model, y0=y0, t=self.time_series, method='rk4', rtol=1e-4, atol=1e-4)

        # print(f'Time taken for ODE solver: {round(time.time() - time_start, 2)} seconds')
        # aux = input("Press Enter to continue...")
        
        # # Update the state variables from the ODE solution
        # print('\n', 100*'*')
        # print('y', y.shape)

        if len(inputs.shape) == 2:

            s_snow_nn  = y[:, 0, 0]
            s_water_nn = y[:, 0, 1]
            # Relu the state variables
            s_snow_nn = torch.maximum(s_snow_nn, torch.tensor(0.0)).to(self.device)
            s_water_nn = torch.maximum(s_water_nn, torch.tensor(0.0)).to(self.device)

            # Stack tensors to create inputs for the neural network
            inputs_nn = torch.stack([s_snow_nn, s_water_nn, self.precp_series, self.tmean_series], dim=-1)

        elif len(inputs.shape) == 3:

            s_snow_nn  = y[:, :, 0].squeeze()
            s_water_nn = y[:, :, 1].squeeze()

            # print('self.s_snow_lstm', self.s_snow_lstm[-5:])
            # print('s_snow_nn', s_snow_nn[-5:])

            # Relu the state variables
            s_snow_nn = torch.maximum(s_snow_nn, torch.tensor(0.0)).to(self.device)
            s_water_nn = torch.maximum(s_water_nn, torch.tensor(0.0)).to(self.device)

            # Extract the window_size-1 first elements from the LSTM state variables
            s_snow_first_batch = self.s_snow[0, :-1]
            s_water_first_batch = self.s_water[0, :-1]

            # Concatenate along the last dimension
            s_snow_combined = torch.cat((s_snow_first_batch, s_snow_nn), dim=0)
            s_water_combined = torch.cat((s_water_first_batch, s_water_nn), dim=0)

            # Create LSTM sequences using self.window_size
            s_snow_nn = self.create_sequences(s_snow_combined)
            s_water_nn = self.create_sequences(s_water_combined)

            # print('s_snow_nn', s_snow_nn.shape)
            # print('s_water_nn', s_water_nn.shape)
            # print('precp_series', self.precp_series.shape)
            # print('tmean_series', self.tmean_series.shape)

            # Stack tensors to create inputs for the neural network
            inputs_nn = torch.stack([s_snow_nn, s_water_nn, self.precp_series, self.tmean_series], dim=-1)


        # print('inputs_nn', inputs_nn.shape)


        #!!!!!!!!This output is already in log space - has to be converted back to normal space when the model is called
        # Assuming nnmodel returns a tensor of shape (batch_size, numOfVars)
        # output = self.pretrainer.nnmodel(inputs_nn.to(self.device), [self.basin]) #.to(self.device)
        output = self.pretrainer.nnmodel(inputs_nn, [self.basin])

        # Extract the last variable (last column) from the output
        q_output = output[:, -1]

        return q_output

    def hybrid_model_mlp(self, t, y):

        # Flush time
        sys.stdout.write(f"t \r{t}")
        sys.stdout.flush()
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        s0 = y[..., 0] #.to(self.device)
        s1 = y[..., 1] #.to(self.device)

        # Interpolate the input variables
        t_np = t.detach().cpu().numpy()
        precp = self.precp_interp(t_np, extrapolate='periodic')
        temp = self.temp_interp(t_np, extrapolate='periodic')
        lday = self.lday_interp(t_np, extrapolate='periodic')

        # Convert to tensor
        precp = torch.tensor(precp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        temp = torch.tensor(temp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        lday = torch.tensor(lday, dtype=self.data_type_torch)

        # # Find left index for the interpolation
        # idx = torch.searchsorted(self.time_series, t, side='right') - 1
        # idx = idx.clamp(max=self.time_series.size(0) - 2)  # Ensure indices do not exceed valid range

        # # Linear interpolation
        # precp = self.precp_series[idx] + (self.precp_series[idx + 1] - self.precp_series[idx]) \
        #     * (t - self.time_series[idx]) / (self.time_series[idx + 1] - self.time_series[idx]).unsqueeze(0)
        # temp = self.tmean_series[idx] + (self.tmean_series[idx + 1] - self.tmean_series[idx]) \
        #     * (t - self.time_series[idx]) / (self.time_series[idx + 1] - self.time_series[idx]).unsqueeze(0)
        # lday = self.lday_series[idx] + (self.lday_series[idx + 1] - self.lday_series[idx]) \
        #     * (t - self.time_series[idx]) / (self.time_series[idx + 1] - self.time_series[idx])
        

        # Compute ET from the pretrainer.nnmodel
        inputs_nn = torch.stack([s0, s1, precp, temp], dim=-1)

        basin = self.basin
        if not isinstance(basin, list):
            basin = [basin]

        m100_outputs = self.pretrainer.nnmodel(inputs_nn, basin)[0] #.to(self.device)

        # Target variables:  Psnow, Prain, M, ET and, Q
        p_snow, p_rain, m, et, q = m100_outputs[0], m100_outputs[1], m100_outputs[2], \
                                   m100_outputs[3], m100_outputs[4]
        
        # Scale back to original values for the ODEs and Relu the Mechanism Quantities    
        if self.cfg.scale_target_vars:
            p_snow = torch.relu(torch.sinh(p_snow) * self.step_function(-temp[0]))
            p_rain = torch.relu(torch.sinh(p_rain))
            m = torch.relu(self.step_function(s0) * torch.sinh(m))
            et = self.step_function(s1) * torch.exp(et) * lday
            q = self.step_function(s1) * torch.exp(q) 

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        return torch.stack([ds0_dt, ds1_dt], dim=-1)

    def hybrid_model_lstm(self, t, y):

        # # Flush time
        # sys.stdout.write(f"t \r{t}")
        # sys.stdout.flush()
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        s0 = y[..., 0]
        s1 = y[..., 1]

        # Interpolate the input variables
        t_np = t.detach().cpu().numpy()
        precp = self.precp_interp(t_np, extrapolate='periodic')
        temp = self.temp_interp(t_np, extrapolate='periodic')
        lday = self.lday_interp(t_np, extrapolate='periodic')

        # Convert to tensor
        precp = torch.tensor(precp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        temp = torch.tensor(temp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        lday = torch.tensor(lday, dtype=self.data_type_torch)

        # Find left index for the interpolation
        idx = torch.searchsorted(self.time_series, t, side='right') - 1
        idx = idx.clamp(max=self.time_series.size(0) - 2)  # Ensure indices do not exceed valid range

        # Prepare entries for LSTM (find the idx position in the 2nd dim and substitute the last value by the new value)
        precp_nn = self.precp_series[idx - self.window_size]
        precp_nn[-1] = precp
        temp_nn = self.tmean_series[idx - self.window_size]
        temp_nn[-1] = temp

        # Prepare s_snow_nn and s_water_nn. Remove the first element and add the new value to the last entry
        self.s_snow_lstm = torch.cat((self.s_snow_lstm[1:], s0), dim=0)
        self.s_water_lstm = torch.cat((self.s_water_lstm[1:], s1), dim=0)

        # Compute ET from the pretrainer.nnmodel
        inputs_nn = torch.stack([self.s_snow_lstm, self.s_water_lstm, precp_nn, temp_nn], dim=-1)

        basin = self.basin
        if not isinstance(basin, list):
            basin = [basin]

        m100_outputs = self.pretrainer.nnmodel(inputs_nn, basin)[0] #.to(self.device)

        # Target variables:  Psnow, Prain, M, ET and, Q
        p_snow, p_rain, m, et, q = m100_outputs[0], m100_outputs[1], m100_outputs[2], \
                                   m100_outputs[3], m100_outputs[4]
        
        # Scale back to original values for the ODEs and Relu the Mechanism Quantities    
        if self.cfg.scale_target_vars:
            p_snow = torch.relu(torch.sinh(p_snow) * self.step_function(-temp[0]))
            p_rain = torch.relu(torch.sinh(p_rain))
            m = torch.relu(self.step_function(s0) * torch.sinh(m))
            et = self.step_function(s1) * torch.exp(et) * lday
            q = self.step_function(s1) * torch.exp(q) 

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        return torch.stack([ds0_dt, ds1_dt], dim=-1)


    def create_sequences(self, data):
        num_sequences = data.size(0) - self.window_size + 1
        sequences = [data[i:i + self.window_size] for i in range(num_sequences)]
        return torch.stack(sequences)


    @staticmethod
    def step_function(x):
        '''
        Step function to be used in the model.
        
        - Args:
            x: tensor, input value.
            
        - Returns:
            tensor, step function applied to input value(s).
        '''
        return (torch.tanh(5.0 * x) + 1.0) * 0.5
