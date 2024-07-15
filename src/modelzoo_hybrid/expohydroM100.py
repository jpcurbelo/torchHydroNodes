
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
    

    def forward(self, inputs, basins, use_grad=True):

        self.use_grad = use_grad

        # print('use_grad', use_grad)
        # print(f"Memory usage before forward pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        

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
            self.time_series, _ = torch.unique(self.time_series_lstm.flatten(), sorted=True, return_inverse=True)
            # Transfer to device
            self.time_series = self.time_series #.to(self.device)

            self.s_snow_lstm = self.s_snow[0, :-1]
            self.s_water_lstm = self.s_water[0, :-1]
            self.precp_lstm = self.precp_series[0, :-1]
            self.tmean_lstm = self.tmean_series[0, :-1]
            # Add the first value of the sequence at the begining of the tensor
            self.s_snow_lstm = torch.cat((self.s_snow_lstm[0].unsqueeze(0), self.s_snow_lstm), dim=0) #.to(self.device)
            self.s_water_lstm = torch.cat((self.s_water_lstm[0].unsqueeze(0), self.s_water_lstm), dim=0) #.to(self.device)
            self.precp_lstm = torch.cat((self.precp_lstm[0].unsqueeze(0), self.precp_lstm), dim=0) #.to(self.device)
            self.tmean_lstm = torch.cat((self.tmean_lstm[0].unsqueeze(0), self.tmean_lstm), dim=0) #.to(self.device)

        # Make basin global to be used in hybrid_model
        self.basins = basins

        # Set the interpolators 
        self.precp_interp = self.interpolators[self.basins]['prcp']
        self.temp_interp = self.interpolators[self.basins]['tmean']
        self.lday_interp = self.interpolators[self.basins]['dayl']

        # # Profile the ODE solver
        # time_start = time.time()

        # Define rtol and atol
        # Higher rtol and atol values will make the ODE solver faster but less accurate
        if self.odesmethod in ['euler', 'rk4', 'midpoint']:
            rtol = 1e-3
            atol = 1e-3
        elif self.odesmethod in ['dopri5', 'fehlberg2', 'dopri8', 'bosh3', 'adaptive_heun', 'heun3']:
            rtol = 1e-3
            atol = 1e-6
        elif self.odesmethod in ['explicit_adams', 'implicit_adams', 'fixed_adams']:
            rtol = 1e-6
            atol = 1e-9

        ode_solver = torchdiffeq.odeint
        if len(inputs.shape) == 2:
            # Set the initial conditions
            y0 = torch.stack([self.s_snow[0], self.s_water[0]], dim=0).unsqueeze(0)
            y = ode_solver(self.hybrid_model_mlp, y0=y0, t=self.time_series, method=self.odesmethod, rtol=rtol, atol=atol)   # 'rk4' 'midpoint'   'euler' 'dopri5' #rtol=1e-6, atol=1e-6
            # y = ode_solver(self.hybrid_model, y0=y0, t=time_series, method='rk4', rtol=1e-3, atol=1e-6)
        elif len(inputs.shape) == 3:
            # Set the initial conditions
            y0 = torch.stack([self.s_snow[0, -1], self.s_water[0, -1]], dim=0).unsqueeze(0)   #.to(self.device)
            y = ode_solver(self.hybrid_model_lstm, y0=y0, t=self.time_series[self.window_size-1:], method=self.odesmethod, rtol=rtol, atol=atol)

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

            # Relu the state variables
            s_snow_nn = torch.maximum(s_snow_nn, torch.tensor(0.0))    #.to(self.device)
            s_water_nn = torch.maximum(s_water_nn, torch.tensor(0.0))  #.to(self.device)

            # Extract the window_size-1 first elements from the LSTM state variables
            s_snow_first_batch = self.s_snow[0, :-1]
            s_water_first_batch = self.s_water[0, :-1]

            # Concatenate along the last dimension
            # print(s_snow_first_batch.device, s_snow_nn.device)
            s_snow_combined = torch.cat((s_snow_first_batch, s_snow_nn), dim=0)
            s_water_combined = torch.cat((s_water_first_batch, s_water_nn), dim=0)

            # Create LSTM sequences using self.window_size
            s_snow_nn = self.create_sequences(s_snow_combined)
            s_water_nn = self.create_sequences(s_water_combined)

            # Stack tensors to create inputs for the neural network
            inputs_nn = torch.stack([s_snow_nn, s_water_nn, self.precp_series, self.tmean_series], dim=-1)   #.to(self.device)

        #!!!!!!!!This output is already in log space - has to be converted back to normal space when the model is called
        # Assuming nnmodel returns a tensor of shape (batch_size, numOfVars)
        # output = self.pretrainer.nnmodel(inputs_nn, [self.basin], use_grad=self.use_grad)
        if self.pretrainer.nnmodel.include_static:
            output = self.pretrainer.nnmodel(inputs_nn.to(self.device), [self.basins], 
                                    static_inputs=self.pretrainer.nnmodel.torch_static[self.basins])
        else:
            output = self.pretrainer.nnmodel(inputs_nn.to(self.device), [self.basins])

        # Extract the last variable (last column) from the output
        q_output = output[:, -1]

        # print(f"Memory usage after forward pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # aux = input("Press Enter to continue...")

        return q_output

    def hybrid_model_mlp(self, t, y):

        # # Flush time
        # sys.stdout.write(f"t \r{t}")
        # sys.stdout.flush()
         
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

        basins = self.basins
        if not isinstance(basins, list):
            basins = [basins]

        # m100_outputs = self.pretrainer.nnmodel(inputs_nn, basin)[0] #.to(self.device)
        # Forward pass
        if self.pretrainer.nnmodel.include_static:
            m100_outputs = self.pretrainer.nnmodel(inputs_nn.to(self.device), basins, 
                                        static_inputs=self.pretrainer.nnmodel.torch_static[basins[0]])
        else:
            m100_outputs = self.pretrainer.nnmodel(inputs_nn.to(self.device), basins)

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

        # print(f"Initial memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB - {t}d")
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        s0 = y[..., 0]  #.to(self.device)
        s1 = y[..., 1]  #.to(self.device)
        # print(f"Memory usage after unpacking state variables: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Interpolate the input variables
        t_np = t.detach().cpu().numpy()
        precp = self.precp_interp(t_np, extrapolate='periodic')
        temp = self.temp_interp(t_np, extrapolate='periodic')
        lday = self.lday_interp(t_np, extrapolate='periodic')
        # print(f"Memory usage after interpolation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Convert to tensor
        precp = torch.tensor(precp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        temp = torch.tensor(temp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        lday = torch.tensor(lday, dtype=self.data_type_torch).to(self.device)
        # print(f"Memory usage after converting to tensors: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


        # # Find left index for the interpolation
        # idx = torch.searchsorted(self.time_series, t, side='right') - 1
        # idx = idx.clamp(max=self.time_series.size(0) - 2)  # Ensure indices do not exceed valid range

        # # Prepare entries for LSTM (find the idx position in the 2nd dim and substitute the last value by the new value)
        # precp_nn = self.precp_series[idx - self.window_size]
        # precp_nn[-1] = precp
        # temp_nn = self.tmean_series[idx - self.window_size]
        # temp_nn[-1] = temp

        # Prepare lstm inputs. Remove the first element and add the new value to the last entry
        self.s_snow_lstm = torch.cat((self.s_snow_lstm[1:], s0), dim=0)
        self.s_water_lstm = torch.cat((self.s_water_lstm[1:], s1), dim=0)
        self.precp_lstm = torch.cat((self.precp_lstm[1:], precp), dim=0)
        self.tmean_lstm = torch.cat((self.tmean_lstm[1:], temp), dim=0)
        # print(f"Memory usage after preparing LSTM inputs: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Compute ET from the pretrainer.nnmodel
        inputs_nn = torch.stack([self.s_snow_lstm, self.s_water_lstm, self.precp_lstm, self.tmean_lstm], dim=-1)  
        # print(f"Memory usage after stacking inputs: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        basin = self.basin

        basin = self.basin
        if not isinstance(basin, list):
            basin = [basin]
        # print(f"Memory usage after preparing basin: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        m100_outputs = self.pretrainer.nnmodel(inputs_nn, basin, use_grad=self.use_grad)[0] 
        # print(f"Memory usage after nnmodel output: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Target variables:  Psnow, Prain, M, ET and, Q
        p_snow, p_rain, m, et, q = m100_outputs[0], m100_outputs[1], m100_outputs[2], \
                                   m100_outputs[3], m100_outputs[4]
        # print(f"Memory usage after unpacking nnmodel outputs: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Scale back to original values for the ODEs and Relu the Mechanism Quantities    
        if self.cfg.scale_target_vars:
            p_snow = torch.relu(torch.sinh(p_snow) * self.step_function(-temp[0]))
            p_rain = torch.relu(torch.sinh(p_rain))
            m = torch.relu(self.step_function(s0) * torch.sinh(m))
            et = self.step_function(s1) * torch.exp(et) * lday
            q = self.step_function(s1) * torch.exp(q) 
        # print(f"Memory usage after scaling back variables: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        # # print(f"Memory usage after ODE solver: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # print(f"Memory usage before returning results: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        result = torch.stack([ds0_dt, ds1_dt], dim=-1)
        # print(f"Memory usage after returning results: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Clear temporary variables
        del s0, s1, precp, temp, lday, inputs_nn, m100_outputs, p_snow, p_rain, m, et, q, ds0_dt, ds1_dt
        torch.cuda.empty_cache()
        # print(f"Memory usage after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # aux = input("Press Enter to continue...")

        return result

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
