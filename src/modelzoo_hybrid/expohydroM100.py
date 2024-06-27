
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

        s_snow = inputs[:, 0]
        s_water = inputs[:, 1]
        self.precp_series = inputs[:, 2]
        self.tmean_series = inputs[:, 3]
        # self.lday_series = inputs[:, 4]
        self.time_series = inputs[:, 4]

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

        # Set the initial conditions
        y0 = torch.stack([s_snow[0], s_water[0]], dim=0).unsqueeze(0)

        # # Profile the ODE solver
        # time_start = time.time()

        # ode_solver = torchdiffeq.odeint
        # y = ode_solver(self.hybrid_model, y0=y0, t=self.time_series, method=self.odesmethod, rtol=1e-3, atol=1e-6)   # 'rk4' 'midpoint'   'euler' 'dopri5' #rtol=1e-6, atol=1e-6
        # # y = ode_solver(self.hybrid_model, y0=y0, t=time_series, method='rk4', rtol=1e-3, atol=1e-6)

        # Initialize your hybrid model and ODE solver
        # ode_solver =  torchdiffeq.odeint_adjoint
        ode_solver =  torchdiffeq.odeint
        hybrid_model = ExpHydroODEs(
            # self.precp_series,
            # self.tmean_series,
            # self.lday_series,
            # self.time_series,

            self.precp_interp,
            self.temp_interp,
            self.lday_interp,
            self.data_type_torch,
            self.device,

            self.scale_target_vars,
            self.pretrainer,
            self.basin,
            self.step_function
        )
        y = ode_solver(hybrid_model, y0=y0, t=self.time_series, method=self.odesmethod, rtol=1e-3, atol=1e-6)   # 'rk4' 'midpoint'   'euler' 'dopri5' #rtol=1e-6, atol=1e-6
        # y = ode_solver(hybrid_model, y0=y0, t=self.time_series, method='rk4', rtol=1e-4, atol=1e-4)

        # print(f'Time taken for ODE solver: {round(time.time() - time_start, 2)} seconds')
        # aux = input("Press Enter to continue...")
        
        # Update the state variables from the ODE solution
        s_snow_nn  = y[:, 0, 0]
        s_water_nn = y[:, 0, 1]
        # Relu the state variables
        s_snow_nn = torch.maximum(s_snow_nn, torch.tensor(0.0)).to(self.device)
        s_water_nn = torch.maximum(s_water_nn, torch.tensor(0.0)).to(self.device)

        # # Convert other series to tensors if they are not already
        # tmean_series_tensor = torch.tensor(tmean_series, dtype=torch.float32) if not isinstance(tmean_series, torch.Tensor) else tmean_series
        # precp_series_tensor = torch.tensor(precp_series, dtype=torch.float32) if not isinstance(precp_series, torch.Tensor) else precp_series

        # print(s_snow_new.device, s_water_new.device, tmean_series_tensor.device, precp_series_tensor.device)

        # Stack tensors to create inputs for the neural network
        inputs_nn = torch.stack([s_snow_nn, s_water_nn, self.precp_series, self.tmean_series], dim=-1)

        #!!!!!!!!This output is already in log space - has to be converted back to normal space when the model is called
        # Assuming nnmodel returns a tensor of shape (batch_size, numOfVars)
        # output = self.pretrainer.nnmodel(inputs_nn.to(self.device), [self.basin]) #.to(self.device)
        output = self.pretrainer.nnmodel(inputs_nn, [self.basin])
        # output = self.pretrainer.nnmodel(inputs_nn)

        # Extract the last variable (last column) from the output
        q_output = output[:, -1]

        return q_output

    def hybrid_model(self, t, y):

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

        basin = self.basin
        if not isinstance(basin, list):
            basin = [basin]

        m100_outputs = self.pretrainer.nnmodel(inputs_nn, basin)[0] #.to(self.device)
        # m100_outputs = self.pretrainer.nnmodel(inputs_nn)[0]

        # print('inputs_nn', inputs_nn)
        # print('t', t)
        # print('s0', s0)
        # print('s1', s1)
        # print('precp', precp)
        # print('temp', temp)
        # print('lday', lday)
        # print('m100_outputs.shape', m100_outputs.shape)
        # print('m100_outputs', m100_outputs)
        # aux = input("Press Enter to continue AAA...")

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

            # p_snow = torch.relu(torch.sinh(p_snow))
            # p_rain = torch.relu(torch.sinh(p_rain))
            # m = torch.relu(torch.sinh(m))
            # et = torch.relu(torch.exp(et) * lday)
            # q = torch.relu(torch.exp(q))

    

        # # # Relu the Mechanism Quantities
        # # p_snow = torch.relu(p_snow)
        # # p_rain = torch.relu(p_rain)
        # # m = torch.relu(m)
        # # et = torch.relu(et)
        # # q = torch.relu(q)

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        # # Convert to numpy
        # ds0_dt = ds0_dt.cpu().detach().numpy()
        # ds1_dt = ds1_dt.cpu().detach().numpy()

        return torch.stack([ds0_dt, ds1_dt], dim=-1)


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
