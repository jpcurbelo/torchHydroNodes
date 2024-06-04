
import xarray
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp as sp_solve_ivp
import torch
from tqdm import tqdm
import sys
import torch.nn as nn

from src.utils.metrics import loss_name_func_dict
from src.modelzoo_hybrid.basemodel import BaseHybridModel
from src.modelzoo_nn.basepretrainer import NNpretrainer
from src.utils.load_process_data import (
    Config,
    ExpHydroCommon,
)

# Ref: exphydro -> https://hess.copernicus.org/articles/26/5085/2022/
class ExpHydroM100(BaseHybridModel, ExpHydroCommon, nn.Module):

    def __init__(self,
                 cfg: Config,
                 pretrainer: NNpretrainer,
                 ds: xarray.Dataset,
    ):
        BaseHybridModel.__init__(self, cfg, pretrainer, ds)  # Initialize BaseHybridModel
        ExpHydroCommon.__init__(self)  # Initialize ExpHydroCommon
        nn.Module.__init__(self)  # Initialize nn.Module

        # Interpolators
        self.interpolators = self.create_interpolator_dict()

        # Parameters per basin
        self.params_dict = self.get_parameters()

        # # Optimizer and scheduler
        # if hasattr(self.cfg, 'optimizer'):
        #     if self.cfg.optimizer.lower() == 'adam':
        #         optimizer_class = torch.optim.Adam
        #     elif self.cfg.optimizer.lower() == 'sgd':
        #         optimizer_class = torch.optim.SGD
        #     else:
        #         raise NotImplementedError(f"Optimizer {self.cfg.optimizer} not implemented")

        #     if hasattr(self.cfg, 'learning_rate'):
        #         if isinstance(self.cfg.learning_rate, float):
        #             self.optimizer = optimizer_class(self.pretrainer.nnmodel.parameters(), lr=self.cfg.learning_rate)
        #             self.scheduler = None
        #         elif isinstance(self.cfg.learning_rate, dict) and \
        #             'initial' in self.cfg.learning_rate and \
        #             'decay' in self.cfg.learning_rate and \
        #             ('decay_step_fraction' in self.cfg.learning_rate and \
        #                 self.cfg.learning_rate['decay_step_fraction'] <= self.epochs):
        #                 self.optimizer = optimizer_class(self.pretrainer.nnmodel.parameters(), lr=self.cfg.learning_rate['initial'])
        #                 # Learning rate scheduler
        #                 self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
        #                                                                 step_size=self.epochs // self.cfg.learning_rate['decay_step_fraction'],
        #                                                                 gamma=self.cfg.learning_rate['decay'])
        #         else:
        #             raise ValueError("Learning rate not specified correctly in the config (should be a float or a dictionary" +
        #                 "with 'initial', 'decay', and 'decay_step_fraction' keys) and " +
        #                 "'decay_step_fraction' can be at most equal to the number of epochs")
        #     else:
        #         self.optimizer = optimizer_class(self.pretrainer.nnmodel.parameters(), lr=0.001)
        #         self.scheduler = None
        # else:
        #     self.optimizer = torch.optim.Adam(self.pretrainer.nnmodel.parameters(), lr=0.001)
        #     self.scheduler = None

        # # Loss function setup
        # try:
        #     # Try to get the loss function name from configuration
        #     loss_name = self.cfg.loss
        #     self.loss = loss_name_func_dict[loss_name.lower()]
        # except KeyError:
        #     # Handle the case where the loss name is not recognized
        #     raise NotImplementedError(f"Loss function {loss_name} not implemented")
        # except ValueError:
        #     # Handle the case where 'loss' is not specified in the config
        #     # Optionally, set a default loss function
        #     print("Warning! (Inputs): 'loss' not specified in the config. Defaulting to MSELoss.")
        #     self.loss = torch.nn.MSELoss()
    

    def forward(self, inputs, basin):

        s_snow = inputs[:, 0]
        s_water = inputs[:, 1]
        tmean_series = inputs[:, 2]
        precp_series = inputs[:, 3]
        time_series = inputs[:, 4]

        basin_params = self.params_dict[basin]

        self.precp_interp = self.interpolators[basin]['prcp']
        self.temp_interp = self.interpolators[basin]['tmean']
        self.lday_interp = self.interpolators[basin]['dayl']

        # Set the initial conditions
        y0 = np.array([basin_params[0], basin_params[1]])

        # Set the parameters
        params = tuple(basin_params[2:]) + (basin,)

        # Move tensors to CPU and convert to numpy
        time_series_cpu = time_series.cpu().numpy()
        precp_series_cpu = precp_series.cpu().numpy()

        # Now use the CPU-based numpy arrays
        y = sp_solve_ivp(self.hybrid_model, t_span=(time_series_cpu[0], time_series_cpu[-1]), y0=y0, t_eval=time_series_cpu, 
                        args=params, 
                        method=self.odesmethod,
                        # method='DOP853',
                        # rtol=1e-9, atol=1e-12,
                        )
        
        # Update the state variables from the ODE solution
        s_snow_new = torch.tensor(y.y[0], dtype=torch.float32)
        s_water_new = torch.tensor(y.y[1], dtype=torch.float32)
        # Relu the state variables
        s_snow_new = torch.maximum(s_snow_new, torch.tensor(0.0)).to(self.device)
        s_water_new = torch.maximum(s_water_new, torch.tensor(0.0)).to(self.device)

        # Convert other series to tensors if they are not already
        tmean_series_tensor = torch.tensor(tmean_series, dtype=torch.float32) if not isinstance(tmean_series, torch.Tensor) else tmean_series
        precp_series_tensor = torch.tensor(precp_series, dtype=torch.float32) if not isinstance(precp_series, torch.Tensor) else precp_series

        # print(s_snow_new.device, s_water_new.device, tmean_series_tensor.device, precp_series_tensor.device)

        # Stack tensors to create inputs for the neural network
        inputs_nn = torch.stack([s_snow_new, s_water_new, tmean_series_tensor, precp_series_tensor], dim=-1)

        #!!!!!!!!This output is already in log space - has to be converted back to normal space when the model is called
        basin = [basin] if not isinstance(basin, list) else basin
        q_output = self.pretrainer.nnmodel(inputs_nn.to(self.device), basin).to(self.device)[0]

        return q_output

    def hybrid_model(self, t, y, f, smax, qmax, df, tmax, tmin, basin):

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
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        y = torch.tensor(y, dtype=self.data_type_torch)
        s0 = y[0].unsqueeze(0).to(self.device)
        s1 = y[1].unsqueeze(0).to(self.device)

        # Interpolate the input variables
        precp = self.precp_interp(t, extrapolate='periodic')
        temp = self.temp_interp(t, extrapolate='periodic')
        lday = self.lday_interp(t, extrapolate='periodic')

        # Convert to tensor
        precp = torch.tensor(precp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        temp = torch.tensor(temp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)

        # Compute ET from the pretrainer.nnmodel
        inputs_et = torch.stack([s0, s1, temp, precp], dim=-1)

        if not isinstance(basin, list):
            basin = [basin]
        
        m100_outputs = self.pretrainer.nnmodel(inputs_et, basin).to(self.device)[0]
        # Target variables:  Psnow, M, Prain, ET and, Q
        p_snow, p_rain, m, et, q = m100_outputs[0], m100_outputs[1], m100_outputs[2], \
                                   m100_outputs[3], m100_outputs[4]
        
        # Relu the Mechanism Quantities
        p_snow = torch.relu(p_snow)
        p_rain = torch.relu(p_rain)
        m = torch.relu(m)
        et = torch.relu(et)
        q = torch.relu(q)

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        # Convert to numpy
        ds0_dt = ds0_dt.cpu().detach().numpy()
        ds1_dt = ds1_dt.cpu().detach().numpy()

        return [ds0_dt, ds1_dt]
