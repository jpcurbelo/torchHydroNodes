
import xarray
from scipy.integrate import solve_ivp as sp_solve_ivp
import torch
import torch.nn as nn
## https://github.com/rtqichen/torchdiffeq/tree/master
import torchdiffeq

from src.modelzoo_hybrid.basemodel import BaseHybridModel
from src.modelzoo_nn.basepretrainer import NNpretrainer
from src.utils.load_process_data import (
    Config,
    ExpHydroCommon,
    # ExpHydroODEs,
)

MASS_BALANCE_TOLERANCE = 1e-6 #np.sqrt(np.finfo(float).eps)
FIXED_METHODS = ['euler', 'rk4', 'midpoint']

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
    

    def forward(self, inputs, basin, use_grad=True):

        self.use_grad = use_grad

        # print('use_grad', use_grad)
        # print(f"IN - Memory usage before forward pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Extract the state variables
        # If inputs shape is 2D, then is 'mlp' model
        if len(inputs.shape) == 2:  # For MLP models (inputs are 2D):
            self.s_snow = inputs[:, 0]
            self.s_water = inputs[:, 1]
            # self.precp_series = inputs[:, 2]
            # self.tmean_series = inputs[:, 3]
            self.nn_inputs = {}
            for i, var in enumerate(self.pretrainer.input_var_names[2:]):
                self.nn_inputs[var] = inputs[:, i + 2]
            self.time_series = inputs[:, -1]
        # If inputs shape is 3D, then is 'lstm' model
        elif len(inputs.shape) == 3:
            self.window_size = inputs.shape[1]
            self.s_snow = inputs[:, :, 0]
            self.s_water = inputs[:, :, 1]
            # self.precp_series = inputs[:, :, 2]
            # self.tmean_series = inputs[:, :, 3]
            # self.time_series_lstm = inputs[:, :, 4]
            self.nn_inputs = {}
            for i, var in enumerate(self.pretrainer.input_var_names[2:]):
                self.nn_inputs[var] = inputs[:, :, i + 2]
            self.time_series_lstm = inputs[:, :, -1]

            # Flatten self.time_series to have a 1D vector containing the time idxs values
            self.time_series, _ = torch.unique(self.time_series_lstm.flatten(), sorted=True, return_inverse=True)
            # Transfer to device
            self.time_series = self.time_series.to(self.device)

            self.s_snow_lstm = self.s_snow[0, :-1]
            self.s_water_lstm = self.s_water[0, :-1]
            # self.precp_lstm = self.precp_series[0, :-1]
            # self.tmean_lstm = self.tmean_series[0, :-1]
            self.nn_inputs_lstm = {}
            for var in self.nn_inputs:
                self.nn_inputs_lstm[var] = self.nn_inputs[var][0, :-1]

            # Add the first value of the sequence at the begining of the tensor
            self.s_snow_lstm = torch.cat((self.s_snow_lstm[0].unsqueeze(0), self.s_snow_lstm), dim=0)   #.to(self.device)
            self.s_water_lstm = torch.cat((self.s_water_lstm[0].unsqueeze(0), self.s_water_lstm), dim=0)   #.to(self.device)
            # self.precp_lstm = torch.cat((self.precp_lstm[0].unsqueeze(0), self.precp_lstm), dim=0)   #.to(self.device)
            # self.tmean_lstm = torch.cat((self.tmean_lstm[0].unsqueeze(0), self.tmean_lstm), dim=0)    #.to(self.device)
            for var in self.nn_inputs_lstm:
                self.nn_inputs_lstm[var] = torch.cat((self.nn_inputs_lstm[var][0].unsqueeze(0), self.nn_inputs_lstm[var]), dim=0)

        # Make basin global to be used in hybrid_model
        self.basin = basin


        # print('self.pretrainer.input_var_names:', self.model.pretrainer.input_var_names)
        # print('inputs:', inputs.shape)
        # print(inputs[:3, :])
        # print(inputs[-3:, :])
        # print('targets:', targets.shape)
        # print(targets[:3, :])
        # print(targets[-3:, :])
        # aux = input('Press Enter to continue...')

        # print('self.pretrainer.input_var_names[2:]', self.pretrainer.input_var_names[2:])
        # print('self.nn_inputs:', self.nn_inputs.keys())
        # for var in self.nn_inputs:
        #     print(var, self.nn_inputs[var][:3])
        #     print(var, self.nn_inputs[var][-3:])
        # aux = input('Press Enter to continue...')



        # # Set the interpolators 
        # # self.precp_interp = self.interpolators[self.basin]['prcp']
        # # self.temp_interp = self.interpolators[self.basin]['tmean']
        # # self.lday_interp = self.interpolators[self.basin]['dayl']


        # # Profile the ODE solver
        # time_start = time.time()

        # Set the options for the ODE solver
        if self.odesmethod in FIXED_METHODS:
            options = {"step_size": self.time_step, "interp": "cubic"}
        else:
            options = {}

        ode_solver = torchdiffeq.odeint
        # print("About to call the ODE solver")
        if len(inputs.shape) == 2:  # For MLP models (inputs are 2D):
            # Set the initial conditions
            y0 = torch.stack([self.s_snow[0], self.s_water[0]], dim=0).unsqueeze(0)    #.to(self.device)
            y = ode_solver(self.hybrid_model_mlp, y0=y0, t=self.time_series, method=self.odesmethod, 
                           rtol=self.rtol, atol=self.atol, options=options)   # 'rk4' 'midpoint'   'euler' 'dopri5' #rtol=1e-6, atol=1e-6
            # y = ode_solver(self.hybrid_model, y0=y0, t=time_series, method='rk4', rtol=1e-3, atol=1e-6)
        elif len(inputs.shape) == 3:
            # Set the initial conditions
            y0 = torch.stack([self.s_snow[0, -1], self.s_water[0, -1]], dim=0).unsqueeze(0)    #.to(self.device)
            y = ode_solver(self.hybrid_model_lstm, y0=y0, t=self.time_series[self.window_size-1:], method=self.odesmethod, 
                           rtol=self.rtol, atol=self.atol, options=options)

        if len(inputs.shape) == 2:  # For MLP models (inputs are 2D):

            s_snow_nn  = y[:, 0, 0]
            s_water_nn = y[:, 0, 1]

            # print('s_snow_nn:', s_snow_nn.shape)
            # print('s_water_nn:', s_water_nn.shape)

            # for var in self.nn_inputs:
            #     print(var, self.nn_inputs[var].shape)
            
            # Stack tensors to create inputs for the neural network
            # inputs_nn = torch.stack([s_snow_nn, s_water_nn, self.precp_series, self.tmean_series], dim=-1)
            inputs_nn = torch.stack([s_snow_nn, s_water_nn] + [self.nn_inputs[var] for var in self.nn_inputs], dim=-1)   #.to(self.device)

            # print('inputs_nn:', inputs_nn.shape)
            # aux = input('Press Enter to continue...')

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
            # inputs_nn = torch.stack([s_snow_nn, s_water_nn, self.precp_series, self.tmean_series], dim=-1)   #.to(self.device)
            inputs_nn = torch.stack([s_snow_nn, s_water_nn] + [self.nn_inputs[var] for var in self.nn_inputs], dim=-1)   #.to(self.device)


        # # Measure memory usage before and after creating q_output
        # print("Memory usage before creating q_output:")
        # memory_before = torch.cuda.memory_allocated(self.device)
        # print(f"Memory Allocated: {memory_before / (1024 ** 2):.2f} MB")

        #!!!!!!!!This output is already in log space - has to be converted back to normal space when the model is called
        # Assuming nnmodel returns a tensor of shape (batch_size, numOfVars)
        # output = self.pretrainer.nnmodel(inputs_nn, [self.basin], use_grad=self.use_grad)
        # print("About to call the NN model - end of ODE solver")
        if self.pretrainer.nnmodel.include_static:
            q_output = self.pretrainer.nnmodel(inputs_nn, self.basin, 
                                    static_inputs=self.pretrainer.nnmodel.torch_static[self.basin], 
                                    use_grad=self.use_grad)[:, -1]
        else:
            q_output = self.pretrainer.nnmodel(inputs_nn, self.basin, 
                                    use_grad=self.use_grad)[:, -1]
            
        # print("Memory usage after creating q_output:")
        # memory_after = torch.cuda.memory_allocated(self.device)
        # print(f"Memory Allocated: {memory_after / (1024 ** 2):.2f} MB")

        # Clear the cache
        torch.cuda.empty_cache()

        # Return the last variable (last column) from the output + the state variables
        return q_output, s_snow_nn, s_water_nn
    
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
        input_list = [self.interpolators[self.basin][var](t_np, extrapolate='periodic') for var in self.pretrainer.input_var_names[2:]]

        # Convert to tensor
        input_tensors = [torch.tensor(input_, dtype=self.data_type_torch).unsqueeze(0).to(self.device) for input_ in input_list]
        
        # Stack tensors to create inputs for the neural network
        inputs_nn = torch.stack([s0, s1] + input_tensors, dim=-1)

        # Forward pass
        if self.pretrainer.nnmodel.include_static:
            m100_outputs = self.pretrainer.nnmodel(inputs_nn.to(self.device), self.basin, 
                                        static_inputs=self.pretrainer.nnmodel.torch_static[self.basin],
                                        use_grad=self.use_grad)[0]
        else:
            m100_outputs = self.pretrainer.nnmodel(inputs_nn.to(self.device), self.basin,
                                        use_grad=self.use_grad)[0]

        # Unpacking target variables from m100_outputs list
        p_snow, p_rain, m, et, q = m100_outputs  # Psnow, Prain, M, ET, Q
        
        # Scale back to original values for the ODEs and Relu the Mechanism Quantities    
        if self.cfg.scale_target_vars:
            # Find index of temp in the input_var_names
            # If tmean is not in the input_var_names, then the find tmin and tmax and calculate tmean
            if 'tmean' in self.pretrainer.input_var_names:
                idx_temp = self.pretrainer.input_var_names[2:].index('tmean')
                temp = input_tensors[idx_temp]
            elif 'tmin' in self.pretrainer.input_var_names and 'tmax' in self.pretrainer.input_var_names:
                idx_tmin = self.pretrainer.input_var_names[2:].index('tmin')
                idx_tmax = self.pretrainer.input_var_names[2:].index('tmax')
                temp = (input_tensors[idx_tmin] + input_tensors[idx_tmax]) / 2
            else:
                raise ValueError("Temperature variable not found in the input_var_names - tmean or tmin and tmax")
            
            # Find lday from interterpolators
            lday = self.interpolators[self.basin]['dayl'](t_np, extrapolate='periodic')
            lday = torch.tensor(lday, dtype=self.data_type_torch).unsqueeze(0).to(self.device)

            p_snow = torch.relu(torch.sinh(p_snow) * self.step_function(-temp[0]))
            p_rain = torch.relu(torch.sinh(p_rain))
            m = torch.relu(self.step_function(s0) * torch.sinh(m))
            et = self.step_function(s1) * torch.exp(et) * lday
            q = self.step_function(s1) * torch.exp(q) 

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        # print(f"OUT-Memory usage after converting to tensors: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # aux = input("Press Enter to continue...")

        # Clear temporary variables
        # del s0, s1, precp, temp, lday, inputs_nn, m100_outputs, p_snow, p_rain, m, et, q
        del s0, s1, input_tensors, inputs_nn, m100_outputs, p_snow, p_rain, m, et, q
        # Clear the cache
        torch.cuda.empty_cache()

        return torch.stack([ds0_dt, ds1_dt], dim=-1)

    def hybrid_model_lstm(self, t, y):

        # # Flush time
        # sys.stdout.write(f"t \r{t}")
        # sys.stdout.flush()

        # print(f"Initial memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB - {t}d")
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        s0 = y[..., 0]    #.to(self.device)
        s1 = y[..., 1]    #.to(self.device)
        # print(f"Memory usage after unpacking state variables: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Interpolate the input variables
        t_np = t.detach().cpu().numpy()
        input_list = [self.interpolators[self.basin][var](t_np, extrapolate='periodic') for var in self.pretrainer.input_var_names[2:]]

        # Convert to tensor
        input_tensors = [torch.tensor(input_, dtype=self.data_type_torch).unsqueeze(0).to(self.device) for input_ in input_list]

        # Prepare lstm inputs. Remove the first element and add the new value to the last entry
        self.s_snow_lstm = torch.cat((self.s_snow_lstm[1:], s0), dim=0)
        self.s_water_lstm = torch.cat((self.s_water_lstm[1:], s1), dim=0)
        for i, var in enumerate(self.nn_inputs_lstm):
            self.nn_inputs_lstm[var] = torch.cat((self.nn_inputs_lstm[var][1:], input_tensors[i]), dim=0)

        # Stack tensors to create inputs for the neural network
        inputs_nn = torch.stack([self.s_snow_lstm, self.s_water_lstm] + [self.nn_inputs_lstm[var] for var in self.nn_inputs_lstm], dim=-1)   #.to(self.device)

        # Forward pass
        if self.pretrainer.nnmodel.include_static:
            m100_outputs = self.pretrainer.nnmodel(inputs_nn.to(self.device), self.basin, 
                                        static_inputs=self.pretrainer.nnmodel.torch_static[self.basin],
                                        use_grad=self.use_grad)[0]
        else:
            m100_outputs = self.pretrainer.nnmodel(inputs_nn.to(self.device), self.basin,
                                        use_grad=self.use_grad)[0]

        # Target variables:  Psnow, Prain, M, ET and, Q
        p_snow, p_rain, m, et, q = m100_outputs[0], m100_outputs[1], m100_outputs[2], \
                                   m100_outputs[3], m100_outputs[4]
        
        # Scale back to original values for the ODEs and Relu the Mechanism Quantities    
        if self.cfg.scale_target_vars:
            # Find index of temp in the input_var_names
            # If tmean is not in the input_var_names, then the find tmin and tmax and calculate tmean
            if 'tmean' in self.pretrainer.input_var_names:
                idx_temp = self.pretrainer.input_var_names[2:].index('tmean')
                temp = input_tensors[idx_temp]
            elif 'tmin' in self.pretrainer.input_var_names and 'tmax' in self.pretrainer.input_var_names:
                idx_tmin = self.pretrainer.input_var_names[2:].index('tmin')
                idx_tmax = self.pretrainer.input_var_names[2:].index('tmax')
                temp = (input_tensors[idx_tmin] + input_tensors[idx_tmax]) / 2
            else:
                raise ValueError("Temperature variable not found in the input_var_names - tmean or tmin and tmax")
            
            # Find lday from interterpolators
            lday = self.interpolators[self.basin]['dayl'](t_np, extrapolate='periodic')
            lday = torch.tensor(lday, dtype=self.data_type_torch).unsqueeze(0).to(self.device)

            p_snow = torch.relu(torch.sinh(p_snow) * self.step_function(-temp[0]))
            p_rain = torch.relu(torch.sinh(p_rain))
            m = torch.relu(self.step_function(s0) * torch.sinh(m))
            et = self.step_function(s1) * torch.exp(et) * lday
            q = self.step_function(s1) * torch.exp(q) 
        # print(f"Memory usage after scaling back variables: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Clear temporary variables
        del s0, s1, input_tensors, inputs_nn, m100_outputs

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q


        # # Check mass balance
        # total_input = p_snow + p_rain
        # total_output = et + q
        # change_in_storage = ds0_dt + ds1_dt
        # mass_balance_error = (total_input - total_output - change_in_storage).item()

        # # Print or log mass balance error
        # if abs(mass_balance_error) > MASS_BALANCE_TOLERANCE:
        #     print(f"Mass balance error at time {t_np}: {abs(mass_balance_error):.2e}")

        # print(f"Memory usage before returning results: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        result = torch.stack([ds0_dt, ds1_dt], dim=-1)

        # Clear temporary variables
        del p_snow, p_rain, m, et, q, ds0_dt, ds1_dt
        torch.cuda.empty_cache()
        # print(f"Memory usage after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

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
