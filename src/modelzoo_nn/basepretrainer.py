import os
import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import time

from src.modelzoo_nn.basemodel import BaseNNModel
from src.datasetzoo.basedataset import BaseDataset
from src.utils.metrics import loss_name_func_dict
from src.utils.load_process_data import (
    # BatchSampler, 
    CustomDatasetToNN,
    BasinBatchSampler,
    ExpHydroCommon,
)
from src.utils.metrics import (
    NSE_eval,
    compute_all_metrics,
)

from src.utils.load_process_data import EarlyStopping

class NNpretrainer(ExpHydroCommon):

    def __init__(self, nnmodel: BaseNNModel, fulldataset: BaseDataset):
                # # loss='mse', lr=0.001, batch_size=-1, epochs=100):
        
        self.fulldataset = fulldataset
        self.nnmodel = nnmodel
        self.cfg = self.nnmodel.concept_model.cfg
        
        self.dataset = self.nnmodel.concept_model.dataset
        self.basins = self.dataset.basin.values

        # # Set random seeds
        # self._set_seeds()

        # Check if log_n_basins exists and is either a positive integer or a non-empty list
        if hasattr(self.cfg, 'log_n_basins') and (
            (isinstance(self.cfg.log_n_basins, int) and self.cfg.log_n_basins > 0) or
            (isinstance(self.cfg.log_n_basins, list) and len(self.cfg.log_n_basins) > 0)
        ):
            if isinstance(self.cfg.log_n_basins, int):
                sample_size = min(len(self.basins), self.cfg.log_n_basins)
                self.basins_to_log = random.sample(list(self.basins), sample_size)
            elif isinstance(self.cfg.log_n_basins, list):
                self.basins_to_log = self.cfg.log_n_basins
        else:
            self.basins_to_log = None
        
        self.log_every_n_epochs = self.cfg.log_every_n_epochs

        # Batch size
        self.batch_size = self.cfg.batch_size

        # Epochs
        if hasattr(self.cfg, 'epochs'):
            self.epochs = self.cfg.epochs
        else:
            self.epochs = 100

        # Device
        self.device = self.nnmodel.device

        # Data type
        self.dtype = self.cfg.precision['torch']

        # Number of workers
        if hasattr(self.cfg, 'num_workers'):
            self.num_workers = self.cfg.num_workers
        else:
            self.num_workers = 8

        # Input/output variables
        self.input_var_names = self.cfg.nn_dynamic_inputs #+ ['dayl']
        # print('self.input_var_names:', self.input_var_names)
        self.output_var_names = self.cfg.nn_mech_targets
        self.target = self.cfg.target_variables[0]

        # Scale the target variables
        if self.cfg.scale_target_vars:
            self.scale_target_vars(is_trainer=False)

        # Create the dataloader
        self.dataloader = self.create_dataloaders()
        self.num_batches = len(self.dataloader)

        # Optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()

        # Loss function setup
        try:
            # Try to get the loss function name from configuration
            loss_name = self.cfg.loss
            self.loss = loss_name_func_dict[loss_name.lower()]
        except KeyError:
            # Handle the case where the loss name is not recognized
            raise NotImplementedError(f"Loss function {loss_name} not implemented")
        except ValueError:
            # Handle the case where 'loss' is not specified in the config
            # Optionally, set a default loss function
            print("Warning! (Inputs): 'loss' not specified in the config. Defaulting to MSELoss.")
            self.loss = torch.nn.MSELoss()

        # ## Pretrainer Initialized
        # # loss_pretrain: mse
        # # lr_pretrain: 0.001
        # # batch_size_pretrain: -1
        # # epochs_pretrain: 100
        # print('Pretrainer Initialized')
        # print(f'loss_pretrain: {self.loss}')
        # print(f'lr_pretrain: {self.optimizer.param_groups[0]["lr"]}')
        # print(f'batch_size_pretrain: {self.batch_size}')
        # print(f'epochs_pretrain: {self.epochs}')

    def setup_optimizer_and_scheduler(self):
        # Determine optimizer class based on configuration
        if hasattr(self.cfg, 'optimizer'):
            optimizer_name = self.cfg.optimizer.lower()
            if optimizer_name == 'adam':
                optimizer_class = torch.optim.Adam
            elif optimizer_name == 'sgd':
                optimizer_class = torch.optim.SGD
            else:
                raise NotImplementedError(f"Optimizer {self.cfg.optimizer} not implemented")
        else:
            optimizer_class = torch.optim.Adam  # Default to Adam if no optimizer is specified

        # Determine learning rate and scheduler
        if hasattr(self.cfg, 'learning_rate'):
            lr = self.cfg.learning_rate
            if isinstance(lr, float):
                optimizer = optimizer_class(self.nnmodel.parameters(), lr=lr)
                scheduler = None
            elif isinstance(lr, dict) and \
                'initial' in lr and \
                'decay' in lr and \
                ('decay_step_fraction' in lr and \
                    lr['decay_step_fraction'] <= self.epochs):
                optimizer = optimizer_class(self.nnmodel.parameters(), lr=lr['initial'])
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                            step_size=self.epochs // lr['decay_step_fraction'],
                                                            gamma=lr['decay'])
            else:
                raise ValueError("Learning rate not specified correctly in the config. "
                                 "It should be a float or a dictionary with 'initial', 'decay', and 'decay_step_fraction' keys. "
                                 "'decay_step_fraction' can be at most equal to the number of epochs.")
        else:
            optimizer = optimizer_class(self.nnmodel.parameters(), lr=0.001)  # Default learning rate if none is specified
            scheduler = None

        return optimizer, scheduler

    def create_dataloaders(self, is_trainer=False):
        '''Create the dataloaders for the pretrainer'''

        # Convert xarray DataArrays to PyTorch tensors and store in a dictionary
        tensor_dict = {var: torch.tensor(self.dataset[var].values, dtype=self.dtype) for var in self.dataset.data_vars}

        # Create a list of input and output tensors based on the variable names
        if is_trainer:
            time_series = self.dataset['date'].values
            time_idx = torch.linspace(0, len(time_series) - 1, len(time_series), dtype=self.dtype)
            # Reshape time_idx to match the shape of other tensors (number_of_basins, time_idx)
            time_idx = time_idx.repeat(self.cfg.number_of_basins, 1)  #.to(self.device)
            input_var_names = self.input_var_names + ['time_idx'] 
            tensor_dict['time_idx'] = time_idx
            # Substitute the last of the output variables with the target variable
            output_var_names = self.output_var_names[:-1] + [self.target]

            # Scale self.target variable
            if self.cfg.scale_target_vars:
                tensor_dict[self.target] = torch.log(tensor_dict[self.target] + 1e-6)

        else:
            input_var_names = self.input_var_names
            output_var_names = self.output_var_names
        
        input_tensors = [tensor_dict[var] for var in input_var_names]
        output_tensors = [tensor_dict[var] for var in output_var_names]

        # Keep basin IDs as a list of strings
        num_dates = len(self.dataset.date)
        basin_ids = [basin for basin in self.basins for _ in range(num_dates)]

        if self.cfg.scale_target_vars:
            self.nnmodel.torch_input_mins = {}
            for idx, basin in enumerate(self.basins):
                target_vals = []
                for outvar in output_var_names:
                    # if outvar == 'q_bucket':
                    #     print('outvar:', outvar, torch.min(tensor_dict[outvar][idx] + np.finfo(float).eps))

                    target_vals.append(torch.min(tensor_dict[outvar][idx] + np.finfo(float).eps))

                self.nnmodel.torch_input_mins[basin] = torch.tensor(target_vals, dtype=self.dtype)

        # Ensure input and output tensors are wrapped into single composite tensors if needed
        if self.cfg.nn_model == 'lstm':

            # Create sequences
            input_tensor = torch.stack(input_tensors, dim=2).permute(1, 0, 2)  # Shape: [num_dates, num_basins, num_vars]
            output_tensor = torch.stack(output_tensors, dim=2).permute(1, 0, 2)  # Shape: [num_dates, num_basins, num_vars]

            # Convert to sequences
            input_sequences, output_sequences, sequence_basin_ids = [], [], []
            for i in range(input_tensor.size(1)):  # Iterate over basins
                for j in range(0, num_dates - self.cfg.seq_length + 1):  # Create sequences
                    input_sequences.append(input_tensor[j:j + self.cfg.seq_length, i, :])
                    output_sequences.append(output_tensor[j + self.cfg.seq_length - 1, i, :])
                    sequence_basin_ids.append(basin_ids[i * num_dates + j])

            input_tensor = torch.stack(input_sequences)   #.to(self.device)  # Shape: [num_sequences, seq_length, num_vars]
            output_tensor = torch.stack(output_sequences)   #.to(self.device)  # Shape: [num_sequences, num_output_vars]
            basin_ids = sequence_basin_ids

        elif self.cfg.nn_model == 'mlp':
            input_tensor = torch.stack(input_tensors, dim=2).view(-1, len(input_var_names)) if \
                len(input_tensors) > 1 else input_tensors[0].view(-1, 1)   #.to(self.device)
            output_tensor = torch.stack(output_tensors, dim=2).view(-1, len(output_var_names)) if \
                len(output_tensors) > 1 else output_tensors[0].view(-1, 1)   #.to(self.device)

        # Ensure that the number of samples (first dimension) is the same across all tensors
        assert input_tensor.shape[0] == output_tensor.shape[0] == len(basin_ids), "Size mismatch between tensors"

        # Create a custom dataset with the input and output tensors and basin IDs
        dataset = CustomDatasetToNN(input_tensor, output_tensor, basin_ids)

        if self.batch_size == -1:
            self.batch_size = num_dates
        
        batch_sampler = BasinBatchSampler(basin_ids, self.batch_size)

        # Create DataLoader with custom batch sampler
        # kwargs = {'pin_memory': pin_memory, 'num_workers': self.num_workers} if 'cuda' in str(self.device) else {}
        kwargs = {}  # Runs faster without pin_memory and num_workers
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)

        return dataloader
    
    def train(self, max_nan_batches=10, loss=None, lr=None, epochs=None, any_log=True):

        # Determine the training behavior based on whether loss, lr, or epochs have been provided
        if loss is not None:
            self.loss = loss_name_func_dict[loss.lower()]
        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr
            self.scheduler = None
        if epochs is not None:
            self.epochs = epochs
        if any([loss, lr, epochs]):
            verbose = False
            disable_pbar = True
            save_model_results = False
        else:
            verbose = self.cfg.verbose
            disable_pbar = self.cfg.disable_pbar
            save_model_results = True

        early_stopping = EarlyStopping(patience=self.cfg.patience)

        if self.cfg.verbose:
            print('-' * 60)
            print(f"-- Pretraining the neural network model -- ({self.device})")
            print('-' * 60)

        nan_count = 0
        for epoch in range(self.epochs):

            # Clear CUDA cache at the beginning of each epoch
            torch.cuda.empty_cache()

            pbar = tqdm(self.dataloader, disable=disable_pbar, file=sys.stdout)
            pbar.set_description(f'# Epoch {epoch + 1:05d}')

            epoch_loss = 0.0
            num_batches_seen = 0

            for (inputs, targets, basin_ids) in pbar:

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                if self.nnmodel.include_static:
                    predictions = self.nnmodel(inputs, basin_ids[0], 
                                            static_inputs=self.nnmodel.torch_static[basin_ids[0]])
                else:
                    predictions = self.nnmodel(inputs, basin_ids[0])

                # print('predictions:', predictions.shape)
                # print(predictions[:4, :])
                # aux = input('Press Enter to continue...')
                
                nan_mask = torch.isnan(predictions)
                if nan_mask.any():
                    nan_count += 1
                    predictions = predictions[~nan_mask]
                    targets = targets[~nan_mask]
                    if nan_count > max_nan_batches:
                        print(f"Exceeded {max_nan_batches} allowed NaN batches. Stopping training.")
                        return False

                loss = self.loss(targets, predictions)
                # print(nan_count, loss)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.nnmodel.parameters(), self.cfg.clip_gradient_norm)

                # Update the weights
                self.optimizer.step()

                # Accumulate the loss
                epoch_loss += loss.item()
                num_batches_seen += 1

                # Update progress bar with current average loss
                avg_loss = epoch_loss / num_batches_seen
                pbar.set_postfix({'Loss': f'{avg_loss:.4e}'})

                # Delete variables to free memory
                del inputs, targets, predictions, loss
                torch.cuda.empty_cache()

            pbar.close()

            if save_model_results and (epoch == 0 or ((epoch + 1) % self.log_every_n_epochs == 0)):
                if verbose and any_log:
                    print(f"-- Saving the model weights and plots (epoch {epoch + 1}) --")
                # Save the model weights
                self.save_model()
                self.save_plots(epoch=epoch+1)

            # Early stopping check
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1} with loss {avg_loss:.4e}")
                break

            # Learning rate scheduler
            if (self.scheduler is not None) and epoch < self.epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                # Check if learning rate has changed
                new_lr = self.optimizer.param_groups[0]['lr']
                if verbose and current_lr != new_lr:
                    print(f"Learning rate updated to {new_lr}")

            # Print the average loss for the epoch if not verbose (pbar is disabled)
            # if not verbose and save_model_results:
            if not verbose and any_log:
                print(f"Epoch {epoch + 1} Loss: {avg_loss:.4e}")

        # Save the final model weights and plots
        if verbose and any_log:
            print("-- Saving final plots --")
        self.save_plots()
        if verbose and any_log:
            print("-- Saving final model weights --")
        self.save_model()
        if save_model_results:
            if verbose and any_log:
                print("-- Evaluating the model --")
            self.evaluate()

        # Reset optimizer and scheduler to the initial learning rate for the next training
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()

        return True

    def save_model(self):
        '''Save the model weights'''

        # Create a directory to save the model weights if it does not exist
        model_dir = 'model_weights'
        model_path = self.cfg.run_dir / model_dir
        model_path.mkdir(parents=True, exist_ok=True)

        # Save the model weights
        torch.save(self.nnmodel.state_dict(), model_path / f'pretrainer_{self.cfg.nn_model}_{len(self.basins)}basins.pth')

    def save_plots(self, epoch=None):
        '''
        Save the plots of the observed and predicted values for a random subset of basins
        and for each dataset period
        
        Args:
            epoch (int): The current epoch number
            
        Returns:
            None
        '''

        # Clear CUDA cache to free memory
        torch.cuda.empty_cache()

        # Extract keys that start with 'ds_'
        ds_periods = [key for key in self.fulldataset.__dict__.keys() if key.startswith('ds_') \
                           and 'static' not in key]

        # Check if log_n_basins exists and is either a positive integer or a non-empty list
        if self.basins_to_log is not None:

            # Progress bar for basins
            pbar_basins = tqdm(self.basins_to_log, disable=self.cfg.disable_pbar, file=sys.stdout)
            for basin in pbar_basins:
                pbar_basins.set_description(f'* Plotting basin {basin}')

                if basin not in self.basins:
                    print(f"Basin {basin} not found in the dataset. Skipping...")
                    continue

                # Create a directory to save the plots if it does not exist
                basin_dir = Path(self.cfg.plots_dir) / basin
                basin_dir.mkdir(parents=True, exist_ok=True)

                for dsp in ds_periods:

                    ds_period = getattr(self.fulldataset, dsp)
                    ds_basin = ds_period.sel(basin=basin)

                    # Get model outputs
                    inputs = self.get_model_inputs(ds_basin, self.input_var_names, basin, is_trainer=False)

                    # Get model outputs
                    # Forward pass
                    if self.nnmodel.include_static:
                        outputs = self.nnmodel(inputs, basin, static_inputs=self.nnmodel.torch_static[basin], use_grad=False)
                    else:
                        outputs = self.nnmodel(inputs, basin, use_grad=False)
                        # outputs = run_job_with_memory_check(self.nnmodel, ds_basin,  self.input_var_names, basin, inputs.shape, inputs.dtype(), use_grad=False)

                    # Reshape outputs
                    outputs = self.reshape_outputs(outputs)

                    # Scale back outputs
                    if self.cfg.scale_target_vars:
                        outputs = self.scale_back_simulated(outputs, ds_basin)

                        # If period is ds_train, also scale back the observed variables
                        if dsp == 'ds_train':
                            ds_basin = self.scale_back_observed(ds_basin)

                    # Save the results as a CSV file
                    period_name = dsp.split('_')[-1]

                    if epoch is None:
                        var_list = self.output_var_names
                    else:
                        var_list = [self.output_var_names[-1]]

                    for vi, var in enumerate(var_list):

                        y_obs = ds_basin[var.lower()].values
                        if epoch is None:
                            y_bucket = outputs[:, vi].detach().cpu().numpy()
                        else:
                            y_bucket = outputs[:, -1].detach().cpu().numpy()

                        plt.figure(figsize=(10, 6))
                        plt.plot(ds_basin.date, y_obs, label='Observed')
                        plt.plot(ds_basin.date, y_bucket, label='Predicted', alpha=0.7)
                        plt.xlabel('Date')
                        plt.ylabel(var)
                        plt.legend()

                        if vi == len(var_list) - 1:
                                nse_val = NSE_eval(y_obs, y_bucket)
                                plt.title(f'{var} - {basin} - {period_name} | $NSE = {nse_val:.3f}$')
                        else:
                            plt.title(f'{var} - {basin} - {period_name}')

                        plt.tight_layout()
                        
                        if epoch is not None:
                            plt.savefig(basin_dir / f'{var}_{basin}_{period_name}_epoch{epoch}.png', dpi=75)
                        else:
                            plt.savefig(basin_dir / f'_{var}_{basin}_{period_name}.png', dpi=75)
                        plt.close('all')

                # Clear CUDA cache to free memory
                torch.cuda.empty_cache()
            
            pbar_basins.close()

    def evaluate(self):

        metrics_dir = self.cfg.run_dir / 'model_metrics'
        if not metrics_dir.exists():
            metrics_dir.mkdir()

        # Extract keys that start with 'ds_'
        ds_periods = [key for key in self.fulldataset.__dict__.keys() if key.startswith('ds_') \
                           and 'static' not in key]

        for dsp in ds_periods:
            ds_period = getattr(self.fulldataset, dsp)
            ds_basins = ds_period.basin.values
            
            results = []
            
            # Progress bar for basins
            pbar_basins = tqdm(ds_basins, disable=self.cfg.disable_pbar, file=sys.stdout)
            for basin in pbar_basins:
                pbar_basins.set_description(f'* Evaluating basin {basin} ({dsp})')

                if basin not in self.basins:
                    print(f"Basin {basin} not found in the dataset. Skipping...")
                    continue
                
                ds_basin = ds_period.sel(basin=basin)
                
                inputs = self.get_model_inputs(ds_basin, self.input_var_names, basin, is_trainer=False)

                # Get model outputs
                # Forward pass
                if self.nnmodel.include_static:
                    outputs = self.nnmodel(inputs, basin, static_inputs=self.nnmodel.torch_static[basin], use_grad=False)
                else:
                    outputs = self.nnmodel(inputs, basin, use_grad=False)

                # Reshape outputs
                outputs = self.reshape_outputs(outputs)

                # Scale back outputs
                if self.cfg.scale_target_vars:
                    outputs = self.scale_back_simulated(outputs, ds_basin)

                    # If period is ds_train, also scale back the observed variables
                    if dsp == 'ds_train':
                        ds_basin = self.scale_back_observed(ds_basin)
            
                # Always the last variable in the list -> target variable
                y_obs = ds_basin[self.output_var_names[-1].lower()].values
                y_bucket = outputs[:, -1].detach().cpu().numpy()
                
                # Extract dates
                dates = ds_basin.date.values

                # Save results to a CSV file
                # Create a DataFrame with the required columns
                results_df = pd.DataFrame({
                    'date': dates,
                    'y_obs': y_obs,
                    'y_sim': y_bucket
                })

                # Save results to a CSV file
                period_name = dsp.split('_')[-1]
                results_file = f'{basin}_results_{period_name}.csv'
                results_file_path = Path(self.cfg.results_dir) / results_file

                # Ensure the results directory exists
                results_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the DataFrame to a CSV file
                results_df.to_csv(results_file_path, index=False)

                # Compute all evaluation metrics
                metrics = compute_all_metrics(y_obs, y_bucket, dates, self.cfg.metrics)
                
                # Store results in a dictionary
                result = {'basin': basin}
                result.update(metrics)
                results.append(result)
            
            # Convert results to a DataFrame
            df_results = pd.DataFrame(results)

            # Sort the DataFrame by basin name
            df_results = df_results.sort_values(by='basin').reset_index(drop=True)
            
            # Save results to a CSV file
            period_name = dsp.split('_')[-1]
            metrics_file = f'evaluation_metrics_{period_name}.csv'
            metrics_file_path = metrics_dir / metrics_file

            # Save the results to a CSV file
            df_results.to_csv(metrics_file_path, index=False)

############################################################################################################