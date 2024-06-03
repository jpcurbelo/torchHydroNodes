import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.modelzoo_nn.basemodel import BaseNNModel
from src.datasetzoo.basedataset import BaseDataset
from src.utils.metrics import loss_name_func_dict
from src.utils.load_process_data import (
    BatchSampler, 
    CustomDatasetToNN,
    BasinBatchSampler,
    ExpHydroCommon,
)
from src.utils.metrics import (
    NSE_eval,
    compute_all_metrics,
)


class NNpretrainer(ExpHydroCommon):

    def __init__(self, nnmodel: BaseNNModel, fulldataset: BaseDataset):
        
        self.fulldataset = fulldataset
        self.nnmodel = nnmodel
        self.cfg = self.nnmodel.concept_model.cfg
        
        self.dataset = self.nnmodel.concept_model.ds
        self.basins = self.dataset.basin.values

        # # Set random seeds
        # self._set_random_seeds()

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
        self.input_var_names = self.cfg.nn_dynamic_inputs
        self.output_var_names = self.cfg.nn_mech_targets

        # Scale the target variables
        self.dataset = self.scale_target_vars()

        # Create the dataloader
        self.dataloader = self.create_dataloaders()
        self.num_batches = len(self.dataloader)

        # Optimizer and scheduler
        if hasattr(self.cfg, 'optimizer'):
            if self.cfg.optimizer.lower() == 'adam':
                optimizer_class = torch.optim.Adam
            elif self.cfg.optimizer.lower() == 'sgd':
                optimizer_class = torch.optim.SGD
            else:
                raise NotImplementedError(f"Optimizer {self.cfg.optimizer} not implemented")

            if hasattr(self.cfg, 'learning_rate'):
                if isinstance(self.cfg.learning_rate, float):
                    self.optimizer = optimizer_class(self.nnmodel.parameters(), lr=self.cfg.learning_rate)
                    self.scheduler = None
                elif isinstance(self.cfg.learning_rate, dict) and \
                    'initial' in self.cfg.learning_rate and \
                    'decay' in self.cfg.learning_rate and \
                    ('decay_step_fraction' in self.cfg.learning_rate and \
                        self.cfg.learning_rate['decay_step_fraction'] <= self.epochs):
                        self.optimizer = optimizer_class(self.nnmodel.parameters(), lr=self.cfg.learning_rate['initial'])
                        # Learning rate scheduler
                        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                        step_size=self.epochs // self.cfg.learning_rate['decay_step_fraction'],
                                                                        gamma=self.cfg.learning_rate['decay'])
                else:
                    raise ValueError("Learning rate not specified correctly in the config (should be a float or a dictionary" +
                        "with 'initial', 'decay', and 'decay_step_fraction' keys) and " +
                        "'decay_step_fraction' can be at most equal to the number of epochs")
            else:
                self.optimizer = optimizer_class(self.nnmodel.parameters(), lr=0.001)
                self.scheduler = None
        else:
            self.optimizer = torch.optim.Adam(self.nnmodel.parameters(), lr=0.001)
            self.scheduler = None

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

    def create_dataloaders(self, trainer=False):
        '''Create the dataloaders for the pretrainer'''

        # Convert xarray DataArrays to PyTorch tensors and store in a dictionary
        tensor_dict = {var: torch.tensor(self.dataset[var].values, dtype=self.dtype) for var in self.dataset.data_vars}

        print(self.dataset)

        print(self.input_var_names)

        # Create a list of input and output tensors based on the variable names
        input_var_names = self.input_var_names + ['time_series'] if trainer else self.input_var_names
        input_tensors = [tensor_dict[var] for var in input_var_names]
        output_tensors = [tensor_dict[var] for var in self.output_var_names]

        # Keep basin IDs as a list of strings
        num_dates = len(self.dataset.date)
        basin_ids = [basin for basin in self.basins for _ in range(num_dates)]

        # Ensure input and output tensors are wrapped into single composite tensors if needed
        input_tensor = torch.stack(input_tensors, dim=2).view(-1, len(self.input_var_names)) if \
            len(input_tensors) > 1 else input_tensors[0].view(-1, 1)
        output_tensor = torch.stack(output_tensors, dim=2).view(-1, len(self.output_var_names)) if \
            len(output_tensors) > 1 else output_tensors[0].view(-1, 1)

        # Ensure that the number of samples (first dimension) is the same across all tensors
        assert input_tensor.shape[0] == output_tensor.shape[0] == len(basin_ids), "Size mismatch between tensors"

        # Create a custom dataset with the input and output tensors and basin IDs
        dataset = CustomDatasetToNN(input_tensor, output_tensor, basin_ids)

        pin_memory = True if 'cuda' in str(self.device) else False

        # Create custom batch sampler
        batch_sampler = BatchSampler(len(dataset), self.batch_size, shuffle=False)
        # batch_sampler = BasinBatchSampler(basin_ids, self.batch_size, shuffle=False)

        # Create DataLoader with custom batch sampler
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, 
                                pin_memory=pin_memory, num_workers=self.num_workers)

        return dataloader
    
    def train(self, max_nan_batches=10):

        if self.cfg.verbose:
            print("-- Pretraining the neural network model --")

        nan_count = 0  # Counter for NaN occurrences
        for epoch in range(self.epochs):

            epoch_loss = 0
            pbar = tqdm(self.dataloader, disable=self.cfg.disable_pbar, file=sys.stdout)
            pbar.set_description(f'# Epoch {epoch + 1:05d}')

            num_batches_seen = 0
            for (inputs, targets, basin_ids) in pbar:

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.nnmodel(inputs.to(self.device), basin_ids)

                # Move targets to the same device as predictions
                targets = targets.to(self.device)

                # Find the indices of NaNs in the predictions
                nan_mask = torch.isnan(predictions)
                nan_indices = torch.where(nan_mask)[0]

                # print("Basins in this batch:", set(basin_ids))

                if len(nan_indices) > 0:
                    nan_count += 1

                    # print(predictions.shape, targets.shape, nan_mask.shape, nan_indices.shape)

                    # Find the basins with NaNs in the predictions
                    nan_basins = set([basin_ids[i] for i in nan_indices])
                    print(f"Basins with NaNs in the predictions: {nan_basins}")
                    return

                    # Drop the NaNs from the predictions and targets
                    predictions = predictions[~nan_mask]
                    targets = targets[~nan_mask]

                    print(f"Dropped {len(nan_indices)} NaNs from predictions in batch {num_batches_seen + 1}")

                    if nan_count > max_nan_batches:
                        print(f"Exceeded {max_nan_batches} allowed NaN batches. Stopping training.")
                        return

                # Compute the loss
                loss = self.loss(predictions, targets) 

                # if torch.isnan(loss).any():
                #     nan_count += 1
                   
                #     # Find the indices of NaNs in the loss
                #     nan_indices = torch.where(torch.isnan(loss))[0]

                #     # Find the basins with NaNs in the loss
                #     nan_basins = [basin_ids[i] for i in nan_indices]
                #     print(f"Basins with NaNs in the loss: {nan_basins}")  

                #     # # Adjust learning rate
                #     # for param_group in self.optimizer.param_groups:
                #     #     param_group['lr'] *= 0.5
                #     # print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']}")

                #     # Reduce gradient clipping norm
                #     if self.cfg.clip_gradient_norm is not None:
                #         self.cfg.clip_gradient_norm *= 0.5

                #     # Skip the batch if NaNs are found
                #     print(f"Skipping batch {num_batches_seen + 1} due to NaNs in loss")
                #     if nan_count > max_nan_batches:
                #         print(f"Exceeded {max_nan_batches} allowed NaN batches. Stopping training.")
                #         return
                    
                #     num_batches_seen += 1
                #     continue

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

            pbar.close()

            if (epoch == 0 or ((epoch + 1) % self.log_every_n_epochs == 0)):
                if self.cfg.verbose:
                    print(f"-- Saving the model weights and plots (epoch {epoch + 1}) --")
                # Save the model weights
                self.save_model()
                self.save_plots(epoch=epoch+1)

            # Learning rate scheduler
            if (self.scheduler is not None) and epoch < self.epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                # Check if learning rate has changed
                new_lr = self.optimizer.param_groups[0]['lr']
                if self.cfg.verbose and current_lr != new_lr:
                    print(f"Learning rate updated to {new_lr}")

        # # Save the final model weights and plots
        # if self.cfg.verbose:
        #     print("-- Saving final model weights --")
        # self.save_model()
        # if self.cfg.verbose:
        #     print("-- Saving final plots --")
        # self.save_plots()
        if self.cfg.verbose:
            print("-- Evaluating the model --")
        self.evaluate()

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

        # Extract keys that start with 'ds_'
        ds_periods = [key for key in self.fulldataset.__dict__.keys() if key.startswith('ds_')]

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

                    inputs = torch.cat([torch.tensor(ds_basin[var.lower()].values).unsqueeze(0) \
                        for var in self.input_var_names], dim=0).t().to(self.device)

                    basin_list = [basin for _ in range(inputs.shape[0])]
                    outputs = self.nnmodel(inputs, basin_list)

                    # Scale back outputs
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
                            plt.savefig(basin_dir / f'{var}_{basin}_{period_name}.png', dpi=75)
                        plt.close('all')
            
            pbar_basins.close()

    def evaluate(self):
        # Extract keys that start with 'ds_'
        ds_periods = [key for key in self.fulldataset.__dict__.keys() if key.startswith('ds_')]

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
                
                inputs = torch.cat([torch.tensor(ds_basin[var.lower()].values).unsqueeze(0)
                                    for var in self.input_var_names], dim=0).t().to(self.device)
                
                basin_list = [basin for _ in range(inputs.shape[0])]
                outputs = self.nnmodel(inputs, basin_list)
                
                # Always the last variable in the list -> target variable
                y_obs = ds_basin[self.output_var_names[-1].lower()].values
                y_bucket = outputs[:, -1].detach().cpu().numpy()
                
                # Extract dates
                dates = ds_basin.date.values
                
                metrics = compute_all_metrics(y_obs, y_bucket, dates, self.cfg.metrics)
                
                # Store results in a list
                result = {'basin': basin}
                result.update(metrics)
                results.append(result)
            
            # Convert results to a DataFrame
            df_results = pd.DataFrame(results)

            # Sort the DataFrame by basin name
            df_results = df_results.sort_values(by='basin').reset_index(drop=True)
            
            # Save results to a CSV file
            period_name = dsp.split('_')[-1]
            output_file = f'evaluation_metrics_{period_name}.csv'
            output_file_path = Path(self.cfg.results_dir) / output_file

            # Save the results to a CSV file
            df_results.to_csv(output_file_path, index=False)

    # def _set_random_seeds(self):
    #     '''
    #     Set random seeds for reproducibility
    #     '''
    #     if self.cfg.seed is None:
    #         self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

    #     # fix random seeds for various packages
    #     random.seed(self.cfg.seed)
    #     np.random.seed(self.cfg.seed)
    #     torch.cuda.manual_seed(self.cfg.seed)
    #     torch.manual_seed(self.cfg.seed)


