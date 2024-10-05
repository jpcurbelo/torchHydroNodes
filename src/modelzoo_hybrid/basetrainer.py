from tqdm import tqdm
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv
import tracemalloc  # For CPU memory tracking
import gc

from src.utils.metrics import loss_name_func_dict
from src.utils.metrics import (
    NSElossNH,
    NSE_eval,
    compute_all_metrics,
)

from src.utils.load_process_data import (
    EarlyStopping,
    # calculate_tensor_memory,
    # get_free_gpu_memory,
    # run_job_with_memory_check
)

class BaseHybridModelTrainer:

    def __init__(self, model):
        
        self.model = model
        self.device_to_train = self.model.device

        self.target = self.model.cfg.concept_target[0]
        # print('BaseHybridModelTrainer - self.target:', self.target)
        self.hybrid_model = (self.model.cfg.hybrid_model).lower()
        self.nnmodel_name = (self.model.pretrainer.nnmodel.__class__.__name__).lower()
        self.basins = self.model.dataset.basin.values
        self.number_of_basins = self.model.cfg.number_of_basins

        # Extract keys that start with 'ds_'
        self.ds_periods = [key for key in self.model.pretrainer.fulldataset.__dict__.keys() if key.startswith('ds_') \
                           and 'static' not in key]

        # Loss function setup
        try:
            # Try to get the loss function name from configuration
            loss_name = self.model.cfg.loss
            self.loss = loss_name_func_dict[loss_name.lower()]
        except KeyError:
            # Handle the case where the loss name is not recognized
            raise NotImplementedError(f"Loss function {loss_name} not implemented")
        except ValueError:
            # Handle the case where 'loss' is not specified in the config
            # Optionally, set a default loss function
            print("Warning! (Inputs): 'loss' not specified in the config. Defaulting to MSELoss.")
            self.loss = torch.nn.MSELoss()

    def train(self, is_resume=False, max_nan_batches=5):

        early_stopping = EarlyStopping(patience=self.model.cfg.patience)

        # if self.model.cfg.verbose:
        print('-' * 60)
        print(f"-- Training the hybrid model on {self.device_to_train} --")
        print(f'Initial learning rate: {self.model.optimizer.param_groups[0]["lr"]:.2e}')
        print('-' * 60)

        # # Save the model weights - Epoch 0
        # self.save_model()
        # self.save_plots(epoch=0)

        best_loss = float('inf')  # Initialize best loss to a very high value
        nan_count = 0
        stop_training = False  # Flag to signal when to stop both loops

        # Check if training is on GPU
        is_gpu = 'cuda' in self.device_to_train.type

        for epoch in range(self.model.epochs):

            # Clear CUDA cache at the beginning of each epoch if running on GPU
            if is_gpu:
                torch.cuda.empty_cache()

            pbar = tqdm(self.model.dataloader, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            pbar.set_description(f'# Epoch {epoch + 1:05d} ')

            epoch_loss_sum = 0.0
            num_batches_seen = 0

            if self.model.cfg.carryover_state:
                # Create tensors to store the s_snow and s_water values to be used in the next batch
                carryover_s_snow = torch.zeros(self.model.pretrainer.batch_size).to(self.device_to_train)
                carryover_s_water = torch.zeros(self.model.pretrainer.batch_size).to(self.device_to_train)

            first_batch = True  # Flag to check if it's the first batch

            for (inputs, targets, basin_ids) in pbar:

                # Zero the parameter gradients
                self.model.optimizer.zero_grad()

                # Transfer to device
                inputs = inputs.to(self.device_to_train, non_blocking=True)
                targets = targets.to(self.device_to_train, non_blocking=True) 

                # print('inputs.shape:', inputs.shape)
                # print(' self.pretrainer.input_var_names:', self.model.pretrainer.input_var_names)
                # aux = input('Press Enter to continue...')

                if self.model.cfg.carryover_state and not first_batch:
                    # Update the s_snow and s_water values for the current batch
                    if len(inputs.shape) == 2: # For LSTM models (inputs are 2D)
                        # Update the first timestep (index 0) of the sequence for the carryover features
                        inputs[0, 0] = carryover_s_snow
                        inputs[0, 1] = carryover_s_water
                    elif len(inputs.shape) == 3:  # For LSTM models (inputs are 3D)
                        # Ensure that carryover state is sliced to match the batch size of the inputs
                        current_batch_size = inputs.shape[0]  # Get current batch size (might be smaller for last batch)
                        inputs[:, 0, 0] = carryover_s_snow[:current_batch_size]  # Adjust to match current batch size
                        inputs[:, 0, 1] = carryover_s_water[:current_batch_size]  # Adjust to match current batch size

                # Forward pass
                q_sim, s_snow, s_water = self.model(inputs, basin_ids[0])

                nan_mask = torch.isnan(q_sim)
                if nan_mask.any():
                    nan_count += 1
                    q_sim = q_sim[~nan_mask]
                    targets = targets[~nan_mask]
                    if nan_count > max_nan_batches:
                        print(f"Exceeded {max_nan_batches} allowed NaN batches. Stopping training.")
                        stop_training = True  # Set flag to stop both loops
                        break  # Break the inner loop

                # Update the s_snow and s_water values for the next batch
                if self.model.cfg.carryover_state:
                    if len(inputs.shape) == 2:  # For MLP models (inputs are 2D)
                        carryover_s_snow = s_snow[-1].clone().detach()
                        carryover_s_water = s_water[-1].clone().detach()
                    elif len(inputs.shape) == 3:  # For LSTM models (inputs are 3D)
                        carryover_s_snow = s_snow[:, -1].clone().detach()
                        carryover_s_water = s_water[:, -1].clone().detach()

                # Compute loss
                if isinstance(self.loss, NSElossNH):
                    std_val = self.model.scaler['ds_feature_std'][self.target].sel(basin=basin_ids[0]).values
                    std_val = torch.tensor(std_val, dtype=self.model.data_type_torch).to(self.device_to_train)
                    loss = self.loss(targets[:, -1], q_sim, std_val)   ####torch.exp(
                else:
                    if self.model.cfg.scale_target_vars:
                        loss = self.loss(torch.exp(targets[:, -1]), torch.exp(q_sim))
                        # loss = self.loss(torch.exp(targets[:, -1]), torch.exp(q_sim), 
                        #                  self.model.pretrainer.nnmodel.torch_target_means[basin_ids[0]])
                    else:
                        loss = self.loss(targets[:, -1], q_sim)
                        # loss = self.loss(targets[:, -1], q_sim,
                        #                  self.model.pretrainer.nnmodel.torch_target_means[basin_ids[0]])

                ##############################################################
                # # Backward pass if batch is not the last one
                # if num_batches_seen < len(self.model.dataloader) - 1 or first_batch:

                # Backward pass
                loss.backward()
                # Update the weights
                self.model.optimizer.step()
                # Gradient clipping
                if self.model.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.pretrainer.nnmodel.parameters(), self.model.cfg.clip_gradient_norm)
               
                # Accumulate the loss
                epoch_loss_sum += loss.item()
                num_batches_seen += 1

                # Update progress bar with current average loss
                avg_loss = epoch_loss_sum / num_batches_seen
                pbar.set_postfix({'Loss': f'{avg_loss:.4e}'})

                # Delete variables to free memory
                del inputs, targets, q_sim, loss
                # torch.cuda.empty_cache()

                first_batch = False  # Set the flag to False after processing the first batch

            pbar.close()

            # Break the outer loop if stopping is signaled
            if stop_training:
                break

            # Save the model weights and plots
            if ((epoch == 0 or ((epoch + 1) % self.model.cfg.log_every_n_epochs == 0))) and epoch < self.model.epochs - 1:               
                if self.model.cfg.verbose:
                    print(f"-- Saving the basin plots (epoch {epoch + 1}) | --")
                # Save plots 
                self.save_plots(epoch=epoch+1)

            # Check for the best model and save it
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()
                if self.model.cfg.verbose:
                    print(f"-- Best model updated at epoch {epoch + 1} with loss {avg_loss:.4e}")

            # Early stopping check
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1} with loss {avg_loss:.4e}")
                break

            # Learning rate scheduler
            if (self.model.scheduler is not None) and epoch < self.model.epochs - 1:
                current_lr = self.model.optimizer.param_groups[0]['lr']
                self.model.scheduler.step()

                # Check if learning rate has changed
                new_lr = self.model.optimizer.param_groups[0]['lr']
                if self.model.cfg.verbose and new_lr != current_lr:
                    print(f"Learning rate updated from {current_lr:.2e} to {new_lr:.2e}")

            # Print the average loss for the epoch if not verbose (pbar is disabled)
            if not self.model.cfg.verbose:
                print(f"Epoch {epoch + 1} Loss: {avg_loss:.4e}")

        # Evaluate and save the final plots
        if not stop_training:
            # Save the final model weights and plots
            if self.model.cfg.verbose:
                print("-- Training completed | Evaluating the model --")
            self.evaluate()
            if is_gpu:
                torch.cuda.empty_cache()
            self.save_plots(epoch=epoch + 1)

    ################################################# 
    def train_finetune(self, is_resume=False, max_nan_batches=5):
        ''' 
        Train the model with no (or minimal) logging and plotting just to profile and fine-tune the model.
        ''' 

        # For performance tracking
        total_train_time = 0.0
        epoch_times = []

        # Open CSV file for writing epoch stats (time, loss, memory)
        with open(self.model.cfg.run_dir / 'epoch_stats.csv', 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'epoch_time_seg', 'avg_loss', 'cpu_peak_memory_mb']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            nan_count = 0
            stop_training = False  # Flag to signal when to stop both loops
            
            # Check if training is on CPU
            is_cpu = 'cpu' in self.device_to_train.type

            # Start tracemalloc only if running on CPU
            if is_cpu:
                tracemalloc.start()

            for epoch in range(self.model.epochs):

                # Reset CPU peak memory stats at the beginning of each epoch if running on CPU
                if is_cpu:
                    tracemalloc.clear_traces()

                # Start time for epoch
                start_time = time.time()

                # Reset epoch statistics
                epoch_loss_sum = 0.0
                num_batches_seen = 0

                for (inputs, targets, basin_ids) in self.model.dataloader:

                    # Zero the parameter gradients
                    self.model.optimizer.zero_grad()

                    # Transfer to device
                    inputs = inputs.to(self.device_to_train, non_blocking=True)
                    targets = targets.to(self.device_to_train, non_blocking=True) 

                    # Forward pass
                    q_sim, _, _ = self.model(inputs, basin_ids[0])

                    # Check for NaN values
                    nan_mask = torch.isnan(q_sim)
                    if nan_mask.any():
                        nan_count += 1
                        q_sim = q_sim[~nan_mask]
                        targets = targets[~nan_mask]
                        if nan_count > max_nan_batches:
                            print(f"Exceeded {max_nan_batches} allowed NaN batches. Stopping training.")
                            stop_training = True  # Set flag to stop both loops
                            break  # Break the inner loop

                    # Compute loss
                    if isinstance(self.loss, NSElossNH):
                        std_val = self.model.scaler['ds_feature_std'][self.target].sel(basin=basin_ids[0]).values
                        std_val = torch.tensor(std_val, dtype=self.model.data_type_torch).to(self.device_to_train)
                        loss = self.loss(targets[:, -1], q_sim, std_val)   
                    else:
                        if self.model.cfg.scale_target_vars:
                            loss = self.loss(torch.exp(targets[:, -1]), torch.exp(q_sim))
                        else:
                            loss = self.loss(targets[:, -1], q_sim)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping if specified
                    if self.model.cfg.clip_gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.pretrainer.nnmodel.parameters(), self.model.cfg.clip_gradient_norm)

                    # Update the weights
                    self.model.optimizer.step()

                    # Accumulate the loss
                    epoch_loss_sum += loss.item()
                    num_batches_seen += 1

                    # Free memory after each batch
                    del inputs, targets, q_sim, loss
                    torch.cuda.empty_cache()  # If using GPU
                    gc.collect()

                # Break the outer loop if stopping is signaled
                if stop_training:
                    break

                # Compute the average loss for the epoch
                avg_loss = epoch_loss_sum / num_batches_seen

                # Performance: End time for epoch and track time
                epoch_time = time.time() - start_time
                epoch_times.append(epoch_time)
                total_train_time += epoch_time

                # Track CPU memory if running on CPU
                cpu_peak_memory_mb = 'N/A'
                if is_cpu:
                    _, cpu_peak_memory_kb = tracemalloc.get_traced_memory()
                    cpu_peak_memory_mb = cpu_peak_memory_kb / 1024  # Convert KB to MB

                # Write time, loss, and memory stats to CSV with formatted values
                writer.writerow({
                    'epoch': epoch + 1, 
                    'epoch_time_seg': f"{epoch_time:.2f}",  # Time rounded to 2 decimal places
                    'avg_loss': f"{avg_loss:.4e}",  # Loss in scientific notation with 4 significant digits
                    'cpu_peak_memory_mb': f"{cpu_peak_memory_mb:.2f}" if is_cpu else 'N/A'
                })

                # Learning rate scheduler
                if (self.model.scheduler is not None) and epoch < self.model.epochs - 1:
                    current_lr = self.model.optimizer.param_groups[0]['lr']
                    self.model.scheduler.step()

                    # Check if learning rate has changed
                    new_lr = self.model.optimizer.param_groups[0]['lr']
                    if self.model.cfg.verbose and new_lr != current_lr:
                        print(f"Learning rate updated from {current_lr:.2e} to {new_lr:.2e}")

                # Save the model weights and plots
                if ((epoch == 0 or ((epoch + 1) % self.model.cfg.log_every_n_epochs == 0))) \
                    and epoch < self.model.epochs - 1:
                    self.save_plots(epoch=epoch+1)

                # Clear CUDA cache after the epoch if running on GPU
                torch.cuda.empty_cache()

            # Stop tracemalloc if running on CPU
            if is_cpu:
                tracemalloc.stop()

            if not stop_training:
                # Save the final model weights and plots
                self.save_model()

                # Start tracemalloc only if running on CPU
                if is_cpu:
                    tracemalloc.start()

                # Time tracking for evaluation
                eval_start_time = time.time()
                self.evaluate()
                eval_time = time.time() - eval_start_time

                # Track CPU memory during evaluation
                cpu_peak_memory_mb = 'N/A'
                if is_cpu:
                    _, cpu_peak_memory_kb = tracemalloc.get_traced_memory()
                    cpu_peak_memory_mb = cpu_peak_memory_kb / 1024  # Convert KB to MB

                # Write the evaluation times and memory stats to the CSV
                writer.writerow({
                    'epoch': 'evaluation',
                    'epoch_time_seg': f"{eval_time:.2f}",
                    'avg_loss': 'N/A',
                    'cpu_peak_memory_mb': f"{cpu_peak_memory_mb:.2f}" if is_cpu else 'N/A'
                })

                # Stop tracemalloc if running on CPU
                if is_cpu:
                    tracemalloc.stop()

                # Start tracemalloc only if running on CPU
                if is_cpu:
                    tracemalloc.start()
                    
                # Time tracking for final plotting
                plot_start_time = time.time()
                self.save_plots(epoch=epoch + 1)
                plot_time = time.time() - plot_start_time

                # Track CPU memory during final plotting
                cpu_peak_memory_mb = 'N/A'
                if is_cpu:
                    _, cpu_peak_memory_kb = tracemalloc.get_traced_memory()
                    cpu_peak_memory_mb = cpu_peak_memory_kb / 1024  # Convert KB to MB

                # Write the final plotting times and memory stats to the CSV
                writer.writerow({
                    'epoch': 'final_plot',
                    'epoch_time_seg': f"{plot_time:.2f}",
                    'avg_loss': 'N/A',
                    'cpu_peak_memory_mb': f"{cpu_peak_memory_mb:.2f}" if is_cpu else 'N/A'
                })

                # Stop tracemalloc if running on CPU
                if is_cpu:
                    tracemalloc.stop()

                return True
            
            else:
                return False

    #################################################

    def save_model(self):
        '''Save the model weights and plots'''

        # Create a directory to save the model weights if it does not exist
        model_dir = 'model_weights'
        model_path = self.model.cfg.run_dir / model_dir
        model_path.mkdir(parents=True, exist_ok=True)

        # Save the model weights
        torch.save(self.model.pretrainer.nnmodel.state_dict(), model_path / f'trainer_{self.hybrid_model}_{self.nnmodel_name}_{self.number_of_basins}basins.pth')

    def save_plots(self, epoch=None):
        '''
        Save the plots of the observed and predicted values for a random subset of basins
        and for each dataset period
        '''

        # Check if log_n_basins exists and is either a positive integer or a non-empty list
        if self.model.pretrainer.basins_to_log is not None:

            # Progress bar for basins
            pbar_basins = tqdm(self.model.pretrainer.basins_to_log, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            for basin in pbar_basins:

                pbar_basins.set_description(f'* Plotting basin {basin}')

                if basin not in self.basins:
                    print(f"Basin {basin} not found in the dataset. Skipping...")
                    continue

                # Create a directory to save the plots if it does not exist
                basin_dir = Path(self.model.cfg.plots_dir) / basin
                basin_dir.mkdir(parents=True, exist_ok=True)

                for dsp in self.ds_periods:

                    period_name = dsp.split('_')[-1]

                    ds_period = getattr(self.model.pretrainer.fulldataset, dsp)
                    ds_basin = ds_period.sel(basin=basin)

                    time_series = ds_basin['date'].values
                    time_idx = self.get_time_idx(dsp, time_series) # Get the time index

                    input_var_names = self.model.pretrainer.input_var_names + ['time_idx']

                    # Add time_idx to the dataset, making sure to match the correct dimensions
                    ds_basin['time_idx'] = (('date',), time_idx)

                    # Get model outputs
                    inputs = self.model.get_model_inputs(ds_basin, input_var_names, basin, is_trainer=True)

                    # No GPU memory control - this is standard
                    outputs, _, _ = self.model(inputs, basin, use_grad=False)
                    # outputs = run_job_with_memory_check(self.model, ds_basin,  input_var_names, basin, inputs.shape, inputs.dtype, use_grad=False)

                    # Reshape outputs
                    outputs = self.model.reshape_outputs(outputs)

                    # Scale back outputs
                    if self.model.cfg.scale_target_vars:
                        outputs = self.model.scale_back_simulated(outputs, ds_basin, is_trainer=True)

                        # If period is ds_train, also scale back the observed variables
                        if dsp == 'ds_train':
                            ds_basin = self.model.scale_back_observed(ds_basin, is_trainer=True)

                    # Get the simulated values in numpy format
                    y_sim = outputs.detach().cpu().numpy()

                    # Get the observed values
                    y_obs = ds_basin[self.target.lower()].values

                    # Plot the observed and predicted values
                    plt.figure(figsize=(10, 6))
                    plt.plot(ds_basin.date, y_obs, label='Observed')
                    plt.plot(ds_basin.date, y_sim, label='Predicted', alpha=0.7)
                    plt.xlabel('Date')
                    plt.ylabel(self.target)
                    plt.legend()

                    nse_val = NSE_eval(y_obs, y_sim)

                    plt.title(f'{self.target} - {basin} - {period_name} | $NSE = {nse_val:.3f}$')

                    plt.tight_layout()
                    plt.savefig(basin_dir / f'{self.target}_{basin}_{period_name}_epoch{epoch}.png', dpi=75)
                    plt.close('all')

                    # Free up memory by deleting large variables after each period
                    del inputs, outputs, y_sim, y_obs, ds_basin

                # Clear cache after processing all periods for a basin
                torch.cuda.empty_cache()  # Free up unused GPU memory
                gc.collect()  # Free up unused CPU memory

            pbar_basins.close()

        # Final cleanup after all basins are processed
        torch.cuda.empty_cache()  # Free up unused GPU memory
        gc.collect()  # Free up unused CPU memory

    def evaluate(self, load_best_model=True):
        '''
        Evaluate the model on the test dataset
        '''

        if load_best_model:
            # Define the path to the best model saved during training
            best_model_path = self.model.cfg.run_dir / 'model_weights' / f'trainer_{self.hybrid_model}_{self.nnmodel_name}_{self.number_of_basins}basins.pth'

            # Check if the best model exists
            if best_model_path.exists():
                # Load best model saved during training
                self.model.pretrainer.nnmodel.load_state_dict(torch.load(best_model_path))
                if self.model.cfg.verbose:
                    print(f"Loaded best model from {best_model_path}")
            else:
                print(f"Best model file not found at {best_model_path}. Evaluation will not be performed.")
                return

        metrics_dir = self.model.cfg.run_dir / 'model_metrics'
        if not metrics_dir.exists():
            metrics_dir.mkdir()

        # Extract keys that start with 'ds_'
        ds_periods = [key for key in self.model.pretrainer.fulldataset.__dict__.keys() if key.startswith('ds_') \
                           and 'static' not in key]

        for dsp in ds_periods:
            ds_period = getattr(self.model.pretrainer.fulldataset, dsp)
            ds_basins = ds_period['basin'].values

            results = []

            # Progress bar for basins
            pbar_basins = tqdm(ds_basins, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            for basin in pbar_basins:
                pbar_basins.set_description(f'* Evaluating basin {basin} ({dsp})')

                if basin not in self.basins:
                    print(f"Basin {basin} not found in the dataset. Skipping...")
                    continue

                ds_basin = ds_period.sel(basin=basin)

                # Clear GPU/CPU cache before processing the next basin
                torch.cuda.empty_cache()  # Clear GPU memory
                gc.collect()  # Clear CPU memory

                time_series = ds_basin['date'].values
                time_idx = self.get_time_idx(dsp, time_series) # Get the time index

                input_var_names = self.model.pretrainer.input_var_names + ['time_idx']

                # Add time_idx to the dataset, making sure to match the correct dimensions
                ds_basin['time_idx'] = (('date',), time_idx)

                # Get the outputs from the hybrid model
                inputs = self.model.get_model_inputs(ds_basin, input_var_names, basin, is_trainer=True)

                # Get model outputs
                outputs, _, _ = self.model(inputs, basin, use_grad=False)

                # Reshape outputs
                outputs = self.model.reshape_outputs(outputs)

                # Scale back outputs
                if self.model.cfg.scale_target_vars:
                    outputs = self.model.scale_back_simulated(outputs, ds_basin, is_trainer=True)
                    # If period is ds_train, also scale back the observed variables
                    if dsp == 'ds_train':
                        ds_basin = self.model.scale_back_observed(ds_basin, is_trainer=True)

                # Get the simulated values in numpy format
                y_sim = outputs.detach().cpu().numpy()

                # Get the observed values
                y_obs = ds_basin[self.target.lower()].values

                # Extract dates
                dates = ds_basin['date'].values

                # Save results to a CSV file
                self.save_basin_results(basin, dsp, dates, y_obs, y_sim)
                # self.save_basin_results_hdf5(basin, dsp, dates, y_obs, y_sim)

                # Compute all evaluation metrics
                metrics = compute_all_metrics(y_obs, y_sim, dates, self.model.cfg.metrics)

                # Store the results in a dictionary
                result = {'basin': basin}
                result.update(metrics)
                results.append(result)

                # Delete large variables to free memory
                del inputs, outputs, y_sim, ds_basin
                torch.cuda.empty_cache()  # Free GPU memory
                gc.collect()  # Free CPU memory

            # Save metrics to CSV
            self.save_metrics_to_csv(dsp, results, metrics_dir)

        # Final cleanup
        torch.cuda.empty_cache()  # Free GPU memory
        gc.collect()  # Free CPU memory

    def get_time_idx(self, dsp, time_series):
        '''Helper function to get time index.'''

        # If the period is not ds_train, then shift the time series by the length of ds_train
        if dsp == 'ds_train':        
            time_idx = np.linspace(0, len(time_series) - 1, len(time_series), dtype=self.model.data_type_np)
        else:
            train_length = len(self.model.pretrainer.fulldataset.ds_train['date'].values)
            test_length = len(time_series)
            time_idx = np.linspace(train_length, train_length + test_length - 1, len(time_series), dtype=self.model.data_type_np)
        
        return time_idx

    def save_basin_results(self, basin, dsp, dates, y_obs, y_sim):
        '''Helper function to save basin results.'''
        results_df = pd.DataFrame({
            'date': dates,
            'y_obs': y_obs,
            'y_sim': y_sim
        })
        period_name = dsp.split('_')[-1]
        results_file = f'{basin}_results_{period_name}.csv'
        results_file_path = Path(self.model.cfg.results_dir) / results_file
        results_file_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_file_path, index=False)

    def save_basin_results_hdf5(self, basin, dsp, dates, y_obs, y_sim):
        '''Helper function to save basin results in HDF5 format.'''
        results_df = pd.DataFrame({
            'date': dates,
            'y_obs': y_obs,
            'y_sim': y_sim
        })
        period_name = dsp.split('_')[-1]
        
        results_file = f'{basin}_results_{period_name}.h5'
        results_file_path = Path(self.model.cfg.results_dir) / results_file
        results_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame as HDF5 with compression
        results_df.to_hdf(results_file_path, key='df', mode='w', complevel=9, complib='zlib')

    def save_basin_results_parquet(self, basin, dsp, dates, y_obs, y_sim, chunk_size=100):
        '''Helper function to save basin results in Parquet format with memory optimizations.'''
        
        # Prepare the file path
        period_name = dsp.split('_')[-1]
        results_file = f'{basin}_results_{period_name}.parquet'
        results_file_path = Path(self.model.cfg.results_dir) / results_file
        results_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data in chunks to save memory
        for i in range(0, len(dates), chunk_size):
            # Chunk the data to process a smaller amount at a time
            date_chunk = dates[i:i+chunk_size]
            y_obs_chunk = y_obs[i:i+chunk_size]
            y_sim_chunk = y_sim[i:i+chunk_size]
            
            # Create a temporary DataFrame for the chunk
            chunk_df = pd.DataFrame({
                'date': date_chunk,
                'y_obs': y_obs_chunk,
                'y_sim': y_sim_chunk
            })
            
            # Append the chunk to the Parquet file
            chunk_df.to_parquet(results_file_path, compression='snappy', engine='pyarrow', index=False, append=True)

        # After saving, free up memory
        del dates, y_obs, y_sim
        gc.collect()  # Ensure that unused memory is released

    def save_metrics_to_csv(self, dsp, results, metrics_dir):
        '''Helper function to save evaluation metrics to CSV.'''
        df_results = pd.DataFrame(results).sort_values('basin').reset_index(drop=True)
        period_name = dsp.split('_')[-1]
        metrics_file = f'evaluation_metrics_{period_name}.csv'
        metrics_file_path = metrics_dir / metrics_file
        df_results.to_csv(metrics_file_path, index=False)
# # ####################################################################################################