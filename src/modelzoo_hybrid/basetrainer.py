from tqdm import tqdm
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from src.utils.metrics import loss_name_func_dict
from src.utils.metrics import (
    NSElossNH,
    NSE_eval,
    compute_all_metrics,
)

from src.utils.load_process_data import EarlyStopping


class BaseHybridModelTrainer:

    def __init__(self, model):
        
        self.model = model

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

    def train(self, is_resume=False):

        early_stopping = EarlyStopping(patience=self.model.cfg.patience)

        if self.model.cfg.verbose:
            print(f"-- Training the hybrid model on {self.model.device} --")

        # Save the model weights - Epoch 0
        # if not is_resume:
        self.save_model()
        self.save_plots(epoch=0)

        for epoch in range(self.model.epochs):

            # Clear CUDA cache at the beginning of each epoch
            torch.cuda.empty_cache()

            pbar = tqdm(self.model.dataloader, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            pbar.set_description(f'# Epoch {epoch + 1:05d} ')

            epoch_loss = 0.0
            num_batches_seen = 0
            for (inputs, targets, basin_ids) in pbar:

                # Zero the parameter gradients
                self.model.optimizer.zero_grad()

                # Transfer to device
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)

                # Forward pass
                q_sim = self.model(inputs, basin_ids[0])

                if isinstance(self.loss, NSElossNH):
                    std_val = self.model.scaler['ds_feature_std'][self.target].sel(basin=basin_ids[0]).values
                    # To tensor
                    std_val = torch.tensor(std_val, dtype=self.model.data_type_torch)  
                    loss = self.loss(targets[:, -1], q_sim, std_val)
                else:
                    if self.model.cfg.scale_target_vars:
                        loss = self.loss(torch.exp(targets[:, -1]), torch.exp(q_sim))  
                    else:
                        loss = self.loss(targets[:, -1], q_sim) 

                # Accumulate the loss
                epoch_loss += loss.item()
                num_batches_seen += 1

                # Update progress bar with current average loss
                avg_loss = epoch_loss / num_batches_seen
                pbar.set_postfix({'Loss': f'{avg_loss:.4e}'})

                # # Save the model state before optimizer step
                # if num_batches_seen == len(self.model.dataloader):
                #     nnmodel_state_dict = copy.deepcopy(self.model.pretrainer.nnmodel.state_dict())

                ##############################################################
                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.model.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.cfg.clip_gradient_norm)

                # Update the weights
                self.model.optimizer.step()
                ##############################################################

            pbar.close()

            # Save the model weights and plots
            if (epoch == 0 or ((epoch + 1) % self.model.cfg.log_every_n_epochs == 0)):
                
                # aux = input("Press Enter to continue...")
                
                if self.model.cfg.verbose:
                    print(f"-- Saving the model weights and plots (epoch {epoch + 1}) | --")
                # Save the model weights
                self.save_model()
                # self.save_plots(nnmodel_state_dict, epoch=epoch+1)
                self.save_plots(epoch=epoch+1)

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

        # Save the final model weights and plots
        if self.model.cfg.verbose:
            print("-- Training completed | Evaluating the model --")
        self.evaluate()

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

        # # Load the saved model state
        # self.model.pretrainer.nnmodel.load_state_dict(nnmodel_state)

        # Check if log_n_basins exists and is either a positive integer or a non-empty list
        if self.model.pretrainer.basins_to_log is not None:

            # Progress bar for basins
            pbar_basins = tqdm(self.model.pretrainer.basins_to_log, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            for basin in pbar_basins:

                # Clear CUDA cache at the beginning of each basin
                
                torch.cuda.empty_cache()

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
                    # If the period is not ds_train, then shift the time series by the length of ds_train
                    if dsp == 'ds_train':        
                        time_idx = np.linspace(0, len(time_series) - 1, len(time_series), 
                                               dtype=self.model.data_type_np)
                    else:
                        time_series_test = getattr(self.model.pretrainer.fulldataset, dsp)['date'].values
                        train_length = len(self.model.pretrainer.fulldataset.ds_train['date'].values)
                        test_length = len(time_series_test)
                        time_idx = np.linspace(train_length, train_length + test_length - 1, len(time_series), 
                                               dtype=self.model.data_type_np)

                    input_var_names = self.model.pretrainer.input_var_names + ['time_idx']

                    # Add time_idx to the dataset, making sure to match the correct dimensions
                    ds_basin['time_idx'] = (('date',), time_idx)

                    # Get model outputs
                    inputs = self.model.get_model_inputs(ds_basin, input_var_names, basin, is_trainer=True)
                    # Get model outputs
                    outputs = self.model(inputs, basin, use_grad=False)
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
                    print(f'NSE: {nse_val:.3f}')

                    plt.title(f'{self.target} - {basin} - {period_name} | $NSE = {nse_val:.3f}$')

                    plt.tight_layout()
                    plt.savefig(basin_dir / f'{self.target}_{basin}_{period_name}_epoch{epoch}.png', dpi=75)
                    plt.close('all')

            pbar_basins.close()

    def evaluate(self):
        '''
        Evaluate the model on the test dataset
        '''

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

                time_series = ds_basin['date'].values
                # If the period is not ds_train, then shift the time series by the length of ds_train
                if dsp == 'ds_train':        
                    time_idx = np.linspace(0, len(time_series) - 1, len(time_series), dtype=self.model.data_type_np)
                else:
                    time_series_test = getattr(self.model.pretrainer.fulldataset, dsp)['date'].values
                    train_length = len(self.model.pretrainer.fulldataset.ds_train['date'].values)
                    test_length = len(time_series_test)
                    time_idx = np.linspace(train_length, train_length + test_length - 1, len(time_series), dtype=self.model.data_type_np)

                input_var_names = self.model.pretrainer.input_var_names + ['time_idx']

                # Add time_idx to the dataset, making sure to match the correct dimensions
                ds_basin['time_idx'] = (('date',), time_idx)

                # Get the outputs from the hybrid model
                inputs = self.model.get_model_inputs(ds_basin, input_var_names, basin, is_trainer=True)
                # Get model outputs
                outputs = self.model(inputs, basin, use_grad=False)
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
                # Create a DataFrame with the required columns
                results_df = pd.DataFrame({
                    'date': dates,
                    'y_obs': y_obs,
                    'y_sim': y_sim
                })

                # Save results to a CSV file
                period_name = dsp.split('_')[-1]
                results_file = f'{basin}_results_{period_name}.csv'
                results_file_path = Path(self.model.cfg.results_dir) / results_file

                # Ensure the results directory exists
                results_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the DataFrame to a CSV file
                results_df.to_csv(results_file_path, index=False)

                # Compute all evaluation metrics
                metrics = compute_all_metrics(y_obs, y_sim, dates, self.model.cfg.metrics)

                # Store the results in a dictionary
                result = {'basin': basin}
                result.update(metrics)
                results.append(result)

            # Convert the results to a DataFrame
            df_results = pd.DataFrame(results)

            # Sort the DF by basin name
            df_results = df_results.sort_values('basin').reset_index(drop=True)

            # Save the results to a CSV file
            period_name = dsp.split('_')[-1]
            metrics_file = f'evaluation_metrics_{period_name}.csv'
            metrics_file_path = metrics_dir / metrics_file

            # Save the results to a CSV file
            df_results.to_csv(metrics_file_path, index=False)