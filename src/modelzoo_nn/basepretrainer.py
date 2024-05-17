import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from pathlib import Path

from src.modelzoo_nn.basemodel import BaseNNModel
from src.datasetzoo.basedataset import BaseDataset
from src.utils.metrics import loss_name_func_dict
from src.utils.load_process_data import BatchSampler, CustomDatasetToNN
from src.utils.metrics import NSE_eval


class NNpretrainer:

    def __init__(self, nnmodel: BaseNNModel, fulldataset: BaseDataset):
        
        self.fulldataset = fulldataset
        self.nnmodel = nnmodel
        self.cfg = self.nnmodel.concept_model.cfg

        # print(pretrainer.nnmodel.concept_model.ds)
        # print(pretrainer.nnmodel.concept_model.ds.basin.values)
        self.dataset = self.nnmodel.concept_model.ds
        self.basins = self.dataset.basin.values

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
                self.optimizer = optimizer_class(self.nnmodel.parameters(), lr=self.cfg.learning_rate['initial'])
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=self.epochs // self.cfg.learning_rate['decay_step_fraction'],
                                                                gamma=self.cfg.learning_rate['decay'])
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
            self.loss = loss_name_func_dict[loss_name]
        except KeyError:
            # Handle the case where the loss name is not recognized
            raise NotImplementedError(f"Loss function {loss_name} not implemented")
        except ValueError:
            # Handle the case where 'loss' is not specified in the config
            # Optionally, set a default loss function
            print("Warning! (Inputs): 'loss' not specified in the config. Defaulting to MSELoss.")
            self.loss = torch.nn.MSELoss()

    def create_dataloaders(self):
        '''Create the dataloaders for the pretrainer'''

        # Convert xarray DataArrays to PyTorch tensors and store in a dictionary
        tensor_dict = {var: torch.tensor(self.dataset[var].values, dtype=torch.float32) for var in self.dataset.data_vars}

        # Create a list of input and output tensors based on the variable names
        input_tensors = [tensor_dict[var] for var in self.input_var_names]
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
        batch_sampler = BatchSampler(len(dataset), self.batch_size, shuffle=True)

        # Create DataLoader with custom batch sampler
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, 
                                pin_memory=pin_memory, num_workers=self.num_workers)

        return dataloader
    
    def train(self):

        if self.cfg.verbose:
            print("-- Pretraining the neural network model --")

        # for epoch in tqdm(range(self.epochs), disable=self.cfg.disable_pbar, file=sys.stdout):
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

                # Compute the loss
                loss = self.loss(predictions.to(self.device), targets.to(self.device)) 

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                # Accumulate the loss
                epoch_loss += loss.item()
                num_batches_seen += 1

                # Update progress bar with current average loss
                avg_loss = epoch_loss / num_batches_seen
                pbar.set_postfix({'Loss': f'{avg_loss:.4e}'})

            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()


        if self.cfg.verbose:
            print("-- Saving the model weights and plots --")
        # Save the model weights
        # torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'{self.basinID}_modelM100.pth'))
        self.save_model()
        self.save_plots()

    def save_model(self):
        '''Save the model weights'''

        # # torch.save(self.nnmodel.state_dict(), model_path)
        # print(self.cfg._cfg.keys())
        # print(self.cfg.periods)
        # print('run_dir', self.cfg.run_dir)
        # print('config_dir', self.cfg.config_dir)
        # print('results_dir', self.cfg.results_dir)
        # print('plots_dir', self.cfg.plots_dir)

        # Create a directory to save the model weights if it does not exist
        model_dir = 'model_weights'
        model_path = self.cfg.run_dir / model_dir
        model_path.mkdir(parents=True, exist_ok=True)

        # Save the model weights
        torch.save(self.nnmodel.state_dict(), model_path / f'pretrained_{self.cfg.nn_model}_{len(self.basins)}basins.pth')

    def save_plots(self):

        # Extract keys that start with 'ds_'
        ds_periods = [key for key in self.fulldataset.__dict__.keys() if key.startswith('ds_')]

        # Check if log_n_figures exists and is greater than 0
        if hasattr(self.cfg, 'log_n_figures') and self.cfg.log_n_figures > 0:

            # Generate a list of random basin IDs to plot
            random.seed(self.cfg.seed)
            sample_size = min(len(self.basins), self.cfg.log_n_figures)
            random_basins = random.sample(list(self.basins), sample_size)

            for basin in random_basins:

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

                    # Save the results as a CSV file
                    period_name = dsp.split('_')[-1]
                    for vi, var in enumerate(self.output_var_names):

                        q_obs = ds_basin[var.lower()].values
                        q_bucket = outputs[:, vi].detach().cpu().numpy()

                        plt.figure(figsize=(10, 6))
                        plt.plot(ds_basin.date, q_obs, label='Observed')
                        plt.plot(ds_basin.date, q_bucket, label='Predicted')
                        plt.xlabel('Date')
                        plt.ylabel(var)
                        plt.legend()

                        if vi == len(self.output_var_names) - 1:
                                nse_val = NSE_eval(q_obs, q_bucket)
                                plt.title(f'{var} - {basin} - {period_name} | $NSE = {nse_val:.3f}$')
                        else:
                            plt.title(f'{var} - {basin} - {period_name}')

                        plt.tight_layout()
                        plt.savefig(basin_dir / f'{var}_{basin}_{period_name}.png', dpi=75)
                        





