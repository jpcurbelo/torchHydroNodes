import xarray as xr
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys

from src.utils.metrics import loss_name_func_dict
from src.utils.load_process_data import BatchSampler, CustomDatasetToNN

class NNpretrainer:

    def __init__(self, nnmodel) -> None:
        
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

        for epoch in range(self.epochs):

            epoch_loss = 0
            for bi, (inputs, targets, basin_ids) in enumerate(self.dataloader):

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.nnmodel(inputs.to(self.device), basin_ids)

                # Compute the loss
                loss = self.loss(predictions.to(self.device), targets.to(self.device))

                # sys.stdout.write(f'epoch {epoch + 1:3d}/{self.epochs:3d} batch {bi + 1:3d}/{self.num_batches:3d} loss {loss:6.4e}\r')
                # sys.stdout.flush()   

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                # Accumulate the loss
                epoch_loss += loss.item()

            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            print(f'Epoch {epoch+1:5d}, Loss: {epoch_loss / self.num_batches:6.4e}')

