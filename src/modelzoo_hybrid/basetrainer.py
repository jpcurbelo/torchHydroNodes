from tqdm import tqdm
import sys
import torch

from src.utils.metrics import loss_name_func_dict
from src.utils.metrics import (
    NSElossNH,
)


class BaseHybridModelTrainer:

    def __init__(self, model):
        
        self.model = model
        self.target = self.model.cfg.concept_target[0]

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

    def train(self):

        if self.model.cfg.verbose:
            print("-- Training the hybrid model --")

        for epoch in range(self.model.epochs):

            pbar = tqdm(self.model.dataloader, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            pbar.set_description(f'# Epoch {epoch + 1:05d}')

            epoch_loss = 0.0
            num_batches_seen = 0
            for (inputs, targets, basin_ids) in pbar:

                # Zero the parameter gradients
                self.model.optimizer.zero_grad()

                # Forward pass
                q_sim = self.model(inputs.to(self.model.device), basin_ids[0])

                if isinstance(self.loss, NSElossNH):
                    std_val = self.model.scaler['ds_feature_std'][self.target].sel(basin=basin_ids[0]).values
                    # To tensor
                    std_val = torch.tensor(std_val, dtype=self.model.data_type_torch).to(self.model.device)
                    loss = self.loss(q_sim, targets.to(self.model.device), std_val)
                else:
                    loss = self.loss(q_sim, targets.to(self.model.device))

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.model.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.cfg.clip_gradient_norm)

                # Update the weights
                self.model.optimizer.step()

                # Accumulate the loss
                epoch_loss += loss.item()
                num_batches_seen += 1

                # Update progress bar with current average loss
                avg_loss = epoch_loss / num_batches_seen
                pbar.set_postfix({'Loss': f'{avg_loss:.4e}'})


            pbar.close()