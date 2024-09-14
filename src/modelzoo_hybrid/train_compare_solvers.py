import time
import torch

def train_compare_solvers(self, is_resume=False):

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=self.model.cfg.patience)

    # For performance tracking
    total_train_time = 0.0
    epoch_times = []
    
    # Accuracy/loss tracking
    best_loss = float('inf')

    for epoch in range(self.model.epochs):

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

        # Compute the average loss for the epoch
        avg_loss = epoch_loss_sum / num_batches_seen

        # Performance: End time for epoch and track time
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        # Accuracy Profiling: Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            self.save_model()

        # Early stopping
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1} with best loss {best_loss:.4e}")
            break

        # Learning rate scheduler step
        if self.model.scheduler is not None and epoch < self.model.epochs - 1:
            self.model.scheduler.step()

        # Performance reporting for current epoch
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds | Loss: {avg_loss:.4e}")

    # Final performance report
    print(f"Total training time: {total_train_time:.2f} seconds")
    print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f} seconds")
    print(f"Best Loss achieved: {best_loss:.4e}")

    # Optionally evaluate the model after training completes
    self.evaluate()