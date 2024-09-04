

def train_compare_solvers(self, is_resume=False):

    early_stopping = EarlyStopping(patience=self.model.cfg.patience)

    for epoch in range(self.model.epochs):

        for (inputs, targets, basin_ids) in self.model.dataloader:

            # Zero the parameter gradients
            self.model.optimizer.zero_grad()

            # Transfer to device
            inputs = inputs.to(self.device_to_train, non_blocking=True)
            targets = targets.to(self.device_to_train, non_blocking=True) 

            # Forward pass
            q_sim, s_snow, s_water = self.model(inputs, basin_ids[0])

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

            # Backward pass
            loss.backward()
            # Update the weights
            self.model.optimizer.step()
            # Gradient clipping
            if self.model.cfg.clip_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.pretrainer.nnmodel.parameters(), self.model.cfg.clip_gradient_norm)
               