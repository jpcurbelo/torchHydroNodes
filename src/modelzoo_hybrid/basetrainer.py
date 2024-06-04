from tqdm import tqdm
import sys


class BaseHybridModelTrainer:

    def __init__(self, model):
        
        self.model = model

    def train(self):

        if self.model.cfg.verbose:
            print("-- Training the hybrid model --")

        for epoch in range(self.model.epochs):

            pbar = tqdm(self.model.dataloader, disable=self.model.cfg.disable_pbar, file=sys.stdout)
            pbar.set_description(f'# Epoch {epoch + 1:05d}')

            for (inputs, targets, basin_ids) in pbar:

                # Zero the parameter gradients
                self.model.optimizer.zero_grad()

                # Forward pass
                Q_model = self.model(inputs.to(self.model.device), basin_ids[0])

                # print('Q_model', Q_model)


            pbar.close()