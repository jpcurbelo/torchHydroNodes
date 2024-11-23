import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict


class HydrologyPINN_v0(nn.Module):
    """
    A PyTorch neural network model for hydrology-based physics-informed neural networks (PINNs).
    """

    def __init__(self, input_size, output_size, 
                 hidden_layers, 
                 params_bounds=None, scaler=None):
        """
        Initialize the neural network with variable hidden layers.
        
        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output features.
            hidden_layers (list of int): A list where each element represents the number of neurons in each hidden layer.
            params_bounds (list of tuples): A list of tuples where each tuple represents the lower and upper bounds for each parameter
        """
        super(HydrologyPINN_v0, self).__init__()

        # Save the scaler
        self.scaler = scaler
        
        # Set up the neural network layers
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.Tanh()]
        
        # Add hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.Tanh())
        
        # Add output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        # Relu activation function
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        
        # Now initialize parameters with the same order guaranteed
        self.params = nn.ParameterDict(OrderedDict({
            name: nn.Parameter(
                torch.Tensor(1).uniform_(bounds[0], bounds[1]), requires_grad=True
            ) for name, bounds in params_bounds.items()
        }))
        self.params_bounds = params_bounds  # Store bounds for clamping

    def forward(self, x):
        # Scale the input if a scaler is provided
        if self.scaler is not None:
            x = torch.tensor(self.scaler.transform(x.cpu().numpy()), dtype=x.dtype, device=x.device)
        return self.network(x)

    def get_clamped_params(self):
        """
        Return parameters clamped within their bounds.
        """
        return {name: torch.clamp(param, *self.params_bounds[name]) for name, param in self.params.items()}
 

class HydrologyPINN_v1(nn.Module):
    """
    A PyTorch neural network model for hydrology-based physics-informed neural networks (PINNs).
    """

    def __init__(self, input_size, hidden_layers, 
                 params_bounds=None, scaler=None, step_function=None):
        """
        Initialize the neural network with variable hidden layers.
        
        Args:
            input_size (int): The number of input features.
            hidden_layers (list of int): A list where each element represents the number of neurons in each hidden layer.
            params_bounds (dict): Dictionary where each key-value pair represents a parameter and its lower/upper bounds.
            scaler (optional): Scaler to normalize/denormalize inputs.
            step_function (callable): Function to enforce positive outputs in Qb and Qs calculations.
        """
        super(HydrologyPINN_v1, self).__init__()

        # Save the scaler and step function
        self.scaler = scaler

        # Set up the neural network layers
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.Tanh()]

       # Add hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.Tanh())
  
        # Add output layer with 2 outputs: S0 and S1
        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.network = nn.Sequential(*layers)

        # Initialize parameters using bounds
        self.params = nn.ParameterDict(OrderedDict({
            name: nn.Parameter(
                torch.Tensor(1).uniform_(bounds[0], bounds[1]), requires_grad=True
            ) for name, bounds in params_bounds.items()
        }))

        # basinID,S0,S1,f,Smax,Qmax,Df,Tmax,Tmin,UNKNOWN
        # 1.0135e6,0.0,1303.0042478479704,0.0167447802633775,1709.4610152413964,18.46996175240424,2.674548847651345,0.17573919612506747,-2.0929590840638728,0.8137969540102923
        self.params['S0'] = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.params['S1'] = nn.Parameter(torch.tensor([1303.0042478479704]), requires_grad=False)
        self.params['f'] = nn.Parameter(torch.tensor([0.0167447802633775]), requires_grad=False)
        self.params['Smax'] = nn.Parameter(torch.tensor([1709.4610152413964]), requires_grad=False)
        self.params['Qmax'] = nn.Parameter(torch.tensor([18.46996175240424]), requires_grad=False)
        self.params['Df'] = nn.Parameter(torch.tensor([2.674548847651345]), requires_grad=False)
        self.params['Tmax'] = nn.Parameter(torch.tensor([0.17573919612506747]), requires_grad=False)
        self.params['Tmin'] = nn.Parameter(torch.tensor([-2.0929590840638728]), requires_grad=False)


        self.params_bounds = params_bounds  # Store bounds for clamping

    def step_function(self, x):
        """
        Step function to enforce positive outputs in Qb and Qs calculations.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output with step function applied.
        """
        # return (torch.tanh(5.0 * x) + 1.0) * 0.5
        return torch.sigmoid(5.0 * x)  # Smooth transition for step-like behavior

    def Qb(self, s1):
        """
        Calculate Qb using learned parameters f, smax, qmax, and the step function.
        """
        f = self.params['f']
        smax = self.params['Smax']
        qmax = self.params['Qmax']

        # # Calculate exp_term 
        # exp_term = torch.exp(-f * (smax - s1))
        # # Calculate result with step function application
        # result = (self.step_function(s1) * self.step_function(s1 - smax) * qmax +
        #         self.step_function(s1) * self.step_function(smax - s1) * qmax * exp_term)

        # # Calculate Qmax * exp(-f * (Smax - S1)) for S1 <= Smax, otherwise Qmax
        # exp_term = qmax * torch.exp(-f * (smax - s1))
        # result = torch.where(s1 <= smax, exp_term, qmax)

        # print('s1:', s1[:5], s1[-5:])
        # print('exp_term:', exp_term)
        # print('result:', result)
        # # aux = input('Press Enter to continue...')

        # print('s1:', s1[:2], s1[-2:])
        # result = s1 / 2

        # # Smooth approximation for the condition S1 <= Smax
        # smooth_transition = torch.sigmoid(500 * (smax - s1))  # Adjust scaling factor as needed
        # # Approximate Q_bucket using a soft switch
        # exp_term = qmax * torch.exp(-f * (smax - s1))
        # result = smooth_transition * exp_term + (1 - smooth_transition) * qmax

        # print('s1:', s1[:2], s1[-2:])
        # print('smooth_transition:', smooth_transition[:2], smooth_transition[-2:])
        # print('exp_term:', exp_term[:2], exp_term[-2:])
        # print('result:', result[:2], result[-2:])

        # return step_fct(s1) * step_fct(s1 - smax) * qmax + step_fct(s1) * step_fct(smax - s1) * qmax * exp_term
        print('smax:', smax)
        print('s1:', s1[:2], s1[-2:])
        print('smax - s1', (smax - s1)[:2], (smax - s1)[-2:])
        exp_term = torch.exp(torch.clamp(-f * (smax - s1), min=-50, max=50))
        result = self.step_function(s1) * self.step_function(s1 - smax) * qmax + self.step_function(s1) * (1 - self.step_function(s1 - smax)) * qmax * exp_term
        # result = exp_term

        print('result:', result[:2], result[-2:])

        return result

    def Qs(self, s1):
        """
        Calculate Qs using learned parameter smax and the step function.
        """
        smax = self.params['Smax']
        
        # # Calculate result with step function application 
        # result = self.step_function(s1) * self.step_function(s1 - smax) * (s1 - smax)

        # # Calculate 0 for S1 <= Smax, otherwise S1 - Smax
        # spill_term = s1 - smax
        # result = torch.where(s1 <= smax, torch.zeros_like(s1), spill_term)

        # result = s1 / 2
        result = torch.where(s1 <= smax, torch.zeros_like(s1), s1 - smax)

        # # Smooth approximation for the condition S1 > Smax
        # smooth_transition = torch.sigmoid(500 * (s1 - smax))  # Adjust scaling factor as needed
        # # Approximate Q_spill using a soft switch
        # result = (s1 - smax) * smooth_transition

        # print('s1:', s1[:2], s1[-2:])
        # print('smooth_transition:', smooth_transition[:2], smooth_transition[-2:])
        # print('result:', result[:2], result[-2:])

        return result
    
    def Q_total(self, s1):
        """
        Calculate total discharge Q as the sum of Q_bucket and Q_spill.
        """
        return self.Qb(s1) + self.Qs(s1)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor.
            Returns:
                S0, S1, q_bucket (torch.Tensor): Outputs of the model.
        """
        # Pass through the network to get S0 and S1
        S0, S1 = self.network(x).split(1, dim=-1)

        # Apply ReLU to enforce non-negative values for storages
        S0 = torch.relu(S0)
        S1 = torch.relu(S1)

        # # S0 = torch.nn.functional.softplus(S0)
        # # S1 = torch.nn.functional.softplus(S1)

        # Calculate q_bucket using S1, based on Q_total
        q_bucket = self.Q_total(S1)

        # Apply ReLU to enforce non-negative values for q_bucket
        q_bucket = torch.relu(q_bucket)
        
        # Ensure S1 retains its gradient for inspection
        S0.retain_grad()
        S1.retain_grad()
        q_bucket.retain_grad()

        return S0, S1, q_bucket
    
    def get_clamped_params(self):
        """
        Return parameters clamped within their bounds.
        """
        return {name: torch.clamp(param, *self.params_bounds[name]) for name, param in self.params.items()}
 




def setup_optimizer_and_scheduler(model, lr, epochs):
    """
    Set up the optimizer and scheduler for the model.
    
    Args:
        model (nn.Module): The PyTorch model to optimize.
        lr (float or dict): The initial learning rate for the optimizer or a dictionary with scheduler parameters.
        epochs (int): Total number of training epochs, used to set the scheduler's step size.
    
    Returns:
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): The learning rate scheduler, if applicable.
    """
    
    # Check if lr is a scalar (float or int)
    if isinstance(lr, (float, int)):
        # Optimizer with scalar learning rate
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None  # No scheduler for scalar lr

    else:
        # Optimizer with initial learning rate from lr dictionary
        optimizer = optim.Adam(model.parameters(), lr=lr["initial"])

        # Scheduler with decay based on epochs and decay_step_fraction
        step_size = max(1, epochs // lr["decay_step_fraction"])  # Ensure step_size is at least 1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr["decay"])

    return optimizer, scheduler

# Define the data loss and physics-based loss functions
def data_loss(predicted, observed):

    return nn.MSELoss()(predicted, observed)

def physics_loss(model, basin, predicted_params, observed, collocation_indices):
    """
    Calculate the physics-based loss on collocation points.
    
    Args:
        model: The ODE model for simulation.
        basin: The current basin being processed.
        predicted_params: Parameters predicted by the PINN.
        observed (torch.Tensor): Observed data tensor.
        collocation_indices (torch.Tensor): Indices for collocation points.
    
    Returns:
        torch.Tensor: The calculated physics loss.
    """
    # Run the model simulation and get the full time series
    simulated = model.run(basin, basin_params=predicted_params, use_grad=True)
    runoff_sim = simulated[-1]  # Extract the runoff component from the model output
    
    # Subset observed and simulated data using collocation indices
    observed_colloc = torch.index_select(observed, 0, collocation_indices)
    runoff_sim_colloc = torch.index_select(runoff_sim, 0, collocation_indices)

    # Calculate the physics-based penalty for the collocation points
    physics_penalty = runoff_sim_colloc - observed_colloc

    return torch.mean(physics_penalty ** 2), runoff_sim

def pinn_loss(predicted, observed, predicted_params, model, basin, 
              epoch, collocation_indices):
    
    # print(f'Predicted: {predicted[-1]}')
    # print(f'Observed: {observed}')

    # print(f'Predicted: len({len(predicted)})', predicted[-1].shape)
    # print(f'Observed: {observed.shape}')
    
    dataloss = data_loss(predicted[-1], observed)

    # physicsloss, simulated_ode = physics_loss(model, basin, predicted_params, observed, collocation_indices)
    # # icloss = nn.MSELoss()(predicted[0], observed[0])
    # icloss = nn.MSELoss()(predicted[:10], observed[:10])

    # print(f'DataLoss: {dataloss:.3e}, PhysicsLoss: {physicsloss:.3e}, IC Loss: {icloss:.3e}')
    # plot_results(observed, predicted, simulated_ode, basin, epoch, period='train') 

    if (epoch % 100) == 0:
        plot_results(observed, predicted[-1], None, basin, epoch, period='train') 
    # return dataloss + physicsloss + icloss

    return dataloss

def plot_results(observed, predicted, simulated_ode, basin, epoch, period='train'):

    nse_nn  = nse_loss(predicted, observed)
    # nse_ode = nse_loss(simulated_ode, observed)

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(observed, label='Observed', color='blue')
    plt.plot(predicted.detach().cpu().numpy(), label=f'Predicted_NN  (NSE: {nse_nn:.3f})',
             color='red', linestyle='--')
    # plt.plot(simulated_ode.detach().cpu().numpy(), label=f'Simulated_ODE (NSE: {nse_ode:.3f})',
    #          color='green', linestyle=':')
    
    # plt.scatter(range(len(simulated_ode)), simulated_ode.detach().cpu().numpy(), 
    #                 label=f'Simulated_ODE (NSE: {nse_ode:.3f})', color='green', 
    #                 marker='o', s=10)
    
    # # Plot a marker on the first value of simulated ODE
    # plt.scatter(0, simulated_ode.detach().cpu().numpy()[0], color='k', marker='X', s=50)
    
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.title(f'Observed vs. Predicted | {period} | epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'{basin}_pins_simulation.png')
    plt.clf()

def nse_loss(predicted, observed):

    return 1 - torch.sum((predicted - observed) ** 2) / (torch.sum((observed - torch.mean(observed)) ** 2) + torch.finfo(torch.float32).eps)

def get_collocation_indices(total_length, coll_fraction=0.6, seed=42):
    """
    Generate random indices for collocation points based on a fraction of the total time series,
    ensuring that the points are distributed over the entire domain.
    
    Args:
        total_length (int): Total length of the time series.
        coll_fraction (float): Fraction of time series to use as collocation points.
        seed (int): Random seed for reproducibility.
        
    Returns:
        torch.Tensor: Tensor containing the random indices for collocation points.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Calculate the number of collocation points based on the fraction
    num_collocation_points = int(total_length * coll_fraction)
    
    # Randomly sample indices across the full range of the time series
    collocation_indices = random.sample(range(total_length), num_collocation_points)
    
    # Sort indices to maintain order in time (optional)
    collocation_indices.sort()
    
    # print(f'Number of collocation points: {num_collocation_points}')
    # print(f'Max index: {max(collocation_indices)}')
    # print(f'Min index: {min(collocation_indices)}')

    return torch.tensor(collocation_indices, dtype=torch.long)

##################################################




if __name__ == "__main__":
    pass