import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import torch
torch.set_printoptions(precision=10)
import numpy as np
import torch.nn.functional as F
import random

# Function to set the seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: restrict multi-threading
    torch.set_num_threads(1)

# Set the seed for reproducibility
set_seed(42)

device = torch.device("cpu")  # Explicitly set to CPU (ensure no GPU is used)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size[0]
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out  

def generate_sine_wave_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        end = start + seq_length * np.pi / 10
        sequence = np.sin(np.linspace(start, end, seq_length + 1))
        X.append(sequence[:-1])
        y.append(sequence[-1])
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == 1:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def test_models():
    seq_length = 10
    num_samples = 1000
    X_train, y_train = generate_sine_wave_data(seq_length, num_samples)
    X_train_lstm = X_train.unsqueeze(-1)
    input_size = seq_length
    hidden_size = 64
    output_size = 1
    lstm_model = LSTM(input_size=1, hidden_size=[hidden_size], output_size=output_size)
    mlp_model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    lstm_model.to(device)
    mlp_model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_train_lstm = X_train_lstm.to(device)
    print("Training LSTM Model")
    train_model(lstm_model, X_train_lstm, y_train)
    print("Training MLP Model")
    train_model(mlp_model, X_train, y_train)
    lstm_model.eval()
    mlp_model.eval()
    with torch.no_grad():
        lstm_predictions = lstm_model(X_train_lstm).squeeze().numpy()
        mlp_predictions = mlp_model(X_train).squeeze().numpy()
    lstm_mse = mean_squared_error(y_train.numpy(), lstm_predictions)
    mlp_mse = mean_squared_error(y_train.numpy(), mlp_predictions)
    print(f"LSTM Model MSE: {lstm_mse}")
    print(f"MLP Model MSE: {mlp_mse}")

if __name__ == '__main__':
    print('torch version:', torch.__version__)
    test_models()
