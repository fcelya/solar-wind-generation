import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Example usage
# Define hyperparameters
input_size = 1  # input size is 1 for univariate time series
hidden_size = 64
output_size = 1
num_layers = 2
seq_length = 10  # length of input sequence (L)
predict_length = 5  # length of prediction (H)

# Generate dummy data
data = np.sin(np.arange(1000) * 0.1) + np.random.randn(1000) * 0.1  # dummy sine wave data

# Prepare data
def prepare_data(data, seq_length, predict_length):
    X, y = [], []
    for i in range(len(data) - seq_length - predict_length):
        seq_in = data[i:i + seq_length]
        seq_out = data[i + seq_length:i + seq_length + predict_length]
        X.append(seq_in)
        y.append(seq_out)
    return torch.tensor(X).unsqueeze(-1).float(), torch.tensor(y).float()

# Prepare data
X, y = prepare_data(data, seq_length, predict_length)

# Split data into train and test sets
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Initialize model, loss function, and optimizer
model = LSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(train_X)
    loss = criterion(outputs, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    test_outputs = model(test_X)
    test_loss = criterion(test_outputs, test_y)
    print(f'Test Loss: {test_loss.item():.4f}')

# Making predictions
future = 100  # number of future time steps to predict
with torch.no_grad():
    pred = data[-seq_length:].tolist()  # initial sequence to start predicting
    for _ in range(future):
        seq = torch.tensor(pred[-seq_length:]).view(1, seq_length, 1).float()
        model_output = model(seq)
        pred.append(model_output.item())

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(range(len(data) - future, len(data)), pred[seq_length:], label='Predictions')
plt.legend()
plt.show()