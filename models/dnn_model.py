import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.relu4 = nn.ReLU()
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        y_pred = self.fc4(x)
        y_pred = self.relu4(y_pred)
        # y_pred shape: (batch_size, output_size)
        return y_pred
