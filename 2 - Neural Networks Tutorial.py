# %%%%%%%%%%%%%%%%%%%% IMPORTING THE PACKAGES %%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# %%%%%%%%%%%%%%%%% CREATING FULLY CONNECTED NEURAL NETWORKS %%%%%%%%%%%%%%%
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # size = 784 (28x8x)
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return x


# %%%%%%%%%%%%%%%%% SET THE DEVICE %%%%%%%%%%%%%%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%%%%%%%%%%%%%%%% HYPERPARAMETERS %%%%%%%%%%%%%%%
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# %%%%%%%%%%%%%%%%% LOADING DATA %%%%%%%%%%%%%%%
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

tes_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# %%%%%%%%%%%%%%%%% INITIALISING THE NETWORK %%%%%%%%%%%%%%%
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# %%%%%%%%%%%%%%%%% TRAINING NEURAL NETWORKS %%%%%%%%%%%%%%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%%%%%%%%%%%%%%%% TRAINING NEURAL NETWORKS %%%%%%%%%%%%%%%
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        print(data.shape)