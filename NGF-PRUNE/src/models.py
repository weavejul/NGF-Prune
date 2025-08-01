from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SIMPLE_MLP_DEFAULTS, LENET5_DEFAULTS, POLICY_MLP_DEFAULTS

class SimpleMLP(nn.Module):
    """Simple MLP for MNIST/FashionMNIST classification."""
    def __init__(self, 
                 input_dim: int = SIMPLE_MLP_DEFAULTS['input_dim'], 
                 hidden_dim1: int = SIMPLE_MLP_DEFAULTS['hidden_dim1'], 
                 hidden_dim2: int = SIMPLE_MLP_DEFAULTS['hidden_dim2'], 
                 output_dim: int = SIMPLE_MLP_DEFAULTS['output_dim']) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5(nn.Module):
    """LeNet-5 for MNIST/FashionMNIST classification."""
    def __init__(self, num_classes=LENET5_DEFAULTS['num_classes']):
        super().__init__()
        self.conv1 = nn.Conv2d(1, LENET5_DEFAULTS['conv1_out_channels'], 
                              kernel_size=LENET5_DEFAULTS['conv_kernel_size'], 
                              stride=1, padding=2)
        self.conv2 = nn.Conv2d(LENET5_DEFAULTS['conv1_out_channels'], 
                              LENET5_DEFAULTS['conv2_out_channels'], 
                              kernel_size=LENET5_DEFAULTS['conv_kernel_size'], 
                              stride=1, padding=0)
        self.fc1 = nn.Linear(LENET5_DEFAULTS['fc1_input_size'], LENET5_DEFAULTS['fc1_output_size'])
        self.fc2 = nn.Linear(LENET5_DEFAULTS['fc1_output_size'], LENET5_DEFAULTS['fc2_output_size'])
        self.fc3 = nn.Linear(LENET5_DEFAULTS['fc2_output_size'], num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyMLP(nn.Module):
    """A simple MLP policy network for discrete action spaces like CartPole."""
    def __init__(self, 
                 input_dim=POLICY_MLP_DEFAULTS['input_dim'], 
                 hidden_dim=POLICY_MLP_DEFAULTS['hidden_dim'], 
                 output_dim=POLICY_MLP_DEFAULTS['output_dim']):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)
        return action_logits