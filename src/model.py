"""
MNIST Models for Federated Learning

Two model options:
1. MNISTLogisticRegression - Simple linear classifier (FedProx paper baseline)
2. MNISTCNN - Small CNN for better feature extraction (recommended)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    Small CNN for MNIST classification.

    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - MaxPool: 2x2
    - FC1: 9216 -> 128
    - FC2: 128 -> 10

    Total parameters: ~1.2M (more expressive than logistic regression)
    """

    def __init__(self, num_classes: int = 10):
        super(MNISTCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # After 2 conv layers with padding=1 and 2 max pools: 28->14->7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Ensure input is (batch, 1, 28, 28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)  # -> (batch, 64*7*7)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class MNISTLogisticRegression(nn.Module):
    """
    Multinomial Logistic Regression for MNIST classification.

    This is a simple linear model: y = softmax(Wx + b)
    - Input: 784 features (flattened 28x28 images)
    - Output: 10 classes (digits 0-9)
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super(MNISTLogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Single linear layer
        self.linear = nn.Linear(input_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)

        Returns:
            Logits of shape (batch_size, 10)
        """
        # Flatten input if necessary
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Linear transformation (logits)
        logits = self.linear(x)

        return logits


def create_model(model_type: str = "logistic") -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: "logistic" for MNISTLogisticRegression, "cnn" for MNISTCNN

    Returns:
        Initialized model
    """
    if model_type.lower() == "cnn":
        return MNISTCNN()
    elif model_type.lower() == "logistic":
        return MNISTLogisticRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'logistic' or 'cnn'")


def get_model_params_count(model: nn.Module) -> int:
    """Get total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    for model_type in ["logistic", "cnn"]:
        print(f"\n{'='*50}")
        print(f"Model Type: {model_type.upper()}")
        print('='*50)

        model = create_model(model_type)
        print(f"Architecture:\n{model}")
        print(f"Total parameters: {get_model_params_count(model):,}")

        # Test forward pass
        dummy_input = torch.randn(32, 1, 28, 28)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
