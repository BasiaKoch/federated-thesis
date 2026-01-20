"""
Multinomial Logistic Regression Model for MNIST

This is a simple linear classifier as specified in the FedProx paper:
- Input: Flattened 28x28 = 784 dimensional vectors
- Output: 10 class logits
- No hidden layers, no CNNs
"""

import torch
import torch.nn as nn


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


def create_model() -> nn.Module:
    """Factory function to create the model."""
    return MNISTLogisticRegression()


def get_model_params_count(model: nn.Module) -> int:
    """Get total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_model()
    print(f"Model: {model}")
    print(f"Total parameters: {get_model_params_count(model):,}")

    # Test forward pass
    dummy_input = torch.randn(32, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
