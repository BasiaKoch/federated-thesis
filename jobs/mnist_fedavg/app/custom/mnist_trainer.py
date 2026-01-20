"""
NVFlare MNIST Trainer with FedProx Support

This trainer implements local training for federated learning on MNIST
with optional FedProx proximal term regularization.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from model import MNISTLogisticRegression


class MNISTDataset(Dataset):
    """Custom Dataset for MNIST loaded from npz file."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: list):
        """
        Args:
            images: Full array of images (N, 28, 28)
            labels: Full array of labels (N,)
            indices: List of indices for this client's subset
        """
        self.images = images[indices]
        self.labels = labels[indices]

        # Normalize: MNIST mean=0.1307, std=0.3081
        self.images = self.images.astype(np.float32) / 255.0
        self.images = (self.images - 0.1307) / 0.3081

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class MNISTTrainer(Executor):
    """
    NVFlare Executor for MNIST federated training with FedProx.

    Attributes:
        epochs: Number of local epochs per round
        batch_size: Training batch size
        lr: Learning rate
        mu: FedProx proximal term coefficient (0 = FedAvg)
        data_file: Path to mnist.npz file
        partition_file: Path to the partition JSON file
    """

    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        mu: float = 0.0,
        data_file: str = "../data/mnist.npz",
        partition_file: str = "../data/partitions/mnist_noniid_partition.json",
    ):
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mu = mu
        self.data_file = data_file
        self.partition_file = partition_file

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.device: Optional[torch.device] = None
        self.client_id: Optional[int] = None

        self._initialized = False

    def _initialize(self, fl_ctx: FLContext):
        """Initialize model, data, and training components."""
        if self._initialized:
            return

        # Get client name from context
        client_name = fl_ctx.get_identity_name()
        # Extract client ID from name (e.g., "site-1" -> 0)
        try:
            self.client_id = int(client_name.split("-")[-1]) - 1
        except (ValueError, IndexError):
            self.client_id = 0

        self.log_info(fl_ctx, f"Initializing trainer for client {self.client_id}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info(fl_ctx, f"Using device: {self.device}")

        # Initialize model
        self.model = MNISTLogisticRegression()
        self.model.to(self.device)

        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()

        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.0  # No momentum as per FedProx paper
        )

        # Load data
        self._load_data(fl_ctx)

        self._initialized = True

    def _load_data(self, fl_ctx: FLContext):
        """Load MNIST data for this client based on partition file."""
        # Load MNIST from npz file
        data_path = Path(self.data_file)
        if not data_path.exists():
            self.log_error(fl_ctx, f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.log_info(fl_ctx, f"Loading MNIST from {data_path}")
        mnist_data = np.load(data_path)
        x_train = mnist_data['x_train']
        y_train = mnist_data['y_train']
        x_test = mnist_data['x_test']
        y_test = mnist_data['y_test']

        # Load partition file
        partition_path = Path(self.partition_file)
        if not partition_path.exists():
            self.log_error(fl_ctx, f"Partition file not found: {partition_path}")
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        with open(partition_path, 'r') as f:
            partition_data = json.load(f)

        # Get indices for this client
        client_key = str(self.client_id)
        if client_key not in partition_data["train_partition"]:
            self.log_error(fl_ctx, f"Client {self.client_id} not in partition")
            raise ValueError(f"Client {self.client_id} not found in partition")

        train_indices = partition_data["train_partition"][client_key]
        test_indices = partition_data["test_partition"][client_key]

        # Get assigned digits for logging
        digits = partition_data["client_digits"][client_key]
        self.log_info(
            fl_ctx,
            f"Client {self.client_id}: digits {digits}, "
            f"train samples: {len(train_indices)}, test samples: {len(test_indices)}"
        )

        # Create datasets
        train_dataset = MNISTDataset(x_train, y_train, train_indices)
        test_dataset = MNISTDataset(x_test, y_test, test_indices)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    def _train_one_epoch(self, global_weights: dict, fl_ctx: FLContext) -> tuple:
        """
        Train for one epoch with optional FedProx proximal term.

        Args:
            global_weights: Global model weights for FedProx regularization
            fl_ctx: FL context

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            # Add FedProx proximal term if mu > 0
            if self.mu > 0 and global_weights is not None:
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_weights:
                        global_param = global_weights[name].to(self.device)
                        proximal_term += ((param - global_param) ** 2).sum()
                loss = loss + (self.mu / 2) * proximal_term

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate(self) -> tuple:
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Execute the training task."""
        try:
            # Initialize on first call
            self._initialize(fl_ctx)

            if task_name == AppConstants.TASK_TRAIN:
                return self._do_train(shareable, fl_ctx, abort_signal)
            elif task_name == AppConstants.TASK_VALIDATION:
                return self._do_validate(shareable, fl_ctx, abort_signal)
            elif task_name == AppConstants.TASK_SUBMIT_MODEL:
                return self._do_submit_model(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Unknown task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except Exception as e:
            self.log_exception(fl_ctx, f"Error in execute: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _do_train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        """Perform local training."""
        # Get global model weights from shareable
        try:
            dxo = from_shareable(shareable)
        except Exception as e:
            self.log_error(fl_ctx, f"Failed to extract DXO: {e}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"Expected WEIGHTS, got {dxo.data_kind}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        global_weights = dxo.data

        # Load global weights into model
        self.model.load_state_dict(
            {k: torch.tensor(v) for k, v in global_weights.items()}
        )

        # Store global weights for FedProx
        global_weights_tensors = {
            k: torch.tensor(v).clone() for k, v in global_weights.items()
        }

        # Train for specified epochs
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, 0)
        self.log_info(fl_ctx, f"Starting local training for round {current_round}")

        for epoch in range(self.epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            train_loss, train_acc = self._train_one_epoch(
                global_weights_tensors, fl_ctx
            )
            self.log_info(
                fl_ctx,
                f"Round {current_round}, Epoch {epoch + 1}/{self.epochs}: "
                f"Loss={train_loss:.4f}, Acc={train_acc:.4f}"
            )

        # Get updated weights
        updated_weights = {
            k: v.cpu().numpy() for k, v in self.model.state_dict().items()
        }

        # Create DXO with weight diff
        weight_diff = {}
        for k in updated_weights:
            weight_diff[k] = updated_weights[k] - global_weights[k]

        dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=weight_diff,
            meta={
                "NUM_STEPS_CURRENT_ROUND": len(self.train_loader) * self.epochs
            }
        )

        return dxo.to_shareable()

    def _do_validate(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        """Perform validation."""
        # Get model weights from shareable
        try:
            dxo = from_shareable(shareable)
        except Exception as e:
            self.log_error(fl_ctx, f"Failed to extract DXO: {e}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"Expected WEIGHTS, got {dxo.data_kind}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Load weights
        self.model.load_state_dict(
            {k: torch.tensor(v) for k, v in dxo.data.items()}
        )

        # Evaluate
        val_loss, val_acc = self._evaluate()

        self.log_info(fl_ctx, f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # Return metrics
        metrics_dxo = DXO(
            data_kind=DataKind.METRICS,
            data={
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
        )

        return metrics_dxo.to_shareable()

    def _do_submit_model(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        """Submit the current model weights."""
        weights = {
            k: v.cpu().numpy() for k, v in self.model.state_dict().items()
        }

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
        return dxo.to_shareable()
