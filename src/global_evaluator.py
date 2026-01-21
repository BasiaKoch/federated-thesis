"""
Global Model Evaluator for NVFlare

This widget evaluates the global model on the FULL MNIST test set after each round,
enabling convergence plots (accuracy/loss vs rounds).

This addresses the dissertation requirement:
"Plot global test accuracy vs communication rounds for FedAvg and FedProx"
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants

from model import create_model


class MNISTTestDataset(Dataset):
    """MNIST test dataset."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images.astype(np.float32) / 255.0
        self.images = (self.images - 0.1307) / 0.3081
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class GlobalModelEvaluator(FLComponent):
    """
    Evaluates global model on full MNIST test set after each aggregation round.

    Tracks:
    - Global test accuracy per round
    - Global test loss per round
    - Enables convergence plots

    Args:
        data_file: Path to mnist.npz
        model_type: 'logistic' or 'cnn'
        output_file: Path to save metrics JSON
    """

    def __init__(
        self,
        data_file: str = "mnist.npz",
        model_type: str = "logistic",
        output_file: str = "global_metrics.json"
    ):
        super().__init__()
        self.data_file = data_file
        self.model_type = model_type
        self.output_file = output_file

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.test_loader: Optional[DataLoader] = None
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            'rounds': [],
            'global_test_loss': [],
            'global_test_accuracy': []
        }

        self._initialized = False

    def _initialize(self, fl_ctx: FLContext):
        """Initialize model and test data."""
        if self._initialized:
            return

        self.log_info(fl_ctx, f"Initializing GlobalModelEvaluator")
        self.log_info(fl_ctx, f"Using device: {self.device}")

        # Initialize model
        self.model = create_model(self.model_type)
        self.model.to(self.device)

        # Load test data - resolve path
        data_path = self._resolve_path(self.data_file)
        if not data_path.exists():
            self.log_error(fl_ctx, f"Data file not found: {data_path}")
            return

        self.log_info(fl_ctx, f"Loading MNIST test data from {data_path}")
        data = np.load(data_path)
        x_test = data['x_test']
        y_test = data['y_test']

        test_dataset = MNISTTestDataset(x_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        self.log_info(fl_ctx, f"Loaded {len(test_dataset)} test samples")
        self._initialized = True

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve data file path."""
        path = Path(path_str)
        if path.exists():
            return path

        # Try DATA_ROOT environment variable
        data_root = os.environ.get("DATA_ROOT", "")
        if data_root:
            resolved = Path(data_root) / path.name
            if resolved.exists():
                return resolved

        return path

    def _evaluate_global_model(self, fl_ctx: FLContext) -> tuple:
        """Evaluate current model on full test set."""
        if self.test_loader is None:
            return None, None

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

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle FL events."""
        if event_type == EventType.START_RUN:
            self._initialize(fl_ctx)

        elif event_type == EventType.GLOBAL_WEIGHTS_UPDATED:
            # Called after server aggregates weights each round
            self._on_global_weights_updated(fl_ctx)

        elif event_type == EventType.END_RUN:
            self._save_metrics(fl_ctx)

    def _on_global_weights_updated(self, fl_ctx: FLContext):
        """Evaluate global model after each aggregation."""
        if not self._initialized:
            return

        # Get current round
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, 0)

        # Get global model weights from context
        shareable = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if shareable is None:
            self.log_warning(fl_ctx, "No global model available for evaluation")
            return

        try:
            dxo = from_shareable(shareable)
            weights = dxo.data

            # Load weights into model
            self.model.load_state_dict(
                {k: torch.tensor(v) for k, v in weights.items()}
            )

            # Evaluate
            test_loss, test_acc = self._evaluate_global_model(fl_ctx)

            if test_loss is not None:
                self.metrics['rounds'].append(current_round)
                self.metrics['global_test_loss'].append(test_loss)
                self.metrics['global_test_accuracy'].append(test_acc)

                self.log_info(
                    fl_ctx,
                    f"Round {current_round} - Global Test: Loss={test_loss:.4f}, Acc={test_acc:.4f}"
                )

        except Exception as e:
            self.log_error(fl_ctx, f"Error evaluating global model: {e}")

    def _save_metrics(self, fl_ctx: FLContext):
        """Save metrics to JSON file."""
        if not self.metrics['rounds']:
            self.log_warning(fl_ctx, "No metrics to save")
            return

        # Get workspace path
        run_dir = fl_ctx.get_prop("RUN_DIR", ".")
        output_path = Path(run_dir) / self.output_file

        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        self.log_info(fl_ctx, f"Saved global metrics to {output_path}")

        # Also print summary
        self.log_info(fl_ctx, "=" * 50)
        self.log_info(fl_ctx, "GLOBAL MODEL CONVERGENCE SUMMARY")
        self.log_info(fl_ctx, "=" * 50)
        self.log_info(fl_ctx, f"Total rounds: {len(self.metrics['rounds'])}")
        self.log_info(fl_ctx, f"Final accuracy: {self.metrics['global_test_accuracy'][-1]:.4f}")
        self.log_info(fl_ctx, f"Best accuracy: {max(self.metrics['global_test_accuracy']):.4f}")
        self.log_info(fl_ctx, f"Final loss: {self.metrics['global_test_loss'][-1]:.4f}")
