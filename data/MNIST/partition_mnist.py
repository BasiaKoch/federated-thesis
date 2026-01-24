"""
Non-IID MNIST Data Partitioning Script

This script creates a non-IID partition of MNIST data following the FedProx paper setup:
1. Label Skew: Each client receives data from only 2 digit classes
2. Quantity Skew: Number of samples per client follows a power-law distribution

Usage:
    python partition_mnist.py --num_clients 30 --output_dir ./partitions
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_mnist(data_file: str) -> tuple:
    """Load MNIST dataset from local npz file."""
    print(f"Loading MNIST from {data_file}...")
    data = np.load(data_file)

    x_train = data['x_train']  # (60000, 28, 28)
    y_train = data['y_train']  # (60000,)
    x_test = data['x_test']    # (10000, 28, 28)
    y_test = data['y_test']    # (10000,)

    print(f"  Train: {x_train.shape[0]} samples")
    print(f"  Test: {x_test.shape[0]} samples")

    return (x_train, y_train), (x_test, y_test)


def get_label_indices(labels: np.ndarray) -> dict:
    """Get indices of samples for each label."""
    label_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        label_indices[int(label)].append(idx)

    return dict(label_indices)


def assign_digit_pairs(num_clients: int, num_classes: int = 10, seed: int = 42) -> dict:
    """
    Assign two digit classes to each client cyclically.

    For 30 clients with 10 digit classes:
    - Client 0: digits 0, 1
    - Client 1: digits 2, 3
    - Client 2: digits 4, 5
    - Client 3: digits 6, 7
    - Client 4: digits 8, 9
    - Client 5: digits 0, 1 (cycle repeats)
    - ...and so on
    """
    np.random.seed(seed)

    # Create digit pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    digit_pairs = [(i, i + 1) for i in range(0, num_classes, 2)]

    client_digits = {}
    for client_id in range(num_clients):
        pair_idx = client_id % len(digit_pairs)
        client_digits[client_id] = digit_pairs[pair_idx]

    return client_digits


def generate_power_law_samples(num_clients: int, total_samples: int,
                                alpha: float = 1.5, seed: int = 42) -> np.ndarray:
    """
    Generate number of samples per client following a power-law distribution.

    Args:
        num_clients: Number of clients
        total_samples: Total number of samples to distribute
        alpha: Power-law exponent (higher = more skewed)
        seed: Random seed

    Returns:
        Array of sample counts per client
    """
    np.random.seed(seed)

    # Generate power-law distributed values
    # Using Zipf-like distribution: p(k) ~ k^(-alpha)
    ranks = np.arange(1, num_clients + 1)
    weights = 1.0 / np.power(ranks, alpha)

    # Shuffle to randomize which clients get more data
    np.random.shuffle(weights)

    # Normalize to get proportions
    proportions = weights / weights.sum()

    # Calculate sample counts, ensuring minimum of 100 samples per client
    min_samples = 100
    available_samples = total_samples - (min_samples * num_clients)

    if available_samples < 0:
        raise ValueError(f"Not enough samples. Need at least {min_samples * num_clients}")

    # Distribute extra samples according to power-law
    extra_samples = (proportions * available_samples).astype(int)
    sample_counts = min_samples + extra_samples

    # Adjust for rounding errors
    diff = total_samples - sample_counts.sum()
    if diff > 0:
        # Add remaining samples to largest clients
        top_indices = np.argsort(sample_counts)[-diff:]
        sample_counts[top_indices] += 1
    elif diff < 0:
        # Remove excess from largest clients
        top_indices = np.argsort(sample_counts)[diff:]
        sample_counts[top_indices] -= 1

    return sample_counts


def partition_data(label_indices: dict, client_digits: dict,
                   sample_counts: np.ndarray, seed: int = 42) -> dict:
    """
    Partition data indices among clients based on their assigned digits and sample counts.

    Args:
        label_indices: Dict mapping labels to their indices
        client_digits: Dict mapping client_id to tuple of two assigned digits
        sample_counts: Array of sample counts per client

    Returns:
        Dict mapping client_id to list of data indices
    """
    np.random.seed(seed)

    # Create pools of available indices for each digit
    available_indices = {
        digit: list(indices) for digit, indices in label_indices.items()
    }
    for digit in available_indices:
        np.random.shuffle(available_indices[digit])

    # Track usage pointers for each digit
    digit_pointers = {digit: 0 for digit in range(10)}

    client_indices = {}

    for client_id in range(len(sample_counts)):
        digit1, digit2 = client_digits[client_id]
        num_samples = sample_counts[client_id]

        # Split samples roughly equally between the two digits
        samples_per_digit = num_samples // 2
        extra = num_samples % 2

        indices = []

        # Get samples from first digit
        start = digit_pointers[digit1]
        end = start + samples_per_digit + extra
        pool = available_indices[digit1]

        # Handle wraparound if we run out of samples
        if end <= len(pool):
            indices.extend(pool[start:end])
            digit_pointers[digit1] = end
        else:
            # Take what's available and wrap around
            indices.extend(pool[start:])
            remaining = (samples_per_digit + extra) - (len(pool) - start)
            indices.extend(pool[:remaining])
            digit_pointers[digit1] = remaining

        # Get samples from second digit
        start = digit_pointers[digit2]
        end = start + samples_per_digit
        pool = available_indices[digit2]

        if end <= len(pool):
            indices.extend(pool[start:end])
            digit_pointers[digit2] = end
        else:
            indices.extend(pool[start:])
            remaining = samples_per_digit - (len(pool) - start)
            indices.extend(pool[:remaining])
            digit_pointers[digit2] = remaining

        client_indices[client_id] = indices

    return client_indices


def create_partition(num_clients: int = 30,
                     data_file: str = "./mnist.npz",
                     output_dir: str = "./partitions",
                     alpha: float = 1.5,
                     seed: int = 42) -> dict:
    """
    Create and save the non-IID MNIST partition.

    Args:
        num_clients: Number of federated clients
        data_file: Path to mnist.npz file
        output_dir: Directory to save partition files
        alpha: Power-law exponent for quantity skew
        seed: Random seed for reproducibility

    Returns:
        Partition metadata dictionary
    """
    print(f"Creating non-IID MNIST partition for {num_clients} clients...")

    # Load MNIST from local file
    (x_train, y_train), (x_test, y_test) = load_mnist(data_file)

    # Get label indices
    print("Organizing samples by label...")
    train_label_indices = get_label_indices(y_train)
    test_label_indices = get_label_indices(y_test)

    # Assign digit pairs to clients
    print("Assigning digit pairs to clients...")
    client_digits = assign_digit_pairs(num_clients, seed=seed)

    # Generate power-law sample counts
    total_train_samples = len(y_train)
    print(f"Generating power-law sample distribution (alpha={alpha})...")
    sample_counts = generate_power_law_samples(
        num_clients, total_train_samples, alpha=alpha, seed=seed
    )

    # Partition training data
    print("Partitioning training data...")
    train_partition = partition_data(
        train_label_indices, client_digits, sample_counts, seed=seed
    )

    # For test data, give each client samples from their assigned digits
    # Use equal splits for evaluation
    print("Partitioning test data...")
    test_partition = {}
    for client_id in range(num_clients):
        digit1, digit2 = client_digits[client_id]
        indices = test_label_indices[digit1] + test_label_indices[digit2]
        test_partition[client_id] = indices

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save partitions
    partition_data_dict = {
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "client_digits": {str(k): list(v) for k, v in client_digits.items()},
        "train_partition": {str(k): v for k, v in train_partition.items()},
        "test_partition": {str(k): v for k, v in test_partition.items()},
        "sample_counts": sample_counts.tolist()
    }

    partition_file = output_path / "mnist_noniid_partition.json"
    print(f"Saving partition to {partition_file}...")
    with open(partition_file, 'w') as f:
        json.dump(partition_data_dict, f)

    # Print statistics
    print("\n" + "=" * 60)
    print("PARTITION STATISTICS")
    print("=" * 60)
    print(f"Total training samples: {total_train_samples}")
    print(f"Total test samples: {len(y_test)}")
    print(f"Number of clients: {num_clients}")
    print(f"Power-law alpha: {alpha}")
    print()
    print("Sample distribution across clients:")
    print(f"  Min samples: {sample_counts.min()}")
    print(f"  Max samples: {sample_counts.max()}")
    print(f"  Mean samples: {sample_counts.mean():.1f}")
    print(f"  Std samples: {sample_counts.std():.1f}")
    print()
    print("Client details (first 10):")
    print("-" * 40)
    for i in range(min(10, num_clients)):
        d1, d2 = client_digits[i]
        n = sample_counts[i]
        print(f"  Client {i:2d}: digits [{d1}, {d2}], samples: {n:5d}")
    if num_clients > 10:
        print(f"  ... and {num_clients - 10} more clients")
    print("=" * 60)

    return partition_data_dict


def main():
    parser = argparse.ArgumentParser(
        description="Create non-IID MNIST partition for federated learning"
    )
    parser.add_argument(
        "--num_clients", type=int, default=30,
        help="Number of federated clients (default: 30)"
    )
    parser.add_argument(
        "--data_file", type=str, default="./mnist.npz",
        help="Path to mnist.npz file (default: ./mnist.npz)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./partitions",
        help="Directory to save partition files (default: ./partitions)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.5,
        help="Power-law exponent for quantity skew (default: 1.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    create_partition(
        num_clients=args.num_clients,
        data_file=args.data_file,
        output_dir=args.output_dir,
        alpha=args.alpha,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
