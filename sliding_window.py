"""
Sliding window data loading utilities for LSTM training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def create_sliding_windows(
    data: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None = None,
    window_size: int = 60,
    step_size: int = 1,
    normalize: bool = True,
    drop_nulls: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Create sliding windows from time series data for LSTM input.
    
    Args:
        data: DataFrame with time series data (sorted by time)
        feature_cols: column names to use as features (e.g., ['TIME', 'COUNTS'])
        label_col: column for labels (if None, only returns features)
        window_size: number of timesteps per window (e.g., 60 for 60-minute window)
        step_size: stride for sliding window (1 for overlapping, >1 for sparser)
        normalize: if True, standardize features to mean=0, std=1
        drop_nulls: if True, skip windows with any NaN values
        
    Returns:
        X: (num_windows, window_size, num_features) features array
        y: (num_windows,) labels array, or None if label_col not provided
        
    Example:
        >>> df = load_solexs_data(...)  # TIME, COUNTS columns
        >>> X, y = create_sliding_windows(
        ...     df,
        ...     feature_cols=['COUNTS'],  # Just use counts as feature
        ...     label_col='is_flare',     # Binary flare indicator
        ...     window_size=60            # 60-step window
        ... )
        >>> # X shape: (6000, 60, 1) - 6000 windows, 60 timesteps, 1 feature
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    data = data.reset_index(drop=True)
    features = data[feature_cols].values.astype(np.float32)
    
    # Drop NaN rows if requested
    if drop_nulls:
        valid_idx = ~np.any(np.isnan(features), axis=1)
        features = features[valid_idx]
        if label_col is not None:
            labels = data.loc[valid_idx, label_col].values
        else:
            labels = None
    else:
        labels = data[label_col].values if label_col is not None else None
    
    # Normalize features
    if normalize:
        feature_mean = np.nanmean(features, axis=0, keepdims=True)
        feature_std = np.nanstd(features, axis=0, keepdims=True)
        feature_std[feature_std == 0] = 1  # Avoid division by zero
        features = (features - feature_mean) / feature_std
    
    # Create sliding windows
    num_samples = features.shape[0]
    windows_X = []
    windows_y = []
    
    for i in range(0, num_samples - window_size + 1, step_size):
        window = features[i : i + window_size]
        
        # Skip windows with NaN values
        if drop_nulls and np.any(np.isnan(window)):
            continue
        
        # Get label (flare in next step after window)
        if labels is not None:
            if i + window_size < num_samples:
                windows_X.append(window)
                windows_y.append(labels[i + window_size])
        else:
            windows_X.append(window)
    
    X = np.array(windows_X, dtype=np.float32)  # (num_windows, window_size, num_features)
    y = np.array(windows_y, dtype=np.float32) if windows_y else None
    
    return X, y


class FlareForecastDataset(Dataset):
    """
    PyTorch Dataset for flare forecasting with sliding windows.
    
    Usage:
        >>> X, y = create_sliding_windows(df, ['COUNTS'], 'is_flare', window_size=60)
        >>> dataset = FlareForecastDataset(X, y)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Args:
            X: (num_samples, window_size, num_features) feature array
            y: (num_samples,) target labels, or None for inference
        """
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
        
        if self.X.dim() != 3:
            raise ValueError(f"X must have shape (N, window_size, num_features), got {self.X.shape}")
        
        if self.y is not None and self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X and y must have same first dimension, got {self.X.shape[0]} vs {self.y.shape[0]}")
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def prepare_flare_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Split data into train/val and create dataloaders.
    
    Args:
        X: (num_samples, window_size, num_features)
        y: (num_samples,) labels
        batch_size: batch size for training
        train_split: fraction of data for training (rest for validation)
        shuffle: whether to shuffle training data
        num_workers: number of data loading workers
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    num_samples = X.shape[0]
    train_size = int(num_samples * train_split)
    
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_dataset = FlareForecastDataset(X[train_idx], y[train_idx])
    val_dataset = FlareForecastDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
