"""
Solar Flare Nowcasting Pipeline using PyTorch LSTM.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline import load_solexs_fits_table
from sliding_window import create_sliding_windows, FlareForecastDataset
from models import ImprovedFlareForecasterLSTM


def get_solexs_lc_files(data_dir: Path) -> list[Path]:
    """Find all SoLEXS .lc and .lc.gz files recursively under data_dir."""
    files = []
    # Search for files with suffix .lc or .lc.gz
    for file_path in data_dir.rglob("*"):
        if file_path.name.endswith(".lc") or file_path.name.endswith(".lc.gz"):
            # Only process SDD2 data to avoid saturated SDD1 data
            if "SDD2" in file_path.parts or "sdd2" in file_path.name.lower():
                files.append(file_path)
    
    # Fallback to any .lc or .lc.gz files if SDD2 filters didn't match anything
    if not files:
        for file_path in data_dir.rglob("*"):
            if file_path.name.endswith(".lc") or file_path.name.endswith(".lc.gz"):
                files.append(file_path)
                
    return sorted(files)


def prepare_labeled_data(data_dir: Path) -> pd.DataFrame:
    """Load all SoLEXS light curve files and assign binary is_flare labels."""
    lc_files = get_solexs_lc_files(data_dir)
    if not lc_files:
        raise FileNotFoundError(f"No SoLEXS .lc or .lc.gz files found in {data_dir}")

    print(f"Found {len(lc_files)} SoLEXS light curve files:")
    for f in lc_files:
        print(f"  - {f.relative_to(data_dir) if data_dir in f.parents else f}")

    dfs = []
    for file_path in lc_files:
        try:
            df_file = load_solexs_fits_table(file_path)
            if not df_file.empty:
                dfs.append(df_file)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    if not dfs:
        raise ValueError("Failed to load any data from SoLEXS light curve files.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by="TIME").reset_index(drop=True)
    
    # Clean nulls
    df = df.dropna(subset=["TIME", "COUNTS"])
    
    # Labeling rule based on mean and standard deviation
    mean_counts = df["COUNTS"].mean()
    std_counts = df["COUNTS"].std()
    threshold = mean_counts + 3.0 * std_counts
    
    df["is_flare"] = (df["COUNTS"] > threshold).astype(np.float32)
    
    num_total = len(df)
    num_flares = int(df["is_flare"].sum())
    print(f"Data loading completed. Total records: {num_total}")
    print(f"Labeling threshold (mean + 3*std): {threshold:.2f} counts/sec")
    print(f"Flare timesteps detected: {num_flares} ({100.0 * num_flares / num_total:.2f}%)")
    
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate confusion matrix, true skill statistic (TSS), recall, and precision."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # True Positive Rate / Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    tss = tpr - fpr                                   # True Skill Statistic
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "TSS": float(tss), "Recall": float(tpr), "Precision": float(precision)
    }


def train_model(data_dir: Path, epochs: int, batch_size: int, model_path: Path, metadata_path: Path):
    """Run data loading, preprocessing, windowing, and PyTorch model training."""
    print("Preparing labeled data...")
    df = prepare_labeled_data(data_dir)
    
    feature_cols = ["TIME", "COUNTS"]
    label_col = "is_flare"
    window_size = 60
    
    # Store feature means and stds for normalization during inference
    features_raw = df[feature_cols].values
    feature_means = np.nanmean(features_raw, axis=0)
    feature_stds = np.nanstd(features_raw, axis=0)
    feature_stds[feature_stds == 0] = 1.0
    
    print("Creating sliding windows...")
    X, y = create_sliding_windows(
        df,
        feature_cols=feature_cols,
        label_col=label_col,
        window_size=window_size,
        step_size=1,
        normalize=True,
        drop_nulls=True
    )
    
    if X.size == 0 or y is None or len(y) == 0:
        raise ValueError("Failed to create sliding windows. Try a larger dataset or smaller window size.")
        
    # Align X and y lengths (discard the last window in X since it has no future label)
    X = X[:len(y)]
        
    print(f"Created sliding windows. Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # Train / Val Split (80% / 20%)
    num_samples = len(X)
    split_idx = int(num_samples * 0.8)
    
    # Keep validation data contiguous for time-series evaluation
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # Handle class balance
    print(f"Train samples: {len(X_train)} (Flares: {int(y_train.sum())})")
    print(f"Val samples: {len(X_val)} (Flares: {int(y_val.sum())})")
    
    train_dataset = FlareForecastDataset(X_train, y_train)
    val_dataset = FlareForecastDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = ImprovedFlareForecasterLSTM(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        output_dim=1,
        bidirectional=True
    ).to(device)
    
    # BCELoss is used since the model output has self.sigmoid(logits) applied
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_X).squeeze(-1)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                y_pred = model(batch_X).squeeze(-1)
                loss = criterion(y_pred, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                all_preds.extend(y_pred.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        # Metrics
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        pred_labels = (all_preds >= 0.5).astype(np.float32)
        metrics = calculate_metrics(all_trues, pred_labels)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val TSS: {metrics['TSS']:.4f} | Val Recall: {metrics['Recall']:.4f} | Val Precision: {metrics['Precision']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            
            # Save metadata
            metadata = {
                "input_dim": len(feature_cols),
                "feature_cols": feature_cols,
                "feature_means": feature_means.tolist(),
                "feature_stds": feature_stds.tolist(),
                "window_size": window_size,
                "best_val_loss": float(best_val_loss),
                "metrics": metrics
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
                
            print(f"  --> Saved new best model checkpoint to {model_path}")
            
    print("Training finished successfully.")


class Nowcaster:
    """Inference helper for real-time solar flare nowcasting."""
    
    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.feature_cols = self.metadata["feature_cols"]
        self.feature_means = np.array(self.metadata["feature_means"], dtype=np.float32)
        self.feature_stds = np.array(self.metadata["feature_stds"], dtype=np.float32)
        self.window_size = self.metadata["window_size"]
        
        # Build model and load state dict
        self.model = ImprovedFlareForecasterLSTM(
            input_dim=len(self.feature_cols),
            hidden_dim=128,
            num_layers=3,
            output_dim=1,
            bidirectional=True
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def predict(self, recent_df: pd.DataFrame) -> float:
        """
        Predict the probability of a flare occurring in the next step.
        
        Args:
            recent_df: pd.DataFrame containing the last 60 seconds (cadence steps) of records
                      with columns listed in self.feature_cols (TIME, COUNTS).
        Returns:
            probability float in range [0.0, 1.0].
        """
        if len(recent_df) < self.window_size:
            raise ValueError(f"Nowcasting requires at least {self.window_size} steps, got {len(recent_df)}")
            
        # Select the latest window_size samples
        df_window = recent_df.tail(self.window_size).copy()
        
        # Extract features
        features = df_window[self.feature_cols].values.astype(np.float32)
        
        # Normalise using training stats
        features_norm = (features - self.feature_means) / self.feature_stds
        
        # Convert to tensor and add batch dimension: (1, window_size, input_dim)
        tensor_X = torch.from_numpy(features_norm).unsqueeze(0)
        
        with torch.no_grad():
            prob = self.model(tensor_X).item()
            
        return prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aditya-L1 Nowcasting Command Line Interface")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\USER\Documents\AL1_SLX_L1_20240507_v1.0",
                        help="Path to SoLEXS FITS folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--model_path", type=str, default="nowcast_model.pt", help="Path to save weights")
    parser.add_argument("--metadata_path", type=str, default="nowcast_metadata.json", help="Path to save metadata")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Starting training on SoLEXS data directory: {args.data_dir}")
        train_model(
            data_dir=Path(args.data_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=Path(args.model_path),
            metadata_path=Path(args.metadata_path)
        )
    else:
        parser.print_help()
