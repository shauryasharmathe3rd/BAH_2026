"""
Solar Flare Forecasting Pipeline using Multi-task PyTorch LSTM.
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
from torch.utils.data import Dataset, DataLoader

from data_pipeline import load_solexs_fits_table
from sliding_window import create_sliding_windows
from models import MultiOutputFlareForecaster


class MultiTaskFlareDataset(Dataset):
    """PyTorch Dataset for multi-task flare forecasting."""
    
    def __init__(self, X: np.ndarray, y_occur: np.ndarray, y_class: np.ndarray, y_conf: np.ndarray):
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y_occur = torch.from_numpy(y_occur) if isinstance(y_occur, np.ndarray) else y_occur
        self.y_class = torch.from_numpy(y_class).long() if isinstance(y_class, np.ndarray) else y_class
        self.y_conf = torch.from_numpy(y_conf) if isinstance(y_conf, np.ndarray) else y_conf
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_occur[idx], self.y_class[idx], self.y_conf[idx]


def get_solexs_lc_files(data_dir: Path) -> list[Path]:
    """Find all SoLEXS .lc and .lc.gz files recursively under data_dir."""
    files = []
    for file_path in data_dir.rglob("*"):
        if file_path.name.endswith(".lc") or file_path.name.endswith(".lc.gz"):
            if "SDD2" in file_path.parts or "sdd2" in file_path.name.lower():
                files.append(file_path)
    
    if not files:
        for file_path in data_dir.rglob("*"):
            if file_path.name.endswith(".lc") or file_path.name.endswith(".lc.gz"):
                files.append(file_path)
                
    return sorted(files)


def prepare_labeled_data(data_dir: Path) -> pd.DataFrame:
    """Load all SoLEXS light curve files and assign multi-task forecasting targets."""
    lc_files = get_solexs_lc_files(data_dir)
    if not lc_files:
        raise FileNotFoundError(f"No SoLEXS .lc or .lc.gz files found in {data_dir}")

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
    df = df.dropna(subset=["TIME", "COUNTS"])
    
    # Labeling boundaries
    mean_counts = df["COUNTS"].mean()
    std_counts = df["COUNTS"].std()
    threshold = mean_counts + 3.0 * std_counts
    
    # 1. Occurrence label
    df["is_flare"] = (df["COUNTS"] > threshold).astype(np.float32)
    
    # 2. Magnitude class label (0: None, 1: B, 2: C, 3: M, 4: X-class)
    class_labels = np.zeros(len(df), dtype=np.int64)
    counts = df["COUNTS"].values
    
    class_labels[counts > threshold] = 1
    class_labels[counts > threshold * 2] = 2
    class_labels[counts > threshold * 5] = 3
    class_labels[counts > threshold * 10] = 4
    df["flare_class"] = class_labels
    
    # 3. Confidence target (target is 1.0 for training)
    df["confidence_target"] = np.ones(len(df), dtype=np.float32)
    
    num_total = len(df)
    print(f"Loaded {num_total} rows. Class distribution:")
    for c in range(5):
        cnt = np.sum(class_labels == c)
        print(f"  Class {c}: {cnt} samples ({100.0 * cnt / num_total:.2f}%)")
        
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics for binary occurrence."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tss = tpr - fpr
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "TSS": float(tss), "Recall": float(tpr), "Precision": float(precision)
    }


def train_model(data_dir: Path, epochs: int, batch_size: int, model_path: Path, metadata_path: Path):
    """Run data preparation, sliding windows, and multi-task model training."""
    df = prepare_labeled_data(data_dir)
    
    feature_cols = ["TIME", "COUNTS"]
    window_size = 60
    
    # Save training statistics
    features_raw = df[feature_cols].values
    feature_means = np.nanmean(features_raw, axis=0)
    feature_stds = np.nanstd(features_raw, axis=0)
    feature_stds[feature_stds == 0] = 1.0
    
    print("Creating multi-task sliding windows...")
    X, y_occur = create_sliding_windows(df, feature_cols, "is_flare", window_size=window_size, step_size=1, normalize=True)
    _, y_class = create_sliding_windows(df, feature_cols, "flare_class", window_size=window_size, step_size=1, normalize=True)
    _, y_conf = create_sliding_windows(df, feature_cols, "confidence_target", window_size=window_size, step_size=1, normalize=True)
    
    # Align lengths
    min_len = min(len(X), len(y_occur), len(y_class), len(y_conf))
    X = X[:min_len]
    y_occur = y_occur[:min_len]
    y_class = y_class[:min_len]
    y_conf = y_conf[:min_len]
    
    print(f"Windows created. Shape: {X.shape}")
    
    # Split 80 / 20
    split_idx = int(min_len * 0.8)
    X_train, y_occur_train, y_class_train, y_conf_train = X[:split_idx], y_occur[:split_idx], y_class[:split_idx], y_conf[:split_idx]
    X_val, y_occur_val, y_class_val, y_conf_val = X[split_idx:], y_occur[split_idx:], y_class[split_idx:], y_conf[split_idx:]
    
    train_dataset = MultiTaskFlareDataset(X_train, y_occur_train, y_class_train, y_conf_train)
    val_dataset = MultiTaskFlareDataset(X_val, y_occur_val, y_class_val, y_conf_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = MultiOutputFlareForecaster(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        num_magnitude_classes=5
    ).to(device)
    
    # Loss functions
    criterion_occur = nn.BCELoss()
    criterion_class = nn.CrossEntropyLoss()
    criterion_conf = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_occur, batch_class, batch_conf in train_loader:
            batch_X = batch_X.to(device)
            batch_occur = batch_occur.to(device)
            batch_class = batch_class.to(device)
            batch_conf = batch_conf.to(device)
            
            optimizer.zero_grad()
            pred_occur, pred_class_logits, pred_conf = model(batch_X)
            
            loss_occur = criterion_occur(pred_occur.squeeze(-1), batch_occur)
            loss_class = criterion_class(pred_class_logits, batch_class)
            loss_conf = criterion_conf(pred_conf.squeeze(-1), batch_conf)
            
            # Combine losses
            loss = loss_occur + loss_class + 0.2 * loss_conf
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_pred_occur = []
        all_true_occur = []
        
        with torch.no_grad():
            for batch_X, batch_occur, batch_class, batch_conf in val_loader:
                batch_X = batch_X.to(device)
                batch_occur = batch_occur.to(device)
                batch_class = batch_class.to(device)
                batch_conf = batch_conf.to(device)
                
                pred_occur, pred_class_logits, pred_conf = model(batch_X)
                
                loss_occur = criterion_occur(pred_occur.squeeze(-1), batch_occur)
                loss_class = criterion_class(pred_class_logits, batch_class)
                loss_conf = criterion_conf(pred_conf.squeeze(-1), batch_conf)
                
                loss = loss_occur + loss_class + 0.2 * loss_conf
                val_loss += loss.item() * batch_X.size(0)
                
                all_pred_occur.extend(pred_occur.cpu().numpy())
                all_true_occur.extend(batch_occur.cpu().numpy())
                
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        all_pred_occur = np.array(all_pred_occur)
        all_true_occur = np.array(all_true_occur)
        pred_labels = (all_pred_occur >= 0.5).astype(np.float32)
        metrics = calculate_metrics(all_true_occur, pred_labels)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val TSS: {metrics['TSS']:.4f} | Val Recall: {metrics['Recall']:.4f} | Val Precision: {metrics['Precision']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            
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
                
            print(f"  --> Saved new best multi-task checkpoint to {model_path}")
            
    print("Multi-task forecasting training finished successfully.")


class Forecaster:
    """Inference helper for real-time multi-task solar flare forecasting."""
    
    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Forecasting model weights or metadata not found.")
            
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.feature_cols = self.metadata["feature_cols"]
        self.feature_means = np.array(self.metadata["feature_means"], dtype=np.float32)
        self.feature_stds = np.array(self.metadata["feature_stds"], dtype=np.float32)
        self.window_size = self.metadata["window_size"]
        
        self.model = MultiOutputFlareForecaster(
            input_dim=len(self.feature_cols),
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            num_magnitude_classes=5
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def predict(self, recent_df: pd.DataFrame) -> dict:
        """
        Run multi-task forecasting inference.
        
        Returns:
            dict containing:
              - flare_prob: float (0.0 to 1.0)
              - magnitude_class: int (0 to 4)
              - confidence: float (0.0 to 1.0)
        """
        if len(recent_df) < self.window_size:
            raise ValueError(f"Forecasting requires at least {self.window_size} steps.")
            
        df_window = recent_df.tail(self.window_size).copy()
        features = df_window[self.feature_cols].values.astype(np.float32)
        features_norm = (features - self.feature_means) / self.feature_stds
        tensor_X = torch.from_numpy(features_norm).unsqueeze(0)
        
        with torch.no_grad():
            prob_occur, class_logits, prob_conf = self.model(tensor_X)
            
            prob = float(prob_occur.item())
            pred_class = int(torch.argmax(class_logits, dim=-1).item())
            conf = float(prob_conf.item())
            
        return {
            "flare_prob": prob,
            "magnitude_class": pred_class,
            "confidence": conf
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aditya-L1 Forecasting Command Line Interface")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\USER\Documents\AL1_SLX_L1_20240507_v1.0",
                        help="Path to SoLEXS FITS folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--model_path", type=str, default="forecast_model.pt", help="Path to save weights")
    parser.add_argument("--metadata_path", type=str, default="forecast_metadata.json", help="Path to save metadata")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Starting multi-task training on SoLEXS data directory: {args.data_dir}")
        train_model(
            data_dir=Path(args.data_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=Path(args.model_path),
            metadata_path=Path(args.metadata_path)
        )
    else:
        parser.print_help()
