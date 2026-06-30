"""
architecture.py

This script contains the model definition, training/evaluation pipeline,
and sampling/inference functions exported from the analysis notebook.
It keeps only the main parts of the notebook and provides a clean interface
for predicting solar flares.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sliding_window as sw

# ============================================================================
# 1. Model Architecture Definitions
# ============================================================================

class AttentionLayer(nn.Module):
    """Multi-head self-attention for temporal sequence modeling."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_dim)
        
        # Final linear layer
        output = self.fc_out(context)
        return output, attention_weights


class ImprovedFlareForecasterLSTM(nn.Module):
    """
    Enhanced LSTM for solar flare forecasting with:
    - Bidirectional LSTM for capturing past/future context
    - Attention mechanism for temporal importance weighting
    - Residual connections for improved gradient flow
    - Batch normalization for training stability
    - Multi-layer FC head for better feature extraction
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        output_dim: int = 1,
        bidirectional: bool = True
    ):
        super(ImprovedFlareForecasterLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # ===== Feature Extraction =====
        # Initial batch norm for input stability
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Bidirectional LSTM - captures patterns in both directions
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # ===== Attention Mechanism =====
        # Self-attention to weight important time steps
        self.attention = AttentionLayer(lstm_output_dim, num_heads=num_heads)
        
        # ===== Classification Head =====
        # Multi-layer FC with residual connections for robustness
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.bn1 = nn.BatchNorm1d(lstm_output_dim // 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
        self.bn2 = nn.BatchNorm1d(lstm_output_dim // 4)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(lstm_output_dim // 4, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_attention: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # ===== Step 1: Normalize Input =====
        x_reshaped = x.reshape(-1, self.input_dim)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, self.input_dim)
        
        # ===== Step 2: Bidirectional LSTM =====
        lstm_out, _ = self.lstm(x)
        
        # ===== Step 3: Attention Weighting =====
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # ===== Step 4: Temporal Aggregation =====
        temporal_features = attended_out.mean(dim=1)
        
        # ===== Step 5: Multi-Layer Classification =====
        fc1_out = self.fc1(temporal_features)
        fc1_out = self.bn1(fc1_out)
        fc1_out = torch.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)
        fc2_out = torch.relu(fc2_out)
        fc2_out = self.dropout2(fc2_out)
        
        logits = self.fc_out(fc2_out)
        probability = self.sigmoid(logits)
        
        if return_attention:
            return probability, attention_weights
        return probability


# ============================================================================
# 2. Helper Data Loading & Evaluation Functions
# ============================================================================

def load_fits_to_df(file_path, extension=1):
    """Loads fits file table to pandas DataFrame."""
    with fits.open(file_path) as hdul:
        data = hdul[extension].data
        return pd.DataFrame(data)


def compute_auc(y_true, y_pred):
    """Computes ROC-AUC metric."""
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    asc_idx = np.argsort(y_pred)
    y_true_sorted = y_true[asc_idx]
    pos_ranks = np.where(y_true_sorted == 1)[0] + 1
    u_stat = np.sum(pos_ranks) - (n_pos * (n_pos + 1)) / 2
    return u_stat / (n_pos * n_neg)


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates model performance metrics."""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            if isinstance(outputs, tuple):
                y_pred = outputs[0]
            else:
                y_pred = outputs
                
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y_batch)
            
            test_loss += loss.item() * X_batch.size(0)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    avg_loss = test_loss / len(dataloader.dataset)
    preds_prob = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    preds_bin = (preds_prob >= 0.5).astype(np.float32)
    
    tp = np.sum((preds_bin == 1) & (targets == 1))
    fp = np.sum((preds_bin == 1) & (targets == 0))
    fn = np.sum((preds_bin == 0) & (targets == 1))
    tn = np.sum((preds_bin == 0) & (targets == 0))
    
    accuracy = (tp + tn) / len(targets) if len(targets) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = compute_auc(targets, preds_prob)
    
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0
    
    return {
        "Loss": avg_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "MCC": mcc,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }


# ============================================================================
# 3. Model Training Loop
# ============================================================================

def train_model(lc_path, weights_path='flare_lstm_weights.pth', num_epochs=5, batch_size=64):
    """
    Trains the ImprovedFlareForecasterLSTM model on the FITS light curve data.
    Matches the exact training parameters and chronological splitting from the notebook.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the raw light curve data
    print(f"Loading raw data from {lc_path}...")
    lc_df = load_fits_to_df(lc_path)
    
    # 2. Create target label: 1 if COUNTS > 1000, 0 otherwise
    feature_cols = ['TIME', 'COUNTS']
    label_col = 'is_flare'
    df_clean = lc_df.copy()
    df_clean[label_col] = (df_clean['COUNTS'] > 1000).astype(np.float32)
    
    # 3. Chronological split (80% train, 20% test)
    train_split = 0.8
    split_idx = int(len(df_clean) * train_split)
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()
    
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    
    # 4. Create sliding windows (window size: 60 timesteps, step size: 1)
    window_size = 60
    step_size = 1
    
    print("Creating sliding windows...")
    X_train, y_train = sw.create_sliding_windows(
        train_df, 
        feature_cols=feature_cols, 
        label_col=label_col, 
        window_size=window_size, 
        step_size=step_size,
        normalize=True
    )
    X_test, y_test = sw.create_sliding_windows(
        test_df, 
        feature_cols=feature_cols, 
        label_col=label_col, 
        window_size=window_size, 
        step_size=step_size,
        normalize=True
    )
    
    print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test: {y_test.shape}")
    
    # 5. Create PyTorch datasets and DataLoaders
    train_dataset = sw.FlareForecastDataset(X_train, y_train)
    test_dataset = sw.FlareForecastDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Initialize Model
    print("Initializing model...")
    model = ImprovedFlareForecasterLSTM(
        input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        output_dim=1,
        bidirectional=True
    ).to(device)
    
    # Loss, Optimizer, and Scheduler
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print(f"Starting training for {num_epochs} epochs...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            if isinstance(outputs, tuple):
                y_pred = outputs[0]
            else:
                y_pred = outputs
                
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to stabilize LSTM training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            preds = (y_pred >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"--- Epoch {epoch+1} Complete: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} ---")
        
        scheduler.step(epoch_loss)
        
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, criterion, device)
    print("\n================ EVALUATION RESULTS ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<12}: {v:.4f}")
        else:
            print(f"{k:<12}: {v}")
    print("====================================================")
    
    # Save the trained weights
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Model weights saved for production to: {weights_path}")
    
    return model


# ============================================================================
# 4. Sampling / Inference Function for API Server and Interaction
# ============================================================================

def predict_flare(sequence_data, model=None, model_weights_path='flare_lstm_weights.pth', device=None):
    """
    Predicts the probability of a solar flare occurring in the next timestep
    given a sequence of the last 60 timesteps of ['TIME', 'COUNTS'] data.
    
    Args:
        sequence_data (np.ndarray, list, or pd.DataFrame): 
            Input sequence of shape (60, 2) or (batch_size, 60, 2).
            Features must be in order: [TIME, COUNTS].
        model (nn.Module, optional): An already loaded and instantiated model.
            If provided, avoids loading weights from disk.
        model_weights_path (str): Path to the trained weights file.
            Only used if model is None.
        device (str or torch.device, optional): Device to run inference on.
        
    Returns:
        dict or list[dict]: Dictionary containing 'flare_probability' and 
            'is_flare_predicted', or a list of such dictionaries if batch input is given.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    # Convert input to numpy array
    if isinstance(sequence_data, pd.DataFrame):
        arr = sequence_data[['TIME', 'COUNTS']].values.astype(np.float32)
    elif isinstance(sequence_data, list):
        arr = np.array(sequence_data, dtype=np.float32)
    else:
        arr = np.array(sequence_data, dtype=np.float32)
        
    # Ensure correct dimensions
    if arr.ndim == 2:
        seq_len, num_features = arr.shape
        if seq_len != 60 or num_features != 2:
            raise ValueError(f"Input sequence must have shape (60, 2), got {arr.shape}")
        arr = np.expand_dims(arr, axis=0)  # Shape: (1, 60, 2)
    elif arr.ndim == 3:
        batch_size, seq_len, num_features = arr.shape
        if seq_len != 60 or num_features != 2:
            raise ValueError(f"Input sequence batch must have shape (batch_size, 60, 2), got {arr.shape}")
    else:
        raise ValueError(f"Input sequence must be 2D (60, 2) or 3D (batch_size, 60, 2), got {arr.ndim} dimensions")
        
    # Normalization using the training set statistics calculated from the notebook:
    # Mean: [1.7137284e+09, 4.2081741e+02]
    # Std:  [1.3463169e+06, 5.7675787e+02]
    mean = np.array([1.7137284e+09, 4.2081741e+02], dtype=np.float32)
    std = np.array([1.3463169e+06, 5.7675787e+02], dtype=np.float32)
    
    normalized_arr = (arr - mean) / std
    
    # Convert to PyTorch tensor
    x_tensor = torch.from_numpy(normalized_arr).to(device)
    
    # Load model if not provided
    if model is None:
        model = ImprovedFlareForecasterLSTM(
            input_dim=2,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.3,
            output_dim=1,
            bidirectional=True
        )
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights file not found at {model_weights_path}. Train the model first.")
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        
    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor)
        if isinstance(outputs, tuple):
            probs = outputs[0]
        else:
            probs = outputs
        probs = probs.squeeze(-1).cpu().numpy()
        
    # Ensure it's iterable even for single prediction
    if not isinstance(probs, np.ndarray) or probs.ndim == 0:
        probs = np.array([probs])
        
    results = []
    for p in probs:
        results.append({
            "flare_probability": float(p),
            "is_flare_predicted": bool(p >= 0.5)
        })
        
    if len(results) == 1:
        return results[0]
    return results


# ============================================================================
# 5. Main Script Runner
# ============================================================================

if __name__ == '__main__':
    # Default file path
    default_lc_file = '/home/shaurya/Documents/Antigravity/BAH_Project/solexs_2026Jun29T054518533/AL1_SLX_L1_20240507_v1.0/SDD2/AL1_SOLEXS_20240507_SDD2_L1.lc'
    weights_file = 'flare_lstm_weights.pth'
    
    if os.path.exists(default_lc_file):
        print("Starting model training and evaluation using notebook logic...")
        train_model(default_lc_file, weights_path=weights_file)
    else:
        print(f"FITS data file not found at {default_lc_file}.")
        print("If you have it placed elsewhere, run `train_model(lc_path)` manually.")
