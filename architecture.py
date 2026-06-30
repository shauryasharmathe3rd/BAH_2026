"""
architecture.py

This script contains the model definition, training/evaluation pipeline,
and sampling/inference functions exported from the analysis notebook.
It supports both SoLEXS and HEL1OS datasets with instrument-specific normalization.
"""

import os
import glob
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
        
        if return_attention:
            return logits, attention_weights
        return logits


# ============================================================================
# 2. Helper Data Loading & Evaluation Functions
# ============================================================================

def make_native(df):
    """Safely convert big-endian columns in FITS files to native endianness for pandas."""
    for col in df.columns:
        try:
            val = df[col].values
            if hasattr(val, 'dtype') and not val.dtype.isnative:
                df[col] = val.astype(val.dtype.newbyteorder("="))
        except Exception:
            pass
    return df


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
            probs = torch.sigmoid(y_pred)
            all_preds.append(probs.cpu().numpy())
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

def train_model(data_dir='Data', weights_path='flare_lstm_weights.pth', num_epochs=5, batch_size=64):
    """
    Trains the ImprovedFlareForecasterLSTM model on both SoLEXS and HEL1OS datasets.
    Preprocesses, normalizes, and splits them chronologically separately to avoid data logic mixing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Discover and load SoLEXS and HEL1OS data
    print("Loading data files...")
    solexs_files = sorted(glob.glob(os.path.join(data_dir, 'solexs_*/**/*.lc'), recursive=True))
    hel1os_files = sorted(glob.glob(os.path.join(data_dir, 'hel1os_*/**/lightcurve_*.fits'), recursive=True))
    
    dfs = []
    
    # Load SoLEXS
    for file_path in solexs_files:
        with fits.open(file_path) as hdul:
            df = pd.DataFrame(hdul[1].data)
        df = make_native(df)
        df = df.dropna(subset=['TIME', 'COUNTS'])
        df = df[['TIME', 'COUNTS']].copy()
        df['is_flare'] = (df['COUNTS'] > 1000).astype(np.float32)
        df['source'] = os.path.basename(file_path)
        df['instrument'] = 'solexs'
        dfs.append(df)
        
    # Load HEL1OS
    for file_path in hel1os_files:
        with fits.open(file_path) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.columns is not None and 'CTR' in hdu.columns.names:
                    df = pd.DataFrame(hdu.data)
                    df = make_native(df)
                    
                    if df['ISOT'].dtype == object:
                        isot_col = df['ISOT'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    else:
                        isot_col = df['ISOT']
                    
                    df['TIME'] = pd.to_datetime(isot_col).astype('datetime64[ns]').astype('int64') // 10**9
                    df['COUNTS'] = df['CTR']
                    df = df.dropna(subset=['TIME', 'COUNTS'])
                    df = df[['TIME', 'COUNTS']].copy()
                    df['is_flare'] = (df['COUNTS'] > 200).astype(np.float32)
                    df['source'] = os.path.basename(file_path) + '_' + hdu.name
                    df['instrument'] = 'hel1os'
                    dfs.append(df)
                    
    if not dfs:
        raise ValueError(f"No valid data files found in {data_dir} directory.")
        
    df_clean = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {df_clean.shape[0]}")
    print(df_clean['instrument'].value_counts())
    
    # 2. Define features & sliding windows configurations
    feature_cols = ['TIME', 'COUNTS']
    label_col = 'is_flare'
    window_size = 60
    step_size = 1
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    # Group by instrument to perform separate normalization
    for instrument, inst_grp in df_clean.groupby('instrument'):
        features = inst_grp[feature_cols].values.astype(np.float32)
        valid_idx = ~np.any(np.isnan(features), axis=1)
        features = features[valid_idx]
        labels = inst_grp.loc[valid_idx, label_col].values
        
        # Compute instrument-specific training split stats (using first 80%) using StandardScaler
        split_idx = int(len(features) * 0.8)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(features[:split_idx])
        normalized_features = scaler.transform(features)
        
        inst_grp_clean = inst_grp.loc[valid_idx].copy()
        inst_grp_clean['NORM_TIME'] = normalized_features[:, 0]
        inst_grp_clean['NORM_COUNTS'] = normalized_features[:, 1]
        
        # Group by source file to prevent sliding windows from mixing different files
        for file_path, file_grp in inst_grp_clean.groupby('source'):
            file_features = file_grp[['NORM_TIME', 'NORM_COUNTS']].values.astype(np.float32)
            file_labels = file_grp[label_col].values.astype(np.float32)
            
            # Chronological split within this file (80% train, 20% test)
            f_split_idx = int(len(file_features) * 0.8)
            
            train_feats = file_features[:f_split_idx]
            train_lbls = file_labels[:f_split_idx]
            test_feats = file_features[f_split_idx:]
            test_lbls = file_labels[f_split_idx:]
            
            # Helper to generate sliding windows
            def make_windows(feats, lbls):
                num_samples = feats.shape[0]
                xs, ys = [], []
                for i in range(0, num_samples - window_size + 1, step_size):
                    if i + window_size < num_samples:
                        xs.append(feats[i : i + window_size])
                        ys.append(lbls[i + window_size])
                return xs, ys
                
            tr_x, tr_y = make_windows(train_feats, train_lbls)
            te_x, te_y = make_windows(test_feats, test_lbls)
            
            X_train_list.extend(tr_x)
            y_train_list.extend(tr_y)
            X_test_list.extend(te_x)
            y_test_list.extend(te_y)
            
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)
    
    print(f"Sliding window sequences - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # 3. Create datasets and loaders
    train_dataset = sw.FlareForecastDataset(X_train, y_train)
    test_dataset = sw.FlareForecastDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Model
    model = ImprovedFlareForecasterLSTM(
        input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        output_dim=1,
        bidirectional=True
    ).to(device)
    
    # Calculate pos_weight based on class imbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    weight_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([weight_ratio], dtype=torch.float32).to(device)
    print(f"BCEWithLogitsLoss class pos_weight ratio: {weight_ratio:.4f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            
            outputs = model(X_batch)
            if isinstance(outputs, tuple):
                y_pred = outputs[0]
            else:
                y_pred = outputs
                
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(y_pred) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"--- Epoch {epoch+1} Complete: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} ---")
        
        scheduler.step(epoch_loss)
        
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, criterion, device)
    print("\n================ EVALUATION RESULTS ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<12}: {v:.4f}")
        else:
            print(f"{k:<12}: {v}")
    print("====================================================")
    
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Model weights saved for production to: {weights_path}")
    
    return model


# ============================================================================
# 4. Sampling / Inference Function for API Server and Interaction
# ============================================================================

def predict_flare(sequence_data, instrument='solexs', model=None, model_weights_path='flare_lstm_weights.pth', device=None):
    """
    Predicts the probability of a solar flare occurring in the next timestep
    given a sequence of the last 60 timesteps of ['TIME', 'COUNTS'] data.
    
    Args:
        sequence_data (np.ndarray, list, or pd.DataFrame): 
            Input sequence of shape (60, 2) or (batch_size, 60, 2).
            Features must be in order: [TIME, COUNTS].
        instrument (str): Instrument type, either 'solexs' or 'hel1os'.
            This determines the specific normalization mean and standard deviation.
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
        
    # Normalize features based on instrument statistics
    # SoLEXS Mean: [1.7151514e+09, 722.7386], Std: [91871.33, 1324.5471]
    # HEL1OS Mean: [1.715076e+09, 9.304749], Std: [25588.596, 92.98037]
    instrument = str(instrument).lower().strip()
    if instrument == 'solexs':
        mean = np.array([1.7151514e+09, 722.7386], dtype=np.float32)
        std = np.array([91871.33, 1324.5471], dtype=np.float32)
    elif instrument == 'hel1os':
        mean = np.array([1.715076e+09, 9.304749], dtype=np.float32)
        std = np.array([25588.596, 92.98037], dtype=np.float32)
    else:
        raise ValueError(f"Unknown instrument type '{instrument}'. Choose 'solexs' or 'hel1os'.")
        
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
            logits = outputs[0]
        else:
            logits = outputs
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        
    # Ensure it's iterable
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
    default_data_dir = 'Data'
    weights_file = 'flare_lstm_weights.pth'
    
    if os.path.exists(default_data_dir):
        print("Starting model training and evaluation using combined notebook logic...")
        train_model(default_data_dir, weights_path=weights_file)
    else:
        print(f"Data directory not found at {default_data_dir}.")
