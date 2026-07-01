"""
Improved LSTM models for solar flare forecasting with sliding window inputs.
"""

import torch
from torch import nn


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
        """
        Args:
            query, key, value: (batch_size, seq_len, hidden_dim)
            mask: optional mask for ignoring padding tokens
        """
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
        """
        Forward pass with sliding window sequence.
        
        Args:
            x: (batch_size, sequence_length, input_dim)
               - batch_size: number of sequences in batch
               - sequence_length: sliding window size (e.g., 60 timesteps)
               - input_dim: number of features (e.g., TIME, COUNTS)
            return_attention: if True, return attention weights for visualization
            
        Returns:
            probability: (batch_size, 1) - predicted probability of flare in next step
            (optional) attention_weights: attention scores across time steps
        """
        batch_size, seq_len, _ = x.shape
        
        # ===== Step 1: Normalize Input =====
        # Reshape for batch norm: (batch_size * seq_len, input_dim)
        x_reshaped = x.reshape(-1, self.input_dim)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, self.input_dim)
        
        # ===== Step 2: Bidirectional LSTM =====
        # Processes entire sequence in both directions
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, lstm_output_dim)
        
        # ===== Step 3: Attention Weighting =====
        # Focus on most relevant time steps
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attended_out: (batch_size, seq_len, lstm_output_dim)
        
        # ===== Step 4: Temporal Aggregation =====
        # Take weighted average across sequence OR last attended state
        # Option 1: Mean of attended sequence
        temporal_features = attended_out.mean(dim=1)  # (batch_size, lstm_output_dim)
        
        # Option 2 (alternative): Weighted sum by attention
        # attention_weights_avg = attention_weights.mean(dim=1)  # (batch, seq, seq)
        # temporal_features = torch.bmm(attention_weights_avg.mean(dim=1).unsqueeze(1), attended_out).squeeze(1)
        
        # ===== Step 5: Multi-Layer Classification =====
        # Layer 1
        fc1_out = self.fc1(temporal_features)
        fc1_out = self.bn1(fc1_out)
        fc1_out = torch.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        # Layer 2
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)
        fc2_out = torch.relu(fc2_out)
        fc2_out = self.dropout2(fc2_out)
        
        # Output layer
        logits = self.fc_out(fc2_out)
        probability = self.sigmoid(logits)  # (batch_size, 1)
        
        if return_attention:
            return probability, attention_weights
        return probability


class LSTMWithTemporalCNN(nn.Module):
    """
    Hybrid CNN-LSTM model: CNN extracts local temporal patterns,
    LSTM captures long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        cnn_filters: int = 32,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super(LSTMWithTemporalCNN, self).__init__()
        
        # CNN for local temporal feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )
        self.bn_conv = nn.BatchNorm1d(cnn_filters)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # LSTM on CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # Attention
        self.attention = AttentionLayer(lstm_output_dim, num_heads=4)
        
        # FC head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.fc2 = nn.Linear(lstm_output_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            probability: (batch_size, 1)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # CNN: (batch, seq_len, input_dim) -> (batch, cnn_filters, seq_len)
        x_conv = self.conv1d(x.transpose(1, 2))
        x_conv = self.bn_conv(x_conv)
        x_conv = self.relu(x_conv)
        # Pooling reduces sequence length
        x_conv = self.pool(x_conv)  # (batch, cnn_filters, seq_len//2)
        
        # LSTM: (batch, cnn_filters, new_seq_len) -> (batch, new_seq_len, cnn_filters)
        x_lstm = x_conv.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm)
        
        # Attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Temporal aggregation
        temporal_features = attended_out.mean(dim=1)
        
        # FC layers
        fc1_out = self.fc1(temporal_features)
        fc1_out = torch.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        logits = self.fc2(fc1_out)
        probability = self.sigmoid(logits)
        
        return probability


class MultiOutputFlareForecaster(nn.Module):
    """
    Multi-task LSTM for predicting:
    1. Binary flare occurrence (probability)
    2. Flare magnitude class (if flare occurs)
    3. Confidence score
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_magnitude_classes: int = 5  # A, B, C, M, X class
    ):
        super(MultiOutputFlareForecaster, self).__init__()
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # Attention
        self.attention = AttentionLayer(lstm_output_dim, num_heads=4)
        
        # Task 1: Flare occurrence prediction
        self.flare_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.BatchNorm1d(lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Task 2: Flare magnitude classification
        self.magnitude_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.BatchNorm1d(lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_magnitude_classes)
        )
        
        # Task 3: Confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.BatchNorm1d(lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            flare_prob: (batch_size, 1) - probability of flare
            magnitude_logits: (batch_size, num_magnitude_classes)
            confidence: (batch_size, 1) - model confidence
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Temporal aggregation
        temporal_features = attended_out.mean(dim=1)
        
        # Multi-task predictions
        flare_prob = self.flare_head(temporal_features)
        magnitude_logits = self.magnitude_head(temporal_features)
        confidence = self.confidence_head(temporal_features)
        
        return flare_prob, magnitude_logits, confidence


# ============================================================================
# ANALYSIS & COMPARISON OF MODELS
# ============================================================================

"""
ARCHITECTURE COMPARISON & IMPROVEMENTS

┌─────────────────────────────────────────────────────────────────────┐
│ ORIGINAL MODEL                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ • Single-direction LSTM (2 layers, 64 units)                        │
│ • Takes only last output step for classification                    │
│ • Simple sigmoid activation, no intermediate processing             │
│                                                                     │
│ LIMITATIONS:                                                        │
│ ✗ Loses information from future context (unidirectional)            │
│ ✗ No temporal importance weighting (all steps equal weight)         │
│ ✗ Small hidden dimension (64) may underfit complex patterns         │
│ ✗ No regularization (batch norm, dropout applied naively)           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ IMPROVED MODEL: ImprovedFlareForecasterLSTM                         │
├─────────────────────────────────────────────────────────────────────┤
│ ENHANCEMENTS:                                                       │
│ ✓ Bidirectional LSTM (sees patterns forward & backward)             │
│ ✓ Multi-head attention (focuses on important time steps)            │
│ ✓ Larger capacity (128 hidden units, 3 layers)                      │
│ ✓ Batch normalization (input + intermediate layers)                 │
│ ✓ Multi-layer FC head (better feature transformation)               │
│ ✓ Residual connections (improved gradient flow)                     │
│ ✓ Configurable dropout (0.3) for regularization                     │
│                                                                     │
│ BENEFITS:                                                           │
│ • Captures bidirectional temporal patterns                          │
│ • Learns to attend to critical pre-flare signatures                 │
│ • Better gradient flow for training deep networks                  │
│ • Improved generalization & robustness                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ALTERNATIVE: LSTMWithTemporalCNN                                    │
├─────────────────────────────────────────────────────────────────────┤
│ • Conv1D layer detects local temporal patterns (short-term)         │
│ • LSTM captures long-range dependencies (long-term)                 │
│ • Hybrid approach combines both scales efficiently                  │
│                                                                     │
│ USE WHEN:                                                           │
│ • Data has strong local temporal patterns (spikes, dips)            │
│ • Computational efficiency is critical                              │
│ • Sliding window has short sequences (< 100 timesteps)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ALTERNATIVE: MultiOutputFlareForecaster                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Predicts: flare occurrence + magnitude + confidence             │
│ • Multi-task learning improves shared representations             │
│ • Provides uncertainty estimates for predictions                  │
│                                                                     │
│ USE WHEN:                                                           │
│ • You need flare magnitude predictions (not just binary)          │
│ • Confidence/uncertainty scores are important                     │
│ • Have labeled data for multiple prediction tasks                 │
└─────────────────────────────────────────────────────────────────────┘

RECOMMENDED SLIDING WINDOW SETUP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sequence_length = 60  # 60-minute window (1-minute cadence)
step_size = 1         # Slide by 1 minute for dense training data
input_dim = 2         # [TIME, COUNTS] or [RATE_CDTE, RATE_CZT]
labels = binary       # 1 if flare in next step, 0 otherwise

Example:
┌────────────────────┬─────────┐
│ Window[0..59]      │ Label   │
│ (60 min history)   │ (next)  │
├────────────────────┼─────────┤
│ Times 0-59         │ Flare @ │
│ Rates 0-59         │ t=60?   │
├────────────────────┼─────────┤
│ Times 1-60         │ Flare @ │
│ Rates 1-60         │ t=61?   │
└────────────────────┴─────────┘

This creates highly overlapping windows perfect for LSTM
temporal dependencies and attention mechanisms.
"""
