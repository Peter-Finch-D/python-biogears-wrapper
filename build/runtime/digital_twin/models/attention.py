# digital_twin/models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Mapping from hidden_size*2 (for bidirectional) to 1 for scalar attention weights
        self.attention = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, lstm_output):
        """
        Forward pass for the attention mechanism.
        
        Parameters:
        - lstm_output: Tensor of shape (batch_size, seq_length, hidden_size*2)
        
        Returns:
        - context_vector: Tensor of shape (batch_size, hidden_size*2)
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length, 1)
        
        # Compute context vector as the weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size*2)
        return context_vector
