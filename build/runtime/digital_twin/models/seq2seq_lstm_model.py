# digital_twin/models/seq2seq_lstm_model.py

import torch
import torch.nn as nn
from digital_twin.models.attention import Attention

class Seq2SeqEnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_length=5, num_layers=2, dropout=0.3):
        super(Seq2SeqEnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.bidirectional = True

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size * pred_length)  # Output multiple timesteps

    def forward(self, x):
        """
        Forward pass for the Seq2Seq LSTM model.
        
        Parameters:
        - x: Tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
        - out: Tensor of shape (batch_size, pred_length, output_size)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size*2)
        context = self.attention(lstm_out)  # (batch_size, hidden_size*2)
        out = self.fc(context)  # (batch_size, output_size * pred_length)
        out = out.view(-1, self.pred_length, 3)  # (batch_size, pred_length, output_size)
        return out
