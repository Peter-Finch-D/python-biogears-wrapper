############################################
# Main.py
############################################

# Import the necessary libraries
import os
import numpy as np
import pandas as pd

# We will use PyTorch to build our Transformer model
import torch
import torch.nn as nn
import torch.optim as optim

from train_model import train_transformer_nn


# Import your data processing function
from digital_twin.data_processing import load_and_process_data

###############################################################################
# Define directories
###############################################################################
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
outputs_dir = 'outputs'
model_output_dir = outputs_dir + '/models'
scalers_output_dir = outputs_dir + '/scalers'
visualizations_dir = outputs_dir + '/visualizations'

os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(scalers_output_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

output_csv_path = os.path.join(outputs_dir, 'processed_data.csv')

###############################################################################
# Define feature and target columns
###############################################################################
feature_cols = [
    'time_delta',
    'SkinTemperature(degC)',
    'intensity', 
    'atemp_c', 
    'rh_pct'
]
target_cols = [
    'SkinTemperature(degC)_diff'
]

###############################################################################
# 1. Process the data and save it to a CSV file
###############################################################################
results = load_and_process_data(
    data_dir=data_dir, 
    output_csv_path=output_csv_path,
    feature_cols=feature_cols,
    target_cols=target_cols,
    scaled=True,
    diff=True,
    time_deltas=True
)

df              = results['df']
sequence_length = results['sequence_length']
scaler_X        = results['scaler_X']
scaler_Y        = results['scaler_Y']

print("Original processed DataFrame shape:", df.shape)
print(df.head())

###############################################################################
# 2. Reshape the data for the Transformer
#
# We know from your description:
#   - There are 1810 simulations.
#   - Each simulation is 19 minutes long (1 sample per minute), so 19 samples.
#   - The data is concatenated in df, but each block of 19 rows belongs to one simulation.
#
# A typical Transformer in PyTorch expects input of shape:
#       (sequence_length, batch_size, input_dim)
# or, if using batch_first=True in the TransformerEncoder, then
#       (batch_size, sequence_length, input_dim).
#
# We'll demonstrate a "batch_first=True" approach.
###############################################################################
num_simulations = 1810
samples_per_sim = 19
num_features = len(feature_cols)

# Convert the relevant columns to a NumPy array
data_x = df[feature_cols].values  # shape: (1810*19, num_features)

# Reshape to (batch_size=num_simulations, sequence_length=19, num_features)
data_x = data_x.reshape(num_simulations, samples_per_sim, num_features)

# Convert to a PyTorch tensor
data_x_tensor = torch.tensor(data_x, dtype=torch.float32)

print("\nAfter reshaping for the Transformer:")
print("data_x_tensor.shape =", data_x_tensor.shape, 
      "which corresponds to (batch_size, seq_length, num_features)")
print("\nExample first simulation (shape 19 x num_features):")
print(data_x_tensor[0])

###############################################################################
# 3. Define a simple Transformer Encoder
#
# We'll create a class that has:
#   - An embedding layer or a linear layer to project the input features to a desired dimension.
#   - Positional encoding to give the model a sense of the order in the sequence.
#   - A stack of TransformerEncoder layers.
###############################################################################
class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding function.
    This helps the Transformer understand the order of tokens (time steps).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # DivTerm is a factor used in the sine/cosine formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape = (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # We add the positional encoding to the input embedding
        x = x + self.pe[:, :seq_len, :]
        return x

class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        """
        input_dim: number of input features per time step
        d_model:   internal dimensionality used by the transformer
        nhead:     number of attention heads
        num_layers:number of transformer encoder layers
        dim_feedforward: dimension of the feedforward sublayer
        dropout:   dropout rate
        """
        super(TimeSeriesTransformerEncoder, self).__init__()
        
        # Step 1: Project input features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Step 2: Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Step 3: Define the Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # We'll use batch_first so shape is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_dim)
        """
        # 1) Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)

        # 2) Add positional encoding
        x = self.pos_encoder(x)       # (batch_size, seq_length, d_model)

        # 3) Pass through the Transformer encoder stack
        encoded = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        return encoded

###############################################################################
# 4. Instantiate and run the Transformer Encoder to encode our data
###############################################################################
# Hyperparameters for the Transformer
input_dim = num_features
d_model = 32          # embedding dimension
nhead = 4             # number of attention heads
num_layers = 2        # how many TransformerEncoder layers
dim_feedforward = 64  # feed-forward network dimension
dropout = 0.1

# Create an instance of the model
transformer_encoder = TimeSeriesTransformerEncoder(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)

# Run the data through the encoder
encoded_data = transformer_encoder(data_x_tensor)
# encoded_data shape: (batch_size=1810, seq_length=19, d_model=32)

print("\nEncoded data shape:", encoded_data.shape)
print("This represents the Transformer-encoded representation of each simulation.\n")

###############################################################################
# 5. Explanation of what's happening (in code comments)
#
#   a) The Transformer uses self-attention to learn relationships between all
#      time steps in a sequence. Rather than a strictly left-to-right approach
#      (like an RNN), it can “attend” to any part of the sequence at once.
#
#   b) Each row in 'df' corresponds to a single time step. We grouped them
#      into blocks of 19 to form a sequence for each simulation.
#
#   c) The 'PositionalEncoding' ensures that the model knows the order of these
#      time steps, since a vanilla Transformer is permutation-invariant
#      (it doesn't inherently know which step is first vs. second).
#
#   d) The 'transformer_encoder' processes the entire sequence, building
#      contextual representations at each time step. In other words, the
#      encoding for time step i can incorporate information from time step j.
#
#   e) 'encoded_data' can now be used as features for your downstream task.
#      For instance, you might feed it into a final prediction layer to forecast
#      the next temperature difference or any other target variable.
###############################################################################

###############################################################################
# 6. (Optional) Flatten or select the final time step encoding
#
# Sometimes people take only the last time step's encoding or do an average.
# That depends on your application. For example:
###############################################################################
# Flatten all time steps:
encoded_data_flat = encoded_data.view(num_simulations, -1)
print("Flattened encoded data shape (per simulation):", encoded_data_flat.shape)

# Alternatively, select only the encoding of the final time step:
encoded_data_last_step = encoded_data[:, -1, :]  # shape: (1810, d_model)
print("Last step encoded data shape (per simulation):", encoded_data_last_step.shape)

###############################################################################
# 7. Next steps
# 
# At this point, you could:
#   - Feed 'encoded_data' (or 'encoded_data_flat' / 'encoded_data_last_step') 
#     into a classification or regression head (e.g. a simple feed-forward layer) 
#     to train your neural network on the target of interest.
#   - Adjust hyperparameters (d_model, nhead, num_layers) to find the best 
#     performance for your data.
###############################################################################

###############################################################################
# 8. Save the dataset that is prepared to be fed into the Transformer
###############################################################################
# We have 'data_x_tensor', which is already reshaped into (batch_size=1810, seq_length=19, num_features).
# We also have 'encoded_data', which is the output of the Transformer encoder.

# 8a. Save the reshaped data (input) to a '.pt' file
torch.save(data_x_tensor, os.path.join(outputs_dir, 'transformer_ready_data.pt'))

# 8b. Save the encoded data (output) to a '.pt' file
torch.save(encoded_data, os.path.join(outputs_dir, 'transformer_encoded_data.pt'))

print("\n--- Data saved! ---")
print(f"The reshaped transformer-ready data was saved to: {os.path.join(outputs_dir, 'transformer_ready_data.pt')}")
print(f"The encoded data was saved to: {os.path.join(outputs_dir, 'transformer_encoded_data.pt')}")

###############################################################################
# Optional: How to load the data back
#   - If you need to load these tensors later, just do:
#       data_x_loaded = torch.load('transformer_ready_data.pt')
#       encoded_data_loaded = torch.load('transformer_encoded_data.pt')
###############################################################################


print("\n--- Transformer Encoding Complete ---")

encoded_data_path = os.path.join(outputs_dir, 'transformer_encoded_data.pt')

train_transformer_nn(
    encoded_data_path=encoded_data_path,
    processed_csv_path=output_csv_path,
    feature_cols=feature_cols,
    target_cols=target_cols
)

# Now evaluate the model
