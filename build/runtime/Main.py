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
    'HeartRate(1/min)',
    'intensity', 
    'atemp_c', 
    'rh_pct'
]
target_cols = [
    'HeartRate(1/min)_diff'
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

from my_transformer import TransformerRegressor

# Example hyperparameters:
model = TransformerRegressor(
    feature_cols=feature_cols,
    target_cols=target_cols,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    hidden_dims=[1024, 512, 256],
    outputs_dir="outputs/models"
)
"""
model.train_model(
    df=df,
    seq_length=1,
    epochs=200,
    learning_rate=1e-3,
    test_split=0.2
)
"""

model_save_path = "outputs/models/combined_transformer_regressor.pt"  # same file you saved earlier
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

initial_state = (0, 109, 0.25, 30.0, 30.0)

bg_df = pd.read_csv(outputs_dir + '/biogears_results.csv')
preds, mae = model.evaluate_model(
    initial_state=initial_state,
    df=bg_df,
    scaler_X=scaler_X,
    scaler_Y=scaler_Y,
    time_col='Time(s)',
    target_col='HeartRate(1/min)',
    intensity_col='intensity',
    atemp_col='atemp_c',
    rh_col='rh_pct',
    predict_delta=True,   # or False, depending on how you trained
    plot_results=True
)
print("Final MAE:", mae)