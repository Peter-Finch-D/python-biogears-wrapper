############################################
# Main.py
############################################

# Import the necessary libraries
import os
import torch
import numpy as np
import pandas as pd
from digital_twin.data_processing import load_and_process_data
from my_transformer import TransformerRegressor

###############################################################################
# Define directories
###############################################################################
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
outputs_dir = 'outputs'
model_output_dir = os.path.join(outputs_dir, 'models')
scalers_output_dir = os.path.join(outputs_dir, 'scalers')
visualizations_dir = os.path.join(outputs_dir, 'visualizations')

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
    'CoreTemperature(degC)',
    'SkinTemperature(degC)',
    'intensity', 
    'atemp_c', 
    'rh_pct'
]
target_cols = [
    'HeartRate(1/min)_next',
    'CoreTemperature(degC)_diff',
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
    next=True,
    diff=True,
    time_deltas=True
)

df              = results['df']
sequence_length = results['sequence_length']
scaler_X        = results['scaler_X']
scaler_Y        = results['scaler_Y']

# Import your multi-target Transformer


###############################################################################
# Create (or load) the TransformerRegressor
###############################################################################
model = TransformerRegressor(
    feature_cols=feature_cols,
    target_cols=target_cols,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    hidden_dims=[1024, 512, 256],
    outputs_dir=model_output_dir
)
"""
# Uncomment if you wish to train from scratch:
model.train_model(
    df=df,
    seq_length=1,    # or 19
    epochs=200,
    learning_rate=1e-3,
    test_split=0.2
)
"""

# Otherwise, load the serialized model
model_save_path = os.path.join(model_output_dir, "combined_transformer_regressor.pt")
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

###############################################################################
# Evaluate step-by-step on some new data (BioGears or otherwise)
###############################################################################
# We'll define an initial state for the features (time_delta, CoreTemp, SkinTemp, intensity, atemp_c, rh_pct)
initial_state = (0, 75, 37.0, 33.0, 0.25, 30.0, 30.0)

# Suppose you have a new CSV with real data
bg_df = pd.read_csv(os.path.join(outputs_dir, 'biogears_results.csv'))
predict_delta_mask = [False, True, True]

extra_feature_cols = ['intensity', 'atemp_c', 'rh_pct']  # these match feature_cols[3:]

preds_array, mae_dict = model.evaluate_model(
    initial_state=initial_state,
    df=bg_df,
    scaler_X=scaler_X,
    scaler_Y=scaler_Y,
    target_cols=['HeartRate(1/min)', 'CoreTemperature(degC)', 'SkinTemperature(degC)'],         # multiple target columns
    time_col='Time(s)',
    extra_feature_cols=extra_feature_cols,
    predict_delta_mask=predict_delta_mask,
    outputs_dir=outputs_dir,
    visualizations_dir=visualizations_dir,
    plot_results=True
)

print("Multi-Target Predictions shape:", preds_array.shape)
print("MAE Dictionary:", mae_dict)
print("Overall MAE:", mae_dict['overall'])
