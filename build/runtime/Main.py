# Main.py

import os
import subprocess
import sys
from datetime import datetime
from digital_twin.utils import print_with_timestamp
from digital_twin.data_processing import load_and_process_data
from digital_twin.train_model import train_model
from digital_twin.evaluate_model import run_evaluation

# Define directories
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
outputs_dir = 'outputs'
model_output_dir = outputs_dir+'/models'
scalers_output_dir = outputs_dir+'/scalers'
visualizations_dir = 'visualizations'
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(scalers_output_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)
output_csv_path = os.path.join(outputs_dir, 'processed_data.csv')

# Define feature and target columns
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
    'CoreTemperature(degC)_next',
    'SkinTemperature(degC)_next',
]

feature_cols = [
    'time_delta',
    'HeartRate(1/min)',
    'CardiacOutput(mL/min)',
    'MeanArterialPressure(mmHg)',
    'SystolicArterialPressure(mmHg)',
    'DiastolicArterialPressure(mmHg)',
    'TotalMetabolicRate(kcal/day)',
    'CoreTemperature(degC)',
    'SkinTemperature(degC)',
    'RespirationRate(1/min)',
    'AchievedExerciseLevel',
    'FatigueLevel',
    'TotalMetabolicRate(W)',
    'intensity',
    'atemp_c',
    'rh_pct'
]
target_cols = [
    'HeartRate(1/min)_next',
    'CoreTemperature(degC)_next',
    'SkinTemperature(degC)_next',
    'CardiacOutput(mL/min)_next',
    'MeanArterialPressure(mmHg)_next',
    'SystolicArterialPressure(mmHg)_next',
    'DiastolicArterialPressure(mmHg)_next',
    'TotalMetabolicRate(kcal/day)_next',
    'RespirationRate(1/min)_next',
    'AchievedExerciseLevel_next',
    'FatigueLevel_next',
    'TotalMetabolicRate(W)_next'
]

# Define model hyperparameters

# ... TODO Implement this


"""
# Process the data and save it to a CSV file
scaler_X, scaler_Y = load_and_process_data(
    data_dir=data_dir, 
    output_csv_path=output_csv_path,
    feature_cols=feature_cols,
    target_cols=target_cols,
    simulation_length=10,
    seq_length=4,
    pred_length=5
)

# Train the model and save to output/models directory
train_model(
    processed_csv_path=output_csv_path,
    scalers_dir=scalers_output_dir,
    models_dir=model_output_dir,
    feature_cols=feature_cols,
    target_cols=target_cols,
    num_workers=10,
    epochs=100,
    hidden_size=64,
    num_layers=2,
)
"""
# Evaluate the model against the BioGears engine and save visualizations
run_evaluation(
    models_dir=model_output_dir,
    scalers_dir=scalers_output_dir,
    visualizations_dir=visualizations_dir,
    feature_cols=feature_cols,
    target_cols=target_cols,
    model_filename='final_seq2seq_lstm_model.pth',
)
