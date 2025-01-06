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
    'HeartRate(1/min)_diff',
    'CoreTemperature(degC)_diff',
    'SkinTemperature(degC)_diff',
    'CardiacOutput(mL/min)_diff',
    'MeanArterialPressure(mmHg)_diff',
    'SystolicArterialPressure(mmHg)_diff',
    'DiastolicArterialPressure(mmHg)_diff',
    'TotalMetabolicRate(kcal/day)_diff',
    'RespirationRate(1/min)_diff',
    'AchievedExerciseLevel_diff',
    'FatigueLevel_diff',
    'TotalMetabolicRate(W)_diff'
]

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
    'HeartRate(1/min)_diff', 
    'CoreTemperature(degC)_diff',
    'SkinTemperature(degC)_diff',
]

# Define model hyperparameters

# ... TODO Implement this

# Define initial state of model
"""
initial_state=(
        1,                      # Time delta
        127.0920033,            # Heart rate
        7141.82462835,          # Cardiac output
        95.62668286666667,      # Mean arterial pressure
        110.93833814999999,     # Systolic arterial pressure
        79.2137125,             # Diastolic arterial pressure
        6852.245369266667,      # Total metabolic rate
        37.062846533333335,     # Core temperature
        33.14000563333333,      # Skin temperature
        17.233677316666668,     # Respiration rate
        0.7853819333333334,     # Achieved exercise level
        0.03883333333333334,    # Fatigue level
        150.00,                    # Total metabolic rate
        0.25,                   # Exercise intensity
        22.22222222222222,      # Ambient temperature
        80.0                    # Relative humidity
    )
"""

initial_state = (
    1,                  # Time delta
    127,                # Heart rate
    37.06,              # Core temperature
    33.14,              # Skin temperature
    0.25,               # Exercise intensity
    22.0,               # Ambient temperature
    80.0                # Relative humidity
)

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
    initial_state=initial_state
)
