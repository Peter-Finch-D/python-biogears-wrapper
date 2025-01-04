# Main.py

import os
import subprocess
import sys
from datetime import datetime
from digital_twin.utils import print_with_timestamp

def run_script(script_name):
    """
    Runs a Python script and handles errors.
    """
    try:
        subprocess.check_call([sys.executable, script_name])
        print_with_timestamp(f"Completed {script_name} successfully.")
    except subprocess.CalledProcessError as e:
        print_with_timestamp(f"Error running {script_name}: {e}")
        sys.exit(1)

"""
if __name__ == "__main__":
    # Define script paths
    data_processing_script = os.path.join('digital_twin', 'data_processing.py')
    train_model_script = os.path.join('digital_twin', 'train_model.py')
    evaluate_model_script = os.path.join('digital_twin', 'evaluate_model.py')
    
    # Check if scripts exist
    for script in [data_processing_script, train_model_script, evaluate_model_script]:
        if not os.path.isfile(script):
            print_with_timestamp(f"Required script {script} not found. Exiting.")
            sys.exit(1)
    
    # Run data processing
    print_with_timestamp("Starting Data Processing...")
    run_script(data_processing_script)
    
    # Run model training
    print_with_timestamp("Starting Model Training...")
    run_script(train_model_script)
    
    # Run evaluation
    print_with_timestamp("Starting Model Evaluation...")
    run_script(evaluate_model_script)
    
    print_with_timestamp("All tasks completed successfully.")
"""

from digital_twin.data_processing import load_and_process_data
from digital_twin.train_model import train_model
from digital_twin.evaluate_model import run_evaluation

# Define directories
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
outputs_dir = 'outputs'
model_output_dir = outputs_dir+'/models'
scalers_output_dir = outputs_dir+'/scalers'
visualizations_dir = '/visualizations'
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(scalers_output_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)
output_csv_path = os.path.join(outputs_dir, 'processed_data.csv')

# DATA PROCESSING
"""
# Run data processing
scaler_X, scaler_Y = load_and_process_data(
    data_dir=data_dir, 
    output_csv_path=output_csv_path,
    simulation_length=10,
    seq_length=4,
    pred_length=5
)
"""

"""
# TRAIN MODEL
train_model(
    processed_csv_path=output_csv_path,
    scalers_dir=scalers_output_dir,
    models_dir=model_output_dir,
)
"""

# Evaluate model
run_evaluation(
    models_dir=model_output_dir,
    scalers_dir=scalers_output_dir,
    visualizations_dir=visualizations_dir,
    model_filename='final_seq2seq_lstm_model.pth',
)