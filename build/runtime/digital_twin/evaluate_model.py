# digital_twin/evaluate_model.py

import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from joblib import load
import matplotlib.pyplot as plt

from digital_twin.models import Seq2SeqEnhancedLSTMModel
from digital_twin.utils import print_with_timestamp, sanitize_filename

from biogears_python.xmlscenario import segments_to_xml
from biogears_python.execution import run_biogears, run_lstm_sequence

def run_evaluation(
    models_dir,
    scalers_dir,
    visualizations_dir,
    model_filename='best_seq2seq_lstm_model.pth',
    segments_cold={
        'time': [1.00] * 10,
        'intensity': [0.25] * 10,
        'atemp_c': [22.00] * 10,
        'rh_pct': [80.00] * 10,
    },
    initial_state=(70.0, 37.0, 33.0)
):
    """
    Evaluates the trained model against the BioGears engine on a cold scenario.
    
    Parameters:
    - models_dir: Directory where models are stored.
    - scalers_dir: Directory where scalers are stored.
    - visualizations_dir: Directory to save evaluation visualizations.
    - model_filename: Filename of the trained model to load.
    - segments_cold: Dictionary defining the cold scenario segments.
    - initial_state: Tuple of initial physiological states (HeartRate, CoreTemperature, SkinTemperature).
    """
    # Load scalers
    print_with_timestamp(f"Current directory: {os.getcwd()}")
    scaler_X = load(os.path.join(scalers_dir, 'scaler_X.joblib'))
    scaler_Y = load(os.path.join(scalers_dir, 'scaler_Y.joblib'))
    
    # Define feature and target columns
    feature_cols = [
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
        'SkinTemperature(degC)_next'
    ]
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_with_timestamp(f"Using device: {device}")
    
    # Initialize model
    input_size = len(feature_cols)      # 6
    output_size = len(target_cols)      # 3
    model = Seq2SeqEnhancedLSTMModel(
        input_size=input_size,
        hidden_size=64,  # Should match the training hyperparameters
        output_size=output_size,
        pred_length=5,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Load model weights
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print_with_timestamp(f"Loaded model from {model_path}")
    
    # Run LSTM predictions
    lstm_results = run_lstm_sequence(
        model, 
        segments_cold, 
        scaler_X, 
        scaler_Y, 
        seq_length=4, 
        initial_state=initial_state, 
        device=device
    )
    
    print_with_timestamp("Model test run on cold scenario segments")
    print("LENGTH OF LSTM RESULTS: ", len(lstm_results))
    print(lstm_results)
    
    # Run BioGears for ground truth
    xml_scenario = segments_to_xml(segments_cold)
    biogears_results = run_biogears(xml_scenario, segments_cold)
    
    print_with_timestamp("Using BioGears for ground truth")
    print("LENGTH OF BIOGEARS RESULTS: ", len(biogears_results))
    print(biogears_results)
    
    # Extract predictions and ground truths
    heart_rate_predictions = lstm_results['HeartRate(1/min)'].values
    core_temp_predictions = lstm_results['CoreTemperature(degC)'].values
    skin_temp_predictions = lstm_results['SkinTemperature(degC)'].values
    
    heart_rate_ground_truth = biogears_results['HeartRate(1/min)'].values
    core_temp_ground_truth = biogears_results['CoreTemperature(degC)'].values
    skin_temp_ground_truth = biogears_results['SkinTemperature(degC)'].values
    
    print(len(heart_rate_predictions), len(heart_rate_ground_truth))
    
    # Calculate Mean Absolute Error (MAE) for each metric
    mae_heart_rate = mean_absolute_error(heart_rate_ground_truth, heart_rate_predictions)
    mae_core_temp = mean_absolute_error(core_temp_ground_truth, core_temp_predictions)
    mae_skin_temp = mean_absolute_error(skin_temp_ground_truth, skin_temp_predictions)
    
    # Print the MAE values
    print_with_timestamp(f"Mean Absolute Error (Heart Rate): {mae_heart_rate:.2f}")
    print_with_timestamp(f"Mean Absolute Error (Core Temperature): {mae_core_temp:.2f}")
    print_with_timestamp(f"Mean Absolute Error (Skin Temperature): {mae_skin_temp:.2f}")
    
    # Optionally, visualize the results
    print(visualizations_dir)
    if not os.path.exists(visualizations_dir):
        print("Does not exist!")
    visualize_results(
        lstm_results=lstm_results,
        biogears_results=biogears_results,
        visualizations_dir=visualizations_dir
    )

def visualize_results(lstm_results, biogears_results, visualizations_dir):
    """
    Generates and saves plots comparing LSTM predictions with BioGears ground truth.
    
    Parameters:
    - lstm_results: DataFrame containing LSTM predictions.
    - biogears_results: DataFrame containing BioGears ground truth.
    - visualizations_dir: Directory to save the plots.
    """
    import matplotlib.pyplot as plt
    import re

    def sanitize_filename(label):
        """
        Sanitize the label to create a safe filename.
        Replaces spaces with underscores, '/' with '_per_', and removes other special characters.
        """
        label = label.lower().replace(" ", "_").replace("/", "_per_")
        label = re.sub(r'[^\w\-_.]', '', label)  # Remove any character that is not alphanumeric, '-', '_', or '.'
        return label
    
    labels = ['Heart Rate (1/min)', 'Core Temperature (degC)', 'Skin Temperature (degC)']
    ground_truth_columns = ['HeartRate(1/min)', 'CoreTemperature(degC)', 'SkinTemperature(degC)']
    prediction_columns = ['HeartRate(1/min)', 'CoreTemperature(degC)', 'SkinTemperature(degC)']
    for i, label in enumerate(labels):
        plt.figure(figsize=(10, 5))
        plt.plot(biogears_results[ground_truth_columns[i]], label='Ground Truth', marker='x')
        plt.plot(lstm_results[prediction_columns[i]], label='Predictions', marker='o')
        plt.fill_between(
            range(len(lstm_results[prediction_columns[i]])),
            lstm_results[prediction_columns[i]] - abs(lstm_results[prediction_columns[i]] - biogears_results[ground_truth_columns[i]]),
            lstm_results[prediction_columns[i]] + abs(lstm_results[prediction_columns[i]] - biogears_results[ground_truth_columns[i]]),
            color='gray', alpha=0.2, label='Error Range'
        )
        plt.title(f'{label} - Ground Truth vs Predictions (Cold Scenario)')
        plt.xlabel('Timestep')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Sanitize the label to create a safe filename
        safe_label = sanitize_filename(label)
        
        # Save the figure
        print(os.path.join(visualizations_dir, f'{safe_label}_cold_scenario.png'))
        plt.savefig(os.path.join(visualizations_dir, f'{safe_label}_cold_scenario.png'))
        if os.path.exists(os.path.join(visualizations_dir, f'{safe_label}_cold_scenario.png')):
            print(f"File {safe_label}_cold_scenario.png exists.")
        else:
            print(f"File {safe_label}_cold_scenario.png does not exist.")
        plt.close()
    
    # Residual Analysis
    for i, label in enumerate(labels):
        residuals = biogears_results[ground_truth_columns[i]].values - lstm_results[prediction_columns[i]].values
        plt.figure(figsize=(10, 4))
        plt.plot(residuals, marker='o', linestyle='-', label='Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals for {label}')
        plt.xlabel('Timestep')
        plt.ylabel('Residual')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Sanitize the label to create a safe filename
        safe_label = sanitize_filename(label)
        
        # Save the figure
        plt.savefig(os.path.join(visualizations_dir, f'{safe_label}_residuals_cold_scenario.png'))
        plt.close()

if __name__ == "__main__":
    # Define paths
    outputs_dir = 'outputs'
    models_dir = os.path.join(outputs_dir, 'models')
    scalers_dir = os.path.join(outputs_dir, 'scalers')
    visualizations_dir = 'visualizations'
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Define model filename (ensure it exists)
    model_filename = 'best_seq2seq_lstm_model.pth'
    
    # Define cold scenario segments
    segments_cold = {  # Cold scenario
        'time': [1.00] * 10,
        'intensity': [0.25] * 10,
        'atemp_c': [22.00] * 10,
        'rh_pct': [80.00] * 10,
    }
    
    # Run evaluation
    run_evaluation(
        models_dir=models_dir,
        scalers_dir=scalers_dir,
        visualizations_dir=visualizations_dir,
        model_filename=model_filename,
        segments_cold=segments_cold,
        initial_state=(0.0, 0.0, 0.0)  # Adjust as needed
    )
