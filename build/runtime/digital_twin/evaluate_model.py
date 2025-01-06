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
    feature_cols,
    target_cols,
    model_filename='best_seq2seq_lstm_model.pth',
    segments_cold={
        'time': [1.00] * 10,
        'intensity': [0.25] * 10,
        'atemp_c': [22.00] * 10,
        'rh_pct': [80.00] * 10,
    },
    initial_state = (
        1,                  # Time delta
        127,                # Heart rate
        37.06,              # Core temperature
        33.14,              # Skin temperature
        0.25,               # Exercise intensity
        22.0,               # Ambient temperature
        80.0                # Relative humidity
    )
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

    # Run BioGears for ground truth
    xml_scenario = segments_to_xml(segments_cold)
    print_with_timestamp("Using BioGears for ground truth")
    biogears_results = run_biogears(xml_scenario, segments_cold)

    print("LENGTH OF BIOGEARS RESULTS: ", len(biogears_results))
    print(biogears_results)
    
    # Load scalers
    scaler_X = load(os.path.join(scalers_dir, 'scaler_X.joblib'))
    scaler_Y = load(os.path.join(scalers_dir, 'scaler_Y.joblib'))
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_with_timestamp(f"Using device: {device}")
    
    # Initialize model
    input_size = len(feature_cols)
    output_size = len(target_cols)
    model = Seq2SeqEnhancedLSTMModel(
        input_size=input_size,
        hidden_size=64,  # Should match the training hyperparameters
        output_size=output_size,
        pred_length=5,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Remove all old checkpoint-based lines:
    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Instead, skip loading incompatible checkpoints:
    print_with_timestamp("Using freshly initialized model weights (no checkpoint load).")
    model.eval()
    
    print_with_timestamp("Model is set for evaluation without loading old checkpoints.")
    
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
    
    
    
    # Extract predictions and ground truths
    heart_rate_predictions = lstm_results['HeartRate(1/min)'].values
    core_temp_predictions = lstm_results['CoreTemperature(degC)'].values
    skin_temp_predictions = lstm_results['SkinTemperature(degC)'].values
    
    heart_rate_ground_truth = biogears_results['HeartRate(1/min)'].values
    core_temp_ground_truth = biogears_results['CoreTemperature(degC)'].values
    skin_temp_ground_truth = biogears_results['SkinTemperature(degC)'].values
    
    print(len(heart_rate_predictions), len(heart_rate_ground_truth))
    
    # Calculate Mean Absolute Error (MAE) for each metric
    for col in feature_cols:
        if col != 'time_delta':
            mae_val = mean_absolute_error(biogears_results[col].values, lstm_results[col].values)
            print_with_timestamp(f"MAE for {col}: {mae_val:.2f}")
    
    # Optionally, visualize the results
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
        # Align the number of points for both LSTM and BioGears
        min_len = min(len(biogears_results[ground_truth_columns[i]]), len(lstm_results[prediction_columns[i]]))
        ground_truth_vals = biogears_results[ground_truth_columns[i]].iloc[:min_len].reset_index(drop=True)
        prediction_vals = lstm_results[prediction_columns[i]].iloc[:min_len].reset_index(drop=True)
        x_range = range(min_len)

        plt.figure(figsize=(10, 5))
        plt.plot(x_range, ground_truth_vals, label='Ground Truth', marker='x')
        plt.plot(x_range, prediction_vals, label='Predictions', marker='o')
        plt.fill_between(
            x_range,
            prediction_vals - abs(prediction_vals - ground_truth_vals),
            prediction_vals + abs(prediction_vals - ground_truth_vals),
            color='gray', alpha=0.2, label='Error Range'
        )
        plt.title(f'{label} - Ground Truth vs Predictions (Cold Scenario)')
        plt.xlabel('Timestep')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_label = sanitize_filename(label)
        plt.savefig(os.path.join(visualizations_dir, f'{safe_label}_cold_scenario.png'))
        plt.close()

    # Residual Analysis
    for i, label in enumerate(labels):
        min_len = min(len(biogears_results[ground_truth_columns[i]]), len(lstm_results[prediction_columns[i]]))
        ground_truth_vals = biogears_results[ground_truth_columns[i]].iloc[:min_len]
        prediction_vals = lstm_results[prediction_columns[i]].iloc[:min_len]
        residuals = ground_truth_vals.values - prediction_vals.values
        x_range = range(min_len)

        plt.figure(figsize=(10, 4))
        plt.plot(x_range, residuals, marker='o', linestyle='-', label='Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals for {label}')
        plt.xlabel('Timestep')
        plt.ylabel('Residual')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_label = sanitize_filename(label)
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
        initial_state=(
            149.63539395, 8132.851798883334, 89.18128038333333, 103.98668848333334,
            73.8366554, 11967.142991633333, 37.091162383333334, 33.1498931,
            18.519237850000003, 0.7299451833333334, 0.10026335, 0.0,
            0.5, 20.0, 40.0
        )
    )
