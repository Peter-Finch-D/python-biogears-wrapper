# Main.py

import pandas as pd
import os
from digital_twin.data_processing import load_and_process_data
from digital_twin.models import SimpleNN
import torch

from biogears_python.xmlscenario import segments_to_xml
from biogears_python.execution import run_biogears
from sklearn.metrics import mean_absolute_error


#from digital_twin.train_model import train_model
#from digital_twin.train_model import train_simple_nn_model    
#from digital_twin.evaluate_model import run_evaluation

# Define directories
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
outputs_dir = 'outputs'
model_output_dir = outputs_dir+'/models'
scalers_output_dir = outputs_dir+'/scalers'
visualizations_dir = outputs_dir+'/visualizations'
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
    'HeartRate(1/min)_diff', 
    'CoreTemperature(degC)_diff',
    'SkinTemperature(degC)_diff',
]

# Define initial state of model
initial_state = (
    1,                  # Time delta
    75,                # Heart rate
    37.03,              # Core temperature
    33.07,              # Skin temperature
    0.25,               # Exercise intensity
    22.0,               # Ambient temperature
    90.0                # Relative humidity
)

# Define the exercise trial for the testing of the model run function
n_segments = 10
segments_cool_humid={
    'time': [1.00] * n_segments,
    'intensity': [0.25] * n_segments,
    'atemp_c': [28.00] * n_segments,
    'rh_pct': [30.00] * n_segments
}

# Process the data and save it to a CSV file
df, scaler_X, scaler_Y = load_and_process_data(
    data_dir=data_dir, 
    output_csv_path=output_csv_path,
    feature_cols=feature_cols,
    target_cols=target_cols,
    simulation_length=10,
    seq_length=4,
    pred_length=5
)

# Instantiate the model architecture and train the model
model_save_path = "outputs/models/nn_model.pth"
model_load_path = "outputs/models/nn_model.pth"

nn_model = SimpleNN(input_size=len(feature_cols), hidden_size=7, output_size=len(target_cols))

# Optional: Train the model from scratch
#nn_model.trn_cml_lss(df, feature_cols, target_cols, epochs=50, save=model_save_path, verbose=True, cumulative_loss_weight=0.95, step_loss_weight=0.05)

# Optional: Load the serialized trained weights
nn_model.load_state_dict(torch.load(model_load_path, weights_only=True))

# Evaluate the model against the BioGears engine
bg_results = run_biogears(segments_to_xml(segments_cool_humid), segments_cool_humid)
bg_results.to_csv('outputs/bg_results.csv')
#bg_results = pd.read_csv('outputs/bg_results.csv')
nn_results = nn_model.run_nn(segments_cool_humid, initial_state, scaler_X, scaler_Y)

nn_results = nn_results.iloc[1:].reset_index(drop=True)

print("BioGears Results:")
print(bg_results)

print("Simple NN Results:")
print(nn_results)

# Calculate mean absolute error for each column
mae_results = {}
for col in nn_results.columns:
    mae_results[col] = mean_absolute_error(bg_results[col], nn_results[col])

# Print the mean absolute error results
print("Mean Absolute Error between BioGears and Simple NN results:")
for col, mae in mae_results.items():
    print(f"{col}: {mae}")


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

visualize_results(nn_results, bg_results, visualizations_dir)