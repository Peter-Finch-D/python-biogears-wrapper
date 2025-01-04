# digital_twin/data_processing.py

import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime
from joblib import dump

from digital_twin.dataset import Seq2SeqPhysioDataset
from digital_twin.utils import print_with_timestamp, sanitize_filename


def load_and_process_data(data_dir, output_csv_path, simulation_length=10, seq_length=4, pred_length=5):
    """
    Loads all CSV files from the specified directory, processes them, and saves the combined DataFrame.
    
    Parameters:
    - data_dir: Directory containing CSV files.
    - output_csv_path: Path to save the processed CSV.
    - simulation_length: Expected number of timesteps per CSV file.
    - seq_length: Number of timesteps in input sequences.
    - pred_length: Number of timesteps to predict.
    
    Returns:
    - scaler_X: Fitted StandardScaler for input features.
    - scaler_Y: Fitted StandardScaler for target features.
    """
    print_with_timestamp("Loading all CSV files for training...")
    
    # Grab all CSV paths
    all_csv_paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if len(all_csv_paths) == 0:
        raise FileNotFoundError("No CSV files found in the data directory.")
    
    all_data = []
    for csv_path in all_csv_paths:
        # Read the CSV
        df = pd.read_csv(csv_path)
    
        # Resample to 1-minute intervals
        df["Time(s)"] = pd.to_timedelta(df["Time(s)"])
        df.set_index("Time(s)", inplace=True)
        df = df.resample('1min').mean()
    
        assert len(df) == simulation_length, f"Expected {simulation_length} rows, got {len(df)} in {csv_path}"
    
        # Drop "Time(s)" column as it's now the index
        if 'Time(s)' in df.columns:
            df.drop(columns=['Time(s)'], inplace=True)
    
        # Shift target columns to represent next timestep
        df['HeartRate(1/min)_next'] = df['HeartRate(1/min)'].shift(-1)
        df['CoreTemperature(degC)_next'] = df['CoreTemperature(degC)'].shift(-1)
        df['SkinTemperature(degC)_next'] = df['SkinTemperature(degC)'].shift(-1)
    
        df.dropna(inplace=True)  # Remove the last row with NaN targets
    
        # Ensure all required columns are present
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
        required_columns = feature_cols + target_cols
    
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
    
        # Store the entire DataFrame (one short sequence)
        all_data.append(df)
    
    print_with_timestamp(f"Loaded {len(all_data)} files. Each has {len(all_data[0]) if len(all_data) > 0 else 0} rows after shifting.")
    
    # Concatenate all data for scaling
    concat_df = pd.concat(all_data, ignore_index=True)
    
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
    
    # Separate scalers for inputs and targets
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    # Fit scalers
    scaler_X.fit(concat_df[feature_cols])
    scaler_Y.fit(concat_df[target_cols])
    
    # Save scalers using joblib for later use
    scaler_output_dir = os.path.join(os.path.dirname(output_csv_path), 'scalers')
    os.makedirs(scaler_output_dir, exist_ok=True)
    dump(scaler_X, os.path.join(scaler_output_dir, 'scaler_X.joblib'))
    dump(scaler_Y, os.path.join(scaler_output_dir, 'scaler_Y.joblib'))
    
    # Save processed data
    concat_df.to_csv(output_csv_path, index=False)
    print_with_timestamp(f"Processed data saved to {output_csv_path}")
    
    return scaler_X, scaler_Y

if __name__ == "__main__":
    # Define directories
    data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
    outputs_dir = 'outputs'
    os.makedirs(outputs_dir, exist_ok=True)
    output_csv_path = os.path.join(outputs_dir, 'processed_data.csv')
    
    # Run data processing
    scaler_X, scaler_Y = load_and_process_data(
        data_dir=data_dir, 
        output_csv_path=output_csv_path,
        simulation_length=10,
        seq_length=4,
        pred_length=5
    )
