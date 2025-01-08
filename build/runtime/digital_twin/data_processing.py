import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump

from digital_twin.utils import print_with_timestamp

def load_and_process_data(data_dir, output_csv_path, feature_cols, target_cols, diff=False, scaled=False, next=False, time_deltas=False):
    """
    Loads all CSV files from the specified directory, processes them to predict changes, and saves the combined DataFrame.
    
    Parameters:
    - data_dir: Directory containing CSV files.
    - output_csv_path: Path to save the processed CSV.
    - feature_cols: List of feature columns.
    - target_cols: List of target columns.
    - diff: Boolean to calculate _diff features or not.
    - next: Boolean to calculate _next features or not.
    
    Returns:
    - scaler_X: Fitted StandardScaler for input features.
    - scaler_Y: Fitted StandardScaler for target features.
    """
    results = {}
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

        # Add time_delta conditionally
        if time_deltas:
            df['time_delta'] = list(range(len(df)))
    
        # Drop "Time(s)" column as it's now the index
        if 'Time(s)' in df.columns:
            df.drop(columns=['Time(s)'], inplace=True)

        if diff:
            # Compute differences for target columns to represent change
            columns_to_exclude = ['intensity', 'atemp_c', 'rh_pct']
            columns_to_diff = df.columns.copy()
            for col in columns_to_diff:
                if col not in columns_to_exclude:
                    df[f'{col}_diff'] = df[col].diff()

        if next:
            # Compute next timestep values for feature columns
            columns_to_exclude = ['intensity', 'atemp_c', 'rh_pct']
            columns_to_next = df.columns.copy()
            for col in columns_to_next:
                if col not in columns_to_exclude:
                    df[f'{col}_next'] = df[col].shift(-1)

        df.dropna(inplace=True)  # Remove the first or last row with NaN after next or diff
        required_columns = feature_cols + target_cols
    
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        
        # Drop any columns not in feature_cols or target_cols
        df = df[required_columns]
    
        # Store the entire DataFrame (one short sequence)
        all_data.append(df)
    
    print_with_timestamp(f"Loaded {len(all_data)} files. Each has {len(all_data[0]) if len(all_data) > 0 else 0} rows after computing differences.")
    
    # Concatenate all data for scaling
    concat_df = pd.concat(all_data, ignore_index=True)
    
    # Separate scalers for inputs and targets
    if scaled:
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        # Fit scalers
        scaler_X.fit(concat_df[feature_cols])
        scaler_Y.fit(concat_df[target_cols])
        # Transform the feature and target columns using the fitted scalers
        concat_df[feature_cols] = scaler_X.transform(concat_df[feature_cols])
        concat_df[target_cols] = scaler_Y.transform(concat_df[target_cols])
        # Save scalers to results
        results['scaler_X'] = scaler_X
        results['scaler_Y'] = scaler_Y
        # Save scalers using joblib for later use
        scaler_output_dir = os.path.join(os.path.dirname(output_csv_path), 'scalers')
        os.makedirs(scaler_output_dir, exist_ok=True)
        dump(scaler_X, os.path.join(scaler_output_dir, 'scaler_X.joblib'))
        dump(scaler_Y, os.path.join(scaler_output_dir, 'scaler_Y.joblib'))
    
    # Save processed data
    concat_df.to_csv(output_csv_path, index=False)
    print_with_timestamp(f"Processed data saved to {output_csv_path}")
    
    results['df'] = concat_df
    return results

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
        feature_cols=['feature1', 'feature2'],  # Example feature columns
        target_cols=['target1', 'target2'],  # Example target columns
        diff=True,
        next=True, 
        scaled=True,
        time_deltas=True
    )
