import os
import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from tqdm import tqdm # type: ignore
from datetime import datetime

def print_with_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Directory containing the CSV files
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'

print_with_timestamp("Loading CSV files...")

# Load all CSV files
all_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
all_files.sort()  # Ensure consistent order

# Use the first ten files for training and testing
train_test_files = all_files[:10]
evaluation_files = all_files[10:]

print_with_timestamp("Loading training and testing data...")

# Load training and testing data
train_test_data_frames = [pd.read_csv(os.path.join(data_dir, file)) for file in train_test_files]
train_test_data = pd.concat(train_test_data_frames, ignore_index=True)

# Drop the 'Time(s)' column
train_test_data = train_test_data.drop(columns=['Time(s)'])

# Shift the target columns to create the next timestep prediction
train_test_data['Next_CoreTemperature'] = train_test_data['CoreTemperature(degC)'].shift(-1)
train_test_data['Next_HeartRate'] = train_test_data['HeartRate(1/min)'].shift(-1)
train_test_data['Next_SkinTemperature'] = train_test_data['SkinTemperature(degC)'].shift(-1)

# Drop the last row with NaN values created by the shift
train_test_data = train_test_data.dropna()

# Features and targets
features = train_test_data.drop(columns=['Next_CoreTemperature', 'Next_HeartRate', 'Next_SkinTemperature'])
targets = train_test_data[['Next_CoreTemperature', 'Next_HeartRate', 'Next_SkinTemperature']]

print_with_timestamp("Training the RandomForestRegressor model...")

# Train a RandomForestRegressor model with tqdm progress bar
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Wrap the fit method with tqdm
with tqdm(total=model.n_estimators) as pbar:
    for _ in range(model.n_estimators):
        model.fit(features, targets)
        pbar.update()

print_with_timestamp("Evaluating the model...")

# Evaluate the model on each evaluation file
for eval_file in evaluation_files:
    print_with_timestamp(f"Evaluating file: {eval_file}")
    eval_data = pd.read_csv(os.path.join(data_dir, eval_file))
    
    # Drop the 'Time(s)' column
    eval_data = eval_data.drop(columns=['Time(s)'])
    
    # Shift the target columns to create the next timestep prediction
    eval_data['Next_CoreTemperature'] = eval_data['CoreTemperature(degC)'].shift(-1)
    eval_data['Next_HeartRate'] = eval_data['HeartRate(1/min)'].shift(-1)
    eval_data['Next_SkinTemperature'] = eval_data['SkinTemperature(degC)'].shift(-1)
    
    # Drop the last row with NaN values created by the shift
    eval_data = eval_data.dropna()
    
    # Features and targets
    eval_features = eval_data.drop(columns=['Next_CoreTemperature', 'Next_HeartRate', 'Next_SkinTemperature'])
    eval_targets = eval_data[['Next_CoreTemperature', 'Next_HeartRate', 'Next_SkinTemperature']]
    
    # Predict on the evaluation data
    eval_pred = model.predict(eval_features)
    
    # Evaluate the model
    mse_core_temp = mean_squared_error(eval_targets['Next_CoreTemperature'], eval_pred[:, 0])
    mse_heart_rate = mean_squared_error(eval_targets['Next_HeartRate'], eval_pred[:, 1])
    mse_skin_temp = mean_squared_error(eval_targets['Next_SkinTemperature'], eval_pred[:, 2])
    
    print(f'File: {eval_file}')
    print(f'Mean Squared Error for Core Temperature: {mse_core_temp}')
    print(f'Mean Squared Error for Heart Rate: {mse_heart_rate}')
    print(f'Mean Squared Error for Skin Temperature: {mse_skin_temp}')
    print('---')

print_with_timestamp("Program completed.")