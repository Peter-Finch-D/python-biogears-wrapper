import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

def print_with_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Directory containing the CSV files
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
visualizations_dir = '/opt/biogears/core/build/runtime/visualizations/'

# Create visualizations directory if it doesn't exist
os.makedirs(visualizations_dir, exist_ok=True)

print_with_timestamp("Loading CSV files...")

# Load all CSV files
all_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
all_files.sort()  # Ensure consistent order

# Use files 11 through 1000 for training and the first ten files for evaluation
train_test_files = all_files[10:]
evaluation_files = all_files[:10]

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

print_with_timestamp("Training the XGBoost model...")

# Train an XGBoost model
model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(features, targets)

print_with_timestamp("Evaluating the model...")

# DataFrame to store evaluation results
evaluation_results = pd.DataFrame(columns=['MAE_CoreTemperature', 'MAE_HeartRate', 'MAE_SkinTemperature'])

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
    predictions = model.predict(eval_features)
    
    # Evaluate the model using Mean Absolute Error
    mae_core_temp = mean_absolute_error(eval_targets['Next_CoreTemperature'], predictions[:, 0])
    mae_heart_rate = mean_absolute_error(eval_targets['Next_HeartRate'], predictions[:, 1])
    mae_skin_temp = mean_absolute_error(eval_targets['Next_SkinTemperature'], predictions[:, 2])
    
    # Store the results in the DataFrame
    evaluation_results.loc[eval_file] = [mae_core_temp, mae_heart_rate, mae_skin_temp]

    # Plot true vs predicted core temperature
    plt.figure(figsize=(10, 6))
    plt.plot(eval_targets['Next_CoreTemperature'].values, label='True Core Temperature')
    plt.plot(predictions[:, 0], label='Predicted Core Temperature')
    plt.xlabel('Timestep')
    plt.ylabel('Core Temperature (degC)')
    plt.title(f'True vs Predicted Core Temperature for {eval_file}')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, f'{eval_file}_core_temperature.png'))
    plt.close()

print_with_timestamp("Program completed.")

# Print the evaluation results DataFrame
print(evaluation_results)