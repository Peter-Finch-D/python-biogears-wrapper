import os
import glob
import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore

import matplotlib.pyplot as plt  # type: ignore

# Import your model evaluation on BioGears segments
from biogears_python.modeleval import evaluate_model_on_segments

from biogears_python.execution import run_lstm

def print_with_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ----------------------------------------------
#               HYPERPARAMETERS
# ----------------------------------------------
EPOCHS = 100
SIMULATION_LENGTH = 10 # Each CSV file is assumed to be 10 timesteps
BATCH_SIZE = SIMULATION_LENGTH
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
TEST_SPLIT = 0.2

# ----------------------------------------------
#             1) LOAD AND COMBINE DATA
# ----------------------------------------------
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
visualizations_dir = '/opt/biogears/core/build/runtime/visualizations/'
os.makedirs(visualizations_dir, exist_ok=True)

print_with_timestamp("Loading all CSV files for training...")

# Grab all CSV paths
all_csv_paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
if len(all_csv_paths) == 0:
    raise FileNotFoundError("No CSV files found in the data directory.")

# Each file is presumably about 10 timesteps (or 600 if 1-sec data).
# We'll load them individually so we can treat each file as one short sequence.
all_data = []
for csv_path in all_csv_paths:

    # Read the csv
    df = pd.read_csv(csv_path)

    # Resample to 1-minute intervals
    df["Time(s)"] = pd.to_timedelta(df["Time(s)"])
    df.set_index("Time(s)", inplace=True)
    df = df.resample('1min').mean()

    assert len(df) == 10, f"Expected 10 rows, got {len(df)}"

    # Ensure that "Time(s)" is a string or timedelta index - we won't actually need the index for training
    if 'Time(s)' in df.columns:
        df.drop(columns=['Time(s)'], inplace=True)
    
    # IMPORTANT: Each DF should contain columns:
    # HeartRate(1/min), CoreTemperature(degC), SkinTemperature(degC),
    # intensity, atemp_c, rh_pct
    # If any are missing, adapt as needed.

    # We store the entire DF (one short sequence) in a list
    all_data.append(df)

print_with_timestamp(f"Loaded {len(all_data)} files. Each is one short time series.")

# ----------------------------------------------
# 2) SCALING: Fit a scaler on the entire dataset
# ----------------------------------------------
# To properly scale, we need to see all numeric columns from all files combined.
# We'll assume the columns are always in the same order in each CSV.
concat_df = pd.concat(all_data, ignore_index=True)

# Identify which columns are inputs vs. targets
# Here, we want to PREDICT (HeartRate, CoreTemp, SkinTemp).
# But we also feed them in as part of the input—plus environment columns.
# So the final targets can be the same 3 physiology columns, while the model input = 6 columns total.
feature_cols = [
    'HeartRate(1/min)', 
    'CoreTemperature(degC)',
    'SkinTemperature(degC)',
    'intensity',
    'atemp_c',
    'rh_pct'
]
target_cols = [
    'HeartRate(1/min)', 
    'CoreTemperature(degC)',
    'SkinTemperature(degC)'
]

scaler = StandardScaler()
scaler.fit(concat_df[feature_cols])  # Fit on all numeric input columns

# For the target, we can use a separate scaler or the same, depending on preference.
# Often a separate scaler is used, but let's keep it simple and re-use one for everything.
# If you prefer a separate target scaler:
#   target_scaler = StandardScaler()
#   target_scaler.fit(concat_df[target_cols])
# We'll do a single scaler approach here for brevity.

# ----------------------------------------------
# 3) CREATE A CUSTOM DATASET
#    Each item in the dataset = (X_seq, Y_seq)
#    where X_seq is shape [seq_len, 6], Y_seq is shape [seq_len, 3]
#    We do "teacher forcing" style sequence training:
#    For a sequence of length T, we want to predict each of the T steps
#    (shifted by 1 if you prefer). But short sequences often just do direct mapping.
# ----------------------------------------------
class PhysioDataset(Dataset):
    def __init__(self, list_of_dfs, feature_cols, target_cols, scaler):
        super().__init__()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scaler = scaler
        
        # Convert each DF into (X, Y) arrays
        self.samples = []
        for df in list_of_dfs:
            # If your dataset truly has EXACTLY 10 rows each, you can rely on that.
            # Otherwise, adapt for the actual length.  
            # We'll skip sequences with fewer than 2 timesteps for safety.
            if len(df) < 2:
                continue

            # Scale input columns
            f_scaled = self.scaler.transform(df[self.feature_cols])
            
            # X and Y can be the same length if you want to predict each step from itself
            # or you can shift by 1 if you prefer next-step prediction.
            # Here, let's do "same length" for a simpler teacher-forcing approach:
            X_seq = f_scaled  # shape [seq_len, 6]
            
            # For target columns, we can scale or not.
            # If using same scaler, we just slice from f_scaled
            # target indices in feature_cols = first 3 columns
            # but let's do a fresh transform for clarity:
            # If you have a separate target_scaler, use that here.
            t_scaled = f_scaled[:, 0:3]  # if the first 3 columns match target_cols exactly
            Y_seq = t_scaled  # shape [seq_len, 3]

            # If you want next-step only: Y_seq = t_scaled[1:] and X_seq = X_seq[:-1]
            # For simplicity, let's do the teacher-forcing style: same shape
            self.samples.append((X_seq, Y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_seq, Y_seq = self.samples[idx]
        # Convert to float32 tensors
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        Y_seq = torch.tensor(Y_seq, dtype=torch.float32)
        return X_seq, Y_seq

# Instantiate the full dataset
full_dataset = PhysioDataset(all_data, feature_cols, target_cols, scaler)

# Optional train/test split (set TEST_SPLIT=0.0 if you truly want no test):
if TEST_SPLIT > 0.0:
    train_size = int((1 - TEST_SPLIT) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
else:
    train_dataset = full_dataset
    test_dataset = None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = None
if test_dataset is not None:
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print_with_timestamp(f"Total sequences: {len(full_dataset)}")
print_with_timestamp(f"Training on {len(train_dataset)} sequences. Test on {len(test_dataset) if test_dataset else 0} sequences.")

# ----------------------------------------------
# 4) DEFINE A STACKED LSTM MODEL FOR SEQ2SEQ
#    This model takes an entire sequence and outputs a prediction for each timestep
# ----------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # batch_first=True => input shape: (B, T, F)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # We'll apply a linear layer at each timestep. The easiest approach is to do:
        #   out, _ = self.lstm(x)
        #   predictions = self.fc(out)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (B, T, input_size)
        # out shape: (B, T, hidden_size)
        # hidden states shape: (num_layers, B, hidden_size)
        out, (h, c) = self.lstm(x)
        # Apply FC to each time step
        out = self.fc(out)  # shape: (B, T, output_size)
        return out

# Create the model
input_size = len(feature_cols)      # 6
output_size = len(target_cols)      # 3
model = LSTMModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=output_size,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------------
# 5) TRAIN THE MODEL
# ----------------------------------------------
print_with_timestamp("Beginning LSTM Training...")
model.train()

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    
    for batch_X, batch_Y in train_loader:
        # batch_X: (B, T, 6)
        # batch_Y: (B, T, 3)
        optimizer.zero_grad()
        
        predictions = model(batch_X)  # shape (B, T, 3)
        loss = criterion(predictions, batch_Y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    if epoch % 10 == 0 or epoch == 1:
        print_with_timestamp(f"Epoch {epoch}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Optionally evaluate on test set if TEST_SPLIT>0
if test_loader is not None:
    model.eval()
    test_loss = 0.0
    scenario_count = 0
    labels = ['Heart Rate', 'Core Temperature', 'Skin Temperature']
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            test_loss += loss.item()
            
            # Convert to numpy arrays
            predictions_np = predictions.cpu().numpy()
            ground_truths_np = batch_Y.cpu().numpy()
            
            for i in range(3):  # Assuming 3 outputs: heart rate, core temperature, skin temperature
                # Create a figure with 16 subplots
                fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=300)
                fig.suptitle(f'{labels[i]} - Ground Truth vs Predictions (Scenario {scenario_count})', fontsize=16)
                
                for j in range(16):
                    ax = axes[j // 4, j % 4]
                    start_idx = j * 10
                    end_idx = (j + 1) * 10
                    predictions_flat = predictions_np[:, :, i].reshape(-1)[start_idx:end_idx]
                    ground_truths_flat = ground_truths_np[:, :, i].reshape(-1)[start_idx:end_idx]
                    
                    ax.plot(ground_truths_flat, label='Ground Truth')
                    ax.plot(predictions_flat, label='Predictions')
                    ax.set_title(f'Samples {start_idx} to {end_idx}')
                    ax.set_xlabel('Sample')
                    ax.set_ylabel('Value')
                    ax.legend()
                
                # Save the figure
                visualization_dir = 'visualization'
                os.makedirs(visualization_dir, exist_ok=True)
                plt.savefig(os.path.join(visualization_dir, f'{labels[i].lower().replace(" ", "_")}_ground_truth_vs_predictions_scenario_{scenario_count}.png'))
                plt.close()
            
            scenario_count += 1

    avg_test_loss = test_loss / len(test_loader)
    print_with_timestamp(f"Test Loss: {avg_test_loss:.4f}")

# ----------------------------------------------
# 6) EVALUATE MODEL ON A BIOGEARS SCENARIO
#    (Segments for "cold" scenario, same as original)
#    This will do iterative multi-step inference:
#    see modeleval.py for the logic in evaluate_model_on_segments()
# ----------------------------------------------
print_with_timestamp("Evaluating the model on cold scenario from BioGears...")

# Create a DataFrame with the same columns as the original data
dummy_df = pd.DataFrame({
    'HeartRate(1/min)': [0] * 10,
    'CoreTemperature(degC)': [0] * 10,
    'SkinTemperature(degC)': [0] * 10,
    'intensity': [0.25] * 10,
    'atemp_c': [22.00] * 10,
    'rh_pct': [50.00] * 10,
})

# Transform the DataFrame using the scaler
scaled_dummy_df = pd.DataFrame(scaler.transform(dummy_df), columns=dummy_df.columns)

segments_cold = {  # Cold scenario
    'time': [1.00] * 10,
    'intensity': scaled_dummy_df['intensity'].tolist(),
    'atemp_c': scaled_dummy_df['atemp_c'].tolist(),
    'rh_pct': scaled_dummy_df['rh_pct'].tolist(),
}

segment_results = run_lstm(model, segments_cold, scaler=scaler)

print_with_timestamp("Model test run on cold scenario segments")
print(segment_results)

# Extract the predictions from segment_results
heart_rate_predictions = segment_results['HeartRate(1/min)']
core_temp_predictions = segment_results['CoreTemperature(degC)']
skin_temp_predictions = segment_results['SkinTemperature(degC)']

# Create a figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=300)
fig.suptitle('Predictions for Cold Scenario', fontsize=16)

# Plot heart rate predictions
axes[0].plot(heart_rate_predictions, label='Predicted Heart Rate')
axes[0].set_title('Heart Rate (1/min)')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Heart Rate')
axes[0].legend()

# Plot core temperature predictions
axes[1].plot(core_temp_predictions, label='Predicted Core Temperature')
axes[1].set_title('Core Temperature (degC)')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Core Temperature')
axes[1].legend()

# Plot skin temperature predictions
axes[2].plot(skin_temp_predictions, label='Predicted Skin Temperature')
axes[2].set_title('Skin Temperature (degC)')
axes[2].set_xlabel('Timestep')
axes[2].set_ylabel('Skin Temperature')
axes[2].legend()

# Save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(visualizations_dir, 'cold_scenario_predictions.png'))
plt.close()

print_with_timestamp("Program completed.")
