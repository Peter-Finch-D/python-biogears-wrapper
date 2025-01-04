import os
import glob
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime

from sklearn.model_selection import KFold # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

# Import your model evaluation on BioGears segments
from biogears_python.modeleval import evaluate_model_on_segments
from biogears_python.xmlscenario import segments_to_xml
from biogears_python.execution import run_biogears
from biogears_python.execution import run_lstm_sequence

import re  # For filename sanitization

def print_with_timestamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def sanitize_filename(label):
    """
    Sanitize the label to create a safe filename.
    Replaces spaces with underscores, '/' with '_per_', and removes other special characters.
    """
    label = label.lower().replace(" ", "_").replace("/", "_per_")
    label = re.sub(r'[^\w\-_.]', '', label)  # Remove any character that is not alphanumeric, '-', '_', or '.'
    return label

# ----------------------------------------------
#               HYPERPARAMETERS
# ----------------------------------------------
EPOCHS = 200  # Increased for potentially better training
SIMULATION_LENGTH = 10  # Each CSV file is assumed to be 10 timesteps
SEQ_LENGTH = 4  # Reduced from 8 to allow for valid sample creation
PRED_LENGTH = 5  # Number of timesteps to predict
BATCH_SIZE = 32  # Increased from 1 to 32 for stable training
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64  # Reduced from 128 to prevent overfitting
NUM_LAYERS = 2  # Reduced from 3 for simplicity
DROPOUT = 0.3  # Increased from 0.2 to 0.3 for better regularization
TEST_SPLIT = 0.2
PATIENCE = 10  # For Early Stopping

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

# Each file is presumably about 10 timesteps.
# Load them individually so each file is one short sequence.
all_data = []
for csv_path in all_csv_paths:
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Resample to 1-minute intervals
    df["Time(s)"] = pd.to_timedelta(df["Time(s)"])
    df.set_index("Time(s)", inplace=True)
    df = df.resample('1min').mean()

    assert len(df) == SIMULATION_LENGTH, f"Expected {SIMULATION_LENGTH} rows, got {len(df)}"

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

# ----------------------------------------------
#             2) SCALING
# ----------------------------------------------
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

# ----------------------------------------------
#             3) CREATE CUSTOM SEQ2SEQ DATASET
# ----------------------------------------------
class Seq2SeqPhysioDataset(Dataset):
    def __init__(self, list_of_dfs, feature_cols, target_cols, scaler_X, scaler_Y, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH):
        super().__init__()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.seq_length = seq_length
        self.pred_length = pred_length  # Number of timesteps to predict
        
        self.samples = []
        for df in list_of_dfs:
            if len(df) < self.seq_length + self.pred_length:
                continue  # Ensure enough timesteps for input and target
            
            # Scale input features
            X_scaled = self.scaler_X.transform(df[self.feature_cols])
            
            # Scale target features
            Y_scaled = self.scaler_Y.transform(df[self.target_cols])
            
            # Create sequences
            for i in range(len(df) - self.seq_length - self.pred_length + 1):
                X_seq = X_scaled[i:i + self.seq_length]
                Y_seq = Y_scaled[i + self.seq_length:i + self.seq_length + self.pred_length]
                self.samples.append((X_seq, Y_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X_seq, Y_seq = self.samples[idx]
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        Y_seq = torch.tensor(Y_seq, dtype=torch.float32)
        return X_seq, Y_seq

# Instantiate the Seq2Seq dataset
full_dataset = Seq2SeqPhysioDataset(
    all_data, feature_cols, target_cols, scaler_X, scaler_Y, 
    seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH
)

# Train/Test split using KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
indices = list(range(len(full_dataset)))
train_index, test_index = next(kf.split(indices))
train_dataset = torch.utils.data.Subset(full_dataset, train_index)
test_dataset = torch.utils.data.Subset(full_dataset, test_index)

# Check dataset sizes
print_with_timestamp(f"Total samples: {len(full_dataset)}")
print_with_timestamp(f"Training on {len(train_dataset)} samples. Test on {len(test_dataset)} samples.")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------------
#             4) DEFINE ENHANCED LSTM MODEL
# ----------------------------------------------
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Changed from mapping to 'hidden_size' to mapping to 1 for scalar weights
        self.attention = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size*2)
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size*2)
        return context_vector

class Seq2SeqEnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_length=5, num_layers=2, dropout=0.3):
        super(Seq2SeqEnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.bidirectional = True

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size * pred_length)  # Output multiple timesteps

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size*2)
        context = self.attention(lstm_out)  # (batch_size, hidden_size*2)
        out = self.fc(context)  # (batch_size, output_size * pred_length)
        out = out.view(-1, self.pred_length, len(target_cols))  # (batch_size, pred_length, output_size)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_with_timestamp(f"Using device: {device}")

# Instantiate the model
input_size = len(feature_cols)      # 6
output_size = len(target_cols)      # 3
model = Seq2SeqEnhancedLSTMModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=output_size,
    pred_length=PRED_LENGTH,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Define loss and optimizer with Weight Decay
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Define Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=1e-4, min_lr=1e-6
)

# ----------------------------------------------
#             5) TRAIN THE MODEL WITH EARLY STOPPING
# ----------------------------------------------
print_with_timestamp("Beginning LSTM Training...")
best_loss = np.inf
epochs_no_improve = 0
early_stop = False

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    model.train()
    
    for batch_X, batch_Y in train_loader:
        # batch_X: (B, T, 6)
        # batch_Y: (B, P, 3)
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(batch_X)  # shape: (B, P, 3)
        loss = criterion(predictions, batch_Y)
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print_with_timestamp(f"Epoch {epoch}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Step the scheduler
    scheduler.step(avg_loss)
    
    # Check for improvement
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), os.path.join(visualizations_dir, 'best_seq2seq_lstm_model.pth'))
        print_with_timestamp("Validation loss improved, saving model.")
    else:
        epochs_no_improve += 1
        print_with_timestamp(f"No improvement in validation loss for {epochs_no_improve} epochs.")
    
    if epochs_no_improve >= PATIENCE:
        print_with_timestamp("Early stopping triggered.")
        early_stop = True
        break

if not early_stop:
    # Save the final model if not saved by early stopping
    torch.save(model.state_dict(), os.path.join(visualizations_dir, 'final_seq2seq_lstm_model.pth'))
    print_with_timestamp("Training completed, model saved.")

# ----------------------------------------------
#             6) EVALUATE ON TEST SET
# ----------------------------------------------
if test_loader is not None:
    model.eval()
    test_loss = 0.0
    predictions_all = []
    ground_truths_all = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            predictions = model(batch_X)  # shape: (B, P, 3)
            loss = criterion(predictions, batch_Y)
            test_loss += loss.item()
            
            predictions_all.append(predictions.cpu().numpy())
            ground_truths_all.append(batch_Y.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print_with_timestamp(f"Test Loss: {avg_test_loss:.4f}")
    
    # Concatenate all predictions and ground truths
    predictions_np = np.concatenate(predictions_all, axis=0)  # (N, P, 3)
    ground_truths_np = np.concatenate(ground_truths_all, axis=0)  # (N, P, 3)
    
    # Reshape for evaluation
    predictions_np = predictions_np.reshape(-1, len(target_cols))  # (N*P, 3)
    ground_truths_np = ground_truths_np.reshape(-1, len(target_cols))  # (N*P, 3)
    
    # Inverse transform to original scale
    predictions_original = scaler_Y.inverse_transform(predictions_np)
    ground_truths_original = scaler_Y.inverse_transform(ground_truths_np)
    
    # Calculate MAE, RMSE, R² for Test Set
    mae_test_heart_rate = mean_absolute_error(ground_truths_original[:, 0], predictions_original[:, 0])
    mae_test_core_temp = mean_absolute_error(ground_truths_original[:, 1], predictions_original[:, 1])
    mae_test_skin_temp = mean_absolute_error(ground_truths_original[:, 2], predictions_original[:, 2])
    
    rmse_test_heart_rate = np.sqrt(mean_squared_error(ground_truths_original[:, 0], predictions_original[:, 0]))
    rmse_test_core_temp = np.sqrt(mean_squared_error(ground_truths_original[:, 1], predictions_original[:, 1]))
    rmse_test_skin_temp = np.sqrt(mean_squared_error(ground_truths_original[:, 2], predictions_original[:, 2]))
    
    r2_test_heart_rate = r2_score(ground_truths_original[:, 0], predictions_original[:, 0])
    r2_test_core_temp = r2_score(ground_truths_original[:, 1], predictions_original[:, 1])
    r2_test_skin_temp = r2_score(ground_truths_original[:, 2], predictions_original[:, 2])
    
    # Print Test Set Metrics
    print_with_timestamp(f"Test Set Mean Absolute Error (Heart Rate): {mae_test_heart_rate:.2f}")
    print_with_timestamp(f"Test Set Root Mean Squared Error (Heart Rate): {rmse_test_heart_rate:.2f}")
    print_with_timestamp(f"Test Set R² Score (Heart Rate): {r2_test_heart_rate:.2f}")
    
    print_with_timestamp(f"Test Set Mean Absolute Error (Core Temperature): {mae_test_core_temp:.2f}")
    print_with_timestamp(f"Test Set Root Mean Squared Error (Core Temperature): {rmse_test_core_temp:.2f}")
    print_with_timestamp(f"Test Set R² Score (Core Temperature): {r2_test_core_temp:.2f}")
    
    print_with_timestamp(f"Test Set Mean Absolute Error (Skin Temperature): {mae_test_skin_temp:.2f}")
    print_with_timestamp(f"Test Set Root Mean Squared Error (Skin Temperature): {rmse_test_skin_temp:.2f}")
    print_with_timestamp(f"Test Set R² Score (Skin Temperature): {r2_test_skin_temp:.2f}")
    
    # Visualization for Test Set
    labels = ['Heart Rate (1/min)', 'Core Temperature (degC)', 'Skin Temperature (degC)']
    
    for i, label in enumerate(labels):
        plt.figure(figsize=(10, 5))
        plt.plot(ground_truths_original[:, i], label='Ground Truth', marker='x')
        plt.plot(predictions_original[:, i], label='Predictions', marker='o')
        plt.fill_between(
            range(len(predictions_original[:, i])),
            predictions_original[:, i] - abs(predictions_original[:, i] - ground_truths_original[:, i]),
            predictions_original[:, i] + abs(predictions_original[:, i] - ground_truths_original[:, i]),
            color='gray', alpha=0.2, label='Error Range'
        )
        plt.title(f'{label} - Ground Truth vs Predictions (Test Set)')
        plt.xlabel('Sample')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Sanitize the label to create a safe filename
        safe_label = sanitize_filename(label)
        
        # Save the figure
        plt.savefig(os.path.join(visualizations_dir, f'{safe_label}_test_set_enhanced.png'))
        plt.close()
    
    # Residual Analysis for Test Set
    for i, label in enumerate(labels):
        residuals = ground_truths_original[:, i] - predictions_original[:, i]
        plt.figure(figsize=(10, 4))
        plt.plot(residuals, marker='o', linestyle='-', label='Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals for {label}')
        plt.xlabel('Sample')
        plt.ylabel('Residual')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(visualizations_dir, f'{sanitize_filename(label)}_residuals.png'))
        plt.close()


print_with_timestamp("Evaluating the model on cold scenario from BioGears...")

# Prepare segments for BioGears scenario
segments_cold = {  # Cold scenario
    'time': [1.00] * 10,
    'intensity': [0.25] * 10,
    'atemp_c': [22.00] * 10,
    'rh_pct': [80.00] * 10,
}

# Run LSTM predictions
lstm_results = run_lstm_sequence(
    model, 
    segments_cold, 
    scaler_X, 
    scaler_Y, 
    seq_length=SEQ_LENGTH, 
    initial_state=(0.0, 0.0, 0.0),  # Adjust as needed
    device=device
)

print_with_timestamp("Model test run on cold scenario segments")
print("LENGTH OF LSTM RESULTS: ", len(lstm_results))
print(lstm_results)

xml_scenario = segments_to_xml(segments_cold)
biogears_results = run_biogears(xml_scenario, segments_cold)

print_with_timestamp("Using BioGears for ground truth")
print("LENGTH OF BIOGEARS RESULTS: ", len(biogears_results))
print(biogears_results)

# Extract the predictions from lstm_results
heart_rate_predictions = lstm_results['HeartRate(1/min)'].values
core_temp_predictions = lstm_results['CoreTemperature(degC)'].values
skin_temp_predictions = lstm_results['SkinTemperature(degC)'].values

# Extract the ground truth from biogears_results
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

# Create a figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=300)
fig.suptitle('Predictions vs Ground Truth for Cold Scenario', fontsize=16)

# Plot heart rate predictions and ground truth
axes[0].plot(heart_rate_predictions, label='Predicted Heart Rate')
axes[0].plot(heart_rate_ground_truth, label='Ground Truth Heart Rate')
axes[0].set_title('Heart Rate (1/min)')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Heart Rate')
axes[0].legend()

# Plot core temperature predictions and ground truth
axes[1].plot(core_temp_predictions, label='Predicted Core Temperature')
axes[1].plot(core_temp_ground_truth, label='Ground Truth Core Temperature')
axes[1].set_title('Core Temperature (degC)')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Core Temperature')
axes[1].legend()

# Plot skin temperature predictions and ground truth
axes[2].plot(skin_temp_predictions, label='Predicted Skin Temperature')
axes[2].plot(skin_temp_ground_truth, label='Ground Truth Skin Temperature')
axes[2].set_title('Skin Temperature (degC)')
axes[2].set_xlabel('Timestep')
axes[2].set_ylabel('Skin Temperature')
axes[2].legend()

# Save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(visualizations_dir, 'cold_scenario_predictions_vs_ground_truth.png'))
plt.close()

print_with_timestamp("Program completed.")