# digital_twin/train_model.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load

from digital_twin.dataset import Seq2SeqPhysioDataset
from digital_twin.models import Seq2SeqEnhancedLSTMModel
from digital_twin.utils import print_with_timestamp


def train_model(
    processed_csv_path,
    scalers_dir,
    models_dir,
    epochs=200,
    batch_size=32,
    learning_rate=1e-3,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    patience=10
):
    """
    Trains the Seq2Seq LSTM model using the processed data.
    
    Parameters:
    - processed_csv_path: Path to the processed CSV file.
    - scalers_dir: Directory where scalers are stored.
    - models_dir: Directory to save trained models.
    - visualizations_dir: Directory for saving visualizations (unused in training).
    - epochs: Number of training epochs.
    - batch_size: Training batch size.
    - learning_rate: Learning rate for the optimizer.
    - hidden_size: Number of hidden units in LSTM.
    - num_layers: Number of LSTM layers.
    - dropout: Dropout rate.
    - patience: Patience for early stopping.
    """
    # Load processed data
    df = pd.read_csv(processed_csv_path)
    
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
    
    # Load scalers
    scaler_X = load(os.path.join(scalers_dir, 'scaler_X.joblib'))
    scaler_Y = load(os.path.join(scalers_dir, 'scaler_Y.joblib'))
    
    # Create dataset
    dataset = Seq2SeqPhysioDataset(
        dataframe=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        seq_length=4,
        pred_length=5
    )
    
    # Train/Test split using KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))
    train_index, test_index = next(kf.split(indices))
    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)
    
    print_with_timestamp(f"Total samples: {len(dataset)}")
    print_with_timestamp(f"Training on {len(train_dataset)} samples. Test on {len(test_dataset)} samples.")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_with_timestamp(f"Using device: {device}")
    
    # Define model
    input_size = len(feature_cols)      # 6
    output_size = len(target_cols)      # 3
    model = Seq2SeqEnhancedLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        pred_length=5,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Define loss and optimizer with Weight Decay
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Define Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=1e-4, min_lr=1e-6
    )
    
    # Early Stopping variables
    best_loss = np.inf
    epochs_no_improve = 0
    early_stop = False
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    print_with_timestamp("Beginning LSTM Training...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)  # (B, P, 3)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print_with_timestamp(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Save the best model
            best_model_path = os.path.join(models_dir, 'best_seq2seq_lstm_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print_with_timestamp("Validation loss improved, saving model.")
        else:
            epochs_no_improve += 1
            print_with_timestamp(f"No improvement in validation loss for {epochs_no_improve} epochs.")
        
        # Early Stopping
        if epochs_no_improve >= patience:
            print_with_timestamp("Early stopping triggered.")
            early_stop = True
            break
    
    if not early_stop:
        # Save the final model if not saved by early stopping
        final_model_path = os.path.join(models_dir, 'final_seq2seq_lstm_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print_with_timestamp("Training completed, final model saved.")
