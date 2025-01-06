import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import multiprocessing  # For determining CPU count

from digital_twin.dataset import Seq2SeqPhysioDataset
from digital_twin.models import Seq2SeqEnhancedLSTMModel, CumulativeLoss
from digital_twin.utils import print_with_timestamp

def train_model(
    processed_csv_path,
    scalers_dir,
    models_dir,
    feature_cols,
    target_cols,
    epochs=200,
    batch_size=64,  # Increased batch size for better CPU utilization
    learning_rate=1e-3,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    patience=10,
    num_workers=None  # Added num_workers parameter
):
    """
    Trains the Seq2Seq LSTM model using the processed data.
    
    Parameters:
    - processed_csv_path: Path to the processed CSV file.
    - scalers_dir: Directory where scalers are stored.
    - models_dir: Directory to save trained models.
    - epochs: Number of training epochs.
    - batch_size: Training batch size.
    - learning_rate: Learning rate for the optimizer.
    - hidden_size: Number of hidden units in LSTM.
    - num_layers: Number of LSTM layers.
    - dropout: Dropout rate.
    - patience: Patience for early stopping.
    - num_workers: Number of subprocesses for data loading. Defaults to number of CPU cores.
    """
    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    print_with_timestamp(f"Number of workers set to: {num_workers}")
    
    # Load processed data
    df = pd.read_csv(processed_csv_path)
    
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
    
    # DataLoaders with dynamic num_workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True  # Useful if using GPU; can be set to False if not
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_with_timestamp(f"Using device: {device}")
    
    # Set number of threads for PyTorch to utilize all CPU cores
    torch.set_num_threads(num_workers)
    torch.set_num_interop_threads(num_workers)
    print_with_timestamp(f"Set PyTorch to use {num_workers} threads.")
    
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
    
    # Define custom cumulative loss function and optimizer with Weight Decay
    criterion = CumulativeLoss(step_loss_weight=0.5, cumulative_loss_weight=0.5)
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
            
            # Extract initial state from the last time step of the input sequence
            initial_state = batch_X[:, -1, :output_size]
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y, initial_state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print_with_timestamp(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Update the LR scheduler
        scheduler.step(avg_loss)
        
    # Save the final model after all epochs
    final_model_path = os.path.join(models_dir, 'final_seq2seq_lstm_model.pth')
    torch.save(model.state_dict(), final_model_path)

    print_with_timestamp("Training completed, final model saved.")
