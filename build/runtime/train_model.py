############################################
# train_model.py
############################################

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def train_transformer_nn(
    encoded_data_path: str, 
    processed_csv_path: str,
    feature_cols: list,
    target_cols: list,
    outputs_dir: str = "outputs",
    epochs: int = 5,
    learning_rate: float = 1e-3,
):
    """
    Loads the Transformer-encoded data from a file and trains a simple
    feed-forward neural network on the final time step of each encoded sequence.

    Arguments:
    ----------
    encoded_data_path: str
        The path to the PyTorch file containing the encoded data 
        (e.g., 'outputs/transformer_encoded_data.pt').
    processed_csv_path: str
        The path to the CSV file containing the processed dataset 
        (e.g., 'outputs/processed_data.csv').
    feature_cols: list
        The feature columns used in Main.py.
    target_cols: list
        The target columns used in Main.py.
    outputs_dir: str
        A directory to save model outputs (default: "outputs").
    epochs: int
        Number of training epochs (default: 5).
    learning_rate: float
        The learning rate for the optimizer (default: 1e-3).
    """

    # -------------------------------------------------------------------------
    # 1. Load the Transformer-encoded data
    #    Shape is typically (num_simulations, seq_length, d_model).
    # -------------------------------------------------------------------------
    encoded_data = torch.load(encoded_data_path)
    print(f"\nLoaded encoded data from: {encoded_data_path}")
    print("encoded_data.shape =", encoded_data.shape)

    # Determine shape info
    num_simulations, seq_length, d_model = encoded_data.shape

    # -------------------------------------------------------------------------
    # 2. Load your processed CSV to obtain the target values
    #    We assume your target columns are in this CSV.
    # -------------------------------------------------------------------------
    df = pd.read_csv(processed_csv_path)
    print(f"\nLoaded processed DataFrame from: {processed_csv_path}")
    print("df.shape =", df.shape)

    # Verify we have the correct total rows in df
    expected_rows = num_simulations * seq_length
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows (num_simulations * seq_length), "
            f"but got {df.shape[0]} in {processed_csv_path}."
        )

    # -------------------------------------------------------------------------
    # 3. Build the target tensor
    #    - We'll show an example for a single target column, taking the final
    #      time step from each simulation. 
    #    - If you have multiple targets, you'd adjust here.
    # -------------------------------------------------------------------------
    if len(target_cols) == 0:
        raise ValueError("No target columns specified. Please set target_cols in Main.py.")

    # For simplicity, let's assume there's a single target column:
    target_col = target_cols[0]

    # Reshape the target column to match (num_simulations, seq_length)
    target_array = df[target_col].values.reshape(num_simulations, seq_length)

    # Example: we'll pick the final time step in each simulation
    targets_final_step = target_array[:, -1]  # shape: (num_simulations,)

    target_tensor = torch.tensor(targets_final_step, dtype=torch.float32)
    print(f"\nTarget tensor shape (final step only): {target_tensor.shape}")

    # -------------------------------------------------------------------------
    # 4. Decide which part of your encoded data to feed into the NN
    #    - Option: take the last time step's encoding: shape -> (num_simulations, d_model)
    # -------------------------------------------------------------------------
    encoded_last_step = encoded_data[:, -1, :]  # shape: (num_simulations, d_model)
    print(f"Encoded data (final step) shape: {encoded_last_step.shape}")

    # -------------------------------------------------------------------------
    # 5. Prepare train/test split 
    #    (Here, a simple 80/20 split, no shuffling for demonstration.)
    # -------------------------------------------------------------------------
    train_size = int(0.8 * num_simulations)
    test_size = num_simulations - train_size

    x_train = encoded_last_step[:train_size]
    x_test  = encoded_last_step[train_size:]
    y_train = target_tensor[:train_size]
    y_test  = target_tensor[train_size:]

    print("\nSplit data:")
    print(f"Train size: {train_size}, Test size: {test_size}")

    # -------------------------------------------------------------------------
    # 6. Define a simple feed-forward neural network
    # -------------------------------------------------------------------------
    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super(SimpleRegressor, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze()  # returns shape (batch,)

    model = SimpleRegressor(input_dim=d_model, hidden_dim=64)
    print("\nCreated SimpleRegressor model:")
    print(model)

    # -------------------------------------------------------------------------
    # 7. Define loss function and optimizer
    # -------------------------------------------------------------------------
    criterion = nn.MSELoss()  # Example for a regression task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -------------------------------------------------------------------------
    # 8. Training loop
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    model.train()
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print("Training complete.\n")

    # -------------------------------------------------------------------------
    # 9. Evaluation on the test set
    # -------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        test_loss = criterion(y_pred_test, y_test)
    print(f"Test MSE: {test_loss.item():.6f}")

    # -------------------------------------------------------------------------
    # 10. (Optional) Save the trained model
    # -------------------------------------------------------------------------
    model_path = os.path.join(outputs_dir, "simple_regressor.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}\n")

    print("--- Finished training on Transformer-encoded data ---")
