import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def predict_skin_temperature_recurrently(
    initial_state,
    df,
    model_path,
    # Default column names (customize as needed):
    time_col='Time(s)',
    skin_temp_col='SkinTemperature(degC)',
    intensity_col='intensity',
    atemp_col='atemp_c',
    rh_col='rh_pct',
    # Indicate if your model predicts a *delta* (difference) or a direct temperature:
    predict_delta=True,
    # Visual/Output arguments:
    outputs_dir='outputs',
    visualizations_dir='visualizations',
    plot_results=True
):
    """
    Recurrently predict skin temperature from an initial state and a PyTorch model.
    We assume the model was trained to predict either:
      - The *difference* in skin temperature (predict_delta=True), OR
      - The direct skin temperature (predict_delta=False).
    
    Parameters
    ----------
    initial_state : tuple or list
        The starting state. For example: (time_delta, skin_temperature, intensity, atemp, rh).
    df : pd.DataFrame
        The DataFrame containing time steps and target columns to compare against.
        Expected columns: time_col, skin_temp_col, intensity_col, atemp_col, rh_col.
    model_path : str
        Path to the serialized PyTorch model (e.g. 'outputs/models/simple_regressor.pt').
    time_col : str
        Name of the time column in df (default 'Time(s)').
    skin_temp_col : str
        Name of the target/skin temperature column in df (default 'SkinTemperature(degC)').
    intensity_col : str
        Name of the intensity column in df (default 'intensity').
    atemp_col : str
        Name of the ambient temperature column in df (default 'atemp_c').
    rh_col : str
        Name of the relative humidity column in df (default 'rh_pct').
    predict_delta : bool
        If True, the model output is interpreted as a 'delta' to add to the current skin temp.
        If False, the model output is taken as the predicted skin temperature directly.
    outputs_dir : str
        Directory where results can be saved.
    visualizations_dir : str
        Directory where to save the plot.
    plot_results : bool
        If True, a matplotlib plot comparing ground truth vs. predictions will be generated.
    
    Returns
    -------
    preds : list
        List of predicted skin temperature values (one per row in df).
    mae : float
        Mean Absolute Error comparing to df[skin_temp_col].
    """
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # ------------------- 1. Load the PyTorch model ---------------------
    # We'll assume the architecture matches what you trained. For example:
    #   input_dim=5 -> (time_delta, SkinTemp, intensity, atemp, rh)
    # You may need to replicate the same class definition used when training.
    #
    # Example advanced regressor (you could adapt to your own):
    class MoreAdvancedRegressor(nn.Module):
        def __init__(self, input_dim=5, hidden_dims=[128, 64, 32], dropout=0.2):
            super(MoreAdvancedRegressor, self).__init__()
            layers = []
            in_dim = input_dim
            
            for hd in hidden_dims:
                layers.append(nn.Linear(in_dim, hd))
                layers.append(nn.BatchNorm1d(hd))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hd
            
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            # x shape: (batch_size, input_dim)
            return self.net(x).squeeze()  # shape: (batch_size,)

    # Instantiate the same architecture:
    model = MoreAdvancedRegressor(input_dim=5)  # or whatever your input_dim is
    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to evaluation mode

    print(f"\nModel loaded from {model_path}.")

    # ------------------- 2. Prepare for recurrent prediction -------------------
    # initial_state is something like (time_delta, skin_temp, intensity, atemp, rh)
    state = initial_state

    # We'll store predictions as we go
    preds = []

    # For easy iteration, let's ensure df is sorted by time if needed
    # (In many scenarios, it might already be sorted, but let's be safe.)
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

    # ------------------- 3. Recurrent loop -------------------
    # We'll loop row by row in the df, each row representing the next time step
    for i in range(len(df_sorted)):
        # Convert 'state' -> torch tensor with shape (1, 5)
        # state: (time_delta, skin_temp, intensity, atemp, rh)
        state_tensor = torch.tensor([state], dtype=torch.float32)  # shape: (1, 5)

        # Predict delta or direct temperature
        with torch.no_grad():
            pred = model(state_tensor).item()

        if predict_delta:
            # If the model predicts the delta (change in skin temp),
            # we add it to the current skin temp (index=1 of the state).
            new_skin_temp = state[1] + pred
        else:
            # If the model predicts the direct temperature,
            # we take pred as the new skin temp
            new_skin_temp = pred

        # The next "time" step. Here we do a simple +1 second or +1 minute,
        # or use the DF's time increments. Let's assume we just increment the time by 1
        # from the example code. You can adapt if you want to match real timestamps.
        new_time_delta = state[0] + 1

        # Collect values from df for the next step
        #   intensity, atemp, rh, etc. from the row i
        new_intensity = df_sorted[intensity_col].iloc[i]
        new_atemp     = df_sorted[atemp_col].iloc[i]
        new_rh        = df_sorted[rh_col].iloc[i]

        # Construct the new state
        # The order is (time_delta, skin_temp, intensity, atemp, rh)
        state = (new_time_delta, new_skin_temp, new_intensity, new_atemp, new_rh)

        # Store the predicted skin temperature (not delta!)
        preds.append(new_skin_temp)

    # ------------------- 4. Compare predictions to ground truth -------------------
    # The ground truth is df_sorted[skin_temp_col].
    # We'll assume the length of df_sorted equals the number of time steps we predicted.
    # If your final data has different length or alignment, adapt accordingly.
    ground_truth = df_sorted[skin_temp_col].values

    # Compute MAE
    mae = mean_absolute_error(ground_truth, preds)

    print(f"\nPredictions complete. Computed MAE = {mae:.4f}")

    # ------------------- 5. Optional: Plot results -------------------
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(ground_truth, label='BioGears Skin Temperature (Ground Truth)', color='blue')
        plt.plot(preds, label='Predicted Skin Temperature', color='red', linestyle='--')
        plt.xlabel('Time Step Index')
        plt.ylabel('SkinTemperature(degC)')
        plt.title('BioGears vs Predicted Skin Temperature')
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(visualizations_dir, 'skin_temperature_comparison.png')
        plt.savefig(out_path)
        print(f"Plot saved to: {out_path}")
        plt.show()

    return preds, mae
