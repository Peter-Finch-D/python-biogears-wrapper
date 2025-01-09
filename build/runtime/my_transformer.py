import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


###############################################################################
# 1. Positional Encoding
###############################################################################
class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for a Transformer.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (max_len, d_model)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return x


###############################################################################
# 2. Combined Transformer + Regressor
###############################################################################
class TransformerRegressor(nn.Module):
    """
    A single model that:
      - Projects input features to d_model (optionally auto-calculated)
      - Adds positional encoding
      - Passes through a multi-layer Transformer encoder
      - Applies a feed-forward "head" to produce a single regression output

    Also includes a 'train_model' method with a 'seq_length' argument
    that reshapes your DataFrame into (num_simulations, seq_length, input_dim).
    """
    def __init__(
        self,
        feature_cols=None,  # so we can auto-calculate input_dim & optional d_model
        target_cols=None,   # not strictly needed for d_model, but can be used if you like
        d_model=None,       # if None, we'll compute from input_dim
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        hidden_dims=[64],
        outputs_dir="outputs/models"
    ):
        """
        If d_model is None, we'll auto-calculate it as `input_dim * 4` or something similar.
        We also derive input_dim from len(feature_cols).
        """
        super().__init__()
        
        # Make sure we have feature_cols
        if feature_cols is None or len(feature_cols) == 0:
            raise ValueError("Please provide feature_cols so we can determine input_dim.")

        self.feature_cols = feature_cols
        self.target_cols  = target_cols or []

        input_dim = len(self.feature_cols)
        if d_model is None:
            # Example: pick a rule of thumb for d_model
            d_model = input_dim * 4
            print(f"[INFO] d_model not provided; using d_model={d_model} (input_dim={input_dim}*4).")

        self.outputs_dir = outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

        # 1) Linear projection of input_dim -> d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2) Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch_size, seq_len, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) A feed-forward "head" to go from d_model -> single value
        layers = []
        in_dim = d_model
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hd
        
        layers.append(nn.Linear(in_dim, 1))  # final regression output
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        Returns: a single scalar per example in the batch (shape: (batch_size,))
        """
        # 1) Project input features
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # 2) Positional encoding
        x = self.pos_encoder(x)       # (batch_size, seq_len, d_model)

        # 3) Transformer
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # 4) Take the final time step (example) or you could average
        last_step = encoded[:, -1, :]  # (batch_size, d_model)

        # 5) Regressor head
        out = self.regressor(last_step)  # (batch_size, 1)
        return out.squeeze(dim=-1)       # (batch_size,)

    ###########################################################################
    # 3. Internal training method (with seq_length argument)
    ###########################################################################
    def train_model(
        self,
        df,
        seq_length=19,      # <--- NEW: adjustable sequence length
        epochs=10,
        learning_rate=1e-3,
        test_split=0.2,
        shuffle=False
    ):
        """
        Train this TransformerRegressor on the provided df.

        We'll:
          1) Derive data_x_tensor from df[feature_cols].
          2) Reshape into (num_sims, seq_length, input_dim).
          3) Derive target from df[target_cols], also reshaped to (num_sims, seq_length).
          4) Use the final time step as the label for each simulation.
        """
        if len(self.feature_cols) == 0:
            raise ValueError("feature_cols is empty or None. Please set it in the constructor.")
        if len(self.target_cols) == 0:
            raise ValueError("No target_cols provided. Please pass them in the constructor.")

        target_col = self.target_cols[0]  # assume single target

        # Make sure total rows in df is divisible by seq_length
        total_rows = df.shape[0]
        if total_rows % seq_length != 0:
            raise ValueError(
                f"DataFrame has {total_rows} rows, not divisible by seq_length={seq_length}. "
                "Adjust seq_length or your data."
            )

        # 1) Reshape features -> (num_sims, seq_length, input_dim)
        input_dim = len(self.feature_cols)
        num_sims = total_rows // seq_length

        data_x = df[self.feature_cols].values  # shape: (total_rows, input_dim)
        data_x = data_x.reshape(num_sims, seq_length, input_dim)
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32)

        # 2) Reshape target -> (num_sims, seq_length), then take final step as label
        target_array = df[target_col].values.reshape(num_sims, seq_length)
        y_final = target_array[:, -1]  # shape: (num_sims,)
        y_tensor = torch.tensor(y_final, dtype=torch.float32)

        # 3) Train/test split
        indices = np.arange(num_sims)
        if shuffle:
            np.random.shuffle(indices)
        train_size = int((1 - test_split) * num_sims)
        train_idx = indices[:train_size]
        test_idx  = indices[train_size:]

        x_train = data_x_tensor[train_idx]  # shape: (train_size, seq_length, input_dim)
        y_train = y_tensor[train_idx]       # shape: (train_size,)
        x_test  = data_x_tensor[test_idx]
        y_test  = y_tensor[test_idx]

        print("\n--- Starting TransformerRegressor Training ---")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"Input shape: {x_train.shape}, Target shape: {y_train.shape}")

        # 4) Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # 5) Training loop
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(x_train)  # forward => shape: (train_size,)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        # 6) Evaluate on test set
        self.eval()
        with torch.no_grad():
            y_pred_test = self(x_test)
            test_loss = criterion(y_pred_test, y_test)
        print(f"\nTest MSE: {test_loss.item():.6f}")

        # 7) Save the trained model
        model_save_path = os.path.join(self.outputs_dir, "combined_transformer_regressor.pt")
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
        print("--- Training Complete ---\n")

    ###########################################################################
    # 4. Evaluate model with a "recurrent" approach
    ###########################################################################
    def evaluate_model(
        self,
        initial_state,
        df,
        scaler_X,
        scaler_Y,
        # Column names in df
        time_col='Time(s)',
        skin_temp_col='SkinTemperature(degC)',
        intensity_col='intensity',
        atemp_col='atemp_c',
        rh_col='rh_pct',
        # Model interpretation
        predict_delta=True,
        # Output/visualization
        outputs_dir='outputs',
        visualizations_dir='visualizations',
        plot_results=True
    ):
        """
        Recurrently predict skin temperature from an initial state, using this
        TransformerRegressor, applying scaler_X/scaler_Y to ensure consistent
        input/output scaling as in training.

        Parameters
        ----------
        initial_state : tuple or list
            The starting state in *unscaled real units*, e.g.:
            (time_delta, skin_temperature, intensity, atemp_c, rh_pct).
            This must match the order of columns used in feature_cols.
        df : pd.DataFrame
            DataFrame containing time steps and target columns to compare against.
            Must include skin_temp_col, intensity_col, etc.
        scaler_X : sklearn scaler (for features)
            The same scaler used to transform the input features (time_delta, skin_temp, etc.)
            during training.
        scaler_Y : sklearn scaler (for the target)
            The same scaler used to transform the target (e.g. the difference in skin temp)
            during training.
        time_col : str
            Column name for time in df (default 'Time(s)').
        skin_temp_col : str
            Column name for the ground-truth skin temperature in df (default 'SkinTemperature(degC)').
        intensity_col, atemp_col, rh_col : str
            Columns for intensity, ambient temp, humidity.
        predict_delta : bool
            If True, the model’s output is interpreted as a scaled *delta* in skin temp.
            If False, it is interpreted as a scaled *absolute temperature*.
        outputs_dir : str
            Directory for saving results.
        visualizations_dir : str
            Directory for saving plots.
        plot_results : bool
            If True, will plot a line chart comparing predictions vs. ground truth.

        Returns
        -------
        preds : list
            List of predicted skin temperatures in real units (one per row in df).
        mae : float
            Mean Absolute Error comparing predictions vs. df[skin_temp_col].
        """
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_absolute_error

        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)

        print("\n--- Evaluating (step-by-step) with TransformerRegressor (scaled) ---")
        
        # 1. Sort the DataFrame by time if necessary
        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        ground_truth = df_sorted[skin_temp_col].values  # in real (unscaled) units

        # 2. Keep an *unscaled* 'state' that we update each iteration
        #    initial_state is in real units, e.g. (time_delta=1, skin_temp=33.0, ...).
        state_unscaled = list(initial_state)  # make it mutable
        preds = []

        # For each row in df, do a step
        for i in range(len(df_sorted)):
            # 2a) We need to scale the current 'state_unscaled' so the model sees
            #     the same scale it saw in training.

            # shape: (1, 5) => single example with 5 features
            # Be sure the columns match the order in feature_cols during training.
            # For example, if feature_cols = ['time_delta', 'SkinTemperature(degC)', 'intensity', 'atemp_c', 'rh_pct']
            # then state_unscaled must follow the same order.
            state_array = np.array([state_unscaled], dtype=np.float32)  # shape (1,5)

            # Scale the input features
            state_scaled = scaler_X.transform(state_array)  # still shape (1,5)

            # Reshape to (batch_size=1, seq_len=1, num_features=5) for Transformer
            state_scaled_3d = state_scaled.reshape(1, 1, -1)  # (1,1,5)

            # 2b) Model forward pass in scaled space
            self.eval()
            with torch.no_grad():
                # Convert to torch tensor
                state_tensor = torch.tensor(state_scaled_3d, dtype=torch.float32)
                model_output_scaled = self(state_tensor).item()  # single float in *scaled target* space

            # 2c) Inverse-transform the model output if the target was scaled
            # The model's prediction is shape (1,1) for scaler_Y, so wrap it:
            model_output_scaled_2d = np.array([[model_output_scaled]], dtype=np.float32)  # (1,1)
            model_output_unscaled_2d = scaler_Y.inverse_transform(model_output_scaled_2d) # also shape (1,1)
            model_output_unscaled = model_output_unscaled_2d[0,0]  # single float in real units

            # 2d) Interpret the model output as delta or absolute temperature
            if predict_delta:
                # If the model predicts a *delta* in real units:
                new_skin_temp = state_unscaled[1] + model_output_unscaled
            else:
                # If the model predicts an *absolute* temperature in real units:
                new_skin_temp = model_output_unscaled

            # 2e) Build the next unscaled state
            # For time_delta, we keep incrementing by +1 in real space
            # (though in training it was presumably also scaled).
            new_time_delta = state_unscaled[0] + 1.0

            # Pull the next row's intensity, atemp, rh in real units from the df
            new_intensity = df_sorted[intensity_col].iloc[i]
            new_atemp     = df_sorted[atemp_col].iloc[i]
            new_rh        = df_sorted[rh_col].iloc[i]

            # So the new unscaled state is (time, skin_temp, intensity, atemp, rh)
            state_unscaled = [
                new_time_delta,
                new_skin_temp,
                new_intensity,
                new_atemp,
                new_rh
            ]

            # 2f) Store the predicted *unscaled* skin temperature
            preds.append(new_skin_temp)

        # 3. Compare predictions to the unscaled ground truth
        mae = mean_absolute_error(ground_truth, preds)
        print(f"\nPredictions complete. MAE = {mae:.4f}")

        # 4. Plot
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(ground_truth, label='BioGears Skin Temp (Ground Truth)', color='blue')
            plt.plot(preds, label='Predicted Skin Temp', color='red', linestyle='--')
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
