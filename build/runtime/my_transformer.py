import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
plt.style.use('dark_background')  # Dark mode to be cool 😎

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
    def __init__(
        self,
        feature_cols=None,  # so we can auto-calculate input_dim & optional d_model
        target_cols=None,   # needed to know how many outputs we produce
        d_model=None,       # if None, we'll compute from input_dim
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        hidden_dims=[64],
        outputs_dir="outputs/models"
    ):
        """
        If d_model is None, we'll auto-calculate it as input_dim * 4 or something similar.
        We also derive input_dim from len(feature_cols).
        The final layer dimension is len(target_cols).
        """
        super().__init__()
        
        # Make sure we have feature_cols
        if not feature_cols:
            raise ValueError("Please provide feature_cols so we can determine input_dim.")
        if not target_cols:
            raise ValueError("Please provide target_cols so we can determine output_dim.")

        self.feature_cols = feature_cols
        self.target_cols  = target_cols

        input_dim = len(self.feature_cols)
        output_dim = len(self.target_cols)  # number of targets we want to predict

        # If d_model is None, pick a simple rule-of-thumb
        if d_model is None:
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

        # 4) A feed-forward "head" to go from d_model -> output_dim (multi-target)
        layers = []
        in_dim = d_model
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hd
        
        layers.append(nn.Linear(in_dim, output_dim))  # final regression output (multi-target)
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        Returns: shape (batch_size, output_dim)
        """
        # 1) Project input features
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # 2) Positional encoding
        x = self.pos_encoder(x)       # (batch_size, seq_len, d_model)

        # 3) Transformer
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # 4) Take the final time step
        last_step = encoded[:, -1, :]  # (batch_size, d_model)

        # 5) Regressor => (batch_size, output_dim)
        out = self.regressor(last_step)
        return out  # shape: (batch_size, len(target_cols))

    ###########################################################################
    # 3. Internal training method (with seq_length argument)
    ###########################################################################
    def train_model(
        self,
        df,
        seq_length=19,      # adjustable sequence length
        epochs=10,
        learning_rate=1e-3,
        test_split=0.2,
        shuffle=False,
        num_workers=1       # <--- New argument controlling CPU threads for training
    ):
        """
        Train this TransformerRegressor on the provided df, with multiple targets,
        using multi-threaded (intra-op) CPU parallelism.

        Steps:
         1) Reshape 'df[feature_cols]' -> (num_sims, seq_length, input_dim).
         2) Reshape 'df[target_cols]' -> (num_sims, seq_length, num_targets).
         3) Take final time-step => shape (num_sims, num_targets).
         4) Optionally set the number of CPU threads used by PyTorch.
        """
        if len(self.feature_cols) == 0:
            raise ValueError("feature_cols is empty or None. Please set it in the constructor.")
        if len(self.target_cols) == 0:
            raise ValueError("No target_cols provided. Please set them in the constructor.")

        # --- Set number of CPU threads for MKL/OMP operations ---
        # This typically controls how many CPU cores/threads are used
        # for CPU-bound operations (e.g. matrix multiplication).
        # If you have a multi-core system, you can try setting num_workers to e.g. 4 or 8.
        torch.set_num_threads(num_workers)
        torch.set_num_interop_threads(num_workers)
        # This won't spawn multiple processes for training; it just allows
        # more CPU threads to be used in parallel for linear algebra.

        # Make sure total_rows is divisible by seq_length
        total_rows = df.shape[0]
        if total_rows % seq_length != 0:
            raise ValueError(
                f"DataFrame has {total_rows} rows, which isn't divisible by seq_length={seq_length}. "
                "Adjust seq_length or your data so it divides evenly."
            )

        # 1) Reshape features
        input_dim = len(self.feature_cols)
        num_targets = len(self.target_cols)
        num_sims = total_rows // seq_length

        data_x = df[self.feature_cols].values  # shape: (total_rows, input_dim)
        data_x = data_x.reshape(num_sims, seq_length, input_dim)
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32)

        # 2) Reshape targets
        data_y = df[self.target_cols].values  # shape: (total_rows, num_targets)
        data_y = data_y.reshape(num_sims, seq_length, num_targets)
        y_final = data_y[:, -1, :]  # final step => shape (num_sims, num_targets)
        y_tensor = torch.tensor(y_final, dtype=torch.float32)

        # 3) Train/test split
        indices = np.arange(num_sims)
        if shuffle:
            np.random.shuffle(indices)

        train_size = int((1 - test_split) * num_sims)
        train_idx = indices[:train_size]
        test_idx  = indices[train_size:]

        x_train = data_x_tensor[train_idx]  # (train_size, seq_length, input_dim)
        y_train = y_tensor[train_idx]       # (train_size, num_targets)
        x_test  = data_x_tensor[test_idx]
        y_test  = y_tensor[test_idx]

        print("\n--- Starting TransformerRegressor Training (Multi-Target) ---")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"Input shape: {x_train.shape}, Target shape: {y_train.shape}")
        print(f"Using up to {num_workers} CPU threads for training.\n")

        # 4) Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # 5) Training loop
        self.train()  # set model to training mode
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(x_train)  # shape: (train_size, num_targets)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        # 6) Evaluate on test set
        self.eval()  # set model to eval mode
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
        # The columns in df that are being predicted (multiple!)
        target_cols=None,
        figure_ranges=None,
        # If your input has time_col, intensity_col, etc. to be updated each step:
        time_col='Time(s)',
        # If you want to pass in columns that update each step for the rest of the state:
        extra_feature_cols=None,  # e.g. ['intensity', 'atemp_c', 'rh_pct']
        # Indicate whether each target is delta or absolute:
        predict_delta_mask=None,
        # Output/visualization
        outputs_dir='outputs',
        visualizations_dir='visualizations',
        plot_results=True
    ):
        """
        Step-by-step evaluation for multi-target outputs.

        Arguments
        ---------
        initial_state : list or array
            The starting state in *unscaled* real units, matching self.feature_cols in order.
            e.g. if self.feature_cols = ['time_delta','SkinTemperature(degC)','intensity','atemp_c','rh_pct']
            then initial_state might be [0, 33.0, 0.25, 30.0, 30.0].
        df : pd.DataFrame
            DataFrame containing the features that might get updated each step
            (like time, intensity, etc.) and the ground-truth columns for multiple targets.
        scaler_X : sklearn scaler
            For scaling the input features (the same used during training).
        scaler_Y : sklearn scaler
            For scaling the multiple targets. Must handle shape (N, n_targets).
        target_cols : list of str
            The columns in df that the model is predicting. E.g. ['SkinTemperature(degC)_diff', 'HeartRate(1/min)_diff'].
        time_col : str
            Name of the time column in df (if any). Used if you sort or track time step by step.
        extra_feature_cols : list of str
            Columns you read from df each step to update the input state (excluding the target columns).
            For instance, ['intensity','atemp_c','rh_pct'] if those change each row.
        predict_delta_mask : list of bool
            If you have multiple targets, you might specify for each target whether it’s a delta or absolute.
            e.g. [True, False] if the first target is a delta, second is absolute. Must match len(target_cols).
            If None, we assume all are deltas or all are absolute (you can tweak logic below).
        outputs_dir : str
            Directory where results can be saved.
        visualizations_dir : str
            Directory where to save the plot.
        plot_results : bool
            If True, plots line charts for each target vs. its ground truth.

        Returns
        -------
        preds_array : np.ndarray
            Shape (num_steps, num_targets) with all predicted values in real units.
        mae_dict : dict
            A dictionary containing per-target MAE and overall MAE.
            E.g. {'SkinTemperature(degC)_diff': 0.3, 'HeartRate(1/min)_diff': 5.2, 'overall': 2.7}
        """
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_absolute_error

        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)

        if target_cols is None or len(target_cols) == 0:
            raise ValueError("Please provide target_cols for multi-target evaluation.")

        num_targets = len(target_cols)

        # If no predict_delta_mask provided, assume all are either deltas or all absolute
        if predict_delta_mask is None:
            # e.g. all True => all are deltas
            # or all False => all are absolute
            predict_delta_mask = [False] * num_targets
        if len(predict_delta_mask) != num_targets:
            raise ValueError("predict_delta_mask must match the length of target_cols.")

        print("\n--- Evaluating (step-by-step) with multi-target TransformerRegressor ---")
        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

        # Ground truth for each target => shape (num_steps, num_targets)
        gt_array = df_sorted[target_cols].values

        # We'll keep an *unscaled* input 'state' that we update each iteration,
        # matching the order of self.feature_cols. (e.g. [time_delta, skin_temp, intensity, ...])
        state_unscaled = list(initial_state)
        preds_list = []  # will collect predicted arrays of shape (num_targets,) per step

        # Extra columns to read each step (like intensity, atemp, rh, etc.)
        # If None, we won't do any step updates from the DF.
        if extra_feature_cols is None:
            extra_feature_cols = []

        # Go row by row
        for i in range(len(df_sorted)):
            # 1) Construct a single input for the model in real units => shape (1, input_dim)
            state_array = np.array([state_unscaled], dtype=np.float32)

            # 2) Scale the input features
            state_scaled = scaler_X.transform(state_array)  # (1, input_dim)
            state_scaled_3d = state_scaled.reshape(1, 1, -1)  # (batch_size=1, seq_len=1, input_dim)

            # 3) Model forward pass
            self.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state_scaled_3d, dtype=torch.float32)
                output_scaled = self(state_tensor).cpu().numpy()  # shape (1, num_targets)
            # output_scaled => (1, n_targets)
            output_scaled = output_scaled[0]  # => shape (n_targets,)

            # 4) Inverse transform the model output
            # We must treat shape (1, n_targets) for the scaler
            output_scaled_2d = output_scaled.reshape(1, -1)  # shape (1, n_targets)
            output_unscaled_2d = scaler_Y.inverse_transform(output_scaled_2d)  # also (1, n_targets)
            output_unscaled = output_unscaled_2d[0]  # shape (n_targets,)

            # 5) If some targets are deltas, add them to the appropriate state entries
            # We need to know which state indices correspond to which target outputs.
            # You might map them like: target_cols -> state indices, if your state includes
            # those same variables. For example, if target_cols=[temp_diff, hr_diff] and
            # state_unscaled has [time, temp, hr, intensity, ...], then you'd do something like:
            #    new_temp = old_temp + output_unscaled[0]
            #    new_hr   = old_hr   + output_unscaled[1]
            #
            # For demonstration, let's assume the second entry in 'state_unscaled' is the temperature,
            # the third might be heart rate, etc. We'll do a simple example:

            # You might need a custom mapping from each target_col -> index in state_unscaled
            # e.g. target_map = {'SkinTemperature(degC)_diff': 1, 'HeartRate(1/min)_diff': 2}
            # For now, we'll pretend we have a direct approach:
            # state_unscaled[1] is temperature, state_unscaled[2] is heart rate, etc.

            # Example for a 2-target scenario:
            #    if predict_delta_mask[0]:
            #        state_unscaled[1] += output_unscaled[0]
            #    else:
            #        state_unscaled[1] = output_unscaled[0]
            #    if predict_delta_mask[1]:
            #        state_unscaled[2] += output_unscaled[1]
            #    else:
            #        state_unscaled[2] = output_unscaled[1]

            # In general, you need a known mapping from your target columns to the correct
            # indices in state_unscaled. We'll do a naive example for an n-target scenario:

            # Let's assume each target corresponds to an index offset in state_unscaled after the first one, for example.
            # YOU must adapt this logic to your actual data layout:
            states_unscaled = []
            for j in range(num_targets):
                if predict_delta_mask[j]:
                    # Add the delta
                    state_unscaled[1 + j] += output_unscaled[j]
                    states_unscaled.append(state_unscaled[1 + j])
                else:
                    # Absolute
                    state_unscaled[1 + j] = output_unscaled[j]
                    states_unscaled.append(state_unscaled[1 + j])

            # 6) Update other features from df row i, e.g. intensity, atemp, rh, etc.
            #    For example, if state_unscaled[3] is 'intensity', we set it to df_sorted[intensity_col].iloc[i].
            for col in extra_feature_cols:
                col_idx = self.feature_cols.index(col)
                state_unscaled[col_idx] = df_sorted[col].iloc[i]

            # 7) Maybe increment time by +1.0 or get it from df:
            state_unscaled[0] += 1.0 

            # Store the predicted output (the newly unscaled model output) in preds_list
            preds_list.append(states_unscaled)

        # After the loop, we have len(df_sorted) predictions, each shape (num_targets,)
        preds_array = np.array(preds_list)  # (num_steps, num_targets)

        # Compare with ground truth = shape (num_steps, num_targets)
        # We'll compute a per-target MAE plus an overall.
        mae_dict = {}
        for j, tgt_col in enumerate(target_cols):
            mae_j = mean_absolute_error(gt_array[:, j], preds_array[:, j])
            mae_dict[tgt_col] = mae_j

        # Overall MAE across all targets (flattened)
        overall_mae = mean_absolute_error(gt_array.flatten(), preds_array.flatten())
        mae_dict['overall'] = overall_mae

        print("\n--- Multi-target predictions complete ---")
        for col, val in mae_dict.items():
            print(f"MAE for '{col}': {val:.4f}")

        # Optionally plot each target
        if plot_results:
            num_plots = num_targets
            fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), sharex=True)
            if num_plots == 1:
                axes = [axes]  # make it iterable

            for j, ax in enumerate(axes):
                ax.plot(gt_array[:, j], label=f'GT {target_cols[j]}', color='blue')
                ax.plot(preds_array[:, j], label=f'Pred {target_cols[j]}', color='red', linestyle='--')
                ax.set_title(f"{target_cols[j]}: GT vs. Prediction")
                ax.set_xlabel('Time Step')
                ax.set_ylabel(target_cols[j])
                if figure_ranges:
                    if target_cols[j] in figure_ranges:
                        ax.set_ylim(figure_ranges[target_cols[j]])
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            out_path = os.path.join(visualizations_dir, 'multitarget_comparison.png')
            plt.savefig(out_path)
            print(f"Plot saved to: {out_path}")
            plt.show()

        return preds_array, mae_dict