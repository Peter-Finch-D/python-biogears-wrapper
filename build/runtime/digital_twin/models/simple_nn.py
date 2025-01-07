import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from .cumulative_l1_loss import CumulativeL1Loss

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()

        self.input_size = input_size

        # First layer (Input layer)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()

        # Second layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        # Third layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()

         # Fourth layer
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()

        # Fifth layer (Output layer)
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

    def train_model(self, df, feature_cols, target_cols, epochs, save=False, verbose=False):

        self.feature_cols = feature_cols
        self.target_cols = target_cols

        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            for index, row in df.iterrows():
                features = torch.tensor(row[feature_cols].values, dtype=torch.float32)
                targets = torch.tensor(row[target_cols].values, dtype=torch.float32)

                optimizer.zero_grad()
                outputs = self.forward(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if verbose: print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        if save != False:
            torch.save(self.state_dict(), save)

    def trn_cml_lss(self, df, feature_cols, target_cols, epochs,seq_length=9, step_loss_weight=0.5, cumulative_loss_weight=0.5,save=False, verbose=False):
        """
        Trains the model using the custom CumulativeL1Loss over sequences of length 9.

        Args:
        df: A DataFrame where each simulation is 9 consecutive rows.
        feature_cols: list of columns in df used as input features.
        target_cols:  list of columns in df representing *changes* or targets at each step.
        epochs: number of training epochs.
        seq_length: length of each simulation (9 in your case).
        step_loss_weight: weight for step-wise L1.
        cumulative_loss_weight: weight for cumulative L1.
        save: path to save the model parameters or False if not saving.
        verbose: print progress if True.
        """

        # 1. Create the custom loss and optimizer
        criterion = CumulativeL1Loss(
            step_loss_weight=step_loss_weight,
            cumulative_loss_weight=cumulative_loss_weight
        )
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # 2. Convert df to a list of (feature_tensor, target_tensor, initial_state) per simulation
        #    Each simulation has exactly 9 rows => shape will be (9, len(feature_cols)) for features
        #    and (9, len(target_cols)) for targets.
        simulations = []
        
        # We'll assume df is structured so that every contiguous 9 rows is one simulation
        # If there's a grouping column (like simulation_id), you'd group by that instead.
        num_sims = len(df) // seq_length
        for i in range(num_sims):
            start_idx = i * seq_length
            end_idx = start_idx + seq_length
            sim_df = df.iloc[start_idx:end_idx]

            # Convert features/targets to tensors of shape (seq_length, num_features)
            feature_tensor = torch.tensor(sim_df[feature_cols].values, dtype=torch.float32)
            target_tensor  = torch.tensor(sim_df[target_cols].values, dtype=torch.float32)

            # initial_state: the state at the very start, before applying the first change
            # For example, if the first row of target_cols indicates the changes from this initial state
            # to the next step, you need to store that initial state somewhere in sim_df or pass externally.
            # 
            # Let's assume your initial state is the first row of some state columns in your df.
            # If you store the "current state" in feature_cols, you might do:
            initial_state = torch.tensor((75, 37, 33))

            simulations.append((feature_tensor, target_tensor, initial_state))

        # 3. Train for the specified number of epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (feature_tensor, target_tensor, initial_state) in simulations:
                # We want shape (batch_size=1, seq_length=9, num_features=?)
                # So let's unsqueeze(0) to add a batch dimension
                features_batch  = feature_tensor.unsqueeze(0)  # (1, 9, num_features_in)
                targets_batch   = target_tensor.unsqueeze(0)   # (1, 9, num_features_out)
                init_state_batch = initial_state.unsqueeze(0)  # (1, num_features_in)

                # Zero out the grad
                optimizer.zero_grad()

                # Forward pass: model should return shape (batch_size=1, seq_length=9, num_features_out)
                # For that to happen, your model's forward pass must be able to handle 9 timesteps at once.
                # If your model is purely feedforward (like a Multi-Layer Perceptron) with input_size = single-step features,
                # you may need to reshape or process each time step inside a loop. 
                #
                # As is, your SimpleNN forward pass is expecting shape (batch_size, input_size). 
                # So we either:
                # (A) Reshape (1*9, input_size) => feed it => reshape back, or
                # (B) Modify your model to handle a temporal dimension.
                
                # Let's do (A) for simplicity:
                bsz, seq_len, in_size = features_batch.shape
                flattened_input = features_batch.view(bsz * seq_len, in_size)   # (9, input_size)
                predictions_flat = self.forward(flattened_input)                # (9, output_size)

                # Reshape predictions to (batch_size=1, seq_length=9, output_size)
                bsz2, out_size = predictions_flat.shape
                predictions = predictions_flat.view(bsz, seq_len, out_size)

                # Compute loss
                loss = criterion(predictions, targets_batch, init_state_batch)
                
                # Backprop & update
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                avg_loss = epoch_loss / num_sims
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        # 4. Optional save
        if save:
            torch.save(self.state_dict(), save)

    def run_nn(self, segments, initial_state, scaler_X=None, scaler_Y=None):

        self.eval()

        # Check to make sure the segments all have time steps of 1 minute
        for i in range(1, len(segments['time'])):
            assert segments['time'][i] == 1

        # Check that the initial state has the same number of elements as the input size
        assert len(initial_state) == self.input_size

        # Build a tensor for the initial state to be fed to the model for prediction
        state_tensor = torch.tensor(initial_state, dtype=torch.float32)

        # Create an object to store the results of the predictions
        results = {
            'Time(s)':                      [initial_state[0]],
            'HeartRate(1/min)':             [initial_state[1]],
            'CoreTemperature(degC)':        [initial_state[2]],
            'SkinTemperature(degC)':        [initial_state[3]],
            #'hr_diff':      [0],
            #'ct_diff':      [0],
            #'st_diff':      [0],
            'intensity':                    [initial_state[4]],
            'atemp_c':                      [initial_state[5]],
            'rh_pct':                       [initial_state[6]]
        }

        for i in range(len(segments['time'])):
            # Scale the input features
            state_tensor_scaled = torch.tensor(scaler_X.transform(state_tensor.unsqueeze(0).detach().numpy()), dtype=torch.float32).squeeze(0)

            # Predict the physiological changes with the network
            state_tensor_prime_scaled = self.forward(state_tensor_scaled)

            # Descale the output features
            state_tensor_prime = torch.tensor(scaler_Y.inverse_transform(state_tensor_prime_scaled.unsqueeze(0).detach().numpy()), dtype=torch.float32).squeeze(0)

            # Create the new state tensor
            state_td = state_tensor[0].item() + 1
            state_hr = state_tensor[1].item() + state_tensor_prime[0].item()
            state_ct = state_tensor[2].item() + state_tensor_prime[1].item()
            state_st = state_tensor[3].item() + state_tensor_prime[2].item()
            state_int = segments['intensity'][i]
            state_atemp = segments['atemp_c'][i]
            state_rh = segments['rh_pct'][i]

            # Store the results
            results['Time(s)'].append(state_td)
            results['HeartRate(1/min)'].append(state_hr)
            results['CoreTemperature(degC)'].append(state_ct)
            results['SkinTemperature(degC)'].append(state_st)
            #results['hr_diff'].append(state_tensor_prime[0].item())
            #results['ct_diff'].append(state_tensor_prime[1].item())
            #results['st_diff'].append(state_tensor_prime[2].item())
            results['intensity'].append(state_int)
            results['atemp_c'].append(state_atemp)
            results['rh_pct'].append(state_rh)

            state = (state_td, state_hr, state_ct, state_st, state_int, state_atemp, state_rh)
            state_tensor = torch.tensor(state, dtype=torch.float32)

        df = pd.DataFrame(results)
        df['Time(s)'] = pd.to_timedelta(df['Time(s)'], unit='m')
        df.set_index('Time(s)', inplace=True)
        return df