import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from .cumulative_l1_loss import CumulativeL1Loss

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")

class GRUNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        A simple GRU-based model. 
        Args:
            input_size  (int):   number of input features
            hidden_size (int):   size of the hidden state of the GRU
            output_size (int):   number of output features
            num_layers  (int):   how many stacked GRU layers you want
        """
        super(GRUNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Define the GRU
        # batch_first=True => input shape is (batch_size, seq_length, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Final fully connected layer to map hidden states to output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # You could add more layers here if needed, such as an additional
        # non-linear layer, dropout, etc.

    def forward(self, x, hidden=None):
        """
        Args:
            x      (torch.Tensor): shape (batch_size, seq_length, input_size)
            hidden (torch.Tensor): optional initial hidden state of shape 
                                   (num_layers, batch_size, hidden_size)
        Returns:
            out    (torch.Tensor): shape (batch_size, seq_length, output_size)
            hidden (torch.Tensor): final hidden state, shape (num_layers, batch_size, hidden_size)
        """

        # If no hidden state is provided, PyTorch will internally initialize to zeros.
        out, hidden = self.gru(x, hidden)      # out shape => (batch_size, seq_length, hidden_size)
        out = self.fc(out)                     # out shape => (batch_size, seq_length, output_size)

        return out, hidden

    def train_model(self, df, feature_cols, target_cols, epochs, save=False, verbose=False):
        """
        Example training loop for single-step sequences (seq_length=1). 
        For multi-step sequences, you'll need to chunk your data differently.
        """

        self.feature_cols = feature_cols
        self.target_cols = target_cols

        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # We can do a simplistic approach like your current code:
        for epoch in range(epochs):
            epoch_loss = 0.0

            for index, row in df.iterrows():
                features = torch.tensor(row[feature_cols].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                targets  = torch.tensor(row[target_cols].values,  dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # features shape => (1,1,input_size)
                # targets shape  => (1,1,output_size)

                optimizer.zero_grad()
                outputs, _ = self.forward(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                avg_loss = epoch_loss / len(df)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        if save:
            torch.save(self.state_dict(), save)

    def trn_cml_lss(self, df, feature_cols, target_cols, epochs, initial_state,
                    seq_length=19, step_loss_weight=0.5, cumulative_loss_weight=0.5,
                    save=False, verbose=False, scaler_X=None):
        """
        An updated approach using the custom CumulativeL1Loss over sequences.
        Now we scale the initial_state if a scaler_X is provided.
        """

        criterion = CumulativeL1Loss(
            step_loss_weight=step_loss_weight,
            cumulative_loss_weight=cumulative_loss_weight
        )
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        # Convert df into sequential chunks
        simulations = []
        num_sims = len(df) // seq_length

        # If the user provided a scaler for features, scale the initial_state
        if not isinstance(initial_state, torch.Tensor):
            initial_state = torch.tensor(initial_state, dtype=torch.float32)
        if scaler_X is not None:
            init_np = initial_state.unsqueeze(0).numpy()    # shape (1, input_size)
            init_scaled = scaler_X.transform(init_np)       # scale
            initial_state = torch.tensor(init_scaled, dtype=torch.float32).squeeze(0)

        initial_state = (initial_state[1], initial_state[2], initial_state[3])
        initial_state = torch.tensor(initial_state, dtype=torch.float32)


        for i in range(num_sims):
            start_idx = i * seq_length
            end_idx   = start_idx + seq_length
            sim_df    = df.iloc[start_idx:end_idx]

            feature_tensor = torch.tensor(sim_df[feature_cols].values, dtype=torch.float32)
            target_tensor  = torch.tensor(sim_df[target_cols].values,  dtype=torch.float32)

            # Clone scaled initial_state for each simulation chunk
            simulations.append((feature_tensor, target_tensor, initial_state.clone()))

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (feature_tensor, target_tensor, init_state) in simulations:
                features_batch = feature_tensor.unsqueeze(0)
                targets_batch  = target_tensor.unsqueeze(0)

                optimizer.zero_grad()

                predictions, _ = self.forward(features_batch)

                init_state_batch = init_state.unsqueeze(0)
                loss = criterion(predictions, targets_batch, init_state_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                avg_loss = epoch_loss / num_sims
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        if save:
            torch.save(self.state_dict(), save)

    def run_nn(self, segments, initial_state, scaler_X=None, scaler_Y=None):
        """
        Example of how to run the GRU on a sequence of inputs. 
        Here we assume each row in `segments` is one time-step.
        """
        self.eval()

        # Check to ensure time increments are 1 minute. (Same as your code)
        for i in range(1, len(segments['time'])):
            assert segments['time'][i] == 1

        # Build initial state tensor
        state_tensor = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
        # shape => (batch_size=1, seq_length=1, input_size)

        results = {
            'Time(s)':                [initial_state[0]],
            'HeartRate(1/min)':       [initial_state[1]],
            'CoreTemperature(degC)':  [initial_state[2]],
            'SkinTemperature(degC)':  [initial_state[3]],
            'intensity':              [initial_state[4]],
            'atemp_c':                [initial_state[5]],
            'rh_pct':                 [initial_state[6]],
        }

        hidden = None  # We'll store the hidden state between steps if we want a 'stateful' run

        for i in range(len(segments['time'])):
            # Scale input
            if scaler_X:
                scaled_input = scaler_X.transform(state_tensor[:, :, :].view(1, -1).detach().numpy())
                scaled_input = torch.tensor(scaled_input, dtype=torch.float32).view(1, 1, -1)
            else:
                scaled_input = state_tensor

            # GRU forward
            prediction, hidden = self.forward(scaled_input, hidden)

            # If we used scaling, invert the scaling on the output
            if scaler_Y:
                prediction_numpy = scaler_Y.inverse_transform(prediction.view(1, -1).detach().numpy())
                prediction_unscaled = torch.tensor(prediction_numpy, dtype=torch.float32).view(1, 1, -1)
            else:
                prediction_unscaled = prediction

            # Construct new state by adding the predicted deltas (similar to your code)
            state_td  = state_tensor[0, 0, 0].item() + 1
            state_hr  = state_tensor[0, 0, 1].item() + prediction_unscaled[0, 0, 0].item()
            state_ct  = state_tensor[0, 0, 2].item() + prediction_unscaled[0, 0, 1].item()
            state_st  = state_tensor[0, 0, 3].item() + prediction_unscaled[0, 0, 2].item()
            state_int = segments['intensity'][i]
            state_atemp = segments['atemp_c'][i]
            state_rh   = segments['rh_pct'][i]

            # Store results
            results['Time(s)'].append(state_td)
            results['HeartRate(1/min)'].append(state_hr)
            results['CoreTemperature(degC)'].append(state_ct)
            results['SkinTemperature(degC)'].append(state_st)
            results['intensity'].append(state_int)
            results['atemp_c'].append(state_atemp)
            results['rh_pct'].append(state_rh)

            # Prepare for next iteration
            new_state = torch.tensor([state_td, state_hr, state_ct, state_st,
                                      state_int, state_atemp, state_rh], dtype=torch.float32)
            state_tensor = new_state.view(1, 1, -1)  # keep shape for next step

        df = pd.DataFrame(results)
        df['Time(s)'] = pd.to_timedelta(df['Time(s)'], unit='m')
        df.set_index('Time(s)', inplace=True)
        return df
