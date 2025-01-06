import torch
import torch.nn as nn

class CumulativeLoss(nn.Module):
    def __init__(self, step_loss_weight=0.5, cumulative_loss_weight=0.5):
        """
        Custom loss function to penalize both step-wise prediction errors
        and cumulative errors over the entire prediction sequence.

        Parameters:
        - step_loss_weight: Weight for the step-wise MSE loss.
        - cumulative_loss_weight: Weight for the cumulative MSE loss.
        """
        super(CumulativeLoss, self).__init__()
        self.step_loss_weight = step_loss_weight
        self.cumulative_loss_weight = cumulative_loss_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets, initial_state):
        """
        Compute the combined loss.

        Parameters:
        - predictions: Tensor of shape (batch_size, pred_length, num_features), predicted changes.
        - targets: Tensor of shape (batch_size, pred_length, num_features), true changes.
        - initial_state: Tensor of shape (batch_size, num_features), initial state for each sequence.

        Returns:
        - Combined loss (scalar).
        """
        # Step-wise loss (MSE between predicted changes and true changes)
        step_loss = self.mse_loss(predictions, targets)

        # Cumulative prediction: initial state + cumulative sum of predicted changes
        cumulative_predictions = initial_state.unsqueeze(1) + torch.cumsum(predictions, dim=1)

        # Cumulative target: initial state + cumulative sum of true changes
        cumulative_targets = initial_state.unsqueeze(1) + torch.cumsum(targets, dim=1)

        # Cumulative loss (MSE between cumulative predicted states and true states)
        cumulative_loss = self.mse_loss(cumulative_predictions, cumulative_targets)

        # Combine the two losses
        combined_loss = (self.step_loss_weight * step_loss +
                         self.cumulative_loss_weight * cumulative_loss)

        return combined_loss
