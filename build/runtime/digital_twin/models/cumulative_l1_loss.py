import torch
import torch.nn as nn

class CumulativeL1Loss(nn.Module):
    """
    Custom L1-based cumulative loss function that penalizes both 
    step-wise error (MAE) and cumulative error (MAE of cumsum).
    """
    def __init__(self, step_loss_weight=0.5, cumulative_loss_weight=0.5):
        super(CumulativeL1Loss, self).__init__()
        self.step_loss_weight = step_loss_weight
        self.cumulative_loss_weight = cumulative_loss_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions, targets, initial_state):
        """
        Args:
          predictions:   (batch_size, seq_length, num_features) predicted changes
          targets:       (batch_size, seq_length, num_features) true changes
          initial_state: (batch_size, num_features) initial state before changes
        
        Returns:
          combined_loss: scalar tensor combining step-wise and cumulative MAE
        """
        # 1) Step-wise loss (MAE between predicted changes and true changes)
        step_loss = self.l1_loss(predictions, targets)

        # 2) Cumulative predictions: initial_state + cumsum of predicted changes
        cumulative_predictions = initial_state.unsqueeze(1) + torch.cumsum(predictions, dim=1)

        # 3) Cumulative targets: initial_state + cumsum of true changes
        cumulative_targets = initial_state.unsqueeze(1) + torch.cumsum(targets, dim=1)

        # 4) Cumulative loss (MAE between cumulative predicted and cumulative targets)
        cumulative_loss = self.l1_loss(cumulative_predictions, cumulative_targets)

        # 5) Weighted combination
        combined_loss = (self.step_loss_weight * step_loss 
                         + self.cumulative_loss_weight * cumulative_loss)
        return combined_loss
