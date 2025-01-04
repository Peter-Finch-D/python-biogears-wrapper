# digital_twin/dataset.py

import torch
from torch.utils.data import Dataset

class Seq2SeqPhysioDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_cols, scaler_X, scaler_Y, seq_length=4, pred_length=5):
        super().__init__()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        self.scaled_features = self.scaler_X.transform(dataframe[self.feature_cols])
        self.scaled_targets = self.scaler_Y.transform(dataframe[self.target_cols])
        
        self.samples = []
        for i in range(len(dataframe) - self.seq_length - self.pred_length + 1):
            X_seq = self.scaled_features[i:i + self.seq_length]
            Y_seq = self.scaled_targets[i + self.seq_length:i + self.seq_length + self.pred_length]
            self.samples.append((X_seq, Y_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X_seq, Y_seq = self.samples[idx]
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        Y_seq = torch.tensor(Y_seq, dtype=torch.float32)
        return X_seq, Y_seq
