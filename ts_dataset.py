import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class TSDataset(Dataset):
    """Time series dataset."""
    def __init__(self, csv_file, timestamp_col, value_col, time_window, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            timestamp_col: name of the column containing timestamp
            value_col: name of the column containing values 
            time_window: time window to consider for conditioning/generation
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file)
        df = df.filter([timestamp_col, value_col], axis=1)
        df.rename(columns={value_col:'Value'}, inplace = True)
        df['Timestamp'] = pd.to_datetime(df[timestamp_col].map(str))
        df = df.drop([timestamp_col], axis=1).set_index('Timestamp')
        
        n = (len(df)//time_window) * time_window
        value = df.Value
        arr = np.asarray([value[time_window*i:time_window*i+time_window] for i in range (n//time_window)], dtype=np.float32)
        data = torch.from_numpy(np.expand_dims(arr, -1))
        self.data = self.normalize(data) if normalize else data
        self.seq_len = data.size(1)
        
        #Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = data[:, -1] - data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min() 
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return 2 * (x - x.min())/(x.max() - x.min()) - 1
    
    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x*self.max - x*self.min + self.max + self.min)
    
    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std
    
    def normalize_deltas(self, x):
        return (self.delta_max - self.delta_min) * (x - self.or_delta_min)/(self.or_delta_max - self.or_delta_min) + self.delta_min