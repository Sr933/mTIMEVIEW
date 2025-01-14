import pandas as pd
import pickle
import json
import torch
import numpy as np
import os
from types import SimpleNamespace
from abc import ABC, abstractmethod
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from libutils import calculate_knot_placement, BSplineBasis, BaseDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

def create_dataloader(config, dataset, indices=None, shuffle=True):
    """
    Creates a dataloader for a subset of the dataset described by the list of indices.
    
    Args:
        config: an instance of the Config class
        dataset: an instapynce of the TTSDataset class
        indices: a list of integers that are the indices of the dataset
        shuffle: a boolean that indicates whether to shuffle the indices
    
    Returns:
        dataloader: a torch.utils.data.DataLoader object
    """
    
    # Check that config is an instance of the Config class
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of the Config class")
    
    # Check that dataset is an instance of the TTSDataset class
    if not isinstance(dataset, TTSDataset):
        raise ValueError("dataset must be an instance of the TTSDataset class")
    
    # Check that indices is a list if provided
    if indices is not None and not isinstance(indices, list):
        raise ValueError("indices must be a list")
    
    # Check that shuffle is a boolean
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean")
    
    # Create a random generator with the provided seed
    gen = torch.Generator()
    gen.manual_seed(config.seed)
    
    # Create a subset of the dataset based on indices
    if indices is None:
        subset = dataset
    else:
        subset = Subset(dataset, indices)
    
    # Retrieve the collate_fn from the dataset (assuming the dataset has this method)
    collate_fn = dataset.get_collate_fn()  # Make sure the dataset has this method
    
    # Create the DataLoader
    dataloader = DataLoader(
        subset, 
        batch_size=config.training.batch_size, 
        shuffle=shuffle, 
        generator=gen, 
        collate_fn=collate_fn,
        num_workers=10
    )

    return dataloader



class SimpleDataset(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        
        # Extract static features - convert them to numpy array
        self.X_static = []
        for sample in self.dataset:
            
            self.X_static.append(sample['static_features'])
        self.X_static = np.array(self.X_static)
        # Extract time-series data and convert it into a 2D array (features x time_steps)
        self.X_ts = []
        self.ts = []  # This will now contain the indices of the target series (ys)
        
        for sample in self.dataset:
            time_series_data = sample['time_series']  # Dictionary with each key as a feature and value as time-series
            
            # Convert each time series feature into a row in the 2D array (features x time_steps)
            ts_features = []
            for key in time_series_data.values():
                ts_features.append(key)  # Each key (time-series feature) will be a row in the 2D array
            
            ts_features = np.array(ts_features)  # Shape will be (features, time_steps)
            self.X_ts.append(ts_features)  # Append the 2D array for each sample
            
            # Add the index of the target series (assuming the target series is aligned with the time series)
            self.ts.append(np.arange(len(sample['target_series'])))  # or sample['target_series'] if needed
        self.X_ts=np.array(self.X_ts)
        # Extract target series (ys)
        self.ys = [sample['target_series'] for sample in self.dataset]

    def get_X_ts_ys(self):
        return self.X_static, self.X_ts, self.ys # Now we return static features, time-series (2D array), and target series
    
    def __len__(self):
        return len(self.dataset)
    
    def get_feature_names(self):
        # Since we removed pandas, we don't have a DataFrame anymore.
        # Assuming the static features are directly in the data, we can return their names from the dataset.
        # If static features are stored as a list or dictionary, you might need to handle it differently.
        return list(self.dataset[0]['static_features'].keys())  # Static feature names from the first sample
    
    def get_feature_ranges(self):
        feature_ranges = {}
        for col in range(self.X_static.shape[1]):  # Iterate over the static features
            feature_min = np.min(self.X_static[:, col])
            feature_max = np.max(self.X_static[:, col])
            feature_ranges[f'feature_{col}'] = (feature_min, feature_max)
        return feature_ranges




def is_dynamic_bias_enabled(config):
    if hasattr(config, 'dynamic_bias'):
        return config.dynamic_bias
    else:
        return False

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class TSEncoder(torch.nn.Module):
    def __init__(self, config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        self.config = config
        self.n_features =config.n_static_features  # Number of static features
        self.n_ts_features =config.n_features  # Number of time-series features
        self.n_basis = config.n_basis
        self.ts_length=config.ts_length
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0

        # Number of time series inputs
        self.n_time_series = self.n_ts_features

        # RNN for each time-series feature
        self.rnn_hidden_size = config.rnn.hidden_size
        self.rnns = nn.ModuleList([nn.GRU(input_size=self.ts_length, 
                                          hidden_size=self.rnn_hidden_size, 
                                          num_layers=3, 
                                          batch_first=True) 
                                   for _ in range(self.n_time_series)])

        # Fully connected layer for static features
        self.fc_static = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_sizes[0]),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.ReLU()
        )

        # MLP layers for the concatenated output
        self.mlp_layers = []
        activation = nn.ReLU()

        # Input layer for MLP
        input_size = self.rnn_hidden_size * self.n_time_series + self.hidden_sizes[0]  # Combine RNNs and FC layer output
        self.mlp_layers.append(nn.Linear(input_size, self.hidden_sizes[0]))
        self.mlp_layers.append(nn.BatchNorm1d(self.hidden_sizes[0]))
        self.mlp_layers.append(activation)
        self.mlp_layers.append(nn.Dropout(self.dropout_p))

        # Hidden layers for MLP
        for i in range(len(self.hidden_sizes) - 1):
            self.mlp_layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.mlp_layers.append(nn.BatchNorm1d(self.hidden_sizes[i + 1]))
            self.mlp_layers.append(activation)
            self.mlp_layers.append(nn.Dropout(self.dropout_p))

        # Output layer
        latent_size = self.n_basis
        if is_dynamic_bias_enabled(config):
            latent_size += 1
        self.mlp_layers.append(nn.Linear(self.hidden_sizes[-1], latent_size))

        self.mlp = nn.Sequential(*self.mlp_layers)

    def forward(self, x_static, x_ts):
        """
        Args:
            x_static: Static features, shape (batch_size, n_features)
            x_ts: List of time-series data, shape (batch_size, seq_len, n_ts_features)
        """
        
        
       
        static_out = self.fc_static(x_static)  # Shape: (batch_size, hidden_size)
          #
        if isinstance(x_ts, list):
            x_ts=torch.stack(x_ts) # Convert list to tensor, ensure it's the right shape

        # Check that x_ts is a tensor with the correct shape
        assert len(x_ts.shape) == 3, f"x_ts must have shape (batch_size, seq_len, n_ts_features), but got {x_ts.shape}"
        # Process each time-series input through its corresponding RNN
        rnn_outputs = []
        for i in range(self.n_time_series):
            ts_i = x_ts[:, i,:]  # Time-series for the i-th feature, shape: (batch_size, seq_len)
            
            padded_ts_i = pad_sequence(ts_i, batch_first=True)  # shape: (batch_size, seq_len, 1)
            rnn_out, _ = self.rnns[i](padded_ts_i)  # rnn_out shape: (batch_size, seq_len, rnn_hidden_size)
            
            rnn_outputs.append(rnn_out)  # Take the last time step output (batch_size, rnn_hidden_size)

        # Concatenate the outputs from each RNN and static features
        rnn_out_combined = torch.cat(rnn_outputs, dim=1)  # Shape: (batch_size, rnn_hidden_size * n_time_series)
        x_combined = torch.cat([static_out, rnn_out_combined], dim=1)  # Shape: (batch_size, static + rnn_combined)

        # Pass the combined features through the MLP
        return self.mlp(x_combined)



ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'leaky_relu': torch.nn.LeakyReLU,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop
}


class Config():

    def __init__(self,
                 n_features=10,
                 n_static_features=1,  # Number of static features
                 n_ts_length=25,
                 n_basis=5,
                 T=1,
                 seed=42,
                 encoder={
                     'hidden_sizes': [32, 64, 32],
                     'activation': 'relu',
                     'dropout_p': 0.2
                 },
                 rnn={
                     'hidden_size': 64,  # Hidden size of RNN
                     'num_layers': 1,   # Number of layers in RNN
                     'dropout_p': 0.2,  # Dropout for RNN
                     'cell_type': 'LSTM'  # RNN cell type: 'LSTM' or 'GRU'
                 },
                 training={
                     'optimizer': 'adam',
                     'lr': 1e-3,
                     'batch_size': 32,
                     'weight_decay': 1e-5
                 },
                 dataset_split={
                     'train': 0.8,
                     'val': 0.1,
                     'test': 0.1
                 },
                 dataloader_type='iterative',
                 device='cpu',
                 num_epochs=200,
                 internal_knots=None,
                 n_basis_tunable=False,
                 dynamic_bias=False):

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if not isinstance(n_static_features, int):
            raise ValueError("n_static_features must be an integer >= 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        if rnn['cell_type'] not in ['LSTM', 'GRU']:
            raise ValueError("rnn['cell_type'] must be one of ['LSTM', 'GRU']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.n_static_features = n_static_features
        self.ts_length = n_ts_length
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.rnn = SimpleNamespace(**rnn)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias


class TuningConfig(Config):
    def __init__(
        self,
        trial,
        n_features=1,
        n_static_features=1,  # Number of static features
        n_ts_length=25,
        n_basis=5,
        T=1,
        seed=42,
        dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        dataloader_type='iterative',
        device='cpu',
        num_epochs=200,
        internal_knots=None,
        n_basis_tunable=False,
        dynamic_bias=False,
        rnn={
            'hidden_size': 64,  # Hidden size of RNN
            'num_layers': 1,   # Number of layers in RNN
            'dropout_p': 0.2,  # Dropout for RNN
            'cell_type': 'LSTM'  # RNN cell type: 'LSTM' or 'GRU'
        },
        encoder={
            'hidden_sizes': [32, 64, 32],
            'activation': 'relu',
            'dropout_p': 0.2
        },
        training={
            'optimizer': 'adam',
            'lr': 1e-3,
            'batch_size': 32,
            'weight_decay': 1e-5
        }
    ):
        # Define hyperparameter search space
        hidden_sizes = [trial.suggest_int(f'hidden_size_{i}', 16, 128) for i in range(3)]
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        )
        dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)

        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

        if n_basis_tunable:
            n_basis = trial.suggest_int('n_basis', 5, 16)

        encoder = {
            'hidden_sizes': hidden_sizes,
            'activation': activation,
            'dropout_p': dropout_p
        }
        training = {
            'optimizer': 'adam',
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }
        
        # Validation of configuration values
        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if not isinstance(n_static_features, int):
            raise ValueError("n_static_features must be an integer >= 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError("encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError("encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"encoder['activation'] must be one of {list(ACTIVATION_FUNCTIONS.keys())}")
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError(f"optimizer['name'] must be one of {list(OPTIMIZERS.keys())}")
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError("dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError("dataloader_type must be one of ['iterative','tensor']")
        if rnn['cell_type'] not in ['LSTM', 'GRU']:
            raise ValueError("rnn['cell_type'] must be one of ['LSTM', 'GRU']")

        # Store parameters in object
        self.n_basis = n_basis
        self.n_features = n_features
        self.n_static_features = n_static_features
        self.ts_length=n_ts_length
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.rnn = SimpleNamespace(**rnn)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias

class YNormalizer:
    """Normalize y values."""

    def __init__(self):
        self.fitted = False

    def fit(self, ys):
        """Fit normalization parameters."""
        if self.fitted:
            raise RuntimeError('Already fitted.')
        if isinstance(ys, list):
            Y = np.concatenate(ys, axis=0)
        else:
            Y = ys
        self.y_mean = np.mean(Y)
        self.y_std = np.std(Y)
        self.fitted = True

    def transform(self, ys):
        """Normalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [(y - self.y_mean) / self.y_std for y in ys]
        else:
            return (ys - self.y_mean) / self.y_std
    
    def inverse_transform(self, ys):
        """Denormalize y values."""
        if not self.fitted:
            raise RuntimeError('Call fit before transform.')
        if isinstance(ys, list):
            return [y * self.y_std + self.y_mean for y in ys]
        else:
            return ys * self.y_std + self.y_mean
    
    def save(self, path):
        """Save normalization parameters using json"""
        y_normalization = {'y_mean': self.y_mean, 'y_std': self.y_std}
        full_path = os.path.join(path, "y_normalizer.json")
        with open(full_path, 'w') as f:
            json.dump(y_normalization, f)
    
    def load(path):
        """Load normalization parameters using json"""
        with open(path, 'r') as f:
            y_normalization = json.load(f)
        ynormalizer = YNormalizer()
        ynormalizer.set_params(y_normalization['y_mean'], y_normalization['y_std'])
        return ynormalizer

    def load_from_benchmark(timestamp, name, benchmark_dir='benchmarks'):
        """Load normalization parameters from a benchmark."""
        path = os.path.join(benchmark_dir, timestamp, name, 'y_normalizer.json')
        return YNormalizer.load(path)

    def set_params(self, y_mean, y_std):
        """Set normalization parameters."""
        self.y_mean = y_mean
        self.y_std = y_std
        self.fitted = True

    def fit_transform(self, ys):
        """Fit normalization parameters and normalize y values."""
        self.fit(ys)
        return self.transform(ys)

class TTS(torch.nn.Module):

    def __init__(self, config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.encoder = TSEncoder(self.config)
        if not is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, X_static, X_ts, Phis):
        """
        Args:
            X_static: a tensor of shape (D, M) where D is the number of samples and M is the number of static features
            X_ts: a tensor of shape (D, N_max, T_f) where N_max is the maximum number of time steps and T_f is the number of time-series features
            Phis:
                if dataloader_type = 'tensor': a tensor of shape (D, N_max, B) where D is the number of samples, N_max is the maximum number of time steps, and B is the number of basis functions
                if dataloader_type = 'iterative': a list of D tensors of shape (N_d, B) where N_d is the number of time steps and B is the number of basis functions
        """
        h = self.encoder(X_static, X_ts)
        if is_dynamic_bias_enabled(self.config):
            self.bias = h[:, -1]
            h = h[:, :-1]

        if self.config.dataloader_type == "iterative":
            if is_dynamic_bias_enabled(self.config):
                return [torch.matmul(torch.tensor(Phi, dtype=torch.float32), h[d, :]) + self.bias[d] for d, Phi in enumerate(Phis)]
            else:
                return [torch.matmul(torch.tensor(Phi, dtype=torch.float32), h[d, :]) + self.bias for d, Phi in enumerate(Phis)]
        elif self.config.dataloader_type == "tensor":
            if is_dynamic_bias_enabled(self.config):
                return torch.matmul(Phis, torch.unsqueeze(h, -1)).squeeze(-1) + torch.unsqueeze(self.bias, -1)
            else:
                return torch.matmul(Phis, torch.unsqueeze(h, -1)).squeeze(-1) + self.bias

    def predict_latent_variables(self, X_static, X_ts):
        """
        Args:
            X_static: a numpy array of shape (D, M) where D is the number of samples and M is the number of static features
            X_ts: a numpy array of shape (D, N, T_f) where N is the number of time steps and T_f is the number of time-series features
        Returns:
            a numpy array of shape (D, B) where D is the number of samples and B is the number of basis functions
        """
        device = next(self.encoder.parameters()).device
        X_static = torch.from_numpy(X_static).float().to(device)
        X_ts = torch.from_numpy(X_ts).float().to(device)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(X_static, X_ts)
            if is_dynamic_bias_enabled(self.config):
                return h[:, :-1].cpu().numpy()
            else:
                return h.cpu().numpy()

    def forecast_trajectory(self, x_static, x_ts, t):
        """
        Args:
            x_static: a numpy array of shape (M,) where M is the number of static features
            x_ts: a numpy array of shape (N, T_f) where N is the number of time steps and T_f is the number of time-series features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (N,) where N is the number of time steps
        """
        device = next(self.encoder.parameters()).device
        x_static = torch.unsqueeze(torch.from_numpy(x_static), 0).float().to(device)
        x_ts = torch.unsqueeze(torch.from_numpy(x_ts), 0).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x_static, x_ts)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[0, -1]
                h = h[:, :-1]
            return (torch.matmul(Phi, h[0, :]) + self.bias).cpu().numpy()

    def forecast_trajectories(self, X_static, X_ts, t):
        """
        Args:
            X_static: a numpy array of shape (D, M) where D is the number of samples and M is the number of static features
            X_ts: a numpy array of shape (D, N, T_f) where N is the number of time steps and T_f is the number of time-series features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (D, N) where D is the number of samples and N is the number of time steps
        """
        device = next(self.encoder.parameters()).device
        X_static = torch.from_numpy(X_static).float().to(device)
        X_ts = torch.from_numpy(X_ts).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)  # shape (N, B)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(X_static, X_ts)  # shape (D, B)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[:, -1]
                h = h[:, :-1]
            return (torch.matmul(h, Phi.T) + self.bias).cpu().numpy()  # shape (D, N), broadcasting takes care of bias

class TTSDataset(Dataset):
    def __init__(self, config, X_static_train, X_ts_train, ys_train, ts_train):
        """
        Custom dataset class that handles static features and time-series data.
        Args:
            config: configuration containing various settings (e.g., batch size, time horizon, etc.)
            X_static_train: Static features (numpy array of shape [D, M])
            X_ts_train: Time-series features (list of D numpy arrays)
            ys_train: Target series (list of D numpy arrays)
            ts_train: Time indices for the time-series data (list of D numpy arrays)
        """
        self.config = config
        self.X_static = X_static_train
        self.X_ts = X_ts_train
        self.ys = ys_train
        self.ts = ts_train

        self.T=self.ts
        # Number of samples
        self.D = len(self.X_static)

        # Process data for later use
        self._process_data()

    def _process_data(self):
        """
        Process and prepare the data for use, converting inputs to tensors.
        """
        # Convert static features to tensor
        self.X_static = torch.tensor(self.X_static, dtype=torch.float32)

        # Convert time-series data, target series, and time indices to tensors
        self.X_ts = [torch.tensor(ts, dtype=torch.float32) for ts in self.X_ts]
        self.ys = [torch.tensor(y, dtype=torch.float32) for y in self.ys]
        self.ts = [torch.tensor(t, dtype=torch.float32) for t in self.ts]

        # Determine lengths of time-series and max length
        self.Ns = [len(ts) for ts in self.ts]
        self.N_max = max(self.Ns)

        # Compute spline matrices if required
        if hasattr(self.config, 'n_basis'):
            self.Phis = self._compute_matrices()

    def _compute_matrices(self):
        bspline = BSplineBasis(self.config.n_basis, (0, len(self.T[0])), internal_knots=self.config.internal_knots)
        Phis = list(bspline.get_all_matrices(self.T))
        return Phis

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.D

    def __getitem__(self, idx):
        """
        Returns a single data point for a given index in the desired format.
        """
       
        return {
            'static_features': self.X_static[idx],
            'time_series': self.X_ts[idx],
            'target_series': self.ys[idx],
            'time_indices': self.ts[idx],
            'spline_matrix': self.Phis[idx] if hasattr(self, 'Phis') else None,}
        
    
    def get_collate_fn(self):
        """
        Returns the collate function that will be used by DataLoader to batch the data.
        """
        def collate_fn(batch):
            """
            Custom collate function for batching samples.
            """
            static_features = torch.stack([b['static_features'] for b in batch], dim=0)

            
            time_series = [b['time_series'] for b in batch]
            target_series = [b['target_series'] for b in batch]
            time_indices = [b['time_indices'] for b in batch]
            spline_matrices = [b['spline_matrix'] for b in batch]
            return static_features, time_series, target_series, time_indices, spline_matrices
           

        return collate_fn


class LitTTS(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = TTS(config)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # def forward(self, batch, batch_idx, dataloader_idx=0):
        if self.config.dataloader_type == 'iterative':
            static_features, time_series, target_series, time_indices, spline_matrices = batch
            preds = self.model(static_features, time_series, spline_matrices)
            return preds

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            return pred  # 2D tensor

    def training_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            static_features, time_series, target_series, time_indices, spline_matrices = batch
            preds = self.model(static_features, time_series, spline_matrices)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, target_series)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('train_loss', loss, batch_size=static_features.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        
        
        if self.config.dataloader_type == 'iterative':
            static_features, time_series, target_series, time_indices, spline_matrices = batch
            preds = self.model(static_features, time_series, spline_matrices)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, target_series)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('val_loss', loss, batch_size=static_features.shape[0])

        return loss

    def test_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            static_features, time_series, target_series, time_indices, spline_matrices = batch
            preds = self.model(static_features, time_series, spline_matrices)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, target_series)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('test_loss', loss, batch_size=static_features.shape[0])

        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer



class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    def __init__(self):
        self.name = self.get_name()


    def tune(self, n_trials, seed, benchmarks_dir):
        """Tune the benchmark."""

        print("Tuning")
        def objective(trial):     
            # Get the model with the trial's hyperparameters
            model = self.get_model_for_tuning(trial, seed)
 
            return self.train(model[0], tuning=False)['val_loss'] #Changed from True
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler,direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_hyperparameters = best_trial.params

        print('[Best hyperparameter configuration]:')
        print(best_hyperparameters)

        tuning_dir = os.path.join(benchmarks_dir, self.name, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # Save best hyperparameters
        hyperparam_save_path = os.path.join(tuning_dir, f'hyperparameters.json')
        with open(hyperparam_save_path, 'w') as f:
            json.dump(best_hyperparameters, f)
        
        # Save optuna study
        study_save_path = os.path.join(tuning_dir, f'study_{seed}.pkl')
        with open(study_save_path, 'wb') as f:
            pickle.dump(study, f)

        # Save trials dataframe
        df = study.trials_dataframe()
        df.set_index('number', inplace=True)
        df_save_path = os.path.join(tuning_dir, f'trials_dataframe.csv')
        df.to_csv(df_save_path)

        # save optuna visualizations
        # fig = optuna.visualization.plot_intermediate_values(study)
        # fig.write_image(os.path.join(tuning_dir, 'intermediate_values.png'))

        # fig = optuna.visualization.plot_optimization_history(study)
        # fig.write_image(os.path.join(tuning_dir, 'optimization_history.png'))

        # fig = optuna.visualization.plot_param_importances(study)
        # fig.write_image(os.path.join(tuning_dir, 'param_importance.png'))

        print(f'[Tuning complete], saved tuning results to {tuning_dir}')

        return best_hyperparameters

    def run(self, dataset: BaseDataset, train_indices, val_indices, test_indices, n_trials, n_tune, seed, benchmarks_dir, **kwargs):
        """Run the benchmark."""
        self.benchmarks_dir = benchmarks_dir

        # Create a numpy random generator
        rng = np.random.default_rng(seed)

        # Seed for tuning
        tuning_seed = seed

        # Generate seeds for training
        training_seeds = rng.integers(0, 2**31 - 1, size=n_trials)
        training_seeds = [s.item() for s in training_seeds]

        # Prepare the data
        self.prepare_data(dataset, train_indices, val_indices, test_indices)

        # Tune the model
        if n_tune > 0:
            print(f"[Tuning for {n_tune} trials]")
            best_hyperparameters = self.tune(n_trials=n_tune, seed=tuning_seed, benchmarks_dir=benchmarks_dir)
        else:
            print(f"[No tuning, using default hyperparameters]")
            best_hyperparameters = None

        print(f"[Training for {n_trials} trials with best hyperparameters]")

        # Train the model n_trials times
        test_losses = []
        for i in range(n_trials):
            print(f"[Training trial {i+1}/{n_trials}] seed: {training_seeds[i]}")
            model = self.get_final_model(best_hyperparameters, training_seeds[i])
            test_loss = self.train(model)['test_loss']
            test_losses.append(test_loss)

        # Save the losses
        df = pd.DataFrame({'seed':training_seeds,'test_loss': test_losses})
        results_folder = os.path.join(benchmarks_dir, self.name, 'final')
        os.makedirs(results_folder, exist_ok=True)
        test_losses_save_path = os.path.join(results_folder, f'results.csv')
        df.to_csv(test_losses_save_path, index=False)
        
        return test_losses

    @abstractmethod
    def prepare_data(self, dataset: BaseDataset, train_indices, val_indices, test_indices):
        """Prepare the data for the benchmark."""
        pass

    @abstractmethod
    def train(self, model, tuning=False):
        """
        Train the benchmark. Returns a dictionary with train, validation, and test loss
        Returns:
            dict: Dictionary with train, validation, and test loss {'train_loss': float, 'val_loss': float, 'test_loss': float}
        """
        pass
       
    @abstractmethod
    def get_model_for_tuning(self, trial, seed):
        """Get the model."""
        pass

    @abstractmethod
    def get_final_model(self, hyperparameters, seed):
        """Get the model."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Get the name of the benchmark."""
        pass 
class TTSBenchmark(BaseBenchmark):
    """TTS benchmark."""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        return 'TTS'
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""
        config = TuningConfig(
                            trial,
                            n_features=self.config.n_features,
                            n_static_features=self.config.n_static_features,
                            n_ts_length=self.config.ts_length,
                            n_basis=self.config.n_basis,
                            T=self.config.T,
                            seed=self.config.seed,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs,
                            dataloader_type=self.config.dataloader_type,
                            internal_knots=self.config.internal_knots,
                            n_basis_tunable=self.config.n_basis_tunable,
                            dynamic_bias=self.config.dynamic_bias)
        litmodel = LitTTS(config)
        litmodel = litmodel.to(self.config.device) 
        tuning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
        return (litmodel, tuning_callback)
       
    def get_final_model(self, parameters, seed):
        """Get model for testing."""
        if parameters is not None:
            encoder = {
                'hidden_sizes': [parameters[f'hidden_size_{i}'] for i in range(3)],
                'activation': parameters['activation'],
                'dropout_p': parameters['dropout_p']
            }
            training = {
                'batch_size': parameters['batch_size'],
                'lr': parameters['lr'],
                'weight_decay': parameters['weight_decay'],
                'optimizer': 'adam'
            }
            if self.config.n_basis_tunable:
                n_basis = parameters['n_basis']
            else:
                n_basis = self.config.n_basis

            config = Config(n_features=self.config.n_features,
                            n_static_features=self.config.n_static_features,
                            n_ts_length=self.config.ts_length,
                            n_basis=n_basis,
                            T=self.config.T,
                            seed=seed,
                            encoder=encoder,
                            training=training,
                            dataloader_type=self.config.dataloader_type,
                            device=self.config.device,
                            num_epochs=self.config.num_epochs,
                            internal_knots=self.config.internal_knots,
                            n_basis_tunable=self.config.n_basis_tunable,
                            dynamic_bias=self.config.dynamic_bias
                            )
        else:
            config = self.config

        litmodel = LitTTS(config)
        return litmodel

    

  

   
    def prepare_data(self, dataset, train_indices, val_indices, test_indices):
        """
        Prepares the data by splitting into train, validation, and test sets,
        applying transformations, and scaling the features.
        """
        # Get static features and time-series features from the dataset
        X_static, X_ts, ys = dataset.get_X_ts_ys()

        
        X_static = np.array(X_static)
        
        # Split data for training, validation, and test sets
        X_static_train = X_static[train_indices]  # Use .iloc for correct indexing on DataFrame
        X_static_val = X_static[val_indices]
        X_static_test = X_static[test_indices]

        
        X_ts_train = [X_ts[i] for i in train_indices]
        X_ts_val = [X_ts[i] for i in val_indices]
        X_ts_test = [X_ts[i] for i in test_indices]
        
        ys_train = [ys[i] for i in train_indices]
        ys_val = [ys[i] for i in val_indices]
        ys_test = [ys[i] for i in test_indices]

        # Scale static features
        static_scaler = StandardScaler()
        X_static_train_scaled = static_scaler.fit_transform(X_static_train)
        X_static_val_scaled = static_scaler.transform(X_static_val)
        X_static_test_scaled = static_scaler.transform(X_static_test)

        # Scale time-series data (you can modify this if you need more advanced handling)
        ts_scaler = StandardScaler()
        X_ts_train_scaled = [ts_scaler.fit_transform(np.array(ts)) for ts in X_ts_train]
        X_ts_val_scaled = [ts_scaler.transform(np.array(ts)) for ts in X_ts_val]
        X_ts_test_scaled = [ts_scaler.transform(np.array(ts)) for ts in X_ts_test]

        # Store the processed data in the object
        self.X_static_train = X_static_train_scaled
        self.X_static_val = X_static_val_scaled
        self.X_static_test = X_static_test_scaled

        self.X_ts_train = X_ts_train_scaled
        self.X_ts_val = X_ts_val_scaled
        self.X_ts_test = X_ts_test_scaled

        self.ys_train = ys_train
        self.ys_val = ys_val
        self.ys_test = ys_test

        # Store the time-series indices if needed (ts)
        self.ts_train = [np.arange(len(y)) for y in ys_train]  # Or any custom logic you need
        self.ts_val = [np.arange(len(y)) for y in ys_val]
        self.ts_test = [np.arange(len(y)) for y in ys_test]
        
        # Now create datasets that are compatible with DataLoader
        self.train_dataset = TTSDataset(self.config,self.X_static_train, self.X_ts_train, self.ys_train, self.ts_train)
        self.val_dataset = TTSDataset(self.config,self.X_static_val, self.X_ts_val, self.ys_val, self.ts_val)
        self.test_dataset = TTSDataset(self.config,self.X_static_test, self.X_ts_test, self.ys_test, self.ts_test)

        

        



    def train(self, model, tuning=False):
        """Train model."""
        
        if tuning:
            log_dir = os.path.join(self.benchmarks_dir, self.name, 'tuning', 'logs', f'seed_{model.config.seed}')
        else:
            log_dir =  os.path.join(self.benchmarks_dir, self.name, 'final', 'logs', f'seed_{model.config.seed}')


        tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)

        # Create folder if does not exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save config as a pickle file
        config = model.config
        if not tuning:
            with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
                pickle.dump(config, f)

        # Create callbacks
        best_val_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename='best_val'
        )

        # Early stopping callback if validation loss doesn't improve
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=False,
            mode='min'
        )
        callback_ls = [best_val_checkpoint, early_stop_callback]
        
        
        trainer_dict = {
            'deterministic': True,
            'devices': 1,
            'enable_model_summary': False,
            'enable_progress_bar': False,
            'accelerator': config.device,
            'max_epochs': config.num_epochs,
            'logger': tb_logger,
            'check_val_every_n_epoch': 10,
            'log_every_n_steps': 1,
            'callbacks': callback_ls,
            
        }

        trainer = pl.Trainer(
            **trainer_dict
        )
        
        train_dataloader = create_dataloader(model.config, self.train_dataset, None, shuffle=True)
        val_dataloader = create_dataloader(model.config, self.val_dataset, None, shuffle=False)
        test_dataloader = create_dataloader(model.config, self.test_dataset, None, shuffle=False)

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

        print(f"Finished after {trainer.current_epoch} epochs.")

        train_loss = trainer.logged_metrics['train_loss']
        print(f"train_loss: {train_loss}")
        val_loss = early_stop_callback.best_score
        test_loss = 0

        if not tuning:
            results = trainer.test(model=model, dataloaders=test_dataloader)
            test_loss = results[0]['test_loss']

        return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}

 
def load_dataset(dataset_name):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the script
    dataset_path = os.path.join(script_dir,dataset_name+".pkl")  # Construct relative path

    if not os.path.exists(dataset_path):
        raise ValueError(f"A dataset description file with this name does not exist at {dataset_path}.")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)  # Assuming the dataset is pickled
    return dataset

def make_json_serializable(dictionary):
    if is_json_serializable(dictionary):
        return dictionary
    else:
        for key, value in dictionary.items():
            if is_json_serializable(value):
                continue
            elif isinstance(value, dict):
                dictionary[key] = make_json_serializable(value)
            else:
                dictionary[key] = {
                    'class': value.__class__.__name__,
                    'value': make_json_serializable(value.__dict__) if hasattr(value, '__dict__') else str(value)
                }
    return dictionary



def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
    
def get_baseline(name, parameter_dict=None):
    class_name = name + 'Benchmark'
    return globals()[class_name](**parameter_dict)