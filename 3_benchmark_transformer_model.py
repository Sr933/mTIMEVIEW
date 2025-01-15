import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import pandas as pd
import numpy as np
import time
import argparse

# Dataset Definition
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        """
        Dataset for multiple time-series data combined with static features and target series.
        """
        self.time_series = torch.stack([
            torch.stack([torch.tensor(sample['time_series'][str(key)], dtype=torch.float32)
                         for key in sorted(sample['time_series'].keys())], dim=1)
            for sample in data
        ])  # Shape: (num_samples, seq_len, n_time_series)
        
            
        self.static_features = torch.tensor([sample['static_features'] for sample in data], dtype=torch.float32)
        self.target_series = torch.tensor([sample['target_series'] for sample in data], dtype=torch.float32)

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx], self.static_features[idx], self.target_series[idx]


# Transformer-based Model Definition
class TransformerTimeSeriesPredictor(nn.Module):
    def __init__(self, time_series_length, n_time_series, static_feature_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerTimeSeriesPredictor, self).__init__()

        # Embedding for time-series input and static features
        self.embedding_time_series = nn.Linear(n_time_series, d_model)
        self.embedding_static = nn.Linear(static_feature_dim, d_model)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, time_series_length + 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output projection
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, time_series, static_features):
        batch_size, seq_len, n_time_series = time_series.size()

        # Embed time-series
        time_series_embedded = self.embedding_time_series(time_series)  # (batch_size, seq_len, d_model)

        # Embed static features
        static_embedded = self.embedding_static(static_features).unsqueeze(1)  # (batch_size, 1, d_model)

        # Concatenate static features as a token to the time-series
        combined_sequence = torch.cat([time_series_embedded, static_embedded], dim=1)  # (batch_size, seq_len+1, d_model)

        # Add positional encoding
        combined_sequence += self.positional_encoding[:, :combined_sequence.size(1), :]

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(combined_sequence)  # (batch_size, seq_len+1, d_model)

        # Take the last token's output (static token)
        static_token_output = transformer_output[:, -1, :]  # (batch_size, d_model)

        # Final output projection
        output = self.output_layer(static_token_output)  # (batch_size, output_dim)

        return output


# Training Loop
def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    """
    Trains the model and evaluates on the validation set.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for time_series, static_features, target_series in train_loader:
            time_series, static_features, target_series = (
                time_series.to(device),
                static_features.to(device),
                target_series.to(device),
            )

            optimizer.zero_grad()
            predictions = model(time_series, static_features)
            loss = criterion(predictions, target_series)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for time_series, static_features, target_series in val_loader:
                time_series, static_features, target_series = (
                    time_series.to(device),
                    static_features.to(device),
                    target_series.to(device),
                )
                predictions = model(time_series, static_features)
                loss = criterion(predictions, target_series)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")


# Main Function
def main(pkl_path, batch_size=64, num_epochs=100, lr=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load data from pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Create dataset and split
    dataset = TimeSeriesDataset(data)
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = test_size = (total_samples - train_size) // 2
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model parameters
    time_series_length = dataset.time_series.size(1)
    n_time_series = dataset.time_series.size(2)
    static_feature_dim = dataset.static_features.size(1)
    output_dim = dataset.target_series.size(1)

    model = TransformerTimeSeriesPredictor(
        time_series_length=time_series_length,
        n_time_series=n_time_series,
        static_feature_dim=static_feature_dim,
        d_model=10,
        nhead=2,
        num_layers=2,
        output_dim=output_dim
    )

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, lr, device)

    # Evaluate on test set
    test_loss = 0
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for time_series, static_features, target_series in test_loader:
            time_series, static_features, target_series = (
                time_series.to(device),
                static_features.to(device),
                target_series.to(device),
            )
            predictions = model(time_series, static_features)
            loss = criterion(predictions, target_series)
            test_loss += loss.item()

    final_loss = test_loss / len(test_loader)
    print(f"Test Loss: {final_loss:.4f}")
    return final_loss


def run_model_with_seed(path, seed, num_epochs):
    """
    Run the model with a specific seed and return the loss and runtime.
    """
    np.random.seed(seed)  # Set the random seed
    start_time = time.time()  # Record the start time
    
    # Assuming main() returns the loss of the model on the dataset
    loss = main(path, num_epochs=num_epochs)  # Replace with your model's actual function
    
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate runtime
    
    return loss, runtime

if __name__ == "__main__":
    # Paths to the datasets
    parser = argparse.ArgumentParser(description="Set the dataset name")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--num_epochs', type=int, required=True, help="Number of epochs")
    # Parse arguments
    args = parser.parse_args()

    # Use the dataset_name from the command-line argument
    dataset_name = args.dataset_name
    num_epochs = args.num_epochs
    path = dataset_name[:-3] + ".pkl"
    num_seeds = 5 # Define the number of seeds to run
    
    

    print(f"Running model on dataset: {path}")
    
    seed_losses = []
    seed_runtimes = []
    
    for seed in range(num_seeds):
        loss, runtime = run_model_with_seed(path, seed, num_epochs)
        seed_losses.append(loss)
        seed_runtimes.append(runtime)
    
    # Calculate mean and standard deviation of loss and average runtime for the dataset
    mean_loss = np.mean(seed_losses)
    std_loss = np.std(seed_losses)
    avg_runtime = np.mean(seed_runtimes)
    std_runtime = np.std(seed_runtimes)
    # Append the results to the losses and runtimes lists
    losses={'Dataset': [path], 'Mean Loss': [mean_loss], 'Std Loss': [std_loss]}
    runtimes={'Dataset': [path], 'Average Runtime (seconds)': [avg_runtime], 'Std Runtime': [std_runtime]}

    # Create DataFrames from the results
    loss_df = pd.DataFrame(losses)
    runtime_df = pd.DataFrame(runtimes)

    # Specify the CSV file paths
    loss_csv_file_path = dataset_name +'_transformer_losses.csv'
    runtime_csv_file_path =dataset_name + '_transformer_runtimes.csv'

    # Save the DataFrames to CSV files
    loss_df.to_csv(loss_csv_file_path, index=False)
    runtime_df.to_csv(runtime_csv_file_path, index=False)

    print(f"Results saved to {loss_csv_file_path} and {runtime_csv_file_path}")
