import numpy as np
import pickle
import library


def generate_synthetic_data_gauss(num_samples=1000, time_steps=25, save_path='gaussian_data.pkl'):
    """
    Generates synthetic data for a dataset with time series and static features, based on Gaussian distribution
    with random mean and standard deviation, shifting them with random parameters.

    Args:
        num_samples (int): Number of samples to generate.
        time_steps (int): Number of time steps for each time series.
        save_path (str): Path to save the generated data as a pickle file.

    Returns:
        None
    """
    data = []  # List to store the synthetic data samples

    for _ in range(num_samples):
        # Randomly generate parameters for Gaussian distribution
        mean_shift = np.random.uniform(-5, 5)  # Shift for the mean
        std_shift = np.random.uniform(0.5, 2.0)  # Shift for the standard deviation

        # Use random base mean and std for Gaussian distribution
        mean = np.random.uniform(0, 10)  # Base mean of the Gaussian distribution
        std = np.random.uniform(0.5, 3.0)  # Base standard deviation of the Gaussian distribution

        # Apply the shifts
        shifted_mean = mean + mean_shift
        shifted_std = std + std_shift

        # Generate Gaussian time series using the shifted parameters
        t = np.linspace(-10, 10, time_steps)  # Time vector
        gauss_series = np.exp(-0.5 * ((t - mean) ** 2) / (std ** 2))  # Gaussian series

        gauss_series_shifted = np.exp(-0.5 * ((t - shifted_mean) ** 2) / (shifted_std ** 2)) 
        
        # Combine all data for this sample
        sample = {
            'time_series': {
                '1': gauss_series.tolist(),  # Gaussian distributed time series
            },
            'static_features': [mean_shift, std_shift],  # Static features: mean and std shifts
            'target_series': gauss_series_shifted.tolist(),  # Target data (same as the Gaussian series)
        }

        data.append(sample)  # Add this sample to the dataset

    # Save data as pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Synthetic Gaussian data saved to {save_path}")
    create_datset(save_path)  # Create and save the datase



def generate_synthetic_data_sin(num_samples=1000, time_steps=25, save_path='sin_data.pkl'):
    """
    Generates synthetic data for a dataset with time series and static features, based on sine function
    with random amplitude and frequency, shifting them with random parameters.

    Args:
        num_samples (int): Number of samples to generate.
        time_steps (int): Number of time steps for each time series.
        save_path (str): Path to save the generated data as a pickle file.

    Returns:
        None
    """
    data = []  # List to store the synthetic data samples

    for _ in range(num_samples):
        # Randomly generate parameters for sine function
        amplitude_shift = np.random.uniform(-2, 2)  # Shift for the amplitude
        freq_shift = np.random.uniform(-0.5, 0.5)  # Shift for the frequency

        # Use random base amplitude and frequency for sine wave
        amplitude = np.random.uniform(1, 5)  # Base amplitude of the sine wave
        frequency = np.random.uniform(0.1, 1.0)  # Base frequency of the sine wave

        # Apply the shifts
        shifted_amplitude = amplitude + amplitude_shift
        shifted_frequency = frequency + freq_shift

        # Generate sine time series using the shifted parameters
        t = np.linspace(0, 2 * np.pi, time_steps)  # Time vector over a full sine wave period
        sine_series = amplitude * np.sin(frequency * t)  # Base sine series

        sine_series_shifted = shifted_amplitude * np.sin(shifted_frequency * t)  # Target data (shifted)

        # Combine all data for this sample
        sample = {
            'time_series': {
                '1': sine_series.tolist(),  # Sine wave time series
            },
            'static_features': [amplitude_shift, freq_shift],  # Static features: amplitude and frequency shifts
            'target_series': sine_series_shifted.tolist(),  # Target data (shifted sine wave)
        }

        data.append(sample)  # Add this sample to the dataset

    # Save data as pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Synthetic Sine data saved to {save_path}")
    create_datset(save_path)  # Call to create and save the dataset, assuming it's a valid function


    

def create_datset(dataset_path):
    """
    This function loads the dataset, creates a SimpleDataset object, and saves it as a pickle file.

    Args:
        dataset_path (str): Path to the generated synthetic data.

    Returns:
        None
    """
    # Create a SimpleDataset object using the library
    dataset = library.SimpleDataset(dataset_path)

    # Save the dataset as a pickle file
    output_path = dataset_path[:-4] + "set.pkl"  
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {output_path}")



def generate_synthetic_data_exp(
    num_samples=1000, 
    time_steps=25, 
    save_path='exp_data.pkl'
):
    """
    Generates synthetic data where the task is to predict a modified sine wave 
    with an added or subtracted exponential series based on a static feature.

    Args:
        num_samples (int): Number of samples to generate.
        time_steps (int): Number of time steps for each time series.
        scale_param (float): Scale factor for the exponential component.
        save_path (str): Path to save the generated data as a pickle file.

    Returns:
        None
    """
    data = []  # List to store the synthetic data samples

    for _ in range(num_samples):
        # Generate the time vector
        t = np.linspace(0, 1, time_steps)

        # Generate random amplitude and frequency for the sine wave
        
        freq = np.random.uniform(0.1, 1.0)  # Random frequency
        amplitude= np.random.uniform(1, 5)  # Random amplitude
        scale_param = np.random.uniform(0.1, 1.0)  # Random scale parameter for the exponential component
        # Generate the sine wave
        sin_series = amplitude* np.sin(2 * freq * np.pi * t)

        # Generate the exponential component
        exp_series = scale_param * np.exp(t)

        # Randomly choose whether to add or subtract the exponential series
        operation = np.random.choice([0, 1])  # 0 for subtract, 1 for add

        # Modify the sine wave based on the operation
        if operation == 1:
            modified_series = sin_series + exp_series
        else:
            modified_series = sin_series - exp_series

        # Create a sample with time series, static features, and target series
        sample = {
            'time_series': {
                '1': sin_series.tolist(),  # Original sine wave (input)
                '2': exp_series.tolist()   # Exponential component
            },
            'static_features': [operation],  # Operation and sine wave parameters
            'target_series': modified_series.tolist()         # Modified sine wave (output)
        }

        data.append(sample)  # Add this sample to the dataset

    # Save the data as a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Synthetic data with exponential series saved to {save_path}")
    create_datset(save_path)  # Create and save the dataset







if __name__ == "__main__":
    # Generate synthetic datasets with different types of time-series data
    generate_synthetic_data_gauss(num_samples=500, time_steps=25)  # Combined sine and exponential data
    generate_synthetic_data_sin(num_samples=500, time_steps=25)  # Sine wave data
    generate_synthetic_data_exp(num_samples=500, time_steps=25)  # Exponential series data