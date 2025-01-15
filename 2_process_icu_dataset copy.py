import os
import pandas as pd
import numpy as np
import pickle
import library
def generate_dataset_from_files(directory, save_path='icu_data.pkl'):
    """
    Generates a dataset using files in the specified directory.
    The dataset includes static features (age, gender), input time series (RESP, SpO2),
    and the target series (HR). Missing values are interpolated.

    Args:
        directory (str): Path to the directory containing the Fix and Numerics files.
        save_path (str): Path to save the generated dataset as a pickle file.

    Returns:
        None
    """
    data = []  # List to store the processed samples

    # Process files and extract relevant data
    results = process_files(directory)

    for result in results:
        # Static features
        age = result['age']
        gender_binary = result['gender_binary']
        
        if age >= 0:  # Skip samples with missing age
            # Data from the CSV
            df = result['data']

            # Interpolate missing values in the CSV data
            df.interpolate(method='linear', inplace=True)
            # Forward-fill remaining NaN values
            df.ffill(inplace=True)

            # Backward-fill any remaining NaN values
            df.bfill(inplace=True)

            # Extract RESP, SpO2 as inputs, and HR as the target series
            
            def resample_series(series, target_length=50):
                """
                Resamples a series to a fixed number of samples using linear interpolation.

                Args:
                    series (array-like): The original series.
                    target_length (int): The desired number of samples.

                Returns:
                    np.ndarray: The resampled series.
                """
                original_indices = np.linspace(0, 1, len(series))
                target_indices = np.linspace(0, 1, target_length)
                return np.interp(target_indices, original_indices, series)

            # Extract and resample each series
            resp_series = resample_series(df[' RESP'].values, target_length=100)
            spo2_series = resample_series(df[' SpO2'].values, target_length=100)
            hr_series = resample_series(df[' HR'].values, target_length=100)

            # Create a single sample
            sample = {
                'time_series': {
                    '1': resp_series.tolist(),  # Input series 1
                    '2': spo2_series.tolist()  # Input series 2
                },
                'static_features': [age, gender_binary],
                'target_series': hr_series.tolist()  # Output series
            }
            data.append(sample)
            
   

    # Save the dataset as a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Dataset saved to {save_path}")
    create_datset(save_path)  # Create and save the dataset
    

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


# Supporting functions
def process_files(directory):
    results = []

    # Iterate through all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith("_Fix.txt"):
            # Extract age and gender from Fix file
            fix_file_path = os.path.join(directory, file_name)
            age, gender_binary = extract_age_and_gender(fix_file_path)
            
            # Look for the corresponding Numerics CSV file
            base_name = file_name.replace("_Fix.txt", "")
            csv_file_name = f"{base_name}_Numerics.csv"
            csv_file_path = os.path.join(directory, csv_file_name)
            
            if os.path.exists(csv_file_path):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_file_path)
                results.append({
                    "file_base_name": base_name,
                    "age": age,
                    "gender_binary": gender_binary,
                    "data": df
                })
            else:
                print(f"Warning: Corresponding CSV file not found for {fix_file_path}")

    return results

def extract_age_and_gender(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract age
    age_line = next(line for line in content.splitlines() if "Age:" in line)
    age_str = age_line.split(":")[1].strip()
    if age_str.endswith("+"):
        age = 90  # Handle ages like "90+"
    else:
        age = int(age_str)
    
    # Extract gender
    gender_line = next(line for line in content.splitlines() if "Gender:" in line)
    gender = gender_line.split(":")[1].strip()
    gender_binary = 1 if gender == 'M' else 0

    return age, gender_binary

if __name__ == '__main__':
     ##Change directory to your own directory
    directory = "/data/sr933/physionet/physionet.org/files/bidmc/1.0.0/bidmc_csv"
    generate_dataset_from_files(directory, save_path='icu_data.pkl')
