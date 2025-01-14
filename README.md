## mTIMEVIEW

## Environment Setup
1. Ensure Conda is installed.
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate <environment_name>

## Download data from Physionet using
1.Run the following command to download the real dataset:

wget -r -N -c -np https://physionet.org/files/bidmc/1.0.0/

2Change the paths in your scripts to point to the CSV files in the downloaded dataset.

## Run scripts in order

