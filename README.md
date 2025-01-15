## mTIMEVIEW

## Environment Setup
1. Ensure Conda is installed.
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate <environment_name>

## Download data from Physionet using
1. Run the following command to download the real dataset:

   ```bash
   wget -r -N -c -np https://physionet.org/files/bidmc/1.0.0/

2. Change the paths in your 2_process_icu_datset.py script to point to the CSV files folder in the downloaded dataset.

## Run scripts
1. Run the run_all.sh file
2. Alternatively, run the scripts individually changing the datasets and hyperparameters as desired.

