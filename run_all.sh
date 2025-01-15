#Script to run all the scripts in the mTIMEVIEW pipeline

python 1_generate_synthetic_data.py

python 2_process_icu_dataset.py

python 3_benchmark_transformer_model.py --dataset_name "icu_dataset" --num_epochs 100
python 3_benchmark_transformer_model.py --dataset_name "sin_dataset" --num_epochs 100
python 3_benchmark_transformer_model.py --dataset_name "exp_dataset" --num_epochs 100
python 3_benchmark_transformer_model.py --dataset_name "gaussian_dataset" --num_epochs 100

python 4_benchmark_new_model.py --dataset_name "icu_dataset" --num_epochs 100
python 4_benchmark_new_model.py --dataset_name "sin_dataset" --num_epochs 100
python 4_benchmark_new_model.py --dataset_name "exp_dataset" --num_epochs 100
python 4_benchmark_new_model.py --dataset_name "gaussian_dataset" --num_epochs 100

