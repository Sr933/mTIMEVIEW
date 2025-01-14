import os
import numpy as np
import pandas as pd
import pickle
import json
import time
import copy 
from datetime import datetime

from library import load_dataset, make_json_serializable, get_baseline,Config   # Make sure these are properly defined or imported



def run_benchmarks(dataset_name, benchmarks: dict, dataset_split=[0.7, 0.15, 0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='benchmarks', dataset_description_path='dataset_descriptions', notes=''):
    """
    Runs a set of benchmarks on a dataset
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Check if there exists a file summary.csv in the benchmarks directory
    if not os.path.exists(benchmarks_dir):
        os.makedirs(benchmarks_dir)  # Create the directory if it does not exist

    if os.path.exists(os.path.join(benchmarks_dir, 'summary.csv')):
        df = pd.read_csv(os.path.join(benchmarks_dir, 'summary.csv'), index_col=0)
    else:
        df = pd.DataFrame(columns=['timestamp','dataset_name','n_trials', 'n_tune', 'train_size', 'val_size', 'seed'])

    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    df = pd.concat([df, pd.DataFrame({'timestamp': [timestamp], 'dataset_name': [dataset_name], 'n_trials': [n_trials], 
                                      'n_tune': [n_tune], 'train_size': [dataset_split[0]], 'val_size': [dataset_split[1]], 'seed': [seed]})], 
                   ignore_index=True)
    
    df.to_csv(os.path.join(benchmarks_dir, 'summary.csv'))

    # Check if there exists a file summary.json in the benchmarks directory
    if os.path.exists(os.path.join(benchmarks_dir, 'summary.json')):
        with open(os.path.join(benchmarks_dir, 'summary.json'), 'r') as f:
            summary = json.load(f)
    else:
        summary = []
        with open(os.path.join(benchmarks_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    summary.append(
        {
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'n_trials': n_trials,
            'n_tune': n_tune,
            'train_size': dataset_split[0],
            'val_size': dataset_split[1],
            'seed': seed,
            'results': {},
            'notes': notes
        }
    )

    with open(os.path.join(benchmarks_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Generate train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=seed)

    results = {'name':[], 'mean':[], 'std':[], 'time_elapsed':[]}

    # Prepare the benchmark directory for the current run
    benchmarks_dir = os.path.join(benchmarks_dir, timestamp)
    if not os.path.exists(benchmarks_dir):
        os.mkdir(benchmarks_dir)  # Ensure that the directory exists

    # Save the benchmarks to a pickle file
    with open(os.path.join(benchmarks_dir, 'baselines.pkl'), 'wb') as f:
        pickle.dump(benchmarks, f)
    
    # Save the benchmarks as a json file
    with open(os.path.join(benchmarks_dir, 'baselines.json'), 'w') as f:
        benchmarks_to_save = copy.deepcopy(benchmarks)
        json.dump(make_json_serializable(benchmarks_to_save), f, indent=4)
    
    # Run each benchmark
    for baseline_name, parameter_dict in benchmarks.items():
        time_start = time.time()
        benchmark = get_baseline(baseline_name, parameter_dict)
        losses = benchmark.run(dataset, train_indices, val_indices, test_indices, n_trials=n_trials, n_tune=n_tune, seed=seed, benchmarks_dir=benchmarks_dir, timestamp=timestamp)
        time_end = time.time()
        results['name'].append(benchmark.name)
        results['mean'].append(np.mean(losses))
        results['std'].append(np.std(losses))
        results['time_elapsed'].append(time_end - time_start)

        summary[-1]['results'][benchmark.name] = {'mean': np.mean(losses), 'std': np.std(losses), 'time_elapsed': time_end - time_start}

        # Save the summary
        with open(os.path.join(benchmarks_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

    # Create a dataframe with the results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(benchmarks_dir, 'results.csv'), index=False)

    return df


def generate_indices(n, train_size, val_size, seed=0):
    gen = np.random.default_rng(seed)
    train_indices = gen.choice(n, int(n * train_size), replace=False)
    train_indices = [i.item() for i in train_indices]
    val_indices = gen.choice(list(set(range(n)) - set(train_indices)), int(n * val_size), replace=False)
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n)) - set(train_indices) - set(val_indices))
    return train_indices, val_indices, test_indices

if __name__ == "__main__":
    dataset_name = 'icu_dataset'
    #dataset_names = ['icu_dataset', 'exp_dataset', 'sin_dataset', 'pol_dataset', 'gaussian_dataset']  # Example dataset name
    baselines = ['TTS']
    n_trials = 1
    n_tune = 1
    seed = 42
    n_basis = 9
    device = "cpu" 
    print(device)
    # TTS configuration
    tts_T = {'gaussian_dataset': 1.0,
             'sin_dataset': 1.0,
             'pol_dataset': 1.0,
             'exp_dataset': 1.0,
             'icu_dataset': 1.0}
    tts_n_features = {'gaussian_dataset': 1,
             'sin_dataset': 1,
             'pol_dataset': 1,
             'exp_dataset': 2,
             'icu_dataset': 2}
    tts_n_static_features = {'gaussian_dataset': 2,
             'sin_dataset': 2,
             'pol_dataset': 1,
             'exp_dataset': 1,
             'icu_dataset': 2}
    tts_ts_length = {'gaussian_dataset': 25,
                'sin_dataset': 25,
                'pol_dataset': 25,
                'exp_dataset': 25,
                'icu_dataset': 100}
    benchmark_options = {
        'n_trials': n_trials,
        'n_tune': n_tune,
        'seed': seed,
        'dataset_split': [0.7, 0.15, 0.15],
        'benchmarks_dir': 'benchmarks',
        'dataset_description_path': 'dataset_descriptions',
        'notes': f"n_basis={n_basis}"
    }

    
    print(f'Running benchmarks on dataset: {dataset_name}')
    config = Config(
    n_features=tts_n_features[dataset_name],
    n_static_features=tts_n_static_features[dataset_name],
    n_ts_length=tts_ts_length[dataset_name],
    n_basis=n_basis,
    T=tts_T[dataset_name],
    seed=seed,
    dataloader_type='iterative',
    num_epochs=100,
    device=device,
    n_basis_tunable=False,
    dynamic_bias=True
    )
    
    # Run the benchmarks
    run_benchmarks(dataset_name, benchmarks={'TTS': {'config': config}}, **benchmark_options)
