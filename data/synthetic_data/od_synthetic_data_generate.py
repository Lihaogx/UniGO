import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
from functools import partial
import sys
import h5py
import gc
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from community import community_louvain
import itertools
base_config = {
    # Network parameters
    'n': 1000,  # Number of nodes
    'graph_type': 'ba',  # Graph type: 'ba', 'er', 'ws', 'regular', or 'real'
    'm': 3,  # m parameter for BA model
    'p': 0.01,  # Edge probability for ER model or rewiring probability for WS model
    'k': 4,  # Number of neighbors for WS model

    # Simulation parameters
    'num_graphs': 1,  # Number of graphs to generate
    'od_model': ['fj', 'hk'],  # Opinion dynamics model: 'degroot', 'fj', 'hk', 'hk_hetero', 'hk_noise', 'dw', or 'community'
    'iteration': 100,  # Number of iterations

    # Model-specific parameters
    'epsilon': 0.2,  # Confidence threshold for HK model
    'mu': 0.5,  # Influence factor for FJ model
    'p_noise': 0.1,  # Noise probability for noise model
    'confidence_range': [0.1, 0.3],  # Confidence interval for heterogeneous HK model

    # Initial opinion settings
    'random_initial_opinion': True,  # Whether to randomly generate initial opinions
    'initial_opinions': None,  # If not random, provide initial opinion array here

    # Real network settings
    'use_real_network': False,  # Whether to use real network
    'real_network_path': None,  # Path to real network file

    # Time window parameters
    'lookback': 10,  # Length of x time steps
    'horizon': 5,  # Length of y time steps
}
# Add project root directory to system path
sys.path.append('./UniGO/')
from utils import load_config

def create_graph_with_opinions(n, graph_type='ba', m=2, p=0.1, k=4, seed=None):
    """
    Generate various types of networks with random opinion values.
    
    Parameters:
        n (int): Number of nodes in the network.
        graph_type (str): Network type ('ba', 'er', 'ws', 'regular').
        m (int): Number of edges to add for each new node in BA model, or degree for each node in regular graph.
        p (float): Edge probability for ER model or rewiring probability for WS model.
        k (int): Number of neighbors for each node in WS model.
        seed (int, optional): Random number generator seed.
    
    Returns:
        G (networkx.Graph): Network with 'opinion' attribute.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if graph_type == 'ba':
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif graph_type == 'er':
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif graph_type == 'ws':
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    elif graph_type == 'regular':
        G = nx.random_regular_graph(m, n, seed=seed)
    else:
        raise ValueError(f"Unsupported network type: {graph_type}")
    
    for node in G.nodes():
        G.nodes[node]['opinion'] = np.random.uniform(0, 1)
    
    return G

def initialize_model_parameters(G, od_model, epsilon=0.2, mu=0.5, p=0.1, confidence_range=[0.2, 0.6], random_initial_opinion=True, initial_opinions=None):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Initialize basic parameters
    for i, v in enumerate(nodes):
        if random_initial_opinion:
            G.nodes[v]['initial_opinion'] = np.random.uniform(0, 1)
        else:
            if initial_opinions is None or len(initial_opinions) != n:
                raise ValueError("When random_initial_opinion is False, must provide initial_opinions array of correct length")
            G.nodes[v]['initial_opinion'] = initial_opinions[i]
        
        G.nodes[v]['opinion'] = G.nodes[v]['initial_opinion']
        G.nodes[v]['alpha'] = 0    # Default stubbornness
        G.nodes[v]['epsilon'] = 1.0  # Default confidence threshold, accepting all neighbor influences
        G.nodes[v]['beta'] = 1.0
    
    # Set specific parameters based on different models
    if 'degroot' in od_model:
        pass  # DeGroot model doesn't need additional parameters
    
    elif 'fj' in od_model:
        for v in nodes:
            G.nodes[v]['alpha'] = 1 - mu
    
    elif 'hk' in od_model:
        for v in nodes:
            G.nodes[v]['epsilon'] = epsilon
    
    elif 'dw' in od_model:
        for v in nodes:
            G.nodes[v]['epsilon'] = epsilon
    
    elif 'noise' in od_model:
        for v in nodes:
            G.nodes[v]['beta'] = 1 - p
    
    elif 'hetero' in od_model:
        for v in nodes:
            G.nodes[v]['epsilon'] = np.random.uniform(confidence_range[0], confidence_range[1])
            G.nodes[v]['alpha'] = np.random.uniform(0, 1)
    
    elif 'community' in od_model:
        communities = community_louvain.best_partition(G)
        community_params = {}
        for community in set(communities.values()):
            community_params[community] = {
                'epsilon': np.random.uniform(confidence_range[0], confidence_range[1]),
                'alpha': np.random.uniform(0, 1)
            }
        for v in nodes:
            community = communities[v]
            G.nodes[v]['epsilon'] = community_params[community]['epsilon']
            G.nodes[v]['alpha'] = community_params[community]['alpha']
    
    else:
        raise ValueError(f"Unsupported model: {od_model}")
    
    return G

def model(G, eta, iteration):
    """
    Run opinion dynamics using unified model
    
    Parameters:
    G: Network graph, each node should contain the following attributes:
       'opinion': Current opinion
       'initial_opinion': Initial opinion
       'alpha': Stubbornness
       'epsilon': Confidence threshold
       'beta': Resistance to random perturbations
    eta: Random perturbation array, shape (iteration, num_nodes)
    iteration: Number of iterations
    
    Returns:
    opinion_df: DataFrame containing opinions of all nodes at each time step
    convergence_step: Convergence step number, -1 if not converged
    """
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    
    dic_all = {}
    convergence_count = 0
    convergence_step = -1
    
    for t in range(iteration):
        dic_all[t] = {v: G.nodes[v]['opinion'] for v in nodes}
        
        for v in nodes:
            G.nodes[v]['prev_opinion'] = G.nodes[v]['opinion']
        
        for v in nodes:
            neighbor_influence = 0
            effective_neighbors = [u for u in G.neighbors(v) if abs(G.nodes[u]['prev_opinion'] - G.nodes[v]['prev_opinion']) < G.nodes[v]['epsilon']]
            if effective_neighbors:
                neighbor_influence = np.mean([G.nodes[u]['prev_opinion'] for u in effective_neighbors])
            
            G.nodes[v]['opinion'] = (G.nodes[v]['alpha'] * G.nodes[v]['initial_opinion'] + 
                                     (1 - G.nodes[v]['alpha']) * neighbor_influence + 
                                     (1 - G.nodes[v]['beta']) * eta[t, v])
            
            G.nodes[v]['opinion'] = max(0, min(1, G.nodes[v]['opinion']))
        
        opinion_diff = np.mean([abs(G.nodes[v]['opinion'] - G.nodes[v]['prev_opinion']) for v in nodes])
        if opinion_diff < 1e-6:
            convergence_count += 1
            if convergence_count == 5 and convergence_step == -1:
                convergence_step = t - 3
        else:
            convergence_count = 0
    
    opinion_df = pd.DataFrame(dic_all).T
    return opinion_df, convergence_step

def process_and_save_data(data_item, h5f, idx, k=4):
    """
    Process a single data item and save to HDF5 file
    """
    try:
        # print(f"Processing data item {idx}, convergence step: {data_item.convergence_step}")
        if data_item.convergence_step == -1 or data_item.convergence_step > args.convergence_step:
            # Reconstruct time series
            x = data_item.x  # [36, 1000, 15]
            y = data_item.y  # [36, 1000, 50]
            # print(f"Data shape: x={x.shape}, y={y.shape}")
            x0 = x[0]  # [1000, 15]
            y0 = y[0]  # [1000, 50]
            y35 = y[35]  # [1000, 50]
            y35_last35 = y35[:, -35:]  # [1000, 35]
            time_series = torch.cat([x0, y0, y35_last35], dim=1)  # [1000, 100]
            # print(f"Constructed time series: shape={time_series.shape}")

            if time_series.shape[1] != 100:
                # print(f"Time series dimension error: {time_series.shape[1]} != 100")
                return False

            x_input = time_series[:, :10]  # [1000, 10]
            y_output = time_series[:, 10:]  # [1000, 90]
            # print(f"Input/output shape: x_input={x_input.shape}, y_output={y_output.shape}")

            # Clustering processing
            # print(f"Starting KMeans clustering, k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(time_series.numpy())
            # print(f"Clustering completed, label distribution: {np.bincount(labels)}")
            
            cluster_idx = [[] for _ in range(k)]
            for idx_node, label in enumerate(labels):
                cluster_idx[label].append(idx_node)

            cluster_node_indices = []
            cluster_ptr = [0]
            for c in cluster_idx:
                cluster_node_indices.extend(c)
                cluster_ptr.append(cluster_ptr[-1] + len(c))
            # print(f"Cluster pointers: {cluster_ptr}")

            cluster_node_indices = torch.tensor(cluster_node_indices, dtype=torch.long)
            cluster_ptr = torch.tensor(cluster_ptr, dtype=torch.long)

            # Save to HDF5 file
            # print(f"Starting save to HDF5, group name: data_{idx}")
            group = h5f.create_group(f'data_{idx}')
            group.create_dataset('x', data=x_input.numpy(), compression="gzip")
            group.create_dataset('y', data=y_output.numpy(), compression="gzip")
            group.create_dataset('edge_index', data=data_item.edge_index.numpy(), compression="gzip")
            group.create_dataset('cluster_node_indices', data=cluster_node_indices.numpy(), compression="gzip")
            group.create_dataset('cluster_ptr', data=cluster_ptr.numpy(), compression="gzip")
            group.attrs['convergence_step'] = data_item.convergence_step
            # print(f"Data item {idx} saved successfully")
            
            return True
        else:
            # print(f"Data item {idx} does not meet conditions, convergence step is {data_item.convergence_step}")
            return False
    except Exception as e:
        # print(f"Error processing data item {idx}: {e}")
        # import traceback
        # traceback.print_exc()
        return False
    finally:
        # print(f"Cleaning up memory for data item {idx}")
        gc.collect()

def main(n, graph_type, lookback, horizon, m, p, k, num_graphs, od_model, iteration, epsilon, mu, p_noise, confidence_range, 
         random_initial_opinion=True, initial_opinions=None, use_real_network=False, real_network_path=None, output_dir="./UniGO/data/synthetic_data"):
    """
    Main function to generate data and save directly in HDF5 format
    """
    # Generate filename
    if graph_type == 'ba':
        graph_params = f"ba_n{n}_m{m}"
    elif graph_type == 'er':
        graph_params = f"er_n{n}_p{p:.2f}"
    elif graph_type == 'ws':
        graph_params = f"ws_n{n}_k{k}_p{p:.2f}"
    elif graph_type == 'regular':
        graph_params = f"regular_n{n}_m{m}"
    elif graph_type == 'real':
        graph_params = f"real_n{n}"
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    od_params = []
    if 'degroot' in od_model:
        od_params.append("degroot")
    if 'fj' in od_model:
        od_params.append(f"fj_mu{mu:.2f}")
    if 'hk' in od_model:
        od_params.append(f"hk_epsilon{epsilon:.2f}")
    if 'dw' in od_model:
        od_params.append(f"dw_epsilon{epsilon:.2f}")
    if 'noise' in od_model:
        od_params.append(f"noise_p{p_noise:.2f}")
    if 'hetero' in od_model:
        od_params.append(f"hetero_epsilon[{confidence_range[0]:.2f},{confidence_range[1]:.2f}]")
    if 'community' in od_model:
        od_params.append(f"community_epsilon[{confidence_range[0]:.2f},{confidence_range[1]:.2f}]")

    if not od_params:
        raise ValueError(f"Unsupported model: {od_model}")

    od_model_str = "_".join(od_model)
    od_params_str = "_".join(od_params)

    # Create output directory
    base_dir = os.path.join(output_dir, graph_params, od_model_str)
    os.makedirs(base_dir, exist_ok=True)

    # Create HDF5 file
    h5_file = os.path.join(base_dir, f"{od_params_str}_{num_graphs}graphs.h5")
    idx = 0
    valid_count = 0

    with h5py.File(h5_file, 'w') as h5f:
        for i in tqdm(range(num_graphs), desc="Generating graphs"):
            if use_real_network:
                if real_network_path is None:
                    raise ValueError("Must provide real_network_path when using real network")
                G = nx.read_edgelist(real_network_path)
                n = G.number_of_nodes()
                graph_type = "real"
            else:
                G = create_graph_with_opinions(n, graph_type, m, p, k)
            
            G = initialize_model_parameters(G, od_model, epsilon, mu, p_noise, confidence_range, 
                                            random_initial_opinion=random_initial_opinion, 
                                            initial_opinions=initial_opinions)
            
            eta = np.random.uniform(0, 1, size=(iteration, n))
            
            opinion_df, convergence_step = model(G, eta, iteration)
            
            edge_index = torch.tensor(list(G.edges())).t().contiguous().long()
            
            x_values = []
            y_values = []
            for t in range(iteration - lookback - horizon + 1):
                x_values.append([opinion_df.iloc[t:t+lookback][node].tolist() for node in G.nodes()])
                y_values.append([opinion_df.iloc[t+lookback:t+lookback+horizon][node].tolist() for node in G.nodes()])
            
            x = torch.tensor(x_values, dtype=torch.float)
            y = torch.tensor(y_values, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y, convergence_step=convergence_step)
            
            if process_and_save_data(data, h5f, idx):
                valid_count += 1
            idx += 1

    print(f"Generation completed, saved {valid_count} valid data items to {h5_file}")

def generate_and_filter_parameter_grid(param_ranges):
    """
    Generate parameter grid and filter parameters based on od_model, removing duplicate parameter combinations
    """
    param_names = sorted(param_ranges)
    param_values = [param_ranges[name] for name in param_names]
    
    unique_params = []
    seen_combinations = set()
    
    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        od_model = params['od_model']
        
        filtered_params = {}
        if 'fj' in od_model:
            filtered_params['mu'] = params['mu']
        else:
            params['mu'] = None
        
        if 'hk' in od_model or 'dw' in od_model:
            filtered_params['epsilon'] = params['epsilon']
        else:
            params['epsilon'] = None
        
        if 'noise' in od_model:
            filtered_params['p_noise'] = params['p_noise']
        else:
            params['p_noise'] = None
        
        if 'hetero' in od_model or 'community' in od_model:
            filtered_params['confidence_range'] = params['confidence_range']
        else:
            params['confidence_range'] = None
        
        combination_key = (params['n'], params['graph_type'], params['lookback'], params['horizon'], 
                           params['m'], params['p'], params['k'], params['num_graphs'], tuple(map(tuple, params['od_model'])), 
                           params['iteration'], tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in filtered_params.items()), 
                           params['random_initial_opinion'], params['use_real_network'])
        
        if combination_key not in seen_combinations:
            seen_combinations.add(combination_key)
            unique_params.append(params)
    
    return unique_params

def process_config(params, base_config):
    """
    Process a single configuration
    """
    config = base_config.copy()
    config.update(params)
    
    required_keys = ['epsilon', 'mu', 'p_noise', 'confidence_range']
    for key in required_keys:
        if key not in config:
            config[key] = None
    
    print(f"Processing parameter combination: {params}")
    main(**config)
    print(f"Parameter combination processing completed: {params}")

def process_parameter_grid(base_config, param_ranges, use_threading=True):
    """
    Process parameter grid
    """
    unique_params = generate_and_filter_parameter_grid(param_ranges)
    
    if use_threading:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)
        process_func = partial(process_config, base_config=base_config)
        pool.map(process_func, unique_params)
        pool.close()
        pool.join()
    else:
        for params in unique_params:
            process_config(params, base_config)

def merge_h5_files(input_dir, output_file):
    """
    Merge all HDF5 files into a unified file
    
    Parameters:
        input_dir: Input directory containing all HDF5 files to merge
        output_file: Path to output file
    """
    # Get all HDF5 files
    h5_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.h5') and not file.endswith('_merged.h5'):
                h5_files.append(os.path.join(root, file))
    
    if not h5_files:
        print(f"No HDF5 files found in directory {input_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files, starting merge...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Merge files
    total_items = 0
    with h5py.File(output_file, 'w') as h5_out:
        for h5_file in tqdm(h5_files, desc="Merging files"):
            try:
                with h5py.File(h5_file, 'r') as h5_in:
                    # Get all groups
                    groups = list(h5_in.keys())
                    
                    # Copy each group to output file
                    for group_name in groups:
                        # Create new group name to avoid conflicts
                        new_group_name = f"data_{total_items}"
                        h5_in.copy(group_name, h5_out, name=new_group_name)
                        total_items += 1
            except Exception as e:
                print(f"Error processing file {h5_file}: {e}")
                continue
    
    print(f"Merge completed, merged {total_items} data items into {output_file}")

def process_multiple_configs(param_ranges=None, merge_output=True, merged_file_name='merged_data.h5'):
    """
    Process multiple configurations
    
    Parameters:
        param_ranges: Parameter grid range definitions
        merge_output: Whether to merge output files
    """
    process_parameter_grid(base_config, param_ranges)
    # If merge is enabled, merge all generated HDF5 files
    if merge_output:
        output_dir = base_config.get('output_dir', './UniGO/data/synthetic_data')
        merged_file = os.path.join(output_dir, merged_file_name)
        merge_h5_files(output_dir, merged_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process parameter grid.')
    parser.add_argument('--convergence_step', type=int, default=80,
                        help='Lower bound for convergence steps')
    parser.add_argument('--merge_output', action='store_true', default=True,
                        help='Whether to merge output files')
    parser.add_argument('--merged_file_name', type=str, default='merged_data.h5',
                        help='Path to merged file')
    args = parser.parse_args()

    # param_ranges = {
    #     'n': [500],
    #     'graph_type': ['ba', 'er', 'ws'],
    #     'lookback': [15],
    #     'horizon': [50],
    #     'm': [2, 3, 4, 5],
    #     'p': [0.1, 0.2, 0.3],
    #     'k': [4, 6, 8],
    #     'num_graphs': [1],
    #     'od_model': [['degroot'], ['fj'], ['hk'], ['hetero'], ['community'], ['fj', 'hk'], ['fj', 'noise'], ['fj', 'hetero'], ['fj', 'community'], ['hk', 'noise'], ['fj', 'hk', 'noise']],
    #     'iteration': [100],
    #     'epsilon': [0.1, 0.2, 0.3, 0.4],
    #     'mu': [0.3, 0.4, 0.5, 0.6, 0.7],
    #     'p_noise': [0.05, 0.1, 0.15, 0.2, 0.25],
    #     'confidence_range': [[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]],
    #     'random_initial_opinion': [True],
    #     'use_real_network': [False],
    # }
    param_ranges = {
        'n': [500],
        'graph_type': ['ba'],
        'lookback': [15],
        'horizon': [50],
        'm': [2],
        'p': [0.1],
        'k': [4],
        'num_graphs': [1],
        'od_model': [['degroot'], ['fj']],
        'iteration': [100],
        'epsilon': [0.1],
        'mu': [0.3],
        'p_noise': [0.05],
        'confidence_range': [[0.1, 0.3]],
        'random_initial_opinion': [True],
        'use_real_network': [False],
    }
    process_multiple_configs(param_ranges, args.merge_output, args.merged_file_name)