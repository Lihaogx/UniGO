import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data

lookback = 10
horizon = 90
k = 3

def assert_no_nan(df, description):
    if df.isnull().any().any():
        error_message = f"NaN values found after {description}: \n{df.isnull().sum()}"
        raise ValueError(error_message)

def assert_no_nan_in_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN values found in {name} tensor.")
        
def process_opinions_and_edges(folder_path):
    """
    Processes edge and opinion files within a given folder to create and save PyTorch Geometric Data objects.
    Each Data object is saved under 'data/real_data' with a name derived from the opinion file prefix.
    Missing opinion values are first backward filled from the next available time step, then forward filled from the last available time step.

    Args:
        folder_path (str): The path to the folder containing the edge and opinion files.
    """
    # Identify and read the edge information file
    edge_file = [f for f in os.listdir(folder_path) if f.endswith('_edges.txt')]
    if edge_file:
        edge_file_path = os.path.join(folder_path, edge_file[0])
        edges = pd.read_csv(edge_file_path, sep=',', header=None, names=['source', 'target'])
        edges['source'] = edges['source'].astype(str).str.strip("'")
        edges['target'] = edges['target'].astype(str).str.strip("'")
        
        
        # Strip single quotes from the source and target columns
        edges['source'] = edges['source'].apply(pd.to_numeric, errors='coerce')
        edges['target'] = edges['target'].apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with any NaN values (conversion failures)
        edges.dropna(inplace=True)
        
        
        unique_nodes = pd.unique(edges[['source', 'target']].values.ravel('K'))
        node_mapping = {old: new for new, old in enumerate(unique_nodes)}
        edges['source'] = edges['source'].map(node_mapping)
        edges['target'] = edges['target'].map(node_mapping)
        
        
        # Convert dataframe to long tensor
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    else:
        raise FileNotFoundError("No edge file found in the directory.")

    assert_no_nan(edges, "loading and cleaning edge data")
    # Gather all unique nodes mentioned in the edge file

    # Iterate over all opinion files
    opinion_files = [f for f in os.listdir(folder_path) if f.endswith('_opinion.txt')]
    for opinion_file in opinion_files:
        opinion_path = os.path.join(folder_path, opinion_file)
        opinions = pd.read_csv(opinion_path, sep=',', header=None, names=['node', 'time', 'opinion'])
        opinions['node'] = opinions['node'].astype(str).str.strip("'").astype('int64')
        opinions['time'] = opinions['time'].astype(int)
        opinions['opinion'] = opinions['opinion'].astype(float)
        
        assert_no_nan(opinions, "cleaning opinion data")
        
        # Handling duplicates by aggregating opinions
        opinions = opinions.groupby(['node', 'time']).agg({'opinion': 'mean'}).reset_index()
        
        # Create opinion mapping and handle missing values
        opinion_map = opinions.pivot(index='node', columns='time', values='opinion').reindex(unique_nodes)
        opinion_map = opinion_map.bfill().ffill()

        assert_no_nan(opinion_map, "final opinion map creation")
        # Get unique time points
        times = opinion_map.columns.values
        mid_time = times[len(times) // 2]  # Mid-time point

        
        opinion_map = opinion_map.reindex(node_mapping.keys()).rename(index=node_mapping)
        
        # Ensure time steps are sorted in ascending order
        times = sorted(opinion_map.columns.values)

        # # Check if there are enough time steps
        # if len(times) < 65:
        #     print(f"Insufficient time steps (less than 65) in {opinion_file}. Skipping this file.")
        #     continue

        # Get the first 15 time steps and the next 50 time steps
        x_times = times[:lookback]
        y_times = times[lookback:lookback+horizon]
        # Check if y_times is long enough
        if len(y_times) < horizon:
            print(f"y_times length is less than {horizon}, padding with the last time point data")
            last_time = y_times[-1] if len(y_times) > 0 else x_times[-1]
            # Calculate the number of repetitions needed
            repeat_count = horizon - len(y_times)
            # Add repeated time points
            additional_times = [last_time] * repeat_count
            y_times = list(y_times) + additional_times
        # Extract x and y
        x = torch.tensor(opinion_map[x_times].values, dtype=torch.float)  # [num_nodes, 15]
        y = torch.tensor(opinion_map[y_times].values, dtype=torch.float)  # [num_nodes, 50]
        assert_no_nan_in_tensor(x, "x")
        assert_no_nan_in_tensor(y, "y")

        # Calculate node degrees
        num_nodes = x.size(0)
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))

        # Take logarithm of degrees
        degrees_log = torch.log(degrees.float() + 1e-10)  # Avoid log(0)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        degrees_log_np = degrees_log.cpu().numpy().reshape(-1, 1)  # Convert to numpy array

        # Check for NaN or Inf values
        if np.isnan(degrees_log_np).any() or np.isinf(degrees_log_np).any():
            print("Degrees contain NaN or Inf values. Skipping this data item.")
            continue

        # If all degrees are the same, KMeans may not work properly, skip this data
        if len(np.unique(degrees_log_np)) == 1:
            print("All degrees are identical. Skipping this data item.")
            continue

        kmeans.fit(degrees_log_np)
        labels = kmeans.labels_

        # Generate cluster_idx
        cluster_idx = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            cluster_idx[label].append(idx)

        # Convert to cluster_node_indices and cluster_ptr
        cluster_node_indices = []
        cluster_ptr = [0]
        for c in cluster_idx:
            cluster_node_indices.extend(c)
            cluster_ptr.append(cluster_ptr[-1] + len(c))

        cluster_node_indices = torch.tensor(cluster_node_indices, dtype=torch.long)
        cluster_ptr = torch.tensor(cluster_ptr, dtype=torch.long)

        # Create Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            cluster_node_indices=cluster_node_indices,
            cluster_ptr=cluster_ptr
        )

        # Save Data object
        base_name = opinion_file.replace('_opinion.txt', '')
        save_dir = f"data/real_data/{base_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(data, os.path.join(save_dir, "data.pt"))


def process_all_folders(root_dir):
    """
    Iterates through all folders under a given root directory and processes each using `process_opinions_and_edges`.

    Args:
        root_dir (str): The root directory containing folders to process.
    """
    # Iterate through all folders in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing {folder_name}...")
            process_opinions_and_edges(folder_path)
            print(f"Finished processing {folder_name}.")

if __name__ == "__main__":
    # Specify the root directory path
    root_dir = "./data/raw_data"
    process_all_folders(root_dir)
