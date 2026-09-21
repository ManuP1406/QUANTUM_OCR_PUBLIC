"""
Script to filter both original and transformed graphs based on node count, MSE, and maximum distance from origin after centering.
Creates two files with the SAME indices:
- Filtered original graphs
- Filtered transformed graphs
"""

import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
import os
from collections import defaultdict


ORIGINAL_PKL = 'Data/Data_temp/mnist_graphs_nx_compressed.pkl'
TRANSFORMED_PKL = 'Data/mnist_graphs_nx_repulsion_all.pkl'

OUTPUT_ORIGINAL_FILTERED = 'Data/mnist_graphs_nx_filtered.pkl'
OUTPUT_TRANSFORMED_FILTERED = 'Data/mnist_graphs_nx_repulsion_filtered.pkl'

# Filter parameters
MAX_NODES = 9           
MAX_MSE = 1.5           
TARGET_MIN = 4.5
TARGET_MAX = 6.0
MAX_DISTANCE_FROM_CENTER = 50.0  



def scale_distances_to_range(distances, target_min=TARGET_MIN, target_max=TARGET_MAX):
    """
    Scale original distances to target range preserving proportions.
    """
    min_dist = distances.min()
    max_dist = distances.max()
    
    if max_dist > min_dist:
        scaled = target_min + (distances - min_dist) * (target_max - target_min) / (max_dist - min_dist)
    else:
        scaled = np.ones_like(distances) * (target_min + target_max) / 2
    
    return scaled


def compute_correct_mse(G_trans, G_orig):
    """
    Compute MSE correctly: between final distances and target distances.
    Target distances = original distances scaled to [TARGET_MIN, TARGET_MAX].
    """
    edges_trans = list(G_trans.edges())
    edges_orig = list(G_orig.edges())
    
    # Verify same graph structure
    if len(edges_trans) != len(edges_orig):
        return None, None
    
    # Get distances
    dist_trans = np.array([G_trans[u][v]['distance'] for u, v in edges_trans])
    dist_orig = np.array([G_orig[u][v]['distance'] for u, v in edges_orig])
    
    # Target distances (original scaled to range)
    dist_target = scale_distances_to_range(dist_orig, TARGET_MIN, TARGET_MAX)
    
    # MSE between final and target
    mse = np.mean((dist_trans - dist_target)**2)
    
    return mse, dist_target


def get_max_distance_from_center(G):
    """
    Center the graph by subtracting mean position, then compute the maximum distance from origin.
    Returns the maximum Euclidean distance from origin after centering.
    """
    # Get all node positions
    positions = np.array([G.nodes[n]['pos'] for n in G.nodes()])
    
    # Compute center (mean of positions)
    center = positions.mean(axis=0)
    
    # Center positions
    centered_positions = positions - center
    
    # Compute distances from origin
    distances = np.linalg.norm(centered_positions, axis=1)
    
    # Return maximum distance
    return distances.max()


def check_max_distance_condition(G, max_distance=MAX_DISTANCE_FROM_CENTER):
    """
    Check if the graph's maximum distance from origin after centering is <= max_distance.
    """
    try:
        max_dist = get_max_distance_from_center(G)
        return max_dist <= max_distance, max_dist
    except Exception as e:
        print(f"Error computing max distance: {e}")
        return False, None


def filter_by_indices(G_list, y_list, indices_to_keep, name="dataset"):
    """
    Filter lists by keeping only specified indices.
    """
    G_filtered = [G_list[i] for i in indices_to_keep]
    y_filtered = [y_list[i] for i in indices_to_keep]
    
    print(f"\n{name}: kept {len(G_filtered)}/{len(G_list)} graphs ({len(G_filtered)/len(G_list)*100:.1f}%)")
    
    return G_filtered, y_filtered


def main():
    print("="*80)
    print("FILTERING BOTH ORIGINAL AND TRANSFORMED DATASETS")
    print("="*80)
    print(f"Original file:    {ORIGINAL_PKL}")
    print(f"Transformed file: {TRANSFORMED_PKL}")
    print(f"\nFilter parameters:")
    print(f"  Max nodes: {MAX_NODES}")
    print(f"  Max MSE: {MAX_MSE}")
    print(f"  Max distance from center: {MAX_DISTANCE_FROM_CENTER}")
    print(f"  Target range: [{TARGET_MIN}, {TARGET_MAX}]")
    print("="*80)
    
    # Load original data
    print("\nLoading original graphs...")
    with open(ORIGINAL_PKL, 'rb') as f:
        orig_data = pickle.load(f)
    
    G_orig = orig_data['G']
    y_orig = orig_data['y']
    G_test_orig = orig_data['G_test']
    y_test_orig = orig_data['y_test']
    
    # Load transformed data
    print("\nLoading transformed graphs...")
    with open(TRANSFORMED_PKL, 'rb') as f:
        trans_data = pickle.load(f)
    
    G_trans = trans_data['G']
    y_trans = trans_data['y']
    G_test_trans = trans_data['G_test']
    y_test_trans = trans_data['y_test']
    
    # Verify same number of graphs
    assert len(G_orig) == len(G_trans), "Training set size mismatch"
    assert len(G_test_orig) == len(G_test_trans), "Test set size mismatch"
    

    print("\n" + "="*80)
    print("FINDING INDICES TO KEEP (based on TRANSFORMED graphs)")
    print("="*80)
    
    train_indices = []
    train_labels = []
    train_mses = []
    train_max_distances = []
    excluded_train = defaultdict(int)
    
    for idx, (G_nx, label) in enumerate(tqdm(zip(G_trans, y_trans), total=len(G_trans), desc="Training")):
        # Check node count
        if G_nx.number_of_nodes() > MAX_NODES:
            excluded_train['node_count'] += 1
            excluded_train[f'node_count_digit_{label}'] += 1
            continue
        
        # Check maximum distance from center
        passes_distance, max_dist = check_max_distance_condition(G_nx, MAX_DISTANCE_FROM_CENTER)
        if not passes_distance:
            excluded_train['max_distance'] += 1
            excluded_train[f'max_distance_digit_{label}'] += 1
            continue
        
        # Get original graph
        G_orig_nx = G_orig[idx]
        
        # Compute correct MSE
        result = compute_correct_mse(G_nx, G_orig_nx)
        if result[0] is None:
            excluded_train['mse_error'] += 1
            continue
        
        mse, dist_target = result
        
        # Check MSE
        if mse > MAX_MSE:
            excluded_train['mse'] += 1
            excluded_train[f'mse_digit_{label}'] += 1
            continue
        
        # All checks passed
        train_indices.append(idx)
        train_labels.append(label)
        train_mses.append(mse)
        train_max_distances.append(max_dist)
    
    # Test set
    test_indices = []
    test_labels = []
    test_mses = []
    test_max_distances = []
    excluded_test = defaultdict(int)
    
    for idx, (G_nx, label) in enumerate(tqdm(zip(G_test_trans, y_test_trans), total=len(G_test_trans), desc="Test")):
        # Check node count
        if G_nx.number_of_nodes() > MAX_NODES:
            excluded_test['node_count'] += 1
            excluded_test[f'node_count_digit_{label}'] += 1
            continue
        
        # Check maximum distance from center
        passes_distance, max_dist = check_max_distance_condition(G_nx, MAX_DISTANCE_FROM_CENTER)
        if not passes_distance:
            excluded_test['max_distance'] += 1
            excluded_test[f'max_distance_digit_{label}'] += 1
            continue
        
        # Get original graph
        G_orig_nx = G_test_orig[idx]
        
        result = compute_correct_mse(G_nx, G_orig_nx)
        if result[0] is None:
            excluded_test['mse_error'] += 1
            continue
        
        mse, dist_target = result
        
        if mse > MAX_MSE:
            excluded_test['mse'] += 1
            excluded_test[f'mse_digit_{label}'] += 1
            continue
        
        # All checks passed
        test_indices.append(idx)
        test_labels.append(label)
        test_mses.append(mse)
        test_max_distances.append(max_dist)
    
    print(f"\nResults:")
    print(f"  Training: keeping {len(train_indices)}/{len(G_trans)} graphs")
    print(f"  Test: keeping {len(test_indices)}/{len(G_test_trans)} graphs")
    
    if train_mses:
        print(f"\nMSE statistics (kept training graphs):")
        print(f"  Mean: {np.mean(train_mses):.6f}")
        print(f"  Std:  {np.std(train_mses):.6f}")
        print(f"  Min:  {np.min(train_mses):.6f}")
        print(f"  Max:  {np.max(train_mses):.6f}")
    
    if train_max_distances:
        print(f"\nMax distance statistics (kept training graphs):")
        print(f"  Mean: {np.mean(train_max_distances):.6f}")
        print(f"  Std:  {np.std(train_max_distances):.6f}")
        print(f"  Min:  {np.min(train_max_distances):.6f}")
        print(f"  Max:  {np.max(train_max_distances):.6f}")
    
    if test_mses:
        print(f"\nMSE statistics (kept test graphs):")
        print(f"  Mean: {np.mean(test_mses):.6f}")
        print(f"  Std:  {np.std(test_mses):.6f}")
        print(f"  Min:  {np.min(test_mses):.6f}")
        print(f"  Max:  {np.max(test_mses):.6f}")
    
    if test_max_distances:
        print(f"\nMax distance statistics (kept test graphs):")
        print(f"  Mean: {np.mean(test_max_distances):.6f}")
        print(f"  Std:  {np.std(test_max_distances):.6f}")
        print(f"  Min:  {np.min(test_max_distances):.6f}")
        print(f"  Max:  {np.max(test_max_distances):.6f}")
    
    print(f"\nExcluded training:")
    print(f"  Node count: {excluded_train.get('node_count', 0)}")
    print(f"  Max distance > {MAX_DISTANCE_FROM_CENTER}: {excluded_train.get('max_distance', 0)}")
    print(f"  MSE > {MAX_MSE}: {excluded_train.get('mse', 0)}")
    if excluded_train.get('mse_error', 0) > 0:
        print(f"  MSE calculation error: {excluded_train.get('mse_error', 0)}")
    
    print(f"\nExcluded test:")
    print(f"  Node count: {excluded_test.get('node_count', 0)}")
    print(f"  Max distance > {MAX_DISTANCE_FROM_CENTER}: {excluded_test.get('max_distance', 0)}")
    print(f"  MSE > {MAX_MSE}: {excluded_test.get('mse', 0)}")
    if excluded_test.get('mse_error', 0) > 0:
        print(f"  MSE calculation error: {excluded_test.get('mse_error', 0)}")

    print("\n" + "="*80)
    print("FILTERING ORIGINAL DATASET")
    print("="*80)
    
    G_orig_filtered, y_orig_filtered = filter_by_indices(
        G_orig, y_orig, train_indices, "Original training"
    )
    G_test_orig_filtered, y_test_orig_filtered = filter_by_indices(
        G_test_orig, y_test_orig, test_indices, "Original test"
    )
    
    # Save filtered original
    orig_filtered_data = {
        'G': G_orig_filtered,
        'y': np.array(y_orig_filtered),
        'G_test': G_test_orig_filtered,
        'y_test': np.array(y_test_orig_filtered),
        'G_positions': [nx.get_node_attributes(g, 'pos') for g in G_orig_filtered],
        'G_test_positions': [nx.get_node_attributes(g, 'pos') for g in G_test_orig_filtered],
        'filter_params': {
            'max_nodes': MAX_NODES,
            'max_mse': MAX_MSE,
            'target_min': TARGET_MIN,
            'target_max': TARGET_MAX,
            'max_distance_from_center': MAX_DISTANCE_FROM_CENTER
        },
        'mse_stats': {
            'train_mean': np.mean(train_mses) if train_mses else 0,
            'train_std': np.std(train_mses) if train_mses else 0,
            'train_min': np.min(train_mses) if train_mses else 0,
            'train_max': np.max(train_mses) if train_mses else 0,
            'test_mean': np.mean(test_mses) if test_mses else 0,
            'test_std': np.std(test_mses) if test_mses else 0,
            'test_min': np.min(test_mses) if test_mses else 0,
            'test_max': np.max(test_mses) if test_mses else 0
        },
        'max_distance_stats': {
            'train_mean': np.mean(train_max_distances) if train_max_distances else 0,
            'train_std': np.std(train_max_distances) if train_max_distances else 0,
            'train_min': np.min(train_max_distances) if train_max_distances else 0,
            'train_max': np.max(train_max_distances) if train_max_distances else 0,
            'test_mean': np.mean(test_max_distances) if test_max_distances else 0,
            'test_std': np.std(test_max_distances) if test_max_distances else 0,
            'test_min': np.min(test_max_distances) if test_max_distances else 0,
            'test_max': np.max(test_max_distances) if test_max_distances else 0
        }
    }
    
    with open(OUTPUT_ORIGINAL_FILTERED, 'wb') as f:
        pickle.dump(orig_filtered_data, f)
    
    print(f"\nFiltered original saved to: {OUTPUT_ORIGINAL_FILTERED}")
    

    print("\n" + "="*80)
    print("FILTERING TRANSFORMED DATASET")
    print("="*80)
    
    G_trans_filtered, y_trans_filtered = filter_by_indices(
        G_trans, y_trans, train_indices, "Transformed training"
    )
    G_test_trans_filtered, y_test_trans_filtered = filter_by_indices(
        G_test_trans, y_test_trans, test_indices, "Transformed test"
    )
    
    # Save filtered transformed
    trans_filtered_data = {
        'G': G_trans_filtered,
        'y': np.array(y_trans_filtered),
        'G_test': G_test_trans_filtered,
        'y_test': np.array(y_test_trans_filtered),
        'G_positions': [nx.get_node_attributes(g, 'pos') for g in G_trans_filtered],
        'G_test_positions': [nx.get_node_attributes(g, 'pos') for g in G_test_trans_filtered],
        'transformation_params': {
            'target_min': TARGET_MIN,
            'target_max': TARGET_MAX,
            'scale_factor': 15.0,
            'iterations': 200,
            'repulsion_strength': 8.0
        },
        'filter_params': {
            'max_nodes': MAX_NODES,
            'max_mse': MAX_MSE,
            'max_distance_from_center': MAX_DISTANCE_FROM_CENTER
        },
        'mse_stats': {
            'train_mean': np.mean(train_mses) if train_mses else 0,
            'train_std': np.std(train_mses) if train_mses else 0,
            'train_min': np.min(train_mses) if train_mses else 0,
            'train_max': np.max(train_mses) if train_mses else 0,
            'test_mean': np.mean(test_mses) if test_mses else 0,
            'test_std': np.std(test_mses) if test_mses else 0,
            'test_min': np.min(test_mses) if test_mses else 0,
            'test_max': np.max(test_mses) if test_mses else 0
        },
        'max_distance_stats': {
            'train_mean': np.mean(train_max_distances) if train_max_distances else 0,
            'train_std': np.std(train_max_distances) if train_max_distances else 0,
            'train_min': np.min(train_max_distances) if train_max_distances else 0,
            'train_max': np.max(train_max_distances) if train_max_distances else 0,
            'test_mean': np.mean(test_max_distances) if test_max_distances else 0,
            'test_std': np.std(test_max_distances) if test_max_distances else 0,
            'test_min': np.min(test_max_distances) if test_max_distances else 0,
            'test_max': np.max(test_max_distances) if test_max_distances else 0
        }
    }
    
    with open(OUTPUT_TRANSFORMED_FILTERED, 'wb') as f:
        pickle.dump(trans_filtered_data, f)
    
    print(f"\n Filtered transformed saved to: {OUTPUT_TRANSFORMED_FILTERED}")
    

    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    print(f"\nOriginal filtered:")
    print(f"  Training: {len(G_orig_filtered)} graphs")
    print(f"  Test: {len(G_test_orig_filtered)} graphs")
    
    print(f"\nTransformed filtered:")
    print(f"  Training: {len(G_trans_filtered)} graphs")
    print(f"  Test: {len(G_test_trans_filtered)} graphs")
    
  
    assert len(G_orig_filtered) == len(G_trans_filtered), "Final training size mismatch"
    assert len(G_test_orig_filtered) == len(G_test_trans_filtered), "Final test size mismatch"
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()