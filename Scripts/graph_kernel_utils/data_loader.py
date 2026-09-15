"""Data loading and filtering functions."""

import pickle
import numpy as np


def load_graphs_data(filename: str):
    """Load graph data from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    G = data['G']
    y = data['y']
    G_positions = data['G_positions']
    G_test = data['G_test']
    y_test = data['y_test']
    G_test_pos = data['G_test_positions']
    
    G_edge_distances = data.get('G_edge_distances', None)
    G_test_edge_distances = data.get('G_test_edge_distances', None)

    print(f"Loaded {filename}")
    print(f"  Training: {len(G)} graphs  |  Test: {len(G_test)} graphs")
    if G_edge_distances is not None:
        print(f"  Edge distances available for {len(G_edge_distances)} training graphs")
    
    return G, y, G_positions, G_test, y_test, G_test_pos, G_edge_distances, G_test_edge_distances


def filter_and_limit_classes(G, y, G_pos, classes, max_per_class=None, 
                             G_edge_distances=None, dataset_type="training",
                             verbose=True): 
    """
    Keep only graphs with labels in *classes* and limit per class.
    """
    indices_by_class = {c: [] for c in classes}
    
    for i, lab in enumerate(y):
        if lab in classes:
            indices_by_class[lab].append(i)
    
    selected_indices = []
    for c in classes:
        class_indices = indices_by_class[c]
        if max_per_class and len(class_indices) > max_per_class:
            np.random.seed(42)
            selected = np.random.choice(class_indices, max_per_class, replace=False)
            selected_indices.extend(selected)
            if verbose: 
                print(f"  Class {c}: {len(class_indices)} → {max_per_class} (limited)")
        else:
            selected_indices.extend(class_indices)
            if verbose:
                print(f"  Class {c}: {len(class_indices)} (all)")
    
    selected_indices.sort()
    
    if G_edge_distances is not None:
        filtered_edge_distances = [G_edge_distances[i] for i in selected_indices]
    else:
        filtered_edge_distances = None
    
    return ([G[i] for i in selected_indices],
            [y[i] for i in selected_indices],
            [G_pos[i] for i in selected_indices],
            filtered_edge_distances)