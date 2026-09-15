"""Angle-based node label functions."""

import numpy as np
from grakel import Graph


def compute_angle_to_x_axis(vector):
    """Compute angle of a vector relative to the positive X axis."""
    x, y = vector
    angle = np.arctan2(y, x) * 180 / np.pi
    if angle < 0:
        angle += 360
    return angle


def compute_angle_to_y_axis(vector):
    """Compute angle of a vector relative to the positive Y axis."""
    x, y = vector
    angle = np.arctan2(x, y) * 180 / np.pi
    if angle < 0:
        angle += 360
    return angle


def create_angle_based_labels(positions_dict, edges_dict, method='max', axis='x', n_bins=8):
    """
    Create angle-based labels for each node.
    """
    node_labels = {}
    
    for node in positions_dict.keys():
        if node not in edges_dict or len(edges_dict[node]) == 0:
            node_labels[node] = 0
            continue
        
        pos_node = np.array(positions_dict[node])
        
        angles = []
        for nbr in edges_dict[node]:
            if nbr in positions_dict:
                pos_nbr = np.array(positions_dict[nbr])
                vector = pos_nbr - pos_node
                
                if np.linalg.norm(vector) > 0:
                    if axis == 'x':
                        angle = compute_angle_to_x_axis(vector)
                    else:
                        angle = compute_angle_to_y_axis(vector)
                    angles.append(angle)
        
        if not angles:
            node_labels[node] = 0
            continue
        
        if method == 'max':
            agg_angle = max(angles)
        elif method == 'min':
            agg_angle = min(angles)
        elif method == 'mean':
            agg_angle = np.mean(angles)
        elif method == 'sum':
            agg_angle = np.sum(angles) % 360
        else:
            agg_angle = angles[0]
        
        bin_size = 360 / n_bins
        bin_idx = int(agg_angle // bin_size)
        bin_idx = min(bin_idx, n_bins - 1)
        
        node_labels[node] = bin_idx + 1
    
    return node_labels


def convert_graphs_with_angle_labels(G_list, positions_list, method='max', axis='x', n_bins=8):
    """
    Convert graphs using angle-based labels.
    """
    converted_graphs = []
    
    for i, (g, positions) in enumerate(zip(G_list, positions_list)):
        edges = g.get_edge_dictionary()
        node_labels = create_angle_based_labels(positions, edges, method, axis, n_bins)
        g_labeled = Graph(edges, node_labels=node_labels)
        converted_graphs.append(g_labeled)
    
    return converted_graphs


