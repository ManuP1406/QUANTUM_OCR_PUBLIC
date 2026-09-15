"""Position-based node label functions."""

import numpy as np
from grakel import Graph


def add_positions_as_node_labels(grakel_graph, positions_dict, image_size=None):
    """
    Add positions as node labels for GraphHopper.
    
    Parameters:
    - positions_dict: dict {node: (x, y)} with coordinates
    - image_size: tuple (width, height) or single value. If None, infers from max coordinates.
    """
    edges = grakel_graph.get_edge_dictionary()
    node_labels = {}
    
    # Determine image dimensions for normalization
    if image_size is None:
        all_x = [x for x, _ in positions_dict.values()]
        all_y = [y for _, y in positions_dict.values()]
        width = max(all_x) + 1
        height = max(all_y) + 1
    elif isinstance(image_size, (int, float)):
        width = height = image_size
    else:
        width, height = image_size
    
    for node, (x, y) in positions_dict.items():
        x_norm = x / width
        y_norm = y / height
        node_labels[node] = (x_norm, y_norm)
    
    return Graph(edges, node_labels=node_labels)


def create_discrete_labels_from_positions(positions_dict, grid_size=10, image_size=None, coordinate_type='pixel'):
    """
    Create discrete labels from positions (for Weisfeiler-Lehman kernel)
    
    Parameters:
    - positions_dict: dict {node: (x, y)} with coordinates
    - grid_size: number of grid divisions
    - image_size: tuple (width, height) or single value if square.
                  If None, infers from max coordinate values.
    - coordinate_type: 'pixel' (absolute coordinates) or 'normalized' (0-1 range)
    """
    node_labels = {}
    
    if not positions_dict:
        return node_labels
    
    all_x = [x for x, _ in positions_dict.values()]
    all_y = [y for _, y in positions_dict.values()]
    
    # Determine image dimensions based on coordinate type
    if coordinate_type == 'normalized':
        # Coordinates are already in [0,1] range
        if image_size is None:
            width, height = 1.0, 1.0
        elif isinstance(image_size, (int, float)):
            width = height = float(image_size)
        else:
            width, height = float(image_size[0]), float(image_size[1])
    else:
        # Pixel coordinates
        if image_size is None:
            width = max(all_x) + 1 if all_x else 1
            height = max(all_y) + 1 if all_y else 1
        elif isinstance(image_size, (int, float)):
            width = height = image_size
        else:
            width, height = image_size
    
    if grid_size == 1:
        # All nodes get the same label
        for node in positions_dict.keys():
            node_labels[node] = 1
    else:
        for node, (x, y) in positions_dict.items():
            if coordinate_type == 'normalized':
                # For normalized coordinates [0,1]
                region_x = int(x * grid_size / width)
                region_y = int(y * grid_size / height)
            else:
                # For pixel coordinates
                cell_width = width / grid_size
                cell_height = height / grid_size
                region_x = int(x // cell_width)
                region_y = int(y // cell_height)
            
            # Clamp to valid range
            region_x = max(0, min(region_x, grid_size - 1))
            region_y = max(0, min(region_y, grid_size - 1))
            
            # Create a unique label for each region
            node_labels[node] = region_x * grid_size + region_y
    
    return node_labels


def convert_graphs_with_position_labels(G_list, positions_list, method='continuous', 
                                        grid_size=10, image_size=None, 
                                        coordinate_type='pixel', normalize=True):
    """
    Convert a list of GraKeL graphs to graphs with position-based labels.
    
    Parameters:
    - G_list: list of GraKeL graphs
    - positions_list: list of position dicts
    - method: 'continuous' or 'discrete'
    - grid_size: for discrete method
    - image_size: tuple (width, height) or single value. If None, infers from coordinates.
    - coordinate_type: 'pixel' or 'normalized' (for discrete method)
    - normalize: if True, normalize continuous labels to [0,1] range
    """
    converted_graphs = []
    
    # Determine global image dimensions if not provided
    if image_size is None and normalize:
        # Find max coordinates across all graphs for consistent normalization
        all_x = []
        all_y = []
        for positions in positions_list:
            for x, y in positions.values():
                all_x.append(x)
                all_y.append(y)
        if all_x and all_y:
            width = max(all_x) + 1
            height = max(all_y) + 1
            image_size = (width, height)
    
    for g, positions in zip(G_list, positions_list):
        edges = g.get_edge_dictionary()
        
        if method == 'continuous':
            node_labels = {}
            for node, (x, y) in positions.items():
                if normalize and image_size:
                    if isinstance(image_size, (int, float)):
                        width = height = image_size
                    else:
                        width, height = image_size
                    x_norm = x / width
                    y_norm = y / height
                else:
                    x_norm, y_norm = x, y
                node_labels[node] = (x_norm, y_norm)
        else:  # discrete
            node_labels = create_discrete_labels_from_positions(
                positions, grid_size=grid_size, 
                image_size=image_size, coordinate_type=coordinate_type
            )
        
        g_labeled = Graph(edges, node_labels=node_labels)
        converted_graphs.append(g_labeled)
    
    return converted_graphs


def get_image_dimensions(positions_list, coordinate_type='pixel'):
    """
    Helper function to get image dimensions from a list of position dicts.
    
    Returns:
        (width, height) tuple
    """
    all_x = []
    all_y = []
    for positions in positions_list:
        for x, y in positions.values():
            all_x.append(x)
            all_y.append(y)
    
    if coordinate_type == 'normalized':
        return 1.0, 1.0
    else:
        return max(all_x) + 1 if all_x else 28, max(all_y) + 1 if all_y else 28