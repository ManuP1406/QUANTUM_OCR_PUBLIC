"""Direction-based node label functions."""

import numpy as np
from grakel import Graph


def create_direction_based_labels(positions_dict, edges_dict, n_directions=8):
    """
    Create direction-based labels for each node.
    
    For each node, finds the closest cardinal/intercardinal direction
    to the average of all neighbor vectors.
    
    Parameters:
    - n_directions: 4 (cardinal) or 8 (cardinal+intercardinal)
    
    Returns:
    - node_labels: {node: label} where label=0 for nodes with no neighbors,
                   otherwise label from 1 to n_directions
    """
    node_labels = {}
    
    # Define direction vectors based on n_directions
    if n_directions == 4:
        directions = [
            (1, 0),   # 0: right
            (0, 1),   # 1: up
            (-1, 0),  # 2: left
            (0, -1)   # 3: down
        ]
    else:  # default to 8 directions
        directions = [
            (1, 0),    # 0: right
            (1, 1),    # 1: up-right
            (0, 1),    # 2: up
            (-1, 1),   # 3: up-left
            (-1, 0),   # 4: left
            (-1, -1),  # 5: down-left
            (0, -1),   # 6: down
            (1, -1)    # 7: down-right
        ]
    
    for node in positions_dict.keys():
        # Skip nodes with no neighbors
        if node not in edges_dict or len(edges_dict[node]) == 0:
            node_labels[node] = 0
            continue
        
        pos_node = np.array(positions_dict[node])
        
        # Collect all vectors to neighbors
        vectors = []
        for nbr in edges_dict[node]:
            if nbr in positions_dict:
                pos_nbr = np.array(positions_dict[nbr])
                vector = pos_nbr - pos_node
                if np.linalg.norm(vector) > 0:
                    vectors.append(vector / np.linalg.norm(vector))
        
        if not vectors:
            node_labels[node] = 0
            continue
        
        # Average all vectors to get average direction
        avg_vector = np.mean(vectors, axis=0)
        if np.linalg.norm(avg_vector) == 0:
            node_labels[node] = 0
            continue
        avg_vector = avg_vector / np.linalg.norm(avg_vector)
        
        # Find closest direction
        best_dir = 0
        best_dot = -1
        
        for d_idx, (dx, dy) in enumerate(directions):
            dir_vec = np.array([dx, dy])
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            dot = np.dot(avg_vector, dir_vec)
            if dot > best_dot:
                best_dot = dot
                best_dir = d_idx
        
        node_labels[node] = best_dir + 1  # +1 to reserve 0 for no neighbors
    
    return node_labels


def convert_graphs_with_direction_labels(G_list, positions_list, n_directions=8):
    """
    Convert a list of GraKeL graphs to graphs with direction-based labels.
    
    Parameters:
    - G_list: list of GraKeL graphs
    - positions_list: list of position dictionaries
    - n_directions: 4 or 8 (number of directions)
    
    Returns:
    - converted_graphs: list of GraKeL graphs with direction labels
    """
    converted_graphs = []
    
    for i, (g, positions) in enumerate(zip(G_list, positions_list)):
        # Get edge structure
        edges = g.get_edge_dictionary()
        
        # Create direction-based labels
        node_labels = create_direction_based_labels(positions, edges, n_directions)
        
        # Create new graph with labels
        g_labeled = Graph(edges, node_labels=node_labels)
        converted_graphs.append(g_labeled)
    
    return converted_graphs


def visualize_directions(n_directions=8):
    """
    Utility function to visualize the direction vectors.
    """
    if n_directions == 4:
        directions = [
            (1, 0, "right"),
            (0, 1, "up"),
            (-1, 0, "left"),
            (0, -1, "down")
        ]
    else:
        directions = [
            (1, 0, "right"),
            (1, 1, "up-right"),
            (0, 1, "up"),
            (-1, 1, "up-left"),
            (-1, 0, "left"),
            (-1, -1, "down-left"),
            (0, -1, "down"),
            (1, -1, "down-right")
        ]
    
    print(f"\nDirection mapping for {n_directions} directions:")
    for idx, (dx, dy, name) in enumerate(directions):
        print(f"  {idx+1}: {name} ({dx}, {dy})")
    
    return directions
