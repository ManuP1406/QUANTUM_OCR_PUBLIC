"""
Graph preprocessing utilities for quantum kernel.
Handles graph conversion and position scaling for Pulser device.
"""

import numpy as np

# ============================================================================
# FUNZIONI PER FEATURE ANGOLARI
# ============================================================================

def compute_angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors (0 to π).
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot)


def compute_half_angle_vector(v1, v2):
    """
    Compute the half-angle vector between two vectors.
    Returns a unit vector pointing to the bisector.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return np.array([0.0, 0.0])
    
    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    
    bisector = v1_norm + v2_norm
    bisector_norm = np.linalg.norm(bisector)
    
    if bisector_norm == 0:
        # Vectors are opposite, choose perpendicular direction
        return np.array([-v1_norm[1], v1_norm[0]])
    
    return bisector / bisector_norm


def get_angle_features(graph, positions):
    """
    Compute angle features for each node.
    
    Rules:
    - 1 edge: [π, 0] (max deviation, zero direction)
    - 2 edges: [deviation, direction] where direction = angle of half-angle vector (0 to π)
    - 3 edges: take the pair with largest ANGLE (not deviation), compute deviation + direction
    - 4 edges: take the 2 pairs with largest ANGLES, sum their deviations and directions
    - 5+ edges: take the 3 pairs with largest ANGLES, sum their deviations and directions
    
    Returns:
        dict: node -> [deviation, direction] where:
            deviation: sum of |π - angle| for selected pairs (0 to n_pairs*π)
            direction: circular mean of half-angle vectors (0 to π)
    """
    node_features = {}
    edges_dict = graph.get_edge_dictionary()
    
   
    all_neighbours = {}
    for node in positions.keys():
        all_neighbours[node] = set()
    
    for u, neighbors in edges_dict.items():
        for v in neighbors:
            all_neighbours[u].add(v)
            all_neighbours[v].add(u)
    
    for node in positions.keys():
        neighbours = list(all_neighbours[node])
        n_neighbors = len(neighbours)
        
        # Caso 1 edge o meno
        if n_neighbors <= 1:
            node_features[node] = [np.pi * 0.90, 0.0]
            continue
        
       
        pairs = []
        for i in range(n_neighbors):
            for j in range(i+1, n_neighbors):
                v1 = np.array(positions[neighbours[i]]) - np.array(positions[node])
                v2 = np.array(positions[neighbours[j]]) - np.array(positions[node])
                
                if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                    continue
                
                angle = compute_angle_between_vectors(v1, v2)
                pairs.append({
                    'angle': angle,
                    'v1': v1,
                    'v2': v2,
                    'pair': (neighbours[i], neighbours[j])
                })
        
        if not pairs:
            node_features[node] = [np.pi, 0.0]
            continue
        
       
        pairs.sort(key=lambda x: x['angle'], reverse=True)
        
   
        if n_neighbors == 2:
            n_select = 1
        elif n_neighbors == 3:
            n_select = 1
        elif n_neighbors == 4:
            n_select = 2
        else: 
            n_select = min(3, len(pairs))
        
        selected = pairs[:n_select]
        
        total_deviation = 0.0
        half_angle_vectors = []
        
        for sel in selected:
            angle = sel['angle']
            deviation = np.abs(np.pi - angle)
            total_deviation += deviation
            
            
            half_vec = compute_half_angle_vector(sel['v1'], sel['v2'])
            half_angle_vectors.append(half_vec)
        
    
        if half_angle_vectors:
      
            avg_vec = np.mean(half_angle_vectors, axis=0)

            direction = np.arctan2(avg_vec[1], avg_vec[0]) % np.pi
        else:
            direction = 0.0
        
        node_features[node] = [total_deviation, direction]
    
    return node_features


def graph_to_node_list(orig_graph, orig_positions, rep_positions, features_type="angle+pos"):
    """
    Convert graphs to a list of nodes for quantum kernel.
    
    Parameters:
    - orig_graph: GraKeL graph (original, for features)
    - orig_positions: positions dict from original graph (for angle features)
    - rep_positions: positions dict from transformed graph (for coordinates)
    - features_type: 
        "angle" -> [deviation, direction] (using orig_positions)
        "pos" -> [x, y] as features (using rep_positions, normalized to [0, π])
        "angle+pos" -> [x, y, deviation, direction] (coordinates + angle features)
        None -> [1.0]
    
    Returns:
    - nodes: list of [x, y, feat1, feat2, ...] for each node
    """
    nodes = []
    features = {}
    EPS = 1e-10
    # Compute features from ORIGINAL graph structure and ORIGINAL positions
    if features_type == "angle":
        # Returns [deviation, direction] per node
        angle_feats = get_angle_features(orig_graph, orig_positions)
        features = angle_feats
            
    elif features_type == "pos":
        all_x = [orig_positions[node][0] for node in rep_positions.keys()]
        all_y = [orig_positions[node][1] for node in rep_positions.keys()]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        if max_range > 0:
            scale_factor = (np.pi/2) / max_range
        else:
            scale_factor = 1.0
        
        for node in rep_positions.keys():
            x_orig, y_orig = orig_positions[node]
            x_norm = (x_orig - x_center) * scale_factor + (np.pi/4)
            y_norm = (y_orig - y_center) * scale_factor + (np.pi/4)
            if x_norm < EPS:
                x_norm = 0.0
            if y_norm < EPS:
                y_norm = 0.0
            features[node] = [x_norm, y_norm]
                
    elif features_type == "angle+pos":
        angle_feats = get_angle_features(orig_graph, orig_positions)
        
        all_x = [orig_positions[node][0] for node in rep_positions.keys()]
        all_y = [orig_positions[node][1] for node in rep_positions.keys()]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        if max_range > 0:
            scale_factor = (np.pi/2) / max_range
        else:
            scale_factor = 1.0
        
        global_scale_factor = 1
        for node in rep_positions.keys():
            x_orig, y_orig = orig_positions[node]
            x_norm = ((x_orig - x_center) * scale_factor + (np.pi/4)) * global_scale_factor
            y_norm = ((y_orig - y_center) * scale_factor + (np.pi/4)) * global_scale_factor
            if x_norm < EPS:
                x_norm = 0.0
            if y_norm < EPS:
                y_norm = 0.0
            angle_vec = angle_feats.get(node, [np.pi, 0.0])
            features[node] = [x_norm, y_norm] + angle_vec # Modifica solo angle_vec
            
    else:
        # Default: all nodes feature = 1
        for node in rep_positions.keys():
            features[node] = [0.0]

    # Collect coordinates from TRANSFORMED graph (for Pulser)
    coords = []
    for node in rep_positions.keys():
        x, y = rep_positions[node]
        coords.append([x, y])

    coords = np.array(coords)
    if len(coords) == 0:
        return []

    # Center coordinates
    center = coords.mean(axis=0)
    coords_centered = coords - center

    # Build node list with features
    nodes = []
    for idx, node in enumerate(rep_positions.keys()):
        x, y = coords_centered[idx]
        feat_list = features.get(node, [1.0])
        
        # Combine: [x, y] + features
        node_entry = [x, y] + feat_list
        nodes.append(node_entry)

    return nodes



def compute_global_scale_factor(all_graphs_nodes, min_distance=4.1):
    """
    Compute a single scaling factor for all graphs based on global minimum distance.
    """
    print("\n📐 Computing global scale factor for Pulser device...")
    global_min = float('inf')
    total_pairs = 0
    
    for graph_idx, graph_nodes in enumerate(all_graphs_nodes):
        if len(graph_nodes) < 2:
            continue
        
        # Extract coordinates (first 2 elements are x,y)
        coords = np.array([[node[0], node[1]] for node in graph_nodes])
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < global_min:
                    global_min = dist
                total_pairs += 1
        
        if graph_idx % 100 == 0 and graph_idx > 0:
            print(f"    Processed {graph_idx} graphs...")
    
    print(f"    Total pairs analyzed: {total_pairs}")
    print(f"    Global minimum distance: {global_min:.6f}")
    
    if global_min >= min_distance:
        print(f"    ✅ No scaling needed (min distance already >= {min_distance})")
        return 1.0
    
    scale = min_distance / global_min
    print(f"    Global scaling factor: {scale:.6f}")
    print(f"    After scaling, min distance will be: {global_min * scale:.6f}")
    
    return scale


def scale_all_graphs(graphs_list, scale_factor):
    """
    Apply the same scaling factor to all graphs.
    Scales only x,y coordinates, leaves features unchanged.
    """
    if scale_factor == 1.0:
        return graphs_list
    
    scaled_graphs = []
    for graph_nodes in graphs_list:
        scaled = []
        for node in graph_nodes:
            # node = [x, y, feat1, feat2, ...]
            scaled.append([node[0]*scale_factor, node[1]*scale_factor] + node[2:])
        scaled_graphs.append(scaled)
    
    return scaled_graphs


def verify_scaling(graphs_list, name="Graphs"):
    """
    Verify the scaling by printing min/max distances.
    """
    all_dists = []
    for graph_nodes in graphs_list[:5]:  # Check first 5 graphs
        if len(graph_nodes) < 2:
            continue
        # Extract coordinates
        coords = np.array([[n[0], n[1]] for n in graph_nodes])
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                all_dists.append(dist)
    
    if all_dists:
        print(f"\n{name} distance verification:")
        print(f"    Min distance: {min(all_dists):.6f}")
        print(f"    Max distance: {max(all_dists):.6f}")
        print(f"    Mean distance: {np.mean(all_dists):.6f}")
        print(f"    Std distance: {np.std(all_dists):.6f}")



def compute_local_scale_factors(all_graphs_nodes, min_distance=4.1):
    """
    Compute a scale factor per graph.
    """
    print("\nComputing local scale factors...")

    scale_factors = []

    for graph_nodes in all_graphs_nodes:

        if len(graph_nodes) < 2:
            scale_factors.append(1.0)
            continue

        coords = np.array([[n[0], n[1]] for n in graph_nodes])

        local_min = float("inf")

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < local_min:
                    local_min = dist

        if local_min >= min_distance:
            scale_factors.append(1.0)
        else:
            scale_factors.append(min_distance / local_min)

    return scale_factors


def scale_all_graphs_local(graphs_list, scale_factors):
    """
    Apply per-graph scaling.
    """
    scaled_graphs = []

    for graph_nodes, s in zip(graphs_list, scale_factors):
        scaled = []

        for node in graph_nodes:
            scaled.append([
                node[0] * s,
                node[1] * s,
                *node[2:]
            ])

        scaled_graphs.append(scaled)

    return scaled_graphs