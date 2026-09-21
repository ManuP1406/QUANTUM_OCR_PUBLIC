"""
Script to apply spring relaxation WITH REPULSION to ALL MNIST graphs.
Creates transformed graphs with edge distances in target range [4.5, 9.0].
WITH EARLY STOPPING based on MSE convergence.
"""

import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
import os


from joblib import Parallel, delayed

INPUT_PKL = 'Data/Data_temp/mnist_graphs_nx_compressed.pkl'
OUTPUT_PKL_TRANSFORMED = 'Data/mnist_graphs_nx_repulsion_all.pkl'

# Transformation parameters
TARGET_MIN = 4.5
TARGET_MAX = 6  # 9 
SCALE_FACTOR = 15.0
MAX_ITERATIONS = 200       
DT = 0.05
STIFFNESS = 10.0
REPULSION_STRENGTH = 8.0

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 30  
EARLY_STOPPING_TOLERANCE = 1e-6  
MIN_ITERATIONS = 50             



def scale_distances_to_range(original_distances):
    """Linearly scale original distances to target range."""
    min_dist = original_distances.min()
    max_dist = original_distances.max()
    
    if max_dist > min_dist:
        scaled = TARGET_MIN + (original_distances - min_dist) * (TARGET_MAX - TARGET_MIN) / (max_dist - min_dist)
    else:
        scaled = np.ones_like(original_distances) * (TARGET_MIN + TARGET_MAX) / 2
    
    return scaled


def compute_mse(positions, edges, target_dict):
    """Compute current MSE between distances and targets."""
    current_distances = []
    target_distances = []
    
    for u, v in edges:
        current = np.linalg.norm(positions[u] - positions[v])
        target = target_dict.get((u, v), target_dict.get((v, u), 5.0))
        current_distances.append(current)
        target_distances.append(target)
    
    return np.mean((np.array(current_distances) - np.array(target_distances))**2)


def spring_relaxation_with_repulsion(G_nx, target_dict, initial_positions, verbose=False):
    """
    Spring relaxation with repulsion between non-connected nodes.
    WITH EARLY STOPPING based on MSE convergence.
    
    Returns:
        positions: final positions
        converged: bool indicating if early stopping was triggered
        iterations: number of iterations performed
        mse_history: list of MSE values over iterations
    """
    positions = np.array(initial_positions).copy()
    edges = list(G_nx.edges())
    nodes = list(G_nx.nodes())
    n_nodes = len(nodes)
    
    if len(edges) == 0:
        return positions, True, 0, []
    
    # Create set of connected pairs
    connected_pairs = set()
    for u, v in edges:
        connected_pairs.add((u, v))
        connected_pairs.add((v, u))
    
    # Tracking for early stopping
    best_mse = float('inf')
    best_positions = positions.copy()
    patience_counter = 0
    mse_history = []
    
    # Initial MSE
    current_mse = compute_mse(positions, edges, target_dict)
    mse_history.append(current_mse)
    best_mse = current_mse
    best_positions = positions.copy()
    
    for iteration in range(MAX_ITERATIONS):
        forces = np.zeros_like(positions)
        
        # Spring forces between connected nodes
        for u, v in edges:
            target = target_dict.get((u, v), target_dict.get((v, u), 5.0))
            
            vec = positions[v] - positions[u]
            current = np.linalg.norm(vec)
            
            if current < 1e-6:
                continue
            
            direction = vec / current
            force_magnitude = -STIFFNESS * (current - target)
            
            force = force_magnitude * direction
            forces[u] -= force * DT
            forces[v] += force * DT
        
        # Repulsion between non-connected nodes
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if (i, j) not in connected_pairs:
                    vec = positions[j] - positions[i]
                    dist = np.linalg.norm(vec)
                    
                    if dist < 1e-6:
                        direction = np.random.randn(2)
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = vec / dist
                    
                    force_magnitude = REPULSION_STRENGTH / (dist + 0.1)
                    
                    force = force_magnitude * direction
                    forces[i] -= force * DT
                    forces[j] += force * DT
        
        positions += forces
        positions -= positions.mean(axis=0)
        
        # Compute MSE after update
        current_mse = compute_mse(positions, edges, target_dict)
        mse_history.append(current_mse)
        
        # Early stopping check
        improvement = best_mse - current_mse
        
        if current_mse < best_mse - EARLY_STOPPING_TOLERANCE:
            # Significant improvement
            best_mse = current_mse
            best_positions = positions.copy()
            patience_counter = 0
            if verbose and iteration % 50 == 0:
                print(f"      Iter {iteration}: MSE = {current_mse:.8f} (new best)")
        else:
            patience_counter += 1
        
        # Stop if no improvement for patience iterations AND we've done min iterations
        if patience_counter >= EARLY_STOPPING_PATIENCE and iteration >= MIN_ITERATIONS:
            if verbose:
                print(f"      Early stopping at iteration {iteration} (MSE = {current_mse:.8f})")
            # Restore best positions
            positions = best_positions
            break
    
    converged = patience_counter < EARLY_STOPPING_PATIENCE
    return positions, converged, iteration + 1, mse_history


def transform_graph(G_nx, verbose=False):
    """
    Apply scaling and spring relaxation WITH REPULSION to a single graph.
    Returns new graph with updated positions and distances, plus convergence info.
    """
    # Get original positions
    pos_dict = nx.get_node_attributes(G_nx, 'pos')
    nodes = list(G_nx.nodes())
    original_pos = np.array([pos_dict[i] for i in nodes])
    
    # Scale positions
    scaled_pos = original_pos * SCALE_FACTOR
    
    # Get edges and original distances
    edges = list(G_nx.edges())
    
    if len(edges) == 0:
        # Create graph with scaled positions only
        G_new = nx.Graph()
        for i, node in enumerate(nodes):
            G_new.add_node(node, pos=tuple(scaled_pos[i]))
        return G_new, True, 0, []
    
    original_distances = np.array([G_nx[u][v]['distance'] for u, v in edges])
    
    # Create target distances
    target_dict, target_distances = create_target_dict(edges, original_distances)
    
    # Apply relaxation with early stopping
    final_pos, converged, iterations, mse_history = spring_relaxation_with_repulsion(
        G_nx, target_dict, scaled_pos, verbose
    )
    
    # Calculate final distances
    final_distances = []
    for u, v in edges:
        dist = np.linalg.norm(final_pos[u] - final_pos[v])
        final_distances.append(dist)
    
    # Create new graph
    G_new = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(nodes):
        G_new.add_node(node, pos=tuple(final_pos[i]))
    
    # Add edges with updated distances
    for (u, v), dist in zip(edges, final_distances):
        G_new.add_edge(u, v, distance=dist)
    
    return G_new, converged, iterations, mse_history


def create_target_dict(edges, original_distances):
    """Create dictionary of target distances for each edge."""
    scaled_distances = scale_distances_to_range(original_distances)
    
    target_dict = {}
    for (u, v), target in zip(edges, scaled_distances):
        target_dict[(u, v)] = target
        target_dict[(v, u)] = target
    
    return target_dict, scaled_distances



def process_graph(G_nx, idx):
    verbose = (idx % 100 == 0)
    return transform_graph(G_nx, verbose)


def main():

  
    output_dir = os.path.dirname(OUTPUT_PKL_TRANSFORMED)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creata cartella di output: {output_dir}")

    
   
    print(f"\n Transformed dataset saved to: {OUTPUT_PKL_TRANSFORMED}")
    print("="*80)
    print("TRANSFORMING ALL GRAPHS WITH REPULSION (EARLY STOPPING)")
    print("="*80)
    print(f"Input file:  {INPUT_PKL}")
    print(f"Output file: {OUTPUT_PKL_TRANSFORMED}")
    print(f"\nParameters:")
    print(f"  Target range: [{TARGET_MIN}, {TARGET_MAX}]")
    print(f"  Repulsion strength: {REPULSION_STRENGTH}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Early stopping tolerance: {EARLY_STOPPING_TOLERANCE}")
    print("="*80)
    
    # Load data
    print("\nLoading original graphs...")
    with open(INPUT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    G = data['G']
    y = data['y']
    G_test = data['G_test']
    y_test = data['y_test']
    
    print(f"Loaded {len(G)} training graphs and {len(G_test)} test graphs")
    
 
    print("\n" + "="*80)
    print("TRANSFORMING TRAINING GRAPHS (PARALLEL)")
    print("="*80)
    
    results = Parallel(n_jobs=-1)(
        delayed(process_graph)(G_nx, idx)
        for idx, G_nx in enumerate(tqdm(G, desc="Dispatching Training Jobs"))
    )
    
    G_transformed = []
    convergence_stats = {'converged': 0, 'max_iter': 0, 'iterations': []}
    
    for G_new, converged, iters, mse_hist in results:
        G_transformed.append(G_new)
        
        if converged:
            convergence_stats['converged'] += 1
        
        convergence_stats['iterations'].append(iters)
        convergence_stats['max_iter'] = max(convergence_stats['max_iter'], iters)
    
  
    print("\n" + "="*80)
    print("TRANSFORMING TEST GRAPHS (PARALLEL)")
    print("="*80)
    
    results_test = Parallel(n_jobs=-1)(
        delayed(process_graph)(G_nx, idx)
        for idx, G_nx in enumerate(tqdm(G_test, desc="Dispatching Test Jobs"))
    )
    
    G_test_transformed = [r[0] for r in results_test]
    
 
    print("\n" + "="*80)
    print("CONVERGENCE STATISTICS")
    print("="*80)
    print(f"Training graphs:")
    print(f"  Converged (early stop): {convergence_stats['converged']}/{len(G)} ({100*convergence_stats['converged']/len(G):.1f}%)")
    print(f"  Max iterations used: {convergence_stats['max_iter']}")
    print(f"  Average iterations: {np.mean(convergence_stats['iterations']):.1f}")
    print(f"  Std iterations: {np.std(convergence_stats['iterations']):.1f}")
    
    print("\n" + "="*80)
    print("SAVING TRANSFORMED DATASET")
    print("="*80)
    
    transformed_data = {
        'G': G_transformed,
        'y': y,
        'G_test': G_test_transformed,
        'y_test': y_test,
        'G_positions': [nx.get_node_attributes(g, 'pos') for g in G_transformed],
        'G_test_positions': [nx.get_node_attributes(g, 'pos') for g in G_test_transformed],
        'transformation_params': {
            'target_min': TARGET_MIN,
            'target_max': TARGET_MAX,
            'scale_factor': SCALE_FACTOR,
            'max_iterations': MAX_ITERATIONS,
            'dt': DT,
            'stiffness': STIFFNESS,
            'repulsion_strength': REPULSION_STRENGTH,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_tolerance': EARLY_STOPPING_TOLERANCE,
            'min_iterations': MIN_ITERATIONS
        },
        'convergence_stats': {
            'training_converged': convergence_stats['converged'],
            'training_total': len(G),
            'training_convergence_rate': convergence_stats['converged'] / len(G),
            'training_avg_iterations': float(np.mean(convergence_stats['iterations'])),
            'training_std_iterations': float(np.std(convergence_stats['iterations'])),
            'training_max_iterations': convergence_stats['max_iter']
        }
    }
    
    with open(OUTPUT_PKL_TRANSFORMED, 'wb') as f:
        pickle.dump(transformed_data, f)
    
    print(f"\n Transformed dataset saved to: {OUTPUT_PKL_TRANSFORMED}")
    print(f"  Training graphs: {len(G_transformed)}")
    print(f"  Test graphs: {len(G_test_transformed)}")
    
    print("\n" + "="*80)
    print("TRANSFORMATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()