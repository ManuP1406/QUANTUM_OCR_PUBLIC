"""
Convert NetworkX graphs to Grakel format.
Reads the spring-relaxed graphs and saves them as Grakel graphs with edge distances.
"""

import pickle
import numpy as np
import networkx as nx
from grakel import Graph
import os
from tqdm import tqdm




#INPUT_PKL = 'Data/mnist_graphs_nx_repulsion_filtered.pkl'
#OUTPUT_PKL = 'Data/mnist_graphs_grakel_repulsion_filtered.pkl'

INPUT_PKL = 'Data/mnist_graphs_nx_filtered.pkl'
OUTPUT_PKL = 'Data/mnist_graphs_grakel_filtered.pkl'



def nx_to_grakel(G_nx):
    """
    Convert a NetworkX graph to Grakel format.
    
    Grakel expects:
    - edges: list of (u, v) tuples
    - node_labels: dictionary {node_index: label_value}
    - edge_labels: optional dictionary {(u, v): weight}  # Note: edge_labels, not edge_weights!
    
    For MNIST, we use:
    - node_labels: all 1.0 (dummy labels)
    - edge_labels: the edge distances
    """
    # Get edges
    edges = list(G_nx.edges())
    
    # Create node labels (all 1.0 as in original script)
    node_labels = {node: 1.0 for node in G_nx.nodes()}
    
    # Create edge labels (distances) - Grakel uses edge_labels, not edge_weights
    edge_labels = {}
    for u, v in edges:
        # Store in both directions for Grakel compatibility
        dist = G_nx[u][v]['distance']
        edge_labels[(u, v)] = dist
        edge_labels[(v, u)] = dist
    
    # Create Grakel graph with edge_labels parameter
    grakel_graph = Graph(edges, node_labels=node_labels, edge_labels=edge_labels)
    
    return grakel_graph


def extract_positions(G_nx):
    """
    Extract node positions from NetworkX graph.
    Returns dictionary {node_index: (x, y)}.
    """
    return nx.get_node_attributes(G_nx, 'pos')


def extract_edge_distances(G_nx):
    """
    Extract edge distances from NetworkX graph.
    Returns dictionary {(u, v): distance}.
    """
    edge_distances = {}
    for u, v in G_nx.edges():
        dist = G_nx[u][v]['distance']
        edge_distances[(u, v)] = dist
        edge_distances[(v, u)] = dist  # Store both directions
    return edge_distances




def main():
    print("="*80)
    print("NETWORKX TO GRAKEL CONVERTER")
    print("="*80)
    print(f"Input file:  {INPUT_PKL}")
    print(f"Output file: {OUTPUT_PKL}")
    print("="*80)
    
    # Check if input file exists
    if not os.path.exists(INPUT_PKL):
        print(f"ERROR: Input file not found: {INPUT_PKL}")
        return
    
    # Load NetworkX graphs
    print("\nLoading NetworkX graphs...")
    with open(INPUT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    G_nx_train = data['G']              # Training graphs (NetworkX)
    y_train = data['y']                  # Training labels
    G_nx_test = data['G_test']           # Test graphs (NetworkX)
    y_test = data['y_test']              # Test labels
    
    # Load transformation parameters if available
    transform_params = data.get('transformation_params', {})
    
    print(f"Loaded {len(G_nx_train)} training graphs")
    print(f"Loaded {len(G_nx_test)} test graphs")
    if transform_params:
        print(f"Transformation: {transform_params}")
    
 
    print("\n" + "="*80)
    print("CONVERTING TRAINING GRAPHS TO GRAKEL")
    print("="*80)
    
    G_grakel_train = []
    G_positions_train = []
    G_edge_distances_train = []
    
    for idx, G_nx in enumerate(tqdm(G_nx_train, desc="Training graphs")):
        # Convert to Grakel
        grakel_graph = nx_to_grakel(G_nx)
        G_grakel_train.append(grakel_graph)
        
        # Extract positions
        positions = extract_positions(G_nx)
        G_positions_train.append(positions)
        
        # Extract edge distances
        edge_distances = extract_edge_distances(G_nx)
        G_edge_distances_train.append(edge_distances)
    

    print("\n" + "="*80)
    print("CONVERTING TEST GRAPHS TO GRAKEL")
    print("="*80)
    
    G_grakel_test = []
    G_positions_test = []
    G_edge_distances_test = []
    
    for idx, G_nx in enumerate(tqdm(G_nx_test, desc="Test graphs")):
        # Convert to Grakel
        grakel_graph = nx_to_grakel(G_nx)
        G_grakel_test.append(grakel_graph)
        
        # Extract positions
        positions = extract_positions(G_nx)
        G_positions_test.append(positions)
        
        # Extract edge distances
        edge_distances = extract_edge_distances(G_nx)
        G_edge_distances_test.append(edge_distances)
    

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    # Training stats
    print(f"\nTRAINING SET:")
    print(f"  Graphs: {len(G_grakel_train)}")
    print(f"  Avg nodes: {np.mean([g.n for g in G_grakel_train]):.1f}")
    print(f"  Avg edges: {np.mean([len(g.get_edges()) for g in G_grakel_train]):.1f}")
    
    # Test stats
    print(f"\nTEST SET:")
    print(f"  Graphs: {len(G_grakel_test)}")
    print(f"  Avg nodes: {np.mean([g.n for g in G_grakel_test]):.1f}")
    print(f"  Avg edges: {np.mean([len(g.get_edges()) for g in G_grakel_test]):.1f}")
    
    # Check first graph
    if G_grakel_train:
        print("\nSample training graph (first):")
        print(f"  Nodes: {G_grakel_train[0].n}")
        print(f"  Edges: {len(G_grakel_train[0].get_edges())}")
        
        # Try to get edge labels (distances)
        try:
            edge_labels = G_grakel_train[0].get_edge_weights()  # Note: get_edge_weights() method
            print(f"  Edge labels sample: {dict(list(edge_labels.items())[:3])}")
        except:
            print("  Edge labels: available")
        
        print(f"  Node labels sample: {dict(list(G_grakel_train[0].get_labels().items())[:3])}")
    

    print("\n" + "="*80)
    print("SAVING GRAKEL DATASET")
    print("="*80)
    
    # Create data dictionary in the same format as original
    output_data = {
        'G': G_grakel_train,
        'y': y_train,
        'G_positions': G_positions_train,
        'G_test': G_grakel_test,
        'y_test': y_test,
        'G_test_positions': G_positions_test,
        'G_edge_distances': G_edge_distances_train,
        'G_test_edge_distances': G_edge_distances_test,
        'source': 'spring_relaxed_nx',
        'transformation_params': transform_params
    }
    
    # Save to file
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n Dataset saved to: {OUTPUT_PKL}")
    print(f"  Training graphs: {len(G_grakel_train)}")
    print(f"  Test graphs: {len(G_grakel_test)}")
    print(f"  Edge distances included: yes")
    
    # Verify save
    print("\nVerifying saved file...")
    with open(OUTPUT_PKL, 'rb') as f:
        check_data = pickle.load(f)
    print(f" Verification successful - loaded {len(check_data['G'])} training graphs")
    print(f"  Sample edge distance: {list(check_data['G_edge_distances'][0].keys())[:2]}")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()