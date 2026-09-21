"""
Build graphs from MNIST dataset and save them using NetworkX format.
This version stores graphs as NetworkX objects instead of grakel Graph objects.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import networkx as nx

from construct_graph import image2graph, plot_graph_attention
from mnist_loader import load_data




def save_graphs_data_nx(G, y, G_positions, G_test, y_test, G_test_positions,
                        G_edge_distances=None, G_test_edge_distances=None,
                        filename="mnist_graphs_nx.pkl"):
    """
    Save all graph data (train + test) to a pickle file using NetworkX format.
    """
    data = {
        'G': G,              # List of NetworkX graphs
        'y': np.array(y),    # Labels as numpy array
        'G_positions': G_positions,  # List of position dicts for each graph
        'G_test': G_test,
        'y_test': np.array(y_test),
        'G_test_positions': G_test_positions,
    }
    
    # Add edge distances if they exist
    if G_edge_distances is not None:
        data['G_edge_distances'] = G_edge_distances
    if G_test_edge_distances is not None:
        data['G_test_edge_distances'] = G_test_edge_distances
        
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")
    print(f"  Training: {len(G)} graphs  –  Test: {len(G_test)} graphs")
    if G_edge_distances is not None:
        print(f"  Edge distances included for training")
    
    # Print sample info
    if G:
        print(f"\nSample NetworkX graph info:")
        print(f"  Type: {type(G[0])}")
        print(f"  Nodes: {G[0].number_of_nodes()}")
        print(f"  Edges: {G[0].number_of_edges()}")
        print(f"  Node attributes: {list(G[0].nodes(data=True))[0]}")
        if G_edge_distances:
            edge_data = list(G[0].edges(data=True))[0]
            print(f"  Edge with distance: {edge_data}")


def compute_edge_distances_nx(nodes, edges):
    """
    Compute Euclidean distance for each edge and return as dict for NetworkX.
    
    Args:
        nodes: list of (x, y) coordinates
        edges: list of (u, v) tuples
    
    Returns:
        dict: {(u, v): distance} for each edge
    """
    edge_distances = {}
    for u, v in edges:
        u, v = int(u), int(v)
        pos_u = np.array(nodes[u])
        pos_v = np.array(nodes[v])
        distance = np.linalg.norm(pos_u - pos_v)
        edge_distances[(u, v)] = float(distance)
    return edge_distances


def create_nx_graph(nodes, edges, node_vals, edge_distances=None):
    """
    Create a NetworkX graph from nodes and edges.
    
    Args:
        nodes: list of [x, y, value] for each node
        edges: list of (u, v) tuples
        node_vals: node values (pixel intensities)
        edge_distances: dict of edge distances (optional)
    
    Returns:
        networkx.Graph with node attributes 'pos' and 'value', 
        and edge attribute 'distance'
    """
    G_nx = nx.Graph()
    
    # Add nodes with attributes
    for i, (node, val) in enumerate(zip(nodes, node_vals)):
        G_nx.add_node(i, pos=(float(node[0]), float(node[1])), 
                      value=float(val))
    
    # Add edges with distance attribute
    for u, v in edges:
        u, v = int(u), int(v)
        if edge_distances and (u, v) in edge_distances:
            dist = edge_distances[(u, v)]
        elif edge_distances and (v, u) in edge_distances:
            dist = edge_distances[(v, u)]
        else:
            # Compute distance if not provided
            pos_u = np.array(nodes[u][:2])
            pos_v = np.array(nodes[v][:2])
            dist = float(np.linalg.norm(pos_u - pos_v))
        
        G_nx.add_edge(u, v, distance=dist)
    
    return G_nx


def plot_nx_graph(G_nx, true_label, pred_label=None, idx=0, 
                  show_values=False, show_distances=False):
    """
    Plot a NetworkX graph with positions.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get positions from node attributes
    pos = nx.get_node_attributes(G_nx, 'pos')
    
    # Get node values for coloring
    node_values = nx.get_node_attributes(G_nx, 'value')
    if node_values:
        node_colors = [node_values[i] for i in G_nx.nodes()]
    else:
        node_colors = 'lightblue'
    
    # Draw nodes
    nodes_draw = nx.draw_networkx_nodes(G_nx, pos, ax=ax, 
                                        node_color=node_colors,
                                        cmap='viridis', node_size=200,
                                        vmin=0, vmax=255)
    
    # Draw edges
    nx.draw_networkx_edges(G_nx, pos, ax=ax, edge_color='gray', alpha=0.7)
    
    # Draw node labels (indices)
    nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=8)
    
    # Add edge labels if distances exist and requested
    if show_distances:
        edge_labels = {(u, v): f"{d['distance']:.1f}" 
                      for u, v, d in G_nx.edges(data=True)}
        nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, 
                                     ax=ax, font_size=6)
    
    # Add node values if requested
    if show_values and node_values:
        for i, (x, y) in pos.items():
            ax.text(x, y+0.05, f"{node_values[i]:.0f}", 
                   fontsize=7, ha='center', va='bottom', color='black')
    
    title = f"Graph {idx} – True label: {true_label}"
    if pred_label is not None:
        title += f", Predicted: {pred_label}"
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add colorbar for node values
    plt.colorbar(nodes_draw, ax=ax, label='Node value (0-255)', shrink=0.8)
    
    fig.tight_layout()
    return fig




def get_samples_per_class_nx(dataloader, n_samples=5, merge_dist=3):
    """
    Extract *n_samples* samples for each digit class (0-9) as NetworkX graphs.
    """
    samples = {i: {'images': [], 'graphs': [], 'edge_distances': []} 
               for i in range(10)}

    for images, labels in dataloader:
        for i in range(len(images)):
            label = labels[i].item()
            if len(samples[label]['images']) >= n_samples:
                continue

            img = images[i].numpy().squeeze()
            img_display = ((img * 0.5 + 0.5) * 255).astype(np.uint8)

            nodes, edges, node_vals = image2graph(
                img_display, width=28, with_processing=True,
                merge_dist_thres=merge_dist, merge_with_center=False,
            )
            
          
            edge_distances = compute_edge_distances_nx(nodes, edges)
            
      
            G_nx = create_nx_graph(nodes, edges, node_vals, edge_distances)

            samples[label]['images'].append(img_display)
            samples[label]['graphs'].append(G_nx)
            samples[label]['edge_distances'].append(edge_distances)

        if all(len(samples[c]['images']) >= n_samples for c in range(10)):
            break

    return samples


def save_sample_figure_nx(samples, selected_classes, merge_dist, images_dir):
    """
    Create a grid figure (classes × samples) and save it to images_dir.
    """
    n_samples = min(len(samples[selected_classes[0]]['images']), 5)
    fig, axes = plt.subplots(len(selected_classes), n_samples,
                             figsize=(3*n_samples, 3*len(selected_classes)))

    for row, cls in enumerate(selected_classes):
        for col in range(n_samples):
            ax = axes[row, col]
            
            # Get NetworkX graph
            G_nx = samples[cls]['graphs'][col]
            
            # Get positions from node attributes
            pos = nx.get_node_attributes(G_nx, 'pos')
            
            # Get node values for coloring
            node_values = nx.get_node_attributes(G_nx, 'value')
            
            # Draw graph
            nx.draw_networkx_nodes(G_nx, pos, ax=ax, 
                                  node_color=list(node_values.values()),
                                  cmap='viridis', node_size=50,
                                  vmin=0, vmax=255)
            nx.draw_networkx_edges(G_nx, pos, ax=ax, alpha=0.5, width=1)
            nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=6)
            
            ax.set_title(f'Class {cls}', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')

    classes_str = "_".join(str(c) for c in selected_classes)
    fig.suptitle(
        f'NetworkX Graphs - Classes {selected_classes} (merge_dist={merge_dist})',
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs(images_dir, exist_ok=True)
    filename = os.path.join(
        images_dir,
        f"samples_nx_classes_{classes_str}_merge{merge_dist}.png",
    )
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Sample figure saved to {filename}")




def _image_tensor_to_uint8(img_tensor):
    """Convert a normalised MNIST tensor to a uint8 numpy image."""
    img = img_tensor.numpy().squeeze()
    img_display = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
    return img_display


def build_and_save_graphs_nx(
    batch_size: int = 4,
    merge_dist: int = 3,
    selected_classes_for_plot: list = None,
    output_pkl: str = "mnist_graphs_nx.pkl",
    images_dir: str = "images",
    include_edge_distances: bool = True,
):
    """
    End-to-end pipeline to build and save NetworkX graphs:
      1. Load MNIST
      2. (Optional) save a sample-visualisation image
      3. Convert every image to a NetworkX graph
      4. Save graphs + positions + edge distances to pickle

    Args:
        batch_size: DataLoader batch size.
        merge_dist: Distance threshold for merging nearby nodes.
        selected_classes_for_plot: List of digit classes to visualise.
        output_pkl: Path for the output pickle file.
        images_dir: Folder where sample images are saved.
        include_edge_distances: Whether to compute and save edge distances.
    """
   
    print("Loading MNIST …")
    trainloader, testloader = load_data(batch_size=batch_size)

    if selected_classes_for_plot is not None:
        print("Extracting 5 samples per class for visualisation …")
        samples = get_samples_per_class_nx(trainloader, n_samples=5,
                                           merge_dist=merge_dist)
        save_sample_figure_nx(samples, selected_classes_for_plot,
                              merge_dist, images_dir)
        # Reload dataloader (consumed by sample extraction)
        trainloader, testloader = load_data(batch_size=batch_size)

 
    G, y = [], []
    G_test, y_test = [], []
    
    # Store positions separately (though they're also in graph attributes)
    G_positions, G_test_positions = [], []
    
    # Store edge distances
    G_edge_distances = [] if include_edge_distances else None
    G_test_edge_distances = [] if include_edge_distances else None

    print("Building training graphs (NetworkX) …")
    for images, labels in trainloader:
        for i in range(len(images)):
            label = labels[i].item()
            img_display = _image_tensor_to_uint8(images[i])

            nodes, edges, node_vals = image2graph(
                img_display, width=28, with_processing=True,
                merge_dist_thres=merge_dist, merge_with_center=False,
            )
            
            # Compute edge distances
            edge_distances = None
            if include_edge_distances:
                edge_distances = compute_edge_distances_nx(nodes, edges)
            
            # Create NetworkX graph
            G_nx = create_nx_graph(nodes, edges, node_vals, edge_distances)

            G.append(G_nx)
            y.append(label)
            
            # Store positions dict (for compatibility with existing code)
            positions_dict = {j: (float(nodes[j][0]), float(nodes[j][1]))
                              for j in range(len(nodes))}
            G_positions.append(positions_dict)
            
            if include_edge_distances:
                G_edge_distances.append(edge_distances)

    print(f"  Training graphs: {len(G)}")

    print("Building test graphs (NetworkX) …")
    for images, labels in testloader:
        for i in range(len(images)):
            label = labels[i].item()
            img_display = _image_tensor_to_uint8(images[i])

            nodes, edges, node_vals = image2graph(
                img_display, width=28, with_processing=True,
                merge_dist_thres=merge_dist, merge_with_center=False,
            )
            
            # Compute edge distances
            edge_distances = None
            if include_edge_distances:
                edge_distances = compute_edge_distances_nx(nodes, edges)
            
            # Create NetworkX graph
            G_nx = create_nx_graph(nodes, edges, node_vals, edge_distances)

            G_test.append(G_nx)
            y_test.append(label)
            
            # Store positions dict
            positions_dict = {j: (float(nodes[j][0]), float(nodes[j][1]))
                              for j in range(len(nodes))}
            G_test_positions.append(positions_dict)
            
            if include_edge_distances:
                G_test_edge_distances.append(edge_distances)

    print(f"  Test graphs:     {len(G_test)}")


    if G:
        print("\nSample NetworkX graph info:")
        print(f"  Nodes: {G[0].number_of_nodes()}")
        print(f"  Edges: {G[0].number_of_edges()}")
        print(f"  Node attributes: {list(G[0].nodes(data=True))[0]}")
        
        if include_edge_distances and G_edge_distances:
            print(f"  Edge distances: {len(G_edge_distances[0])} edges")
            edge_data = list(G[0].edges(data=True))[0]
            print(f"  First edge with distance: {edge_data}")


    save_graphs_data_nx(G, y, G_positions, G_test, y_test, G_test_positions,
                        G_edge_distances, G_test_edge_distances,
                        filename=output_pkl)

    return (G, y, G_positions, G_test, y_test, G_test_positions,
            G_edge_distances, G_test_edge_distances)



if __name__ == "__main__":
    build_and_save_graphs_nx(
        batch_size=4,
        merge_dist=3,
        selected_classes_for_plot=[1, 2, 3, 4, 5, 8],
        output_pkl="mnist_graphs_nx_compressed.pkl",
        images_dir="images",
        include_edge_distances=True,
    )