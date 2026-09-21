"""
Build graphs from MNIST dataset and save them.

This script:
1. Loads MNIST train/test data
2. Converts each image to a graph representation via skeletonization
3. Optionally generates sample visualizations (saved as images, not plotted)
4. Saves all graph data to a pickle file
"""

"""
Build graphs from MNIST dataset and save them with edge distances.
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
from grakel import Graph




def save_graphs_data(G, y, G_positions, G_test, y_test, G_test_positions,
                     G_edge_distances=None, G_test_edge_distances=None,
                     filename="mnist_graphs.pkl"):
    """
    Save all graph data (train + test) to a pickle file.
    Now includes edge distances if provided.
    """
    data = {
        'G': G,
        'y': y,
        'G_positions': G_positions,
        'G_test': G_test,
        'y_test': y_test,
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


def plot_graph_with_positions(grakel_graph, positions_dict,
                              true_label, pred_label=None, idx=0,
                              edge_distances=None):
    """
    Plot a grakel graph using pre-saved node positions.
    Now optionally can show edge distances.
    """
    edges_dict = grakel_graph.get_edge_dictionary()

    G_nx = nx.Graph()
    G_nx.add_nodes_from(positions_dict.keys())
    
    # Add edges with distances as attributes
    for u, neighbors in edges_dict.items():
        for v in neighbors:
            if edge_distances is not None and (u, v) in edge_distances:
                G_nx.add_edge(u, v, distance=edge_distances[(u, v)])
            elif edge_distances is not None and (v, u) in edge_distances:
                G_nx.add_edge(u, v, distance=edge_distances[(v, u)])
            else:
                G_nx.add_edge(u, v, distance=None)

    fig, ax = plt.subplots(figsize=(8, 8))
    title = f"Sample {idx} – True label: {true_label}"
    if pred_label is not None:
        title += f", Predicted: {pred_label}"

    # Draw nodes
    pos = positions_dict
    nx.draw_networkx_nodes(G_nx, pos, ax=ax, node_color='lightblue', 
                          node_size=200)
    
    # Draw edges with optional labels
    nx.draw_networkx_edges(G_nx, pos, ax=ax, edge_color='gray')
    
    # Add edge labels if distances exist
    if edge_distances is not None:
        edge_labels = {(u, v): f"{d['distance']:.1f}" 
                      for u, v, d in G_nx.edges(data=True) 
                      if d['distance'] is not None}
        nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels, 
                                     ax=ax, font_size=6)
    
    nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=8)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig


def compute_edge_distances(nodes, edges):
    """
    Compute Euclidean distance for each edge.
    
    Args:
        nodes: list of (x, y) coordinates
        edges: list of (u, v) tuples
    
    Returns:
        dict: {(u, v): distance} for each edge
    """
    edge_distances = {}
    for u, v in edges:
        # Make sure we store edges consistently
        u, v = int(u), int(v)
        pos_u = np.array(nodes[u])
        pos_v = np.array(nodes[v])
        distance = np.linalg.norm(pos_u - pos_v)
        edge_distances[(u, v)] = float(distance)
    return edge_distances


# ---------------------------------------------------------------------------
# Sample extraction & visualisation
# ---------------------------------------------------------------------------

def get_samples_per_class(dataloader, n_samples=5, merge_dist=3):
    """
    Extract *n_samples* samples for each digit class (0-9).
    Now includes edge distances.
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
            
            # Compute edge distances
            edge_distances = compute_edge_distances(nodes, edges)

            samples[label]['images'].append(img_display)
            samples[label]['graphs'].append((nodes, edges, node_vals))
            samples[label]['edge_distances'].append(edge_distances)

        if all(len(samples[c]['images']) >= n_samples for c in range(10)):
            break

    return samples


def save_sample_figure(samples, selected_classes, merge_dist, images_dir):
    """
    Create a grid figure (classes × samples) and **save** it to *images_dir*
    """
    n_samples = min(len(samples[selected_classes[0]]['images']), 5)
    fig, axes = plt.subplots(len(selected_classes), n_samples * 2,
                             figsize=(12, 5))

    for row, cls in enumerate(selected_classes):
        for col in range(n_samples):
            # Original image
            ax_img = axes[row, col * 2]
            ax_img.imshow(samples[cls]['images'][col], cmap='gray')
            ax_img.axis('off')

            # Graph overlay
            ax_graph = axes[row, col * 2 + 1]
            nodes, edges, node_vals = samples[cls]['graphs'][col]
            edge_distances = samples[cls]['edge_distances'][col]
            
            plt.sca(ax_graph)
            plot_graph_attention(nodes, edges, node_weights=node_vals,
                                node_size=15)
            ax_graph.set_aspect('equal')
            ax_graph.axis('off')

        axes[row, 0].set_ylabel(f'Class {cls}', fontsize=12,
                                fontweight='bold', rotation=0, labelpad=15)

    classes_str = "_".join(str(c) for c in selected_classes)
    fig.suptitle(
        f'Samples for classes {selected_classes}  (merge_dist={merge_dist})',
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs(images_dir, exist_ok=True)
    filename = os.path.join(
        images_dir,
        f"samples_classes_{classes_str}_merge{merge_dist}.png",
    )
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Sample figure saved to {filename}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _image_tensor_to_uint8(img_tensor):
    """Convert a normalised MNIST tensor to a uint8 numpy image."""
    img = img_tensor.numpy().squeeze()
    img_display = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
    return img_display


def build_and_save_graphs(
    batch_size: int = 4,
    merge_dist: int = 3,
    selected_classes_for_plot: list = None,
    output_pkl: str = "mnist_graphs.pkl",
    images_dir: str = "images",
    include_edge_distances: bool = True,  
):
    """
    End-to-end pipeline:
      1. Load MNIST
      2. (Optional) save a sample-visualisation image
      3. Convert every image to a grakel Graph
      4. Save graphs + positions + edge distances to pickle

    Args:
        batch_size:  DataLoader batch size.
        merge_dist:  Distance threshold for merging nearby nodes.
        selected_classes_for_plot: List of digit classes to visualise.
        output_pkl:  Path for the output pickle file.
        images_dir:  Folder where sample images are saved.
        include_edge_distances: Whether to compute and save edge distances.
    """
    # ── 1. Load MNIST ──────────────────────────────────────────────────
    print("Loading MNIST …")
    trainloader, testloader = load_data(batch_size=batch_size)

    # ── 2. Optional: sample visualisation ──────────────────────────────
    if selected_classes_for_plot is not None:
        print("Extracting 5 samples per class for visualisation …")
        samples = get_samples_per_class(trainloader, n_samples=5,
                                        merge_dist=merge_dist)
        save_sample_figure(samples, selected_classes_for_plot,
                           merge_dist, images_dir)
        # Reload dataloader (consumed by sample extraction)
        trainloader, testloader = load_data(batch_size=batch_size)


    G, y, G_positions = [], [], []
    G_test, y_test, G_test_positions = [], [], []
    
  
    G_edge_distances = [] if include_edge_distances else None
    G_test_edge_distances = [] if include_edge_distances else None

    print("Building training graphs …")
    for images, labels in trainloader:
        for i in range(len(images)):
            label = labels[i].item()
            img_display = _image_tensor_to_uint8(images[i])

            nodes, edges, node_vals = image2graph(
                img_display, width=28, with_processing=True,
                merge_dist_thres=merge_dist, merge_with_center=False,
            )

           
            edges_list = [(int(u), int(v)) for u, v in edges]
            node_labels = {j: 1.0 for j in range(len(nodes))}
            grakel_graph = Graph(edges_list, node_labels=node_labels)

            G.append(grakel_graph)
            y.append(label)

            positions_dict = {j: (float(nodes[j][0]), float(nodes[j][1]))
                              for j in range(len(nodes))}
            G_positions.append(positions_dict)
            
           
            if include_edge_distances:
                edge_distances = compute_edge_distances(nodes, edges)
                G_edge_distances.append(edge_distances)

    print(f"  Training graphs: {len(G)}")

    print("Building test graphs …")
    for images, labels in testloader:
        for i in range(len(images)):
            label = labels[i].item()
            img_display = _image_tensor_to_uint8(images[i])

            nodes, edges, node_vals = image2graph(
                img_display, width=28, with_processing=True,
                merge_dist_thres=merge_dist, merge_with_center=False,
            )

            edges_list = [(int(u), int(v)) for u, v in edges]
            node_labels = {j: 1.0 for j in range(len(nodes))}
            grakel_graph = Graph(edges_list, node_labels=node_labels)

            G_test.append(grakel_graph)
            y_test.append(label)

            positions_dict = {j: (float(nodes[j][0]), float(nodes[j][1]))
                              for j in range(len(nodes))}
            G_test_positions.append(positions_dict)
            
        
            if include_edge_distances:
                edge_distances = compute_edge_distances(nodes, edges)
                G_test_edge_distances.append(edge_distances)

    print(f"  Test graphs:     {len(G_test)}")


    if G:
        print("\nSample graph info:")
        print(f"  Nodes : {len(G[0].get_edge_dictionary())}")
        print(f"  Labels: {G[0].get_labels() is not None}")
        first_labels = dict(list(G[0].get_labels().items())[:5])
        print(f"  First 5 labels: {first_labels}")
        
        if include_edge_distances and G_edge_distances:
            print(f"  Edge distances: {len(G_edge_distances[0])} edges")
            first_edge = list(G_edge_distances[0].items())[0]
            print(f"  First edge distance: {first_edge}")

   
    save_graphs_data(G, y, G_positions, G_test, y_test, G_test_positions,
                     G_edge_distances, G_test_edge_distances,
                     filename=output_pkl)

    return (G, y, G_positions, G_test, y_test, G_test_positions,
            G_edge_distances, G_test_edge_distances)



if __name__ == "__main__":
    build_and_save_graphs(
        batch_size=4,
        merge_dist=3,
        selected_classes_for_plot=[1 , 2 , 3, 4, 5, 8],
        output_pkl="mnist_graphs_compressed.pkl",
        images_dir="images",
        include_edge_distances=True, 
    )