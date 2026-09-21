"""Visualization functions for graphs and misclassifications."""

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def save_misclassified(G_test, G_test_pos, y_test, y_pred,
                       kernel_label: str, images_dir: str = "images",
                       n_errors: int = 10, G_test_edge_distances=None):
    """Save misclassified test samples as images."""
    mis_idx = [i for i in range(len(y_test)) if y_pred[i] != y_test[i]]
    print(f"\nTotal misclassified: {len(mis_idx)} / {len(y_test)}")

    dist = {}
    for i in mis_idx:
        key = f"{y_test[i]}→{y_pred[i]}"
        dist[key] = dist.get(key, 0) + 1
    print("Error distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")

    safe_label = kernel_label.replace(' ', '_').replace('=', '').replace(',', '').replace('(', '').replace(')', '')
    out_dir = os.path.join(images_dir, f"misclassified_{safe_label}")
    os.makedirs(out_dir, exist_ok=True)

    n_save = min(n_errors, len(mis_idx))
    print(f"\nSaving {n_save} plots to {out_dir}/ …")

    for rank, idx in enumerate(mis_idx[:n_save]):
        fig = _plot_graph(G_test[idx], G_test_pos[idx],
                          y_test[idx], y_pred[idx], idx,
                          G_test_edge_distances[idx] if G_test_edge_distances else None)
        fname = os.path.join(out_dir, f"err{rank:02d}_idx{idx}_true{y_test[idx]}_pred{y_pred[idx]}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {os.path.basename(fname)}")


def _plot_graph(grakel_graph, positions_dict, true_label, pred_label, idx, edge_distances=None):
    """Plot a single graph with optional edge thickness based on distance."""
    edges_dict = grakel_graph.get_edge_dictionary()
    G_nx = nx.Graph()
    G_nx.add_nodes_from(positions_dict.keys())
    
    for u, neighbors in edges_dict.items():
        for v in neighbors:
            G_nx.add_edge(u, v)
            if edge_distances:
                if (u, v) in edge_distances:
                    G_nx[u][v]['distance'] = edge_distances[(u, v)]
                elif (v, u) in edge_distances:
                    G_nx[u][v]['distance'] = edge_distances[(v, u)]

    fig, ax = plt.subplots(figsize=(6, 6))
    
    nx.draw_networkx_nodes(G_nx, positions_dict, ax=ax, node_color='#4fc3f7', node_size=60)
    
    if edge_distances and len(G_nx.edges()) > 0:
        widths = []
        for u, v in G_nx.edges():
            if 'distance' in G_nx[u][v]:
                w = 3.0 * np.exp(-G_nx[u][v]['distance'] / 10.0)
                widths.append(max(0.5, w))
            else:
                widths.append(1.2)
        nx.draw_networkx_edges(G_nx, positions_dict, ax=ax, edge_color='#b0bec5', width=widths)
    else:
        nx.draw_networkx_edges(G_nx, positions_dict, ax=ax, edge_color='#b0bec5', width=1.2)
    
    ax.set_title(f"idx {idx}  |  True: {true_label}  Pred: {pred_label}", fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig


def plot_graph(grakel_graph, positions_dict, title="", idx=0):
    """Public function to plot a single graph."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    edges_dict = grakel_graph.get_edge_dictionary()
    G_nx = nx.Graph()
    G_nx.add_nodes_from(positions_dict.keys())
    
    for u, neighbors in edges_dict.items():
        for v in neighbors:
            G_nx.add_edge(u, v)
    
    nx.draw(G_nx, positions_dict, ax=ax, with_labels=False,
            node_color='#4fc3f7', node_size=80,
            edge_color='#b0bec5', width=1.5)
    
    ax.set_title(f"{title} (idx {idx})", fontsize=14)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig