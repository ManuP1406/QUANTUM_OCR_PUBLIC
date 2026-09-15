"""Kernel computation functions."""

import time
import numpy as np
from grakel.kernels import (GraphletSampling, WeisfeilerLehman,
                            VertexHistogram, RandomWalk, GraphHopper)

from .position_labels import convert_graphs_with_position_labels
from .angle_labels import convert_graphs_with_angle_labels
from .direction_labels import convert_graphs_with_direction_labels


def compute_grakel_kernel(G_train, G_test, kernel_name: str, 
                          G_train_pos=None, G_test_pos=None,
                          graphlet_k=3,
                          wl_iter=3,
                          rw_steps=3, rw_lambda=0.1,
                          grid_size=10,
                          feature_type='position',
                          angle_method='max',
                          angle_axis='x',
                          angle_bins=8,
                          **kwargs):
    """
    Compute train and test kernel matrices using a grakel kernel.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Initializing {kernel_name} kernel with {feature_type} features...")
    
    # Prepare graphs based on feature type
    if feature_type == 'position':
        if kernel_name == 'graphhopper':
            print("  Converting graphs with continuous node labels (positions)...")
            G_train_conv = convert_graphs_with_position_labels(G_train, G_train_pos, 'continuous')
            G_test_conv = convert_graphs_with_position_labels(G_test, G_test_pos, 'continuous')
        elif kernel_name in ['wl', 'vh']:
            print(f"  Converting graphs with discrete position labels (grid_size={grid_size})...")
            G_train_conv = convert_graphs_with_position_labels(G_train, G_train_pos, 'discrete', grid_size)
            G_test_conv = convert_graphs_with_position_labels(G_test, G_test_pos, 'discrete', grid_size)
        else:
            G_train_conv = G_train
            G_test_conv = G_test
    
    elif feature_type == 'angle':
        if kernel_name in ['wl', 'vh']:
            print(f"  Converting graphs with angle-based labels (method={angle_method}, axis={angle_axis}, bins={angle_bins})...")
            G_train_conv = convert_graphs_with_angle_labels(G_train, G_train_pos, angle_method, angle_axis, angle_bins)
            G_test_conv = convert_graphs_with_angle_labels(G_test, G_test_pos, angle_method, angle_axis, angle_bins)
        else:
            print(f"  Warning: {kernel_name} doesn't support angle features, using original graphs")
            G_train_conv = G_train
            G_test_conv = G_test
    
    elif feature_type == 'direction':
        if kernel_name in ['wl', 'vh']:
            print(f"  Converting graphs with direction-based labels (directions={angle_bins})...")
            G_train_conv = convert_graphs_with_direction_labels(G_train, G_train_pos, angle_bins)
            G_test_conv = convert_graphs_with_direction_labels(G_test, G_test_pos, angle_bins)
        else:
            print(f"  Warning: {kernel_name} doesn't support direction features, using original graphs")
            G_train_conv = G_train
            G_test_conv = G_test
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Initialize the kernel
    if kernel_name == 'graphlet':
        gk = GraphletSampling(normalize=True, sampling=None, k=graphlet_k)
        label = f'GraphletSampling (k={graphlet_k})'
    
    elif kernel_name == 'wl':
        if feature_type == 'position':
            label = f'Weisfeiler-Lehman (iter={wl_iter}, grid={grid_size})'
        elif feature_type == 'angle':
            label = f'Weisfeiler-Lehman (iter={wl_iter}, {angle_method}-angle, bins={angle_bins})'
        else:
            label = f'Weisfeiler-Lehman (iter={wl_iter}, direction, dirs={angle_bins})'
        gk = WeisfeilerLehman(n_iter=wl_iter, base_graph_kernel=VertexHistogram, normalize=True)
    
    elif kernel_name == 'vh':
        if feature_type == 'position':
            label = f'VertexHistogram (grid={grid_size})'
        elif feature_type == 'angle':
            label = f'VertexHistogram ({angle_method}-angle, bins={angle_bins})'
        else:
            label = f'VertexHistogram (direction, dirs={angle_bins})'
        gk = VertexHistogram(normalize=True)
    
    elif kernel_name == 'random_walk':
        gk = RandomWalk(normalize=True, lamda=rw_lambda, step=rw_steps)
        label = f'RandomWalk (steps={rw_steps}, λ={rw_lambda})'
    
    elif kernel_name == 'graphhopper':
        gk = GraphHopper(normalize=True)
        label = f'GraphHopper (with positions)'
    
    else:
        raise ValueError(f"Unknown grakel kernel: {kernel_name}")

    print(f"  Computing on training set ({len(G_train_conv)} graphs)...")
    start = time.time()
    K_train = gk.fit_transform(G_train_conv)
    K_train = np.nan_to_num(K_train, nan=0.0)
    print(f"    Time: {time.time() - start:.2f}s")

    print(f"  Computing on test set ({len(G_test_conv)} graphs)...")
    start = time.time()
    K_test = gk.transform(G_test_conv)
    K_test = np.nan_to_num(K_test, nan=0.0)
    print(f"    Time: {time.time() - start:.2f}s")

    return K_train, K_test, label