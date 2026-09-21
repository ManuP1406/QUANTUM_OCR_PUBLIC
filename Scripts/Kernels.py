"""
Graph Kernel Classification on MNIST digits - Main module.
This module provides the main classification function.
"""

import os
import sys
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score, precision_score,
                             recall_score)

# Add parent directory to path to find graph_kernel_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_kernel_utils.data_loader import load_graphs_data, filter_and_limit_classes
from graph_kernel_utils.kernel_computer import compute_grakel_kernel
from graph_kernel_utils.visualization import save_misclassified


def train_and_evaluate(K_train, K_test, y_train, y_test, kernel_label: str):
    """Grid-search SVM with precomputed kernel and return detailed metrics."""
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000]}

    print(f"\n{'='*60}")
    print(f"  {kernel_label}")
    print(f"{'='*60}")
    print(f"  Training: {len(y_train)} samples  |  Test: {len(y_test)} samples")
    print(f"  Classes: {np.unique(y_train)}")

    K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=-1.0)
    K_test = np.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=-1.0)

    start = time.time()
    n_classes = len(np.unique(y_train))
    cv_folds = min(5, n_classes) if n_classes > 1 else 2
    
    grid = GridSearchCV(
        SVC(kernel='precomputed', max_iter=10000), 
        param_grid,
        cv=cv_folds,
        scoring='accuracy', 
        n_jobs=-1
    )
    
    try:
        grid.fit(K_train, y_train)
        best_c = grid.best_params_['C']
        cv_acc = grid.best_score_
        print(f"  Best C = {best_c}  |  CV acc = {cv_acc:.4f}")
        print(f"  Grid search time: {time.time() - start:.2f}s")

        best_model = grid.best_estimator_
        y_pred = best_model.predict(K_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        class_report = classification_report(y_test, y_pred, digits=4, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"  Test accuracy: {acc:.4f}")
        print(f"  Macro F1-score: {f1_macro:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return {
            'y_pred': y_pred,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'best_c': best_c,
            'cv_accuracy': cv_acc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'best_model': best_model
        }
        
    except Exception as e:
        print(f"  Error: {e}, using fallback...")
        svm = SVC(kernel='precomputed', C=1.0)
        svm.fit(K_train, y_train)
        y_pred = svm.predict(K_test)
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'y_pred': y_pred,
            'accuracy': acc,
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'best_c': 1.0,
            'cv_accuracy': 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, digits=4, output_dict=True),
            'best_model': svm
        }


def run_graph_kernel_classification(
    pkl_path=None,
    kernel='wl',
    classes=[1, 3, 5, 8],
    max_per_class=200,
    max_test_per_class=None,
    max_test=None,
    graphlet_k=3,
    wl_iter=3,
    rw_steps=3,
    rw_lambda=0.1,
    grid_size=10,
    feature_type='position',
    angle_method='max',
    angle_axis='x',
    angle_bins=8,
    save_misclassified_flag=True,
    n_errors=10,
    images_dir='images'
):
    """
    Main function for graph kernel classification with configurable parameters.
    """
    # Display configuration
    print(f"\n{'='*60}")
    print(f"RUNNING CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Kernel: {kernel}")
    print(f"Feature type: {feature_type}")
    if feature_type == 'angle':
        print(f"  Method: {angle_method}, Axis: {angle_axis}, Bins: {angle_bins}")
    elif feature_type == 'direction':
        print(f"  Directions: {angle_bins}")
    print(f"Classes: {classes}")
    print(f"Max per class (training): {max_per_class}")
    if max_test_per_class:
        print(f"Max per class (test): {max_test_per_class}")
    print(f"{'='*60}\n")

    # Resolve pickle path
    if pkl_path is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkl_path = os.path.join(script_dir, 'Data', 'mnist_graphs_grakel_filtered.pkl')

    if not os.path.isfile(pkl_path):
        print(f"ERROR: pickle file not found: {pkl_path}")
        return None

    # Load data
    G, y, G_pos, G_test, y_test, G_test_pos, G_edge_dist, G_test_edge_dist = load_graphs_data(pkl_path)

    # Limit training set
    G, y, G_pos, G_edge_dist = filter_and_limit_classes(
        G, y, G_pos, classes, max_per_class, G_edge_dist, dataset_type="training")
    
    G_test, y_test, G_test_pos, G_test_edge_dist = filter_and_limit_classes(
        G_test, y_test, G_test_pos, classes, None, G_test_edge_dist, dataset_type="test")
    
    if max_test_per_class is not None:
        print(f"\nLimiting test set to {max_test_per_class} samples per class:")
        G_test, y_test, G_test_pos, G_test_edge_dist = filter_and_limit_classes(
            G_test, y_test, G_test_pos, classes, max_test_per_class, G_test_edge_dist, dataset_type="test")

    # Compute kernel with enhanced features
    K_train, K_test, label = compute_grakel_kernel(
        G, G_test, kernel,
        G_train_pos=G_pos,
        G_test_pos=G_test_pos,
        graphlet_k=graphlet_k,
        wl_iter=wl_iter,
        rw_steps=rw_steps,
        rw_lambda=rw_lambda,
        grid_size=grid_size,
        feature_type=feature_type,
        angle_method=angle_method,
        angle_axis=angle_axis,
        angle_bins=angle_bins
    )

    results = train_and_evaluate(K_train, K_test, y, y_test, label)

    if save_misclassified_flag and results:
        save_misclassified(G_test, G_test_pos, y_test, results['y_pred'],
                          label, images_dir=images_dir,
                          n_errors=n_errors,
                          G_test_edge_distances=G_test_edge_dist)

    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")
    
    return {
        'kernel': kernel,
        'kernel_label': label,
        'feature_type': feature_type,
        'accuracy': results['accuracy'],
        'f1_macro': results['f1_macro'],
        'f1_weighted': results['f1_weighted'],
        'precision_macro': results['precision_macro'],
        'recall_macro': results['recall_macro'],
        'best_c': results['best_c'],
        'cv_accuracy': results['cv_accuracy'],
        'confusion_matrix': results['confusion_matrix'],
        'classification_report': results['classification_report'],
        'predictions': results['y_pred'],
        'true_labels': y_test,
        'params': {
            'classes': classes,
            'max_per_class': max_per_class,
            'max_test_per_class': max_test_per_class,
            'graphlet_k': graphlet_k,
            'wl_iter': wl_iter,
            'rw_steps': rw_steps,
            'rw_lambda': rw_lambda,
            'grid_size': grid_size,
            'feature_type': feature_type,
            'angle_method': angle_method,
            'angle_axis': angle_axis,
            'angle_bins': angle_bins
        }
    }


if __name__ == '__main__':
    # Example usage
    result = run_graph_kernel_classification(
        kernel='wl',
        classes=[1, 7],
        max_per_class=2000,
        max_test_per_class=600,
        grid_size=10,
        feature_type='angle',
        wl_iter=3,
        save_misclassified_flag=True
    )
    
    if result:
        print(f"\nResult: {result['accuracy']:.4f}")