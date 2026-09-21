"""
Quantum Kernel Classification module.
Provides run_quantum_kernel_classification function with per-pair optimized kernels.
"""

import os
import sys
import time
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score, precision_score,
                             recall_score)

import pickle
import json

from itertools import combinations

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import quantum kernel
from quantumKernels import QuantumMulticlassClassifier

# Import data loader
from graph_kernel_utils.data_loader import load_graphs_data, filter_and_limit_classes

from quantum_kernel_utils.graph_preprocessing import (
    graph_to_node_list, 
    compute_global_scale_factor, 
    scale_all_graphs, 
    verify_scaling
)

import time 


def load_both_datasets(orig_pkl_path, rep_pkl_path, classes, max_per_class, max_test_per_class):
    """
    Load both original and transformed datasets with the same filtering.
    Returns: (G_orig, y_orig, G_orig_pos, G_rep, y_rep, G_rep_pos, and test sets)
    """
    print("\n" + "="*60)
    print("LOADING BOTH DATASETS")
    print("="*60)
    
    # Load original data
    print(f"\nLoading original graphs from: {orig_pkl_path}")
    G_orig, y_orig, G_orig_pos, G_test_orig, y_test_orig, G_test_orig_pos, _, _ = load_graphs_data(orig_pkl_path)
    
    # Load transformed data
    print(f"\nLoading transformed graphs from: {rep_pkl_path}")
    G_rep, y_rep, G_rep_pos, G_test_rep, y_test_rep, G_test_rep_pos, _, _ = load_graphs_data(rep_pkl_path)
    
    # Verify same number of graphs
    assert len(G_orig) == len(G_rep), f"Training set size mismatch: {len(G_orig)} vs {len(G_rep)}"
    assert len(G_test_orig) == len(G_test_rep), f"Test set size mismatch: {len(G_test_orig)} vs {len(G_test_rep)}"
    assert np.array_equal(y_orig, y_rep), "Training labels mismatch"
    assert np.array_equal(y_test_orig, y_test_rep), "Test labels mismatch"
    
    print(f"\n Datasets aligned: {len(G_orig)} training, {len(G_test_orig)} test graphs")
    
    # Filter training set (same indices for both)
    G_orig, y_orig, G_orig_pos, _ = filter_and_limit_classes(
        G_orig, y_orig, G_orig_pos, classes, max_per_class, None, dataset_type="training")
    
    G_rep, y_rep, G_rep_pos, _ = filter_and_limit_classes(
        G_rep, y_rep, G_rep_pos, classes, max_per_class, None, dataset_type="training", verbose=False)
    
    # Filter test set
    G_test_orig, y_test_orig, G_test_orig_pos, _ = filter_and_limit_classes(
        G_test_orig, y_test_orig, G_test_orig_pos, classes, max_test_per_class, None, dataset_type="test")
    
    G_test_rep, y_test_rep, G_test_rep_pos, _ = filter_and_limit_classes(
        G_test_rep, y_test_rep, G_test_rep_pos, classes, max_test_per_class, None, dataset_type="test" , verbose=False)
    
    return (G_orig, y_orig, G_orig_pos, G_rep, y_rep, G_rep_pos,
            G_test_orig, y_test_orig, G_test_orig_pos, G_test_rep, y_test_rep, G_test_rep_pos)


def run_quantum_kernel_classification(
    orig_pkl_path=None,
    rep_pkl_path=None,
    classes=[1, 3, 5, 8],
    max_per_class=20,
    max_test_per_class=100,
    save_misclassified_flag=False,
    # Quantum kernel specific parameters
    encoding_method="multi_run",
    LAYERS=1,
    gamma=1.0,
    encoding_type="with_hyperfine",
    N_samples=200,
    use_correlation="z",
    n_jobs=4,
    n_jobs_quantum=2,
    optimize_parameters=True,
    fixed_params=None,
    # Model saving/loading  
    save_model_path=None,
    load_model_path=None,
    skip_training=False,
    save_summary_path=None
):
    """Multiclass classification with per-pair optimized kernels.
    
    Args:
        save_model_path: If provided, save the trained model to this path
        load_model_path: If provided, load model from this path instead of training
        skip_training: If True, only load model and test (requires load_model_path)
        save_summary_path: If provided, save summary to this txt file
        n_jobs: Number of parallel jobs for pair training and prediction
        n_jobs_quantum: Number of parallel jobs for quantum simulations (RAM intensive)
    """
    
    # Create summary dictionary
    summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'classes': classes,
        'max_per_class': max_per_class,
        'max_test_per_class': max_test_per_class,
        'LAYERS': LAYERS,
        'gamma': gamma,
        'encoding_type': encoding_type,
        'encoding_method': encoding_method,
        'N_samples': N_samples,
        'use_correlation': use_correlation,
        'n_jobs': n_jobs,
        'n_jobs_quantum': n_jobs_quantum,
        'optimize_parameters': optimize_parameters
    }
    

    if (skip_training or load_model_path) and load_model_path and os.path.exists(load_model_path):
        print(f"\n{'='*60}")
        print(f"LOADING PRE-TRAINED MODEL")
        print(f"{'='*60}")
        print(f"Model path: {load_model_path}")
        
        classifier = QuantumMulticlassClassifier.load(load_model_path)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        orig_pkl_path = os.path.join(base_dir, '..', 'Data', 'mnist_graphs_grakel_filtered.pkl')
        rep_pkl_path = os.path.join(base_dir, '..', 'Data', 'mnist_graphs_grakel_repulsion_filtered.pkl')
        orig_pkl_path = os.path.abspath(orig_pkl_path)
        rep_pkl_path = os.path.abspath(rep_pkl_path)
        
        (_, _, _, 
         _, _, _,
         G_test_orig, y_test_orig, G_test_orig_pos,
         G_test_rep, y_test_rep, G_test_rep_pos) = load_both_datasets(
            orig_pkl_path, rep_pkl_path, classes, max_per_class, max_test_per_class
        )
        
        print("\nConverting test graphs to quantum kernel format...")
        test_graphs_converted = []
        for i, graph_orig in enumerate(G_test_orig):
            orig_pos = G_test_orig_pos[i]
            rep_pos = G_test_rep_pos[i]
            node_list = graph_to_node_list(graph_orig, orig_pos, rep_pos, features_type="angle+pos")
            test_graphs_converted.append(node_list)
        
        print("\n" + "="*60)
        print("PREDICTING ON TEST SET")
        print("="*60)
        start_time = time.time()
        y_pred = classifier.predict(test_graphs_converted)
        prediction_time = time.time() - start_time
        
        acc = accuracy_score(y_test_orig, y_pred)
        f1_macro = f1_score(y_test_orig, y_pred, average='macro')
        f1_weighted = f1_score(y_test_orig, y_pred, average='weighted')
        precision_macro = precision_score(y_test_orig, y_pred, average='macro')
        recall_macro = recall_score(y_test_orig, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test_orig, y_pred)
        
        print(f"\nTest accuracy: {acc:.4f}")
        print(f"Macro F1-score: {f1_macro:.4f}")
        print(f"Prediction time: {prediction_time:.2f}s")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        summary['mode'] = 'load'
        summary['accuracy'] = acc
        summary['f1_macro'] = f1_macro
        summary['f1_weighted'] = f1_weighted
        summary['precision_macro'] = precision_macro
        summary['recall_macro'] = recall_macro
        summary['confusion_matrix'] = conf_matrix.tolist()
        summary['prediction_time'] = prediction_time
        summary['loaded_from'] = load_model_path
        
        if save_summary_path:
            save_summary_to_file(summary, save_summary_path)
        
        return {
            'kernel': 'quantum',
            'kernel_label': f"Quantum (L={LAYERS}, {encoding_method})",
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'true_labels': y_test_orig,
            'prediction_time': prediction_time,
            'loaded_from': load_model_path
        }
    

    print(f"\n{'='*60}")
    print(f"QUANTUM KERNEL CLASSIFICATION (Training from scratch)")
    print(f"{'='*60}")
    print(f"Classes: {classes} ({len(classes)} classes -> {len(list(combinations(classes, 2)))} pairs)")
    print(f"Max per class (training): {max_per_class}")
    print(f"Max per class (test): {max_test_per_class}")
    print(f"\nQuantum Kernel Parameters:")
    print(f"  Layers: {LAYERS}")
    print(f"  Gamma: {gamma}")
    print(f"  Encoding: {encoding_type} / {encoding_method}")
    print(f"  N_samples: {N_samples}")
    print(f"  Use correlation: {use_correlation}")
    print(f"  Optimize parameters: {optimize_parameters}")
    print(f"  n_jobs (pair training/prediction): {n_jobs}")
    print(f"  n_jobs_quantum (simulations): {n_jobs_quantum}")
    if save_model_path:
        print(f"  Model will be saved to: {save_model_path}")
    print(f"{'='*60}\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if orig_pkl_path is None:
        orig_pkl_path = os.path.join(base_dir, '..', 'Data', 'mnist_graphs_grakel_filtered.pkl')
    if rep_pkl_path is None:
        rep_pkl_path = os.path.join(base_dir, '..', 'Data', 'mnist_graphs_grakel_repulsion_filtered.pkl')
    
    orig_pkl_path = os.path.abspath(orig_pkl_path)
    rep_pkl_path = os.path.abspath(rep_pkl_path)
    
    print("Original file path:", orig_pkl_path)
    print("Exists?", os.path.isfile(orig_pkl_path))
    print("Rep file path:", rep_pkl_path)
    print("Exists?", os.path.isfile(rep_pkl_path))
    
    if not os.path.isfile(orig_pkl_path):
        print(f"ERROR: Original file not found: {orig_pkl_path}")
        return None
    if not os.path.isfile(rep_pkl_path):
        print(f"ERROR: Transformed file not found: {rep_pkl_path}")
        return None
    
    (G_orig, y_orig, G_orig_pos, 
     G_rep, y_rep, G_rep_pos,
     G_test_orig, y_test_orig, G_test_orig_pos,
     G_test_rep, y_test_rep, G_test_rep_pos) = load_both_datasets(
        orig_pkl_path, rep_pkl_path, classes, max_per_class, max_test_per_class
    )
    
    print(f"\nFinal training set: {len(G_orig)} graphs")
    print(f"Final test set: {len(G_test_orig)} graphs")
    
    print("\nConverting graphs to quantum kernel format...")
    print("  - Using ORIGINAL positions for angle features")
    print("  - Using TRANSFORMED positions for atomic coordinates")
    
    features_type = "angle+pos"
    
    train_graphs_converted = []
    for i, graph_orig in enumerate(G_orig):
        orig_pos = G_orig_pos[i]
        rep_pos = G_rep_pos[i]
        node_list = graph_to_node_list(graph_orig, orig_pos, rep_pos, features_type=features_type)
        train_graphs_converted.append(node_list)
    
    test_graphs_converted = []
    for i, graph_orig in enumerate(G_test_orig):
        orig_pos = G_test_orig_pos[i]
        rep_pos = G_test_rep_pos[i]
        node_list = graph_to_node_list(graph_orig, orig_pos, rep_pos, features_type=features_type)
        test_graphs_converted.append(node_list)
    
    if train_graphs_converted:
        print(f"\nDEBUG: First converted graph")
        print(f"  Number of nodes: {len(train_graphs_converted[0])}")
        if train_graphs_converted[0]:
            print(f"  First node: {train_graphs_converted[0][0]}")
            print(f"  First node length: {len(train_graphs_converted[0][0])}")
            print(f"  number_of_features = {len(train_graphs_converted[0][0]) - 2}")
    
    if fixed_params == True:
        n_features = len(train_graphs_converted[0][0]) - 2 if train_graphs_converted else 0
        t_params = [500] * LAYERS # 210.2
        theta_params = [0.1 * np.pi] * LAYERS # 1.5
        if encoding_type == "with_hyperfine" and n_features > 0:
            freq_params = [1.0] * (LAYERS * n_features)
        else:
            freq_params = []
        fixed_params = t_params + theta_params + freq_params
        print(f"\nUsing default parameters: {fixed_params}")
    else:
        fixed_params = None
    
    print("\n" + "="*60)
    print("CREATING MULTICLASS CLASSIFIER (Per-pair kernels)")
    print("="*60)
    

    

    
    classifier = QuantumMulticlassClassifier(
        classes=classes,
        n_features=n_features,
        params_in=fixed_params,
        encoding_type=encoding_type,
        encoding_method=encoding_method,
        N_samples=N_samples,
        use_correlation=use_correlation,
        n_jobs=n_jobs,
        n_jobs_quantum=n_jobs_quantum,
        **{'LAYERS': LAYERS, 'gamma': gamma, 'find_optimal_gamma': optimize_parameters, 'use_weights': optimize_parameters}
    )
    
    print("\n" + "="*60)
    print("FITTING QUANTUM MULTICLASS CLASSIFIER")
    print("="*60)
    
    start_time = time.time()
    classifier.fit(train_graphs_converted, y_orig)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f}s")
    
    if save_model_path:
        save_dir = os.path.dirname(save_model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        classifier.save(save_model_path)
    
    print("\n" + "="*60)
    print("PREDICTING ON TEST SET")
    print("="*60)
    
    start_time = time.time()
    y_pred = classifier.predict(test_graphs_converted)
    prediction_time = time.time() - start_time
    
    acc = accuracy_score(y_test_orig, y_pred)
    f1_macro = f1_score(y_test_orig, y_pred, average='macro')
    f1_weighted = f1_score(y_test_orig, y_pred, average='weighted')
    precision_macro = precision_score(y_test_orig, y_pred, average='macro')
    recall_macro = recall_score(y_test_orig, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test_orig, y_pred)
    class_report = classification_report(y_test_orig, y_pred, digits=4, output_dict=True)
    
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    print(f"Prediction time: {prediction_time:.2f}s")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    if save_misclassified_flag:
        from graph_kernel_utils.visualization import save_misclassified
        kernel_label = f"Quantum (L={LAYERS}, {encoding_method})"
        save_misclassified(G_test_rep, G_test_rep_pos, y_test_orig, y_pred,
                          kernel_label, images_dir='images',
                          n_errors=10,
                          G_test_edge_distances=None)
    
    summary['mode'] = 'train'
    summary['training_time'] = training_time
    summary['accuracy'] = acc
    summary['f1_macro'] = f1_macro
    summary['f1_weighted'] = f1_weighted
    summary['precision_macro'] = precision_macro
    summary['recall_macro'] = recall_macro
    summary['confusion_matrix'] = conf_matrix.tolist()
    summary['prediction_time'] = prediction_time
    summary['circuit_params'] = classifier.circuit_params
    summary['n_pair_classifiers'] = len(classifier.pair_classifiers)
    
    if save_model_path:
        summary['model_saved_to'] = save_model_path
    
    if save_summary_path:
        save_summary_to_file(summary, save_summary_path)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")
    
    return {
        'kernel': 'quantum',
        'kernel_label': f"Quantum (L={LAYERS}, {encoding_method})",
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred,
        'true_labels': y_test_orig,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'params': {
            'classes': classes,
            'max_per_class': max_per_class,
            'max_test_per_class': max_test_per_class,
            'LAYERS': LAYERS,
            'gamma': gamma,
            'encoding_type': encoding_type,
            'encoding_method': encoding_method,
            'N_samples': N_samples,
            'use_correlation': use_correlation,
            'n_jobs': n_jobs,
            'n_jobs_quantum': n_jobs_quantum
        }
    }


def save_summary_to_file(summary, filepath):
    """Save summary dictionary to a formatted text file."""
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTUM KERNEL CLASSIFICATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Mode: {summary['mode']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("DATASET CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Classes: {summary['classes']}\n")
        f.write(f"Number of classes: {len(summary['classes'])}\n")
        f.write(f"Max per class (training): {summary['max_per_class']}\n")
        f.write(f"Max per class (test): {summary['max_test_per_class']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("QUANTUM KERNEL CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Layers: {summary['LAYERS']}\n")
        f.write(f"Gamma: {summary['gamma']}\n")
        f.write(f"Encoding type: {summary['encoding_type']}\n")
        f.write(f"Encoding method: {summary['encoding_method']}\n")
        f.write(f"N_samples: {summary['N_samples']}\n")
        f.write(f"Use correlation: {summary['use_correlation']}\n")
        f.write(f"Optimize parameters: {summary['optimize_parameters']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PARALLELISM CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"n_jobs (pair training/prediction): {summary['n_jobs']}\n")
        f.write(f"n_jobs_quantum (simulations): {summary['n_jobs_quantum']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Test Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"Macro F1-score: {summary['f1_macro']:.4f}\n")
        f.write(f"Weighted F1-score: {summary['f1_weighted']:.4f}\n")
        f.write(f"Macro Precision: {summary['precision_macro']:.4f}\n")
        f.write(f"Macro Recall: {summary['recall_macro']:.4f}\n")
        f.write(f"Prediction Time: {summary['prediction_time']:.2f}s\n")
        
        if 'training_time' in summary:
            f.write(f"Training Time: {summary['training_time']:.2f}s\n")
        
        if 'n_pair_classifiers' in summary:
            f.write(f"Number of Pair Classifiers: {summary['n_pair_classifiers']}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        
        cm = summary['confusion_matrix']
        f.write(" " * 8)
        for c in summary['classes']:
            f.write(f"{c:>8}")
        f.write("\n")
        
        for i, c in enumerate(summary['classes']):
            f.write(f"{c:>8}")
            for j in range(len(cm[i])):
                f.write(f"{cm[i][j]:>8}")
            f.write("\n")
        
        if 'circuit_params' in summary:
            f.write("\n" + "-"*70 + "\n")
            f.write("CIRCUIT PARAMETERS\n")
            f.write("-"*70 + "\n")
            params = summary['circuit_params']
            f.write(f"t parameters: {params[:summary['LAYERS']]}\n")
            f.write(f"theta parameters: {params[summary['LAYERS']:2*summary['LAYERS']]}\n")
            if len(params) > 2 * summary['LAYERS']:
                f.write(f"freq parameters (first 10): {params[2*summary['LAYERS']:2*summary['LAYERS']+10]}...\n")
        
        if 'model_saved_to' in summary:
            f.write("\n" + "-"*70 + "\n")
            f.write("MODEL\n")
            f.write("-"*70 + "\n")
            f.write(f"Model saved to: {summary['model_saved_to']}\n")
        
        if 'loaded_from' in summary:
            f.write("\n" + "-"*70 + "\n")
            f.write("MODEL\n")
            f.write("-"*70 + "\n")
            f.write(f"Model loaded from: {summary['loaded_from']}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*70 + "\n")


if __name__ == '__main__':
    date = time.strftime("%d-%m-%Y")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    result = run_quantum_kernel_classification(
        classes=[0,1, 2 , 3 , 4 , 5 , 6 ,  7, 8 , 9], #[0,3,8,9],#
        max_per_class=40,
        max_test_per_class=15,
        LAYERS=1,
        N_samples=500,
        optimize_parameters=True,
        n_jobs=45 , #max(1, os.cpu_count() - 2),
        n_jobs_quantum=49,   #49
        save_model_path=f'models/quantum_model_distOpt_{date}.pkl',
        save_summary_path=f'models/summary_distOpt_{date}.txt',
        fixed_params=True
    )               
    
    if result:
        print(f"\nTest result: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['f1_macro']:.4f}")
        print(f"Confusion Matrix:\n{result['confusion_matrix']}")
    
    # Load and test only
    # result = run_quantum_kernel_classification(
    #     classes=[1, 2, 7],
    #     max_test_per_class=10,
    #     load_model_path=f'models/quantum_model_{date}.pkl',  # Load existing model
    #     skip_training=True,
    #     n_jobs=2
    # )
        