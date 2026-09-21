import pickle

import numpy as np
import warnings
import multiprocessing
from collections import defaultdict
from itertools import combinations

from skopt.callbacks import EarlyStopper

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, cross_validate
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from skopt import gp_minimize
from skopt.space import Real

from quantum_kernel_utils.quantum_core import quantum_loop, kernel_js
from quantum_kernel_utils.pulser_functions import R_interatomic
from quantum_kernel_utils.graph_preprocessing import (
    compute_global_scale_factor, 
    compute_local_scale_factors, 
    scale_all_graphs_local,
    scale_all_graphs,
    verify_scaling

)

from sklearn.model_selection import train_test_split


class QuantumEvolutionKernel:
    def __init__(self, graphs, labels, 
                initial_guess=None,
                bayesian_optimization=True, 
                LAYERS=1, 
                gamma=1.0, 
                grid_search_for_C=True, 
                find_optimal_gamma=True,
                encoding_type="with_hyperfine",
                encoding_method="multi_run",
                N_samples=500,
                n_valuations=20,
                param_freq=False,
                use_correlation=None,
                params_in=None, 
                n_jobs=32,
                n_jobs_quantum=4,       
                use_weights=False,
                min_atom_distance=4.1,
                precomputed_distributions=None,
                n_features=4):
        """
        Initialize the Quantum Kernel model.
        
        Parameters:
        - precomputed_distributions: if provided, skip quantum_loop and use these distributions
        - n_features: number of features per node (default 4 for angle+pos)
        """
        self.bayesian_optimization = bayesian_optimization
        self.graphs = graphs
        self.labels = labels
        self.initial_guess = initial_guess
        self.LAYERS = LAYERS
        assert gamma > 0, "Gamma must be positive."
        self.gamma = -gamma
        self.grid_search_for_C = grid_search_for_C
        self.find_optimal_gamma = find_optimal_gamma
        self.encoding_type = encoding_type
        self.encoding_method = encoding_method
        self.N_samples = N_samples
        self.n_valuations = n_valuations
        self.param_freq = param_freq
        self.params_in = params_in
        self.use_correlation = use_correlation
        self.n_jobs = n_jobs
        self.n_jobs_quantum = n_jobs_quantum
        
        # Set number_of_features
        if precomputed_distributions is not None:
            self.number_of_features = n_features
        elif graphs and len(graphs) > 0:
            self.number_of_features = len(graphs[0][0]) - 2
        else:
            self.number_of_features = n_features
            
        self.use_weights = use_weights
        self.min_atom_distance = min_atom_distance
        self.precomputed_distributions = precomputed_distributions
        
        # Store circuit parameters
        if params_in is not None:
            self.circuit_params = params_in
            print(f"Using fixed circuit parameters: {self.circuit_params}")
        else:
            self.circuit_params = None
            print("Circuit parameters will be optimized during fit")

        if graphs and len(graphs) > 0:
            print("Size of the Graphs dataset: ", len(graphs))
        else:
            print("Using precomputed distributions - no graphs provided")
            
        # ===== APPLY POSITION SCALING FOR PULSER (skip if using precomputed) =====
        if precomputed_distributions is None:
            print("\n" + "="*60)
            print("APPLYING POSITION SCALING FOR PULSER")
            print("="*60)
            
            scale_factors = compute_local_scale_factors(self.graphs, self.min_atom_distance)
            self.graphs_scaled = scale_all_graphs_local(self.graphs, scale_factors)
           
            
       
            print("="*60 + "\n")
        else:
            # Use precomputed distributions, skip scaling
           
            self.graphs_scaled = graphs
            print("Using precomputed distributions - skipping position scaling")
        
        # Calculate expected number of kernels
        if self.encoding_method == "multi_run":
            self.expected_kernels = self.number_of_features
        elif self.encoding_method == "single_run":
            self.expected_kernels = self.number_of_features * 2
        else:
            self.expected_kernels = 2
            
        print("------------------QUANTUM KERNEL------------------")
        if graphs and len(graphs) > 0:
            print("Size of the Graphs dataset: ", len(graphs))
        else:
            print("Size of the Graphs dataset: using precomputed distributions")
        print("Quantum Kernel Initialized")
        print("LAYERS: ", LAYERS)
        print("Gamma: ", gamma)
        print("Grid Search for C: ", grid_search_for_C)
        print("Initial Guess: ", initial_guess)
        print("Bayesian Optimization: ", bayesian_optimization)
        print("Find Optimal Gamma: ", find_optimal_gamma)
        print("Encoding type: ", encoding_type)
        print("Encoding method: ", encoding_method)
        print("Number of features: ", self.number_of_features)
        print("Expected number of kernels: ", self.expected_kernels)
        print("Min atom distance: ", min_atom_distance)
       
        print("Precomputed distributions: ", precomputed_distributions is not None)
        print("-------------------------------------------------")
        print("cpu_count: ", multiprocessing.cpu_count())
        print("n_jobs: ", n_jobs)
        print("R_interatomic: ", R_interatomic)
        print("-------------------------------------------------")
        
        if params_in is not None:
            self.optimal_parameters = params_in
            self.initial_guess = None
            print(f"Using fixed parameters from params_in: {self.optimal_parameters}")
        
        elif initial_guess is not None:
            assert "t" in initial_guess and "theta" in initial_guess, \
                "Initial guess must contain 't' and 'theta' keys."
            self.initial_guess = initial_guess
            self.optimal_parameters = None
            print(f"Using provided initial_guess for optimization: {self.initial_guess}")
            
        else:
            self.initial_guess = {
                "t": np.random.uniform(16, 500, LAYERS), 
                "theta": np.random.uniform(0, np.pi, LAYERS)
            }
            self.optimal_parameters = None
            print(f"No parameters provided. Generated random initial_guess: {self.initial_guess}")
    
    def optimize_gamma_and_weights(self, final_distributions):
        """Optimize gamma, C, and feature weights with nested cross-validation + Bayesian optimization on weights."""
        gammas = np.logspace(2, -1, 5)  
        Cs = np.logspace(-1, 2, 4)    
        best_score = 0
        best_gamma = None
        best_C = None
        best_weights = None
        
        n_runs = len(final_distributions[0])  
        N = n_runs * 2 
        
        
        if N == 0:
            print(f"Warning: No kernels to optimize (N={N}), using default weights")
            return self.gamma, None, 1.0
        
        n_samples = len(self.labels)
        n_folds = min(5, n_samples)
        if n_folds < 2:
            print(f"Warning: Only {n_samples} samples, using default gamma and weights")
            return self.gamma, None, 1.0
        
        outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=1406)
        
        print(f"\nOptimizing with nested cross-validation")
        print(f"  Total weight components: {N}")
        print(f"  Using weights: {self.use_weights}")
        print(f"  n_samples: {n_samples}, n_folds: {n_folds}")
        print(f"  Gammas to test: {len(gammas)}")
        print(f"  C values to test: {len(Cs)}")

        for gamma_val in gammas:
            gamma_val = float(gamma_val)
            print(f"\nTesting gamma = {gamma_val:.4f}")
            
            current_weights = None
            if self.use_weights:
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=1406)
                space = [Real(0.0, 1.0) for _ in range(N)]

                def objective(weights_list):
                    w = np.array(weights_list)
                    w /= (np.sum(w) + 1e-10)

                    fold_scores = []
                    for tr_idx, val_idx in inner_cv.split(final_distributions):
                        tr_dist = [final_distributions[i] for i in tr_idx]
                        val_dist = [final_distributions[i] for i in val_idx]
                        tr_lbl = self.labels[tr_idx]
                        val_lbl = self.labels[val_idx]

                        K_tr = kernel_js(tr_dist, tr_dist, -gamma_val, w)
                        K_val = kernel_js(val_dist, tr_dist, -gamma_val, w)

                        best_c_score = 0.0
                        for c in [0.1, 1.0, 10.0]:
                            try:
                                svm = SVC(kernel='precomputed', C=c)
                                svm.fit(K_tr, tr_lbl)
                                s = f1_score(val_lbl, svm.predict(K_val), average='macro')
                                best_c_score = max(best_c_score, s)
                            except Exception:
                                pass
                        fold_scores.append(best_c_score)

                    return -np.mean(fold_scores)

                try:
                    result = gp_minimize(
                        objective,
                        space,
                        n_calls=30,
                        n_initial_points=8,
                        random_state=1406,
                        verbose=False,
                        acq_func="EI",
                    )
                    current_weights = np.array(result.x)
                    current_weights /= (np.sum(current_weights) + 1e-10)
                    print(f"  Best weights: {np.round(current_weights, 3)}")
                    print(f"  Best inner score: {-result.fun:.4f}")
                except Exception as e:
                    print(f"  Bayesian optimization failed, falling back to uniform weights: {e}")
                    current_weights = np.ones(N) / N
            
            kernel_matrix = kernel_js(final_distributions, final_distributions, -gamma_val, current_weights)
            
            def evaluate_c(c):
                try:
                    svm = SVC(kernel='precomputed', C=c)
                    f1_scores = cross_val_score(svm, kernel_matrix, self.labels, 
                                                cv=outer_cv, scoring='f1_macro', 
                                                n_jobs=1)
                    mean_f1 = np.mean(f1_scores)
                    return c, mean_f1
                except Exception:
                    return c, 0.0
            
            c_results = Parallel(n_jobs=min(len(Cs), 1))(   # n_jobs = 1 per via di problemi con hpc
                delayed(evaluate_c)(c) for c in Cs
            )
            
            best_c_for_this_gamma = None
            best_f1_for_this_gamma = 0
            for c, score in c_results:
                print(f"    C={c:.4f}, F1={score:.4f}")
                if score > best_f1_for_this_gamma:
                    best_f1_for_this_gamma = score
                    best_c_for_this_gamma = c
            
            if best_f1_for_this_gamma > best_score:
                best_score = best_f1_for_this_gamma
                best_gamma = gamma_val
                best_C = best_c_for_this_gamma
                best_weights = current_weights
                print(f"  New best: gamma={best_gamma}, C={best_C}, F1={best_score:.4f}")

        if best_gamma is None:
            print("No valid gamma found, using defaults")
            return self.gamma, None, 1.0

        print(f"\nOptimization complete")
        print(f"  Best gamma: {best_gamma:.4f}")
        print(f"  Best C: {best_C:.4f}")
        print(f"  Best F1: {best_score:.4f}")
        
        return -best_gamma, best_weights, best_C

    def compute_quantum_distributions(self, params):
        """Compute quantum distributions. If precomputed, return them."""
        if self.precomputed_distributions is not None:
            print("Using precomputed distributions")
            return self.precomputed_distributions
        if params is None:
            params = self.circuit_params
        
        if params is None:
            raise ValueError("No circuit parameters provided. Either pass params or ensure circuit_params is set.")
        print("Centering graphs for quantum simulations...")
        bad_graphs = 0

        for graph in self.graphs_scaled:
            coords = np.array([node[:2] for node in graph])
            coords -= coords.mean(axis=0)
            for i, node in enumerate(graph):
                node[0] = coords[i, 0]
                node[1] = coords[i, 1]
            # check distances from origin
            distances = np.linalg.norm(coords, axis=1)
            if np.any(distances > 50):
                bad_graphs += 1

        # after processing all graphs
        if bad_graphs > 0:
            raise ValueError(f"{bad_graphs} graphs have nodes with distance > 50 from origin.")
     

     
            
        print(f"  Running quantum simulations with {self.n_jobs_quantum} parallel jobs")
        distributions = Parallel(n_jobs=self.n_jobs_quantum)(
            delayed(quantum_loop)(
                params, 
                graph, 
                self.LAYERS,
                self.N_samples,
                self.encoding_type,
                self.encoding_method,
                use_correlation=self.use_correlation
            ) for graph in self.graphs_scaled
        )
        return distributions

    def grid_search_C(self, kernel_matrix):
        C_vals = np.logspace(-2, 3, 6)
        best_score = 0
        best_C = C_vals[0]
        cv = KFold(n_splits=5, shuffle=True)

        for c in C_vals:
            svm = SVC(kernel='precomputed', C=c)
            f1 = cross_val_score(svm, kernel_matrix, self.labels, 
                                cv=cv, scoring='f1_macro').mean()
            if f1 > best_score:
                best_score = f1
                best_C = c

        return best_C
    
    def evaluate_svm(self, kernel_matrix, C):
        cv = KFold(n_splits=5, shuffle=True)
        svm = SVC(kernel='precomputed', C=C)
        scores = cross_validate(svm, kernel_matrix, self.labels, cv=cv,
                                scoring=['accuracy', 'f1_macro'])

        acc_scores = scores['test_accuracy']
        f1_scores = scores['test_f1_macro']
        svm.fit(kernel_matrix, self.labels)

        print(f"CV Accuracy: {np.mean(acc_scores):.4f} (±{np.std(acc_scores):.4f})")
        print(f"CV F1: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
        return f1_scores, svm

    def SVM(self, params, C=None, final_distributions=None):
        """Coordinate kernel computation and SVM evaluation."""
        warnings.filterwarnings("ignore", category=UserWarning)

        if final_distributions is None:
            final_distributions = self.compute_quantum_distributions(params)
        
        if self.find_optimal_gamma:
            self.gamma, self.weights, C = self.optimize_gamma_and_weights(final_distributions)
            Kernel_SVM = kernel_js(final_distributions, final_distributions, self.gamma, self.weights)
        else:
            self.weights = None
            Kernel_SVM = kernel_js(final_distributions, final_distributions, self.gamma, self.weights)
            if self.grid_search_for_C and C is None:
                C = self.grid_search_C(Kernel_SVM)

        f1_scores, svm = self.evaluate_svm(Kernel_SVM, C)
        return f1_scores, svm, C

    def compute_optimal_parameters(self):
        """Compute the optimal parameters for the quantum circuit."""
        def objective_function(params):
            final_distributions = self.compute_quantum_distributions(params)
            scores, _, _ = self.SVM(params, final_distributions=final_distributions)
            return -np.mean(scores)
        
        t_bounds = [(25, 100) for _ in range(self.LAYERS)]
        theta_bounds = [(0, 0.4 * np.pi) for _ in range(self.LAYERS)] 
        
        if self.param_freq:
            freq_features_bounds = [(0.8, 1.2) for _ in range(self.LAYERS * self.number_of_features)]
        else:
            freq_features_bounds = []
        bounds = t_bounds + theta_bounds + freq_features_bounds
        
        res = gp_minimize(objective_function, bounds, 
                         n_calls=self.n_valuations, 
                         random_state=1406,
                         verbose=True)
        
        return res.x
    
    def fit(self):
        """Fit the model."""
        if self.params_in is None:
            self.optimal_parameters = self.compute_optimal_parameters()
            self.circuit_params = self.optimal_parameters 
            print("Optimal parameters found: ", self.optimal_parameters)
        else:
            self.circuit_params = self.params_in
            self.optimal_parameters = self.params_in
        
        # Compute distributions
        print("Running quantum simulations on training set...")
        self.train_distributions = self.compute_quantum_distributions(self.circuit_params)
        
        # Optimize kernel
        f1_scores, svm, C = self.SVM(self.optimal_parameters, final_distributions=self.train_distributions) 
        
        self.svm = svm
        print(f"✅ Fit completed. Best C: {C}, Gamma: {self.gamma}")
        if self.weights is not None:
            print(f"Optimized Weights (Beta): {self.weights}")

        # Final validation
        final_kernel = kernel_js(self.train_distributions, self.train_distributions, self.gamma, self.weights)
        
        n_folds = min(5, len(self.labels))
        acc_scores = cross_val_score(self.svm, 
                                    final_kernel, 
                                    self.labels, 
                                    cv=KFold(n_splits=n_folds, shuffle=True, random_state=1406))
        
        print(f"Final Cross-Validation Accuracy: {np.mean(acc_scores):.4f} (±{np.std(acc_scores):.4f})")
    
        return self.svm, np.mean(acc_scores)
    
    def predict_from_distributions(self, test_distributions):
        """
        Predict labels using precomputed test distributions.
        """
        assert hasattr(self, 'svm'), "Model has not been fitted yet."
        
        new_kernel_matrix = kernel_js(test_distributions, self.train_distributions, self.gamma, self.weights)
        return self.svm.predict(new_kernel_matrix)
    
    def predict(self, new_graphs):
        """Predict labels for new graphs."""
        assert hasattr(self, 'svm'), "Model has not been fitted yet."
        assert self.circuit_params is not None, "No circuit parameters available. Model must be fitted first."
        # Scale new graphs
       
        scale_factors_test = compute_local_scale_factors(new_graphs, self.min_atom_distance)
        new_graphs_scaled = scale_all_graphs_local(new_graphs, scale_factors_test)
        # Compute distributions
        new_distributions = Parallel(n_jobs=self.n_jobs_quantum)(
            delayed(quantum_loop)(
                self.circuit_params, 
                graph, 
                self.LAYERS,
                self.N_samples,
                self.encoding_type,
                self.encoding_method,
                use_correlation=self.use_correlation
            ) for graph in new_graphs_scaled
        )
        
        return self.predict_from_distributions(new_distributions)
    
    def score(self, new_graphs, new_labels):
        """Compute accuracy on new data."""
        predictions = self.predict(new_graphs)
        accuracy = np.mean(predictions == new_labels)
        f1 = f1_score(new_labels, predictions, average='macro')
        print("Accuracy: ", accuracy)
        print("F1 Score: ", f1)
        return f1


class QuantumPairClassifier:
    """
    Binary classifier for a specific pair of classes.
    """
    def __init__(self, class_pair, distributions, labels, n_features=4,n_jobs=4, n_jobs_quantum=2, params_in=None, **kernel_kwargs):
        self.class_pair = class_pair
        self.distributions = distributions
        self.labels = labels
        self.n_features = n_features
        self.params_in = params_in  
        self.n_jobs = n_jobs                     
        self.n_jobs_quantum = n_jobs_quantum     
        self.kernel_kwargs = kernel_kwargs
        self.model = None
        self.weights = None
        self.gamma = None
        self.circuit_params = None 
    
    def fit(self):
        """Fit the binary classifier for this pair."""
        c1, c2 = self.class_pair
        
        # Map labels to binary
        label_map = {c1: 0, c2: 1}
        y_binary = np.array([label_map[lab] for lab in self.labels])
        
        # Create kernel model with precomputed distributions
        kernel_model = QuantumEvolutionKernel(
            graphs=[],
            labels=y_binary,
            precomputed_distributions=self.distributions,
            params_in=self.params_in, 
            n_features=self.n_features,
            n_jobs=self.n_jobs,              
            n_jobs_quantum=self.n_jobs_quantum,
            **self.kernel_kwargs
        )
        
        self.model, cv_accuracy = kernel_model.fit()
        self.cv_accuracy = cv_accuracy
        self.weights = kernel_model.weights
        self.gamma = kernel_model.gamma
        self.train_distributions = kernel_model.train_distributions
        self.circuit_params = kernel_model.circuit_params
        return self
    
    def predict(self, test_distributions):
        """Predict using precomputed test distributions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        kernel_matrix = kernel_js(test_distributions, self.train_distributions, self.gamma, self.weights)
        return self.model.predict(kernel_matrix)


import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import hashlib
import pickle
import os

class QuantumMulticlassClassifier:
    """
    Multiclass classifier using one-vs-one strategy with per-pair optimized kernels.
    Computes quantum distributions once, then trains one classifier per class pair.
    """
    def __init__(self, classes, n_features=4, n_jobs=4, n_jobs_quantum=2, params_in=None, **kernel_kwargs):
        self.classes = classes
        self.n_features = n_features
        self.params_in = params_in 
        self.kernel_kwargs = kernel_kwargs
        self.pair_classifiers = {}
        self.train_distributions = None
        self.train_labels = None
        self.circuit_params = None
        self.graphs = None
        
        self.n_jobs = n_jobs             
        self.n_jobs_quantum = n_jobs_quantum
        self.kernel_kwargs.update({        
            'n_jobs': n_jobs,
            'n_jobs_quantum': n_jobs_quantum
        })
    
    def _train_single_pair(self, pair_info):
        """Train a single pair classifier."""
        c1 = pair_info['c1']
        c2 = pair_info['c2']
        dist_pair = pair_info['distributions']
        labels_pair = pair_info['labels']
        
        print(f"  Training {c1} vs {c2} ({len(dist_pair)} samples)")
        
        filtered_kwargs = {k: v for k, v in self.kernel_kwargs.items() 
                          if k not in ['n_jobs', 'n_jobs_quantum']}
        
        pair_clf = QuantumPairClassifier(
            class_pair=(c1, c2),
            distributions=dist_pair,
            labels=labels_pair,
            n_features=self.n_features,
            params_in=self.circuit_params,
            n_jobs=self.n_jobs,
            n_jobs_quantum=self.n_jobs_quantum,
            **filtered_kwargs
        )
        pair_clf.fit()
        
        print(f"    Completed {c1} vs {c2}")
        return (c1, c2), pair_clf
    
    def _predict_single_pair(self, classifier_info, test_distributions, n_test, class_to_idx):
        """Predict using a single pair classifier."""
        (c1, c2), pair_clf = classifier_info
        K_test = kernel_js(test_distributions, pair_clf.train_distributions,
                          pair_clf.gamma, pair_clf.weights)
        pred = pair_clf.model.predict(K_test)   
        votes = np.zeros((n_test, len(class_to_idx)))
        for i, p in enumerate(pred):
            if p == 0:
                votes[i, class_to_idx[c1]] += 1
            else:
                votes[i, class_to_idx[c2]] += 1
        return votes
    
    def _optimize_distributions(self, graphs):
        """Optimize circuit parameters using Bayesian optimization."""
        theta = 0.1 * np.pi
        space = [Real(25, 1000, name='t')]
        
        best_score = -np.inf
        best_params = None
        best_distributions = None
        
     
        memory_cache = {}
        
        @use_named_args(space)
        def objective(t):
            nonlocal best_score, best_params, best_distributions
            
            params = np.array([t, theta] + [1.] * self.n_features)
            
            # Check memory cache
            t_key = round(t, 1)
            if t_key in memory_cache:
                print(f"  Using cached distributions for t={t:.2f}")
                distributions = memory_cache[t_key]
            else:
                print(f"  Computing distributions for t={t:.2f}")
                
                filtered_kwargs = {k: v for k, v in self.kernel_kwargs.items() 
                                  if k not in ['n_jobs', 'n_jobs_quantum']}
                
                temp_model = QuantumEvolutionKernel(
                    graphs=graphs,
                    labels=self.train_labels,
                    params_in=params,  
                    n_features=self.n_features,
                    n_jobs=self.n_jobs,
                    n_jobs_quantum=self.n_jobs_quantum,
                    **filtered_kwargs
                )
                
                distributions = temp_model.compute_quantum_distributions(params)
                memory_cache[t_key] = distributions
            
            # Split data
            train_dist, val_dist, train_lab, val_lab = train_test_split(
                distributions, self.train_labels, test_size=0.2, 
                random_state=1406, stratify=self.train_labels
            )
            
            # Train pair classifiers on validation split
            pairs = list(combinations(self.classes, 2))
            pair_data = []
            
            for c1, c2 in pairs:
                mask = np.isin(train_lab, [c1, c2])
                idx_pair = np.where(mask)[0]
                
                if len(idx_pair) < 2:
                    continue
                
                pair_data.append({
                    'c1': c1, 'c2': c2,
                    'distributions': [train_dist[i] for i in idx_pair],
                    'labels': train_lab[mask]
                })
            
            if not pair_data:
                return 0.0
            
            # Train in parallel
            n_jobs = min(len(pair_data), self.n_jobs)
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self._train_single_pair)(pair_info) for pair_info in pair_data
            )
            
            pair_classifiers = dict(results)
            
            # Predict validation set
            n_test = len(val_dist)
            n_classes = len(self.classes)
            class_to_idx = {c: i for i, c in enumerate(self.classes)}
            
            all_votes = Parallel(n_jobs=min(len(pair_classifiers), self.n_jobs), verbose=10)(
                delayed(self._predict_single_pair)((c1c2, clf), val_dist, n_test, class_to_idx)
                for c1c2, clf in pair_classifiers.items()
            )
            
            total_votes = np.sum(all_votes, axis=0)
            y_pred = [self.classes[np.argmax(v)] for v in total_votes]
            score = f1_score(val_lab, y_pred, average='macro')
            
            print(f"  Evaluated t={t:.2f}, F1={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_distributions = distributions.copy()
                print(f"  *** New best score: {score:.4f} with t={t:.2f} ***")
            
            return -score
        
        # Run Bayesian optimization
        print("\n" + "="*60)
        print("STARTING BAYESIAN OPTIMIZATION")
        print("="*60)
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=30,
            n_initial_points=5,
            acq_func='EI',
            random_state=1406,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print(f"Best t: {result.x[0]:.2f}")
        print(f"Best F1 score: {best_score:.4f}")
        print("="*60)
        
    
        self.optimal_distributions = best_distributions
        
        return best_params if best_params is not None else np.array([100, theta] + [1.] * self.n_features)
    
    def fit(self, graphs, labels, params=None):
        """Fit the multiclass classifier."""
        self.train_labels = np.array(labels)
        self.graphs = graphs
        
        # Get n_features from graphs if not set
        if self.n_features is None and graphs and len(graphs) > 0:
            self.n_features = len(graphs[0][0]) - 2
        
        print(f"Number of features per node: {self.n_features}")
        
       
        print("\n" + "="*60)
        print("DETERMINING CIRCUIT PARAMETERS")
        print("="*60)
        
        if params is not None:
            circuit_params = params
            print("Using provided parameters")
        elif self.params_in is not None:
            circuit_params = self.params_in
            print("Using saved parameters from initialization")
        else:
            print("Optimizing parameters...")
            circuit_params = self._optimize_distributions(graphs)
        
        self.circuit_params = circuit_params  
        print(f"Using circuit parameters: t={circuit_params[0]:.2f}, theta={circuit_params[1]:.2f}")
        
       
        print("\n" + "="*60)
        print("COMPUTING QUANTUM DISTRIBUTIONS")
        print("="*60)
        
      
        if hasattr(self, 'optimal_distributions') and params is None and self.params_in is None:
            self.train_distributions = self.optimal_distributions
            print("Using optimal distributions from optimization (no recomputation!)")
        else:
            filtered_kwargs = {k: v for k, v in self.kernel_kwargs.items() 
                          if k not in ['n_jobs', 'n_jobs_quantum']}
            temp_model = QuantumEvolutionKernel(
                graphs=graphs,
                labels=self.train_labels,
                params_in=circuit_params,  
                n_features=self.n_features,
                n_jobs=self.n_jobs,
                n_jobs_quantum=self.n_jobs_quantum,
                **filtered_kwargs
            )
            self.train_distributions = temp_model.compute_quantum_distributions(circuit_params)
        
        print(f"Computed distributions for {len(graphs)} graphs")
        print(f"   Distributions per graph: {len(self.train_distributions[0])}")
        
        # STEP 3: Train one classifier per class pair
        print("\n" + "="*60)
        print("TRAINING PER-PAIR CLASSIFIERS")
        print("="*60)
        
        pairs = list(combinations(self.classes, 2))
        print(f"Number of pairs: {len(pairs)}")
        
        # Prepare data for each pair
        pair_data = []
        for c1, c2 in pairs:
            mask = np.isin(self.train_labels, [c1, c2])
            idx_pair = np.where(mask)[0]
            
            if len(idx_pair) < 2:
                print(f"Not enough samples for pair {c1}-{c2}, skipping")
                continue
            
            pair_data.append({
                'c1': c1, 'c2': c2,
                'distributions': [self.train_distributions[i] for i in idx_pair],
                'labels': self.train_labels[mask],
                'n_samples': len(idx_pair)
            })
        
        n_jobs = min(len(pair_data), self.n_jobs)
        print(f"Training {len(pair_data)} classifiers with {n_jobs} parallel jobs")
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._train_single_pair)(pair_info) for pair_info in pair_data
        )
        
        for (c1, c2), pair_clf in results:
            self.pair_classifiers[(c1, c2)] = pair_clf
            print(f"Trained {c1} vs {c2} with weights: {pair_clf.weights}")
        
        print(f"\nAll {len(results)} pair classifiers trained successfully")
        return self
    
    def predict(self, test_graphs, test_params=None):
        """Predict classes for test graphs."""
        if not self.pair_classifiers:
            raise ValueError("No classifiers trained")
        if self.circuit_params is None:  
            raise ValueError("No circuit parameters available. Model must be fitted first.")
        
        # Get n_features from test graphs
        n_features = self.n_features
        if test_graphs and len(test_graphs) > 0:
            n_features = len(test_graphs[0][0]) - 2
        
        print("\n" + "="*60)
        print("COMPUTING TEST QUANTUM DISTRIBUTIONS")
        print("="*60)
        
        filtered_kwargs = {k: v for k, v in self.kernel_kwargs.items() 
                      if k not in ['n_jobs', 'n_jobs_quantum']}
        
        temp_model = QuantumEvolutionKernel(
            graphs=test_graphs,
            labels=np.zeros(len(test_graphs)),
            n_features=n_features,
            params_in=self.circuit_params,
            n_jobs=self.n_jobs,
            n_jobs_quantum=self.n_jobs_quantum,
            **filtered_kwargs
        )
        
        test_distributions = temp_model.compute_quantum_distributions(self.circuit_params)
        print(f"Computed distributions for {len(test_graphs)} test graphs")
        
        print("\n" + "="*60)
        print("PREDICTING BY COMBINED VOTING")
        print("="*60)
        
        n_test = len(test_graphs)
        n_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        classifier_list = list(self.pair_classifiers.items())
        n_jobs = min(len(classifier_list), self.n_jobs)
        print(f"Running {len(classifier_list)} predictions with {n_jobs} parallel jobs")
        
        all_votes = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._predict_single_pair)((c1c2, clf), test_distributions, n_test, class_to_idx)
            for c1c2, clf in classifier_list
        )
        
        total_votes = np.sum(all_votes, axis=0)
        y_pred = [self.classes[np.argmax(v)] for v in total_votes]
        
        return np.array(y_pred)
    
    def score(self, test_graphs, test_labels, test_params=None):
        """Compute accuracy on test set."""
        y_pred = self.predict(test_graphs, test_params)
        return np.mean(y_pred == test_labels)
    
    def save(self, filepath):
        """Save the entire classifier to disk."""
        print(f"\nSaving model to {filepath}...")
        
        # Convert pair classifiers to serializable format
        pair_classifiers_data = {}
        for (c1, c2), pair_clf in self.pair_classifiers.items():
            pair_classifiers_data[f"{c1}_{c2}"] = {
                'class_pair': (c1, c2),
                'weights': pair_clf.weights,
                'gamma': pair_clf.gamma,
                'circuit_params': pair_clf.circuit_params,
                'train_distributions': pair_clf.train_distributions,
                'svm_model': pair_clf.model,
                'n_features': pair_clf.n_features,
                'params_in': pair_clf.params_in
            }
        
        save_dict = {
            'classes': self.classes,
            'n_features': self.n_features,
            'params_in': self.params_in,
            'kernel_kwargs': self.kernel_kwargs,
            'circuit_params': self.circuit_params,
            'train_distributions': self.train_distributions,
            'train_labels': self.train_labels,
            'pair_classifiers_data': pair_classifiers_data,
            #'cache_dir': self.cache_dir
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {filepath}")
        print(f"   - {len(self.pair_classifiers)} pair classifiers")
        print(f"   - Circuit parameters: {self.circuit_params[:2]}")
    
    @classmethod
    def load(cls, filepath, **override_kwargs):
        """Load a saved classifier from disk."""
        print(f"\nLoading model from {filepath}...")
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Merge saved kwargs with any overrides
        kernel_kwargs = save_dict['kernel_kwargs'].copy()
        kernel_kwargs.update(override_kwargs)
        
        # Create new classifier instance
        classifier = cls(
            classes=save_dict['classes'],
            n_features=save_dict['n_features'],
            params_in=save_dict['params_in'],
            # cache_dir=save_dict.get('cache_dir', './cache'),
            **kernel_kwargs
        )
        
        # Restore attributes
        classifier.circuit_params = save_dict['circuit_params']
        classifier.train_distributions = save_dict['train_distributions']
        classifier.train_labels = save_dict['train_labels']
        
        # Restore pair classifiers
        for key, data in save_dict['pair_classifiers_data'].items():
            pair_clf = QuantumPairClassifier(
                class_pair=data['class_pair'],
                distributions=None,
                labels=None,
                n_features=data['n_features'],
                params_in=data['params_in'],
                **kernel_kwargs
            )
            pair_clf.weights = data['weights']
            pair_clf.gamma = data['gamma']
            pair_clf.circuit_params = data['circuit_params']
            pair_clf.train_distributions = data['train_distributions']
            pair_clf.model = data['svm_model']
            
            classifier.pair_classifiers[data['class_pair']] = pair_clf
        
        print(f"Model loaded from {filepath}")
        print(f"   - Classes: {classifier.classes}")
        print(f"   - Number of pair classifiers: {len(classifier.pair_classifiers)}")
        print(f"   - Circuit parameters: {classifier.circuit_params[:2]}")
        
        return classifier


class QuantumKernelExtractor:
    """Version for extracting kernel matrices."""
    def __init__(self, graphs, labels=None,
                 LAYERS=1, 
                 gamma=1.0,
                 encoding_type="with_hyperfine",
                 encoding_method="multi_run",
                 N_samples=500,
                 use_correlation=None,
                 n_jobs=32,
                 min_atom_distance=4.0):
        
        self.graphs = graphs
        self.labels = labels
        self.LAYERS = LAYERS
        self.gamma = -gamma
        self.encoding_type = encoding_type
        self.encoding_method = encoding_method
        self.N_samples = N_samples
        self.use_correlation = use_correlation
        self.n_jobs = n_jobs
        self.min_atom_distance = min_atom_distance
        
        print("\n" + "="*60)
        print("APPLYING POSITION SCALING FOR PULSER")
        print("="*60)
        
        scale_factors = compute_local_scale_factors(self.graphs, self.min_atom_distance)
        self.graphs_scaled = scale_all_graphs_local(self.graphs, scale_factors)
      
        
      
        
        if graphs and len(graphs) > 0:
            try:
                self.number_of_features = len(graphs[0][0]) - 2
            except:
                self.number_of_features = 0
        else:
            self.number_of_features = 0
            
        if self.encoding_method == "multi_run":
            self.expected_kernels = (self.number_of_features // 2) * 2
        elif self.encoding_method == "single_run":
            self.expected_kernels = self.number_of_features * 2
        else:
            self.expected_kernels = 2
            
        print("="*60)
        print("QUANTUM KERNEL EXTRACTOR")
        print("="*60)
        print(f"Graphs: {len(graphs)}")
        print(f"Layers: {LAYERS}")
        print(f"Gamma: {gamma}")
        print(f"Encoding: {encoding_type} / {encoding_method}")
        print(f"Features: {self.number_of_features}")
        print(f"Expected kernels: {self.expected_kernels}")
        
        print("="*60)
    
    def compute_distributions(self, params):
        distributions = Parallel(n_jobs=self.n_jobs)(
            delayed(quantum_loop)(
                params, 
                graph, 
                self.LAYERS,
                self.N_samples,
                self.encoding_type,
                self.encoding_method,
                use_correlation=self.use_correlation
            ) for graph in self.graphs_scaled
        )
        return distributions
    
    def compute_kernel_matrix(self, params, distributions1=None, distributions2=None, weights=None):
        if distributions1 is None:
            distributions1 = self.compute_distributions(params)
        
        if distributions2 is None:
            distributions2 = distributions1
        
        kernel_matrix = kernel_js(distributions1, distributions2, self.gamma, weights)
        return kernel_matrix
    
    def compute_train_test_kernel(self, params, train_graphs, test_graphs):
        train_scale_factors = compute_local_scale_factors(train_graphs, self.min_atom_distance)
        test_scale_factors = compute_local_scale_factors(test_graphs, self.min_atom_distance)

        train_graphs_scaled = scale_all_graphs_local(train_graphs, train_scale_factors)
        test_graphs_scaled = scale_all_graphs_local(test_graphs, test_scale_factors)
                
        original_graphs = self.graphs_scaled
        self.graphs_scaled = train_graphs_scaled
        train_distributions = self.compute_distributions(params)
        
        self.graphs_scaled = test_graphs_scaled
        test_distributions = self.compute_distributions(params)
        
        self.graphs_scaled = original_graphs
        
        K_train = kernel_js(train_distributions, train_distributions, self.gamma)
        K_test = kernel_js(test_distributions, train_distributions, self.gamma)
        
        return K_train, K_test, train_distributions, test_distributions
    
    def optimize_gamma(self, params, distributions=None, C_range=np.logspace(-2, 3, 6)):
        if distributions is None:
            distributions = self.compute_distributions(params)
        
        gammas = np.logspace(2, -2, 10)
        best_score = 0
        best_gamma = gammas[0]
        
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score, KFold
        
        cv = KFold(n_splits=5, shuffle=True, random_state=1406)
        
        for gamma_val in gammas:
            original_gamma = self.gamma
            self.gamma = -gamma_val
            
            kernel_matrix = kernel_js(distributions, distributions, self.gamma)
            
            best_C_score = 0
            for C in C_range:
                svm = SVC(kernel='precomputed', C=C)
                scores = cross_val_score(svm, kernel_matrix, self.labels, 
                                        cv=cv, scoring='accuracy')
                mean_score = np.mean(scores)
                if mean_score > best_C_score:
                    best_C_score = mean_score
            
            if best_C_score > best_score:
                best_score = best_C_score
                best_gamma = gamma_val
            
            self.gamma = original_gamma
        
        return best_gamma, best_score


def get_kernel_matrix(params, train_graphs, test_graphs=None, min_atom_distance=4.0, **kwargs):
    """High-level function to obtain kernel matrices."""
    extractor = QuantumKernelExtractor(train_graphs, min_atom_distance=min_atom_distance, **kwargs)
    
    if test_graphs is None:
        K_train = extractor.compute_kernel_matrix(params)
        return K_train
    else:
        K_train, K_test, _, _ = extractor.compute_train_test_kernel(params, train_graphs, test_graphs)
        return K_train, K_test