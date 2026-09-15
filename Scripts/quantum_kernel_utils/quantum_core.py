"""
Core quantum loop and kernel functions.
"""
import numpy as np
from collections import defaultdict
from itertools import combinations
from pulser_simulation import QutipEmulator
from scipy.stats import entropy
from joblib import Parallel, delayed
from pulser_simulation import SimConfig
from .pulser_functions import create_sequence
from .quantum_measures import (compute_mutual_information, 
                               compute_correlations_sigma_z, 
                               compute_correlations)
import gc



def quantum_loop(parameters, nodes, LAYERS=1, N_samples=200, 
                 encoding_type=None, encoding_method="multi_run", 
                 freq_par=False, use_correlation=None):
    """
    Unified quantum loop to extract both occupation distributions 
    and sigma_z correlations in a single simulation pass.
    """ 
    feature_count = len(nodes[0]) - 2 
    t_params = parameters[:LAYERS]
    theta_params = parameters[LAYERS:2*LAYERS]
    
    # Handle frequency parameters for encoding
    if not freq_par:
        freq_params = [[1.0 for _ in range(feature_count)] for _ in range(LAYERS)]
    else:
        freq_params_flat = parameters[2*LAYERS : 2*LAYERS + (LAYERS * feature_count)]
        freq_params = [freq_params_flat[i*feature_count : (i+1)*feature_count] for i in range(LAYERS)]
    
    # Setup encoding combinations
    encoding_feat = encoding_type is not None
    if not encoding_feat:
        nodes = [[node[0], node[1]] for node in nodes]
        encoding_method = None
    
    if encoding_feat and encoding_method == "multi_run":
        feat_indices = list(range(2, 2 + feature_count))
        feature_combinations = [(feat_indices[i], feat_indices[i+1]) for i in range(0, len(feat_indices) - 1, 2)]
    elif encoding_feat and encoding_method == "single_run":
        feature_combinations = list(range(feature_count))
    else:
        feature_combinations = [None]

    all_occupations = []
    all_correlations = []
    
    for feature_item in feature_combinations:
        # Build sequence based on the selected encoding method
        if encoding_method == "multi_run":
            idx1, idx2 = feature_item
            selected_nodes = [[node[0], node[1], node[idx1], node[idx2]] for node in nodes]
            seq = create_sequence(selected_nodes, LAYERS, encoding_method, encoding_type, 
                                  feature_pair=(idx1, idx2), freq_params=freq_params)
        elif encoding_method == "single_run":
            feature_idx = feature_item
            selected_nodes = [[node[0], node[1], node[feature_idx + 2]] for node in nodes]
            seq = create_sequence(selected_nodes, LAYERS, encoding_method, encoding_type, 
                                  feature_idx=feature_idx, freq_params=freq_params)
        else:
            seq = create_sequence(nodes, LAYERS, encoding_method, encoding_type, freq_params=freq_params)
        

           
        # seq.measure(basis="digital")
    
        seq.measure(basis="ground-rydberg") # ground-rydberg measurement for occupation distribution
        

     
        assigned_seqGlobal = seq.build(t_list=t_params, theta_list=theta_params)
        
        

        simul = QutipEmulator.from_sequence(assigned_seqGlobal, sampling_rate=1)
        results = simul.run(progress_bar=True)
        samples = results.sample_final_state(N_samples=N_samples)
        
      
        
        # 1. Extract Occupation Distribution (Excitation Counts)
        excitation_count = defaultdict(int)
        for excitation, count in samples.items():
            excitation_count[excitation.count("1")] += count
        
        max_ex = len(nodes)
        norm_occup = np.array([(float(ex)/max_ex, count/N_samples) for ex, count in sorted(excitation_count.items())])
        all_occupations.append(norm_occup)
        
        # 2. Extract Sigma_z Correlations
        corr_z = compute_correlations_sigma_z(samples, len(nodes))
        all_correlations.append(corr_z)

        del assigned_seqGlobal
        del simul
        del results
        del samples
        gc.collect()
            
    # Return both as a dictionary to avoid redundant runs
    final_output = []
    for i in range(len(all_occupations)):
      
        run_data = {
            "occupations": all_occupations[i],
            "correlations": all_correlations[i] 
        }
        final_output.append(run_data)
            

    

    return final_output



def get_realistic_noise_model():
    """Returns a realistic predefined noise model for Pasqal neutral atom QPUs."""
    return NoiseModel(
        state_prep_error=0.01,
        p_false_pos=0.001,
        p_false_neg=0.001,
        temperature=50.0,
        laser_waist=100.0,
        amp_sigma=0.05,
        detuning_sigma=0.1,
        relaxation_rate=0.01,
        dephasing_rate=0.02,
        hyperfine_dephasing_rate=0.001
    )

from scipy.ndimage import gaussian_filter1d

def bin_distribution(distribution, num_bins=20, sigma=0.3): # 0.6 is not good
    """
    Binning for a single distribution with Gaussian smoothing.
    Bridges the gap between low-node and high-node counts.
    """
  
        
    x, probs = zip(*distribution)
    binned = np.zeros(num_bins)
    
   
    bin_edges = np.linspace(0, 1, num_bins + 1)
    

    for val, p in zip(x, probs):
        # Find the correct bin index
        idx = np.digitize(val, bin_edges) - 1
        idx = max(0, min(idx, num_bins - 1))
        binned[idx] += p
  
    if sigma > 0:
        binned = gaussian_filter1d(binned, sigma=sigma)
        
    
    binned /= (np.sum(binned) + 1e-10)
    
    return binned


def jensen_shannon_divergence(P, Q):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    """
    P = np.array(P) / np.sum(P) if np.sum(P) > 0 else np.array(P)
    Q = np.array(Q) / np.sum(Q) if np.sum(Q) > 0 else np.array(Q)
    M = 0.5 * (P + Q)
    
    return 0.5 * (entropy(P, M) + entropy(Q, M))


from scipy.stats import wasserstein_distance


# It is now used with Wasserstein distance, but the structure is similar to the JSD version for easy switching if needed.
def kernel_js(dist_list1, dist_list2, gamma, weights=None, aggregation="sum", 
              distance_metric="wasserstein", num_bins=10, sigma=0.):
    """
    Compute the Quantum Kernel matrix using various distance metrics.
    
    Parameters:
    -----------
    dist_list1, dist_list2 : list
        Lists of quantum results (dictionaries containing 'occupations' and 'correlations').
    gamma : float
        Exponential scale factor (should be negative for a valid similarity kernel).
    weights : np.array, optional
        Weights for each feature component. Expected size: n_runs * 2.
    aggregation : str, optional
        "sum" for a weighted average of independent kernels (RBF-like sum).
        "product" for an exponential of the weighted sum of distances.
    distance_metric : str, optional
        Type of distance to use:
        - "wasserstein": Wasserstein/Earth Mover's Distance (NO binning)
        - "jensen_shannon": Jensen-Shannon Divergence 
        - "hellinger": Hellinger distance 
        - "l2": L2/Euclidean distance 
    num_bins : int, optional
        Number of bins for binning 
    sigma : float, optional
        Gaussian smoothing sigma for bin_distribution 
    """

    if gamma > 0:
        print("Gamma should be negative!")
        gamma = -gamma

    n1 = len(dist_list1)
    n2 = len(dist_list2)

    if n1 == 0 or n2 == 0:
        return np.array([])

    n_runs = len(dist_list1[0])
    total_components = n_runs * 2
   
    if weights is None:
        weights = np.ones(total_components) / total_components
        print(f"DEBUG kernel_js: created uniform weights of size {len(weights)}")

    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            score = 0.0
            comp_idx = 0

            for r in range(n_runs):
             
                occ_i = dist_list1[i][r]['occupations']
                occ_j = dist_list2[j][r]['occupations']
                
              
                corr_i = dist_list1[i][r]['correlations']
                corr_j = dist_list2[j][r]['correlations']
                
                # Calculate distances based on chosen metric
                d_occ = _compute_distance(occ_i, occ_j, distance_metric, num_bins, sigma)
                
                d_corr = 0.0
                if corr_i is not None and corr_j is not None:
                    
                    d_corr = _compute_distance(corr_i, corr_j, distance_metric, num_bins*2, sigma)

                # --- 3. Aggregation ---
                if aggregation == "sum":
                    score += weights[comp_idx]     * np.exp(gamma * d_occ)
                    score += weights[comp_idx + 1] * np.exp(gamma * d_corr)
                else:
                    score += weights[comp_idx]     * d_occ
                    score += weights[comp_idx + 1] * d_corr

                comp_idx += 2

            if aggregation == "sum":
                K[i, j] = score
            else:
                K[i, j] = np.exp(gamma * score)

    return K


def _compute_distance(data_i, data_j, metric, num_bins=20, sigma=0.3):
    """
    Helper function to compute distance between two distributions.
    Uses your existing bin_distribution function for metrics that need binning.
    """
    
    if metric == "wasserstein":
        # NO binning - works directly on raw data
        # Handle the case where data is list of tuples or numpy array
        if isinstance(data_i, list):
            positions_i = [point[0] for point in data_i]
            weights_i = [point[1] for point in data_i]
            positions_j = [point[0] for point in data_j]
            weights_j = [point[1] for point in data_j]
        else:
            positions_i = data_i[:, 0]
            weights_i = data_i[:, 1]
            positions_j = data_j[:, 0]
            weights_j = data_j[:, 1]
            
        return wasserstein_distance(positions_i, positions_j, weights_i, weights_j)
    
    elif metric == "jensen_shannon":
        # Uses YOUR bin_distribution with Gaussian smoothing
        dist_i = bin_distribution(data_i, num_bins=num_bins, sigma=sigma)
        dist_j = bin_distribution(data_j, num_bins=num_bins, sigma=sigma)
        return jensen_shannon_divergence(dist_i, dist_j)
    
    elif metric == "hellinger":
        # Hellinger distance using your binned distributions
        dist_i = bin_distribution(data_i, num_bins=num_bins, sigma=sigma)
        dist_j = bin_distribution(data_j, num_bins=num_bins, sigma=sigma)
        
        # Add small epsilon to avoid numerical issues
        dist_i = np.maximum(dist_i, 1e-10)
        dist_j = np.maximum(dist_j, 1e-10)
        
        # Hellinger distance: H(P,Q) = (1/√2) * √( Σ (√p_i - √q_i)² )
        hellinger_sq = 0.5 * np.sum((np.sqrt(dist_i) - np.sqrt(dist_j))**2)
        return np.sqrt(hellinger_sq)
    
    elif metric == "l2":
        # L2 distance using your binned distributions
        dist_i = bin_distribution(data_i, num_bins=num_bins, sigma=sigma)
        dist_j = bin_distribution(data_j, num_bins=num_bins, sigma=sigma)
        return np.sqrt(np.sum((dist_i - dist_j)**2))
    
    else:
        raise ValueError(f"Unknown distance metric: {metric}. Choose from: wasserstein, jensen_shannon, hellinger, l2")

def flatten_quantum_data(dist_list):
    flattened_features = []
    for sample in dist_list:
        v_sample = []
        for run in sample:
            v_sample.extend(bin_distribution(run['occupations'], num_bins=10))
            if run['correlations'] is not None:
                v_sample.extend(bin_distribution(run['correlations'], num_bins=20))
        flattened_features.append(v_sample)
    return np.array(flattened_features)