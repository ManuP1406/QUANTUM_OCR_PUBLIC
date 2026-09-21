"""
Functions for computing quantum measures from samples.
"""
import numpy as np
from collections import defaultdict
from itertools import combinations


def compute_mutual_information(samples, num_nodes):
    """
    Compute mutual information distribution from measurement samples.
    """
    total_samples = sum(samples.values())
    mutual_info_values = defaultdict(int)
    total_pairs = 0

    for i, j in combinations(range(num_nodes), 2):
        joint_distribution = defaultdict(int)
        marginal_i = defaultdict(int)
        marginal_j = defaultdict(int)

        for outcome, count in samples.items():
            bit_i = outcome[i]
            bit_j = outcome[j]
            joint_distribution[(bit_i, bit_j)] += count
            marginal_i[bit_i] += count
            marginal_j[bit_j] += count

        for key in joint_distribution:
            joint_distribution[key] /= total_samples
        for key in marginal_i:
            marginal_i[key] /= total_samples
        for key in marginal_j:
            marginal_j[key] /= total_samples

        mutual_info = 0
        for (bit_i, bit_j), p_ij in joint_distribution.items():
            p_i = marginal_i[bit_i]
            p_j = marginal_j[bit_j]
            if p_ij > 0 and p_i > 0 and p_j > 0:
                mutual_info += p_ij * np.log2(p_ij / (p_i * p_j))

        mutual_info_values[mutual_info] += 1
        total_pairs += 1

    distribution = [(mi, count / total_pairs) 
                    for mi, count in sorted(mutual_info_values.items())]
    return np.array(distribution)


def compute_correlations_sigma_z(samples, num_nodes):
    """
    Compute sigma_z correlation distribution from measurement samples.
    """
    total_samples = sum(samples.values())
    
    # Compute <sigma_z> for each node
    avg_sigma = [0.0] * num_nodes
    for outcome, count in samples.items():
        for i in range(num_nodes):
            value = 1 if outcome[i] == "0" else -1
            avg_sigma[i] += count * value
    avg_sigma = [val / total_samples for val in avg_sigma]
    
    # Compute correlations
    correlations = {}
    total_pairs = 0
   
    for i, j in combinations(range(num_nodes), 2):
        avg_zz = 0.0
        for outcome, count in samples.items():
            value_i = 1 if outcome[i] == "0" else -1
            value_j = 1 if outcome[j] == "0" else -1
            avg_zz += count * (value_i * value_j)
        avg_zz /= total_samples
        C_ij = avg_zz - avg_sigma[i] * avg_sigma[j]
        normalized_C = (C_ij + 1) / 2
        
        correlations[normalized_C] = correlations.get(normalized_C, 0) + 1
        total_pairs += 1
    
    distribution = [(corr, count / total_pairs) 
                    for corr, count in sorted(correlations.items())]
    return np.array(distribution)


def compute_correlations(samples, num_nodes):
    """
    Compute occupation number correlation distribution.
    """
    correlations = defaultdict(int)
    total_samples = sum(samples.values())
    
    # Compute <ni> for each node
    avg_n = np.zeros(num_nodes)
    for excitation, count in samples.items():
        for i in range(num_nodes):
            if excitation[i] == "1":
                avg_n[i] += count
    avg_n /= total_samples
    
    # Compute <ninj> and correlation C_ij
    for i, j in combinations(range(num_nodes), 2):
        avg_ninj = 0
        for excitation, count in samples.items():
            if excitation[i] == "1" and excitation[j] == "1":
                avg_ninj += count
        avg_ninj /= total_samples
        C_ij = avg_ninj - avg_n[i] * avg_n[j]
        
        normalized_C = (C_ij + 1) / 2
        correlations[normalized_C] += 1
    
    distribution = [(C, count / len(correlations)) 
                    for C, count in sorted(correlations.items())]
    return np.array(distribution)
