import numpy as np
from scipy.stats import wasserstein_distance

def generate_random_pos_mask(pos_count, total):
    if pos_count > total:
        raise ValueError("The number of True values cannot exceed the size of the mask.")
        
    mask = np.full(total, False, dtype=bool)
    mask[:pos_count] = True
    np.random.shuffle(mask)
    return mask

def permute_pos_and_neg(pos_side, neg_side):
    pos_and_neg = np.concatenate((pos_side, neg_side))
    
    pos_mask = generate_random_pos_mask(len(pos_side), len(pos_and_neg))
    neg_mask = ~pos_mask

    return pos_and_neg[pos_mask], pos_and_neg[neg_mask]

def wasserstein_permutation_test(pos_side, neg_side):
    original_was = wasserstein_distance(pos_side, neg_side)
    
    was_permutation_results = []
    for i in range(1000):
        permuted_pos, permuted_neg = permute_pos_and_neg(pos_side, neg_side)
        was_permutation_results.append(wasserstein_distance(permuted_pos, permuted_neg))
    p_value = np.sum(np.array(was_permutation_results) >= original_was) / 1000

    return original_was, p_value, np.max(was_permutation_results)
