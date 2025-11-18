import numpy as np

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.stats import permutation_test

#see: https://safjan.com/metrics-to-compare-histograms/


def compute_wasserstein_distance2(pos_bin_edge, neg_bin_edge, prob_pos, prob_neg):
    pos_bin_centers = (pos_bin_edge[:-1] + pos_bin_edge[1:])/2
    neg_bin_centers = (neg_bin_edge[:-1] + neg_bin_edge[1:])/2
    
    return wasserstein_distance(pos_bin_centers, neg_bin_centers, prob_pos, prob_neg)

def compute_wasserstein_distance(sample_pos,sample_neg, bin_range, bincount):
    prob_pos, bins = np.histogram(sample_pos,bins=bincount,range=(bin_range[0],bin_range[1]), weights=np.ones_like(sample_pos) / len(sample_pos))
    prob_neg, _ = np.histogram(sample_neg,bins=bincount,range=(bin_range[0],bin_range[1]), weights=np.ones_like(sample_neg) / len(sample_neg))

    bin_centers = (bins[:-1] + bins[1:])/2

    #return wasserstein_distance(bin_centers, bin_centers, prob_pos, prob_neg)
    return wasserstein_distance(sample_pos,sample_neg)

def permutation_test_wasserstein_distance(sample_pos, sample_neg, bin_range, bincount):
    was_lambda = lambda x,y: compute_wasserstein_distance(x, y, bin_range, bincount)
    return permutation_test([sample_pos,sample_neg],was_lambda)

def permutation_test_wasserstein_distance2(pos_bin_edge, neg_bin_edge, prob_pos, prob_neg):
    was_lambda = lambda x,y: compute_wasserstein_distance2(pos_bin_edge, neg_bin_edge, x, y)
    return permutation_test([prob_pos, prob_neg],was_lambda,alternative='greater')

def compute_laplace_smoothed_kld(prob_pos,prob_neg, pos_n, neg_n, alpha=1):
    pos_ni = pos_n*prob_pos
    neg_ni = neg_n*prob_neg
    
    pos_denomenator = pos_n + alpha*len(prob_pos)
    neg_denomenator = neg_n + alpha*len(prob_neg)
    
    prob_pos_prime = (pos_ni + np.ones(np.size(pos_ni))*alpha)/pos_denomenator
    prob_neg_prime = (neg_ni + np.ones(np.size(neg_ni))*alpha)/neg_denomenator
    
    return entropy(prob_pos_prime, prob_neg_prime)

