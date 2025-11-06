import numpy as np

from pyemd import emd
from scipy.stats import entropy
from scipy.stats import chisquare
from scipy.stats import wasserstein_distance

#see: https://safjan.com/metrics-to-compare-histograms/

def compute_wasserstein_distance(pos_bin_edge, neg_bin_edge, prob_pos, prob_neg):
    pos_bin_centers = (pos_bin_edge[:-1] + pos_bin_edge[1:])/2
    neg_bin_centers = (neg_bin_edge[:-1] + neg_bin_edge[1:])/2
    
    return wasserstein_distance(pos_bin_centers, neg_bin_centers, prob_pos, prob_neg)

def compute_laplace_smoothed_kld(prob_pos,prob_neg, pos_n, neg_n, alpha=1):
    pos_ni = pos_n*prob_pos
    neg_ni = neg_n*prob_neg
    
    pos_denomenator = pos_n + alpha*len(prob_pos)
    neg_denomenator = neg_n + alpha*len(prob_neg)
    
    prob_pos_prime = (pos_ni + np.ones(np.size(pos_ni))*alpha)/pos_denomenator
    prob_neg_prime = (neg_ni + np.ones(np.size(neg_ni))*alpha)/neg_denomenator
    
    return entropy(prob_pos_prime, prob_neg_prime)

