import numpy as np
from scipy.stats import gamma, wasserstein_distance, entropy

def shift_data(data, margin=0.0125):
    if np.min(data) > margin:
        return data, 0.0
    elif np.min(data) > 0:
        delta = margin-np.min(data)
        return data+delta, -delta
    else:
        delta = np.min(data) - margin
        return data-delta, delta
    
def fit_gamma_distro(data):
    #shift data so that there is a small positive margin between the min and 0
    data_shifted, loc = shift_data(data)

    #to avoid fitting issues, use the Method of Moments (MoM) estimates for alpha and scale
    #see: https://en.wikipedia.org/wiki/Method_of_moments_(statistics)
    #see: https://docs.analytica.com/index.php/Gamma_distribution
    sample_mean = np.mean(data_shifted)
    sample_variance = np.var(data_shifted)
    initial_alpha = (sample_mean**2) / sample_variance
    initial_beta = sample_variance / sample_mean

    alpha, _, beta = gamma.fit(data_shifted, initial_alpha, floc=0, scale=initial_beta)

    return alpha, loc, beta

def plot_fitted_gamma(alpha, loc, beta, lower=0.0, upper=1.0,count=100):
    x = np.linspace(lower, upper, count)
    y = gamma.pdf(x, alpha, loc=loc, scale=beta)
    return x, y

def bootstrap_wasserstein_distance(pos_data, neg_data, num_bootstraps=1000):
    """
    Bootstraps the Wasserstein distance between two 1D datasets.

    Args:
        pos_data (array-like): Data from the pos side of the ellipse mask.
        data2 (array-like): Data from the neg side of the ellipse mask.
        num_bootstraps (int): The number of bootstrap samples to generate.

    Returns:
        list: A list of bootstrap Wasserstein distances.
    """
    distances = []
    n1 = len(pos_data)
    n2 = len(neg_data)

    for _ in range(num_bootstraps):
        # Resample with replacement from each dataset
        sample1 = np.random.choice(pos_data, n1, replace=True)
        sample2 = np.random.choice(neg_data, n2, replace=True)
        
        # Calculate the Wasserstein distance for the bootstrap samples
        # Scipy's function works with the empirical distributions of the samples
        dist = wasserstein_distance(sample1, sample2)
        
        distances.append(dist)
        
    return distances

def run_wasserstein(pos_data, neg_data, cf_interval = 95):
    if cf_interval >= 100 or cf_interval <= 0:
        raise ValueError("invalid cf_interval, range is (0,100)")
    wd = wasserstein_distance(pos_data, neg_data)

    bootstrap_distances = bootstrap_wasserstein_distance(pos_data, neg_data)
    wd_mean = np.mean(bootstrap_distances)
    wd_conf_interval_lower = np.percentile(bootstrap_distances, (100-cf_interval)/2.0)
    wd_conf_interval_upper = np.percentile(bootstrap_distances, 100-((100-cf_interval)/2.0))

    return [wd, wd_mean, wd_conf_interval_lower, wd_conf_interval_upper]

def compute_laplace_smoothed_kld(prob_pos,prob_neg, pos_n, neg_n, alpha=1):
    pos_ni = pos_n*prob_pos
    neg_ni = neg_n*prob_neg
    
    pos_denomenator = pos_n + alpha*len(prob_pos)
    neg_denomenator = neg_n + alpha*len(prob_neg)
    
    prob_pos_prime = (pos_ni + np.ones(np.size(pos_ni))*alpha)/pos_denomenator
    prob_neg_prime = (neg_ni + np.ones(np.size(neg_ni))*alpha)/neg_denomenator
    
    return entropy(prob_pos_prime, prob_neg_prime)