import numpy as np
from scipy.stats import gamma

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