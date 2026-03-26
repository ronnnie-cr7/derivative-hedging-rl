"""
utils.py — Shared helper functions used across all phases
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    T = np.where(np.asarray(T) <= 0, 1e-10, T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_delta(S, K, T, r, sigma):
    T = np.where(np.asarray(T) <= 0, 1e-10, T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def simulate_gbm(S0, mu, sigma, T, N, paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    prices = np.zeros((N, paths))
    prices[0] = S0
    for t in range(1, N):
        Z = np.random.standard_normal(paths)
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return prices
