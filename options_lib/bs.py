import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

def black_scholes_call(S, K, T, r, sigma):
    """Calculate the Black-Scholes price of a European call option."""
    if T <= 0:
        return max(S - K, 0)
    elif sigma <= 0:
        return max(S - K, 0)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

def black_scholes_delta(S, K, T, r, sigma):
    """Calculate the Black-Scholes delta of a European call option."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    elif sigma <= 0:
        return 1.0 if S > K else 0.0
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)
        return delta

def black_scholes_vega(S, K, T, r, sigma):
    """Calculate the Black-Scholes vega of a European call option."""
    if T <= 0 or sigma <= 0:
        return 0.0
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega

def black_scholes_gamma(S, K, T, r, sigma):
    """Calculate the Black-Scholes gamma of a European call option."""
    if T <= 0 or sigma <= 0:
        return 0.0
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

def implied_volatility(C_market, S, K, T, r, tol=1e-6, max_iter=100):
    """Calculate the implied volatility using bisection method."""
    if C_market <= 0 or T <= 0:
        return 0.0

    low = 0.001
    high = 2.0

    for _ in range(max_iter):
        mid = (low + high) / 2
        C_model = black_scholes_call(S, K, T, r, mid)
        if abs(C_model - C_market) < tol:
            return mid
        elif C_model > C_market:
            high = mid
        else:
            low = mid

    return (low + high) / 2
