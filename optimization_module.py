import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

def expected_profit(s0, T):
    """
    Objective Function:
    Calculates the expected total profit assuming the Z-score is standard normally distributed N(0, 1).
    Profit is roughly proportional to: s0 * T * (1 - Phi(s0))
    Since scipy.minimize finds the minimum, we return the negative expected profit.
    """
    # Cumulative probability Phi(s0) -> probability of Z > s0
    prob_cross = 1.0 - norm.cdf(s0)
    expected_profit = s0 * T * prob_cross
    return -expected_profit

def optimize_threshold(z_scores, burn_in=60):
    """
    Optimum Threshold Design:
    Finds the optimal entry Z-Score threshold s0* that maximizes the expected profit.
    """
    valid_z = z_scores.iloc[burn_in:]
    T = len(valid_z)
    
    if T == 0:
        return 1.5 # Fallback if empty
        
    # We constrain the Z-score optimization space (e.g. searching between 0.1 and 3.0 standard deviations)
    bounds = [(0.1, 3.0)]
    initial_guess = [1.0]
    
    # Run bounded optimization
    result = minimize(expected_profit, initial_guess, args=(T,), bounds=bounds, method='L-BFGS-B')
    
    s0_opt = result.x[0]
    return s0_opt

