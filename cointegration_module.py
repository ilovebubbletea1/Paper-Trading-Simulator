import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def run_kalman_filter(y1, y2, burn_in=60):
    """
    Runs a time-varying Kalman Filter.
    Observation equation: y1_t = gamma_t * y2_t + mu_t + v_t
    State equation: [gamma_t, mu_t]^T = [gamma_{t-1}, mu_{t-1}]^T + w_t
    
    Calibration / Burn-in: We initialize Q and R and run the filter. 
    The Z-score signals are constrained to 0 during the burn_in period, allowing the system to converge.
    """
    n = len(y1)
    
    # Initialize states x = [gamma, mu]^T
    x = np.zeros((2, 1))
    x[0, 0] = y1.iloc[0] / y2.iloc[0] if y2.iloc[0] != 0 else 1.0 # Initial guess for hedge ratio (gamma)
    x[1, 0] = 0.0 # Initial guess for spread mean (mu)
    
    # Covariance matrices
    P = np.eye(2) # Initial State covariance
    Q = np.eye(2) * 1e-4 # State transition noise (allows parameters to drift)
    R = 1e-3 # Observation noise covariance
    
    gammas = np.zeros(n)
    mus = np.zeros(n)
    raw_spreads = np.zeros(n)
    obs_variances = np.zeros(n)
    z_scores = np.zeros(n)
    
    for t in range(n):
        H = np.array([[y2.iloc[t], 1.0]])
        
        # Prediction step (Random walk assumption x_pred = x)
        P_pred = P + Q
        
        # Observation prediction
        y_pred = H @ x
        
        # Prediction Error (innovation / raw spread)
        e = y1.iloc[t] - y_pred[0, 0]
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman Gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update step (incorporate new information)
        x = x + K * e
        P = (np.eye(2) - K @ H) @ P_pred
        
        # Store variables
        gammas[t] = x[0, 0]
        mus[t] = x[1, 0]
        raw_spreads[t] = e
        obs_variances[t] = S[0, 0]
        
        # Hard Burn-In standard deviation filter
        if t < burn_in:
            z_scores[t] = 0.0 # Burn-in period: No trading signal to allow convergence
        else:
            # We strictly standardize the spread (Z-score) using Kalman's time-varying observation standard deviation
            std_t = np.sqrt(obs_variances[t])
            if std_t > 0:
                z_scores[t] = raw_spreads[t] / std_t
            else:
                z_scores[t] = 0.0
            
    results = pd.DataFrame({
        'gamma': gammas,
        'mu': mus,
        'raw_spread': raw_spreads,
        'obs_variance': obs_variances,
        'z_score': z_scores
    }, index=y1.index)
    
    return results

def check_cointegration(z_scores, burn_in=60):
    """
    Checks if the spread (Z-Score) is stationary using the ADF test.
    Only evaluates strictly after the convergence burn-in period.
    """
    valid_z = z_scores.iloc[burn_in:]
    if len(valid_z) < 30:
        return False, 1.0 # Insufficient valid data
        
    adf_result = adfuller(valid_z)
    p_value = adf_result[1]
    
    is_cointegrated = p_value < 0.05
    return is_cointegrated, p_value
