import numpy as np
import pandas as pd

def simulate_backtest(y1, y2, kf_results, s0_opt, hard_stop=4.0, fee_bps=5.0):
    """
    Simulates the pairs trading strategy.
    Allocates gross exposure proportionally.
    Long Spread: Buy $Y_1$, Short $Y_2$ with weight scaled by $\gamma_t$
    """
    n = len(y1)
    
    z_scores = kf_results['z_score'].values
    gammas = kf_results['gamma'].values
    
    positions = np.zeros(n) 
    
    current_pos = 0
    stop_loss_triggered = False
    
    for t in range(1, n):
        z = z_scores[t-1] # Use yesterday's signal for trading today
        
        # Hard Stop-Loss: Structural Break Failsafe
        if current_pos != 0 and abs(z) >= hard_stop:
            current_pos = 0
            stop_loss_triggered = True
            positions[t] = 0
            continue
            
        # Reset stop-loss lock when spread reverting
        if stop_loss_triggered:
            if abs(z) < 0.5: # Sufficiently reverted
                stop_loss_triggered = False
            positions[t] = 0
            continue
            
        # Signal Generation
        if current_pos == 0:
            if z > s0_opt:
                current_pos = -1 # Short spread (Overvalued Y1 relative to Y2)
            elif z < -s0_opt:
                current_pos = 1  # Long spread (Undervalued Y1 relative to Y2)
        elif current_pos == 1:
            if z >= 0:
                current_pos = 0  # Reversion complete, close Long
        elif current_pos == -1:
            if z <= 0:
                current_pos = 0  # Reversion complete, close Short
                
        positions[t] = current_pos
        
    ret_y1 = y1.pct_change().fillna(0).values
    ret_y2 = y2.pct_change().fillna(0).values
    
    dr_spread = np.zeros(n)
    
    # Portfolio Return Construction
    for t in range(1, n):
        gamma_t = gammas[t-1]
        # Normalize weights to sum to 1 gross leverage
        total_weight = 1.0 + abs(gamma_t)
        w1 = 1.0 / total_weight
        w2 = abs(gamma_t) / total_weight
        
        pos = positions[t-1] 
        
        if pos == 1:
            # Long Spread: +W1 * R1 - W2 * R2
            dr_spread[t] = (w1 * ret_y1[t]) - (w2 * ret_y2[t])
        elif pos == -1:
            # Short Spread: -W1 * R1 + W2 * R2
            dr_spread[t] = (-w1 * ret_y1[t]) + (w2 * ret_y2[t])
            
    # Transaction Friction
    pos_changes = np.abs(np.diff(positions, prepend=0))
    tc = pos_changes * (fee_bps / 10000.0)
    
    dr_spread = dr_spread - tc
    cum_pnl = np.cumsum(dr_spread)
    
    results = pd.DataFrame({
        'Position': positions,
        'Daily_Return': dr_spread,
        'Cum_PnL': cum_pnl
    }, index=y1.index)
    
    return results
