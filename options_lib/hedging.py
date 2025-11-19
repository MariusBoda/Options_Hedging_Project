import numpy as np
import pandas as pd
from options_lib.bs import black_scholes_delta, implied_volatility

def simple_delta_hedging(df, start_date, end_date, option_col, K, r, maturity, freq=1):
    start_idx = df.index.get_loc(start_date)
    end_idx = df.index.get_loc(end_date) + 1
    
    df_hedge = df.iloc[start_idx:end_idx]
    OP = df[option_col].values[start_idx:end_idx]
    RE = df['Close'].values[start_idx:end_idx]
    n = len(df_hedge)

    deltas = np.zeros(n)
    A_errors = np.zeros(n - 1)
    iv_values = np.zeros(n)

    for i in range(n):
        T = (maturity - df_hedge.index[i]).days / 365
        iv = implied_volatility(OP[i], RE[i], K, T, r)
        iv_values[i] = iv
        deltas[i] = black_scholes_delta(RE[i], K, T, r, iv)

    for i in range(n-1):
        delta_idx = (i // freq) * freq
        current_delta = deltas[delta_idx]
        dC = OP[i+1] - OP[i]
        dR = RE[i+1] - RE[i]
        A_errors[i] = dC - current_delta * dR

    E = np.mean(A_errors**2)
    print(f"Mean Squared Hedging Error: {E:.4f}")

    return df_hedge, deltas, OP, RE, iv_values, A_errors

def delta_hedging(df, start_date, end_date, option_col, K, r, maturity, freq=1, 
                  transaction_cost_per_share=0.0, transaction_cost_percentage=0.0):
    
    start_idx = df.index.get_loc(start_date)
    end_idx = df.index.get_loc(end_date) + 1
    
    df_hedge = df.iloc[start_idx:end_idx]
    OP = df[option_col].values[start_idx:end_idx]
    RE = df['Close'].values[start_idx:end_idx]
    n = len(df_hedge)

    deltas = np.zeros(n)
    iv_values = np.zeros(n)
    shares_held = np.zeros(n)
    cash_position = np.zeros(n)
    portfolio_values = np.zeros(n)
    cumulative_costs = np.zeros(n)
    pnl = np.zeros(n)

    for i in range(n):
        T = (maturity - df_hedge.index[i]).days / 365
        iv = implied_volatility(OP[i], RE[i], K, T, r)
        iv_values[i] = iv
        deltas[i] = black_scholes_delta(RE[i], K, T, r, iv)

    shares_held[0] = -deltas[0]  # Short position for delta hedging
    cash_position[0] = deltas[0] * RE[0] - OP[0]  # Cash from shorting shares minus call premium
    portfolio_values[0] = OP[0] + shares_held[0] * RE[0] + cash_position[0]  # Should be ~0
    pnl[0] = 0.0

    for i in range(1, n):
        if i % freq == 0 or i == n-1:
            target_shares = -deltas[i]
            shares_to_trade = target_shares - shares_held[i-1]
            
            trade_value = abs(shares_to_trade) * RE[i]
            cost = (abs(shares_to_trade) * transaction_cost_per_share + 
                   trade_value * transaction_cost_percentage)
            
            cash_position[i] = cash_position[i-1] - cost - shares_to_trade * RE[i]
            shares_held[i] = target_shares
            cumulative_costs[i] = cumulative_costs[i-1] + cost
        else:
            shares_held[i] = shares_held[i-1]
            cash_position[i] = cash_position[i-1]
            cumulative_costs[i] = cumulative_costs[i-1]
        
        portfolio_values[i] = OP[i] + shares_held[i] * RE[i] + cash_position[i]
        pnl[i] = portfolio_values[i] - portfolio_values[0] 

    A_errors = np.zeros(n - 1)
    for i in range(n-1):
        delta_idx = (i // freq) * freq
        current_delta = deltas[delta_idx]
        dC = OP[i+1] - OP[i]
        dR = RE[i+1] - RE[i]
        A_errors[i] = dC - current_delta * dR

    E = np.mean(A_errors**2)

    return (df_hedge, deltas, OP, RE, iv_values, A_errors, 
            shares_held, cash_position, portfolio_values, cumulative_costs, pnl)

def run_hedging_intervals(df, maturity, interval_length=45, step_size=5, num_intervals=10, 
                         option_col="C400", K=400, r=0.05, freq=1,
                         transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005):
    
    results = []
    
    for i in range(num_intervals):
        start_idx = i * step_size
        end_idx = start_idx + interval_length
        
        if end_idx > len(df):
            break 
        
        interval_data = df.iloc[start_idx:end_idx]
        if interval_data[[option_col, 'Close']].isna().any().any():
            continue
        
        start_date = df.index[start_idx]
        end_date = df.index[end_idx - 1]
        
        calendar_days = (end_date - start_date).days
            
        result = delta_hedging(df, start_date, end_date, option_col, K, r, maturity, freq,
                             transaction_cost_per_share, transaction_cost_percentage)
        
        stats = {
            'interval': len(results),
            'start_date': start_date,
            'end_date': end_date,
            'data_points': interval_length,
            'calendar_days': calendar_days,
            'mean_squared_error': np.mean(result[5]**2),
            'total_costs': result[9][-1],
            'final_pnl': result[10][-1],
            'portfolio_volatility': np.std(result[8]),
            'max_portfolio_value': np.max(result[8]),
            'min_portfolio_value': np.min(result[8]),
            'pnl_percentage': (result[10][-1] / result[2][0] * 100) if result[2][0] != 0 else 0
        }
        results.append(stats)
    
    return pd.DataFrame(results)
