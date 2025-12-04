import numpy as np
import pandas as pd
from options_lib.bs import black_scholes_delta, implied_volatility, black_scholes_vega, black_scholes_gamma

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

def delta_vega_hedging(df1, df2, start_date, end_date, option_primary, option_vega, 
                       K_primary, K_vega, r=0.05, maturity1=None, maturity2=None, freq=1,
                       transaction_cost_per_share=0.0, transaction_cost_percentage=0.0):
    """
    Simulates a delta-vega hedging strategy for a primary option using another option for vega hedging
    and the underlying asset for delta hedging.
    """
    
    # Extract data for the specified period (assuming dates exist in both)
    df_hedge = df1.loc[start_date:end_date]
    OP_primary = df1.loc[start_date:end_date, option_primary].values
    OP_vega = df2.loc[start_date:end_date, option_vega].values
    RE = df1.loc[start_date:end_date, 'Close'].values
    n = len(df_hedge)

    # Greeks for both options
    deltas_primary = np.zeros(n)
    deltas_vega = np.zeros(n)
    vegas_primary = np.zeros(n)
    vegas_vega = np.zeros(n)
    gammas_primary = np.zeros(n)
    gammas_vega = np.zeros(n)
    iv_primary = np.zeros(n)
    iv_vega = np.zeros(n)
    alphas = np.zeros(n)  # vega hedge ratio
    net_deltas = np.zeros(n)

    # Portfolio tracking
    shares_held = np.zeros(n)
    vega_option_held = np.zeros(n)
    cash_position = np.zeros(n)
    portfolio_values = np.zeros(n)
    cumulative_costs = np.zeros(n)
    pnl = np.zeros(n)

    # Calculate Greeks
    for i in range(n):
        T1 = (maturity1 - df_hedge.index[i]).days / 365
        T2 = (maturity2 - df_hedge.index[i]).days / 365
        
        iv_primary[i] = implied_volatility(OP_primary[i], RE[i], K_primary, T1, r)
        deltas_primary[i] = black_scholes_delta(RE[i], K_primary, T1, r, iv_primary[i])
        gammas_primary[i] = black_scholes_gamma(RE[i], K_primary, T1, r, iv_primary[i])
        
        iv_vega[i] = implied_volatility(OP_vega[i], RE[i], K_vega, T2, r)
        deltas_vega[i] = black_scholes_delta(RE[i], K_vega, T2, r, iv_vega[i])
        vegas_vega[i] = black_scholes_vega(RE[i], K_vega, T2, r, iv_vega[i])
        gammas_vega[i] = black_scholes_gamma(RE[i], K_vega, T2, r, iv_vega[i])

        # Hedge ratio to minimize gamma, subject to vega neutrality
        if abs(vegas_vega[i]) > 1e-6:
            # Standard vega hedge ratio
            alpha_v = -vegas_primary[i] / vegas_vega[i]
            
            # Gamma of vega-hedged portfolio
            net_gamma_v = gammas_primary[i] + alpha_v * gammas_vega[i]
            
            # If net gamma is high, adjust alpha to reduce it.
            # This is a simplified approach; a true multi-objective optimization would be more complex.
            # Here, we just blend in a gamma-hedging objective.
            if abs(gammas_vega[i]) > 1e-6:
                alpha_g = -gammas_primary[i] / gammas_vega[i]
                # Use a weighted average of the two alphas. 0.5 is an arbitrary weight.
                alphas[i] = 0.5 * alpha_v + 0.5 * alpha_g
            else:
                alphas[i] = alpha_v
        else:
            alphas[i] = 0.0
        
        # Clip for safety
        alphas[i] = np.clip(alphas[i], -5.0, 5.0)
        
        # Net delta after vega (and partial gamma) hedging
        net_deltas[i] = deltas_primary[i] + alphas[i] * deltas_vega[i]

    # Initialize portfolio
    shares_held[0] = -net_deltas[0]
    vega_option_held[0] = alphas[0]
    cash_position[0] = (net_deltas[0] * RE[0] - alphas[0] * OP_vega[0]) - OP_primary[0]
    portfolio_values[0] = OP_primary[0] + shares_held[0] * RE[0] + vega_option_held[0] * OP_vega[0] + cash_position[0]
    pnl[0] = 0.0

    # Simulate hedging
    for i in range(1, n):
        if i % freq == 0 or i == n-1:
            target_shares = -net_deltas[i]
            target_vega_option = alphas[i]
            shares_to_trade = target_shares - shares_held[i-1]
            vega_option_to_trade = target_vega_option - vega_option_held[i-1]
            
            trade_value_shares = abs(shares_to_trade) * RE[i]
            trade_value_vega = abs(vega_option_to_trade) * OP_vega[i]
            cost = (abs(shares_to_trade) * transaction_cost_per_share + 
                   trade_value_shares * transaction_cost_percentage +
                   abs(vega_option_to_trade) * transaction_cost_per_share + 
                   trade_value_vega * transaction_cost_percentage)
            
            cash_position[i] = cash_position[i-1] - cost - shares_to_trade * RE[i] - vega_option_to_trade * OP_vega[i]
            shares_held[i] = target_shares
            vega_option_held[i] = target_vega_option
            cumulative_costs[i] = cumulative_costs[i-1] + cost
        else:
            shares_held[i] = shares_held[i-1]
            vega_option_held[i] = vega_option_held[i-1]
            cash_position[i] = cash_position[i-1]
            cumulative_costs[i] = cumulative_costs[i-1]
        
        portfolio_values[i] = OP_primary[i] + shares_held[i] * RE[i] + vega_option_held[i] * OP_vega[i] + cash_position[i]
        pnl[i] = portfolio_values[i] - portfolio_values[0]

    # Calculate hedging errors (delta hedging error for primary option)
    A_errors = np.zeros(n - 1)
    for i in range(n-1):
        idx = (i // freq) * freq
        current_net_delta = net_deltas[idx]
        current_alpha = alphas[idx]
        dC_primary = OP_primary[i+1] - OP_primary[i]
        dC_vega = OP_vega[i+1] - OP_vega[i]
        dR = RE[i+1] - RE[i]
        A_errors[i] = dC_primary - current_net_delta * dR - current_alpha * dC_vega

    E = np.mean(A_errors**2)

    return (df_hedge, net_deltas, alphas, OP_primary, OP_vega, RE, iv_primary, iv_vega, A_errors, 
            shares_held, vega_option_held, cash_position, portfolio_values, cumulative_costs, pnl)

def delta_vega_gamma_hedging(df1, df2, df3, start_date, end_date, 
                             option_primary, option_vega, option_gamma,
                             K_primary, K_vega, K_gamma, 
                             r=0.05, 
                             maturity1=None, maturity2=None, maturity3=None, 
                             freq=1,
                             transaction_cost_per_share=0.0, 
                             transaction_cost_percentage=0.0):
    """
    Simulates a delta-vega-gamma hedging strategy for a primary option using two other options
    for vega and gamma hedging, and the underlying asset for delta hedging.
    """
    
    df_hedge = df1.loc[start_date:end_date]
    OP_primary = df1.loc[start_date:end_date, option_primary].values
    OP_vega = df2.loc[start_date:end_date, option_vega].values
    OP_gamma = df3.loc[start_date:end_date, option_gamma].values
    RE = df1.loc[start_date:end_date, 'Close'].values
    n = len(df_hedge)

    # Greeks for all three options
    deltas = np.zeros((n, 3))
    vegas = np.zeros((n, 3))
    gammas = np.zeros((n, 3))
    ivs = np.zeros((n, 3))
    
    options_params = [
        (OP_primary, K_primary, maturity1),
        (OP_vega, K_vega, maturity2),
        (OP_gamma, K_gamma, maturity3)
    ]

    for i in range(n):
        for j, (OP, K, maturity) in enumerate(options_params):
            T = (maturity - df_hedge.index[i]).days / 365
            ivs[i, j] = implied_volatility(OP[i], RE[i], K, T, r)
            deltas[i, j] = black_scholes_delta(RE[i], K, T, r, ivs[i, j])
            vegas[i, j] = black_scholes_vega(RE[i], K, T, r, ivs[i, j])
            gammas[i, j] = black_scholes_gamma(RE[i], K, T, r, ivs[i, j])

    # Hedging positions
    shares_held = np.zeros(n)
    vega_option_held = np.zeros(n)
    gamma_option_held = np.zeros(n)
    
    # Portfolio tracking
    cash_position = np.zeros(n)
    portfolio_values = np.zeros(n)
    cumulative_costs = np.zeros(n)
    pnl = np.zeros(n)

    # Initialize portfolio
    A = np.array([
        [vegas[0, 1], vegas[0, 2]],
        [gammas[0, 1], gammas[0, 2]]
    ])
    b = np.array([-vegas[0, 0], -gammas[0, 0]])
    try:
        hedge_ratios = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        hedge_ratios = np.zeros(2)

    vega_option_held[0] = hedge_ratios[0]
    gamma_option_held[0] = hedge_ratios[1]
    net_delta = deltas[0, 0] + hedge_ratios[0] * deltas[0, 1] + hedge_ratios[1] * deltas[0, 2]
    shares_held[0] = -net_delta
    
    cash_position[0] = (shares_held[0] * RE[0] - vega_option_held[0] * OP_vega[0] - gamma_option_held[0] * OP_gamma[0]) - OP_primary[0]
    portfolio_values[0] = (OP_primary[0] + shares_held[0] * RE[0] + 
                           vega_option_held[0] * OP_vega[0] + gamma_option_held[0] * OP_gamma[0] + 
                           cash_position[0])
    pnl[0] = 0.0

    # Simulate hedging
    for i in range(1, n):
        if i % freq == 0 or i == n - 1:
            A = np.array([
                [vegas[i, 1], vegas[i, 2]],
                [gammas[i, 1], gammas[i, 2]]
            ])
            b = np.array([-vegas[i, 0], -gammas[i, 0]])
            try:
                hedge_ratios = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                hedge_ratios = np.array([vega_option_held[i-1], gamma_option_held[i-1]])

            target_vega_option = hedge_ratios[0]
            target_gamma_option = hedge_ratios[1]
            
            net_delta = deltas[i, 0] + target_vega_option * deltas[i, 1] + target_gamma_option * deltas[i, 2]
            target_shares = -net_delta

            shares_to_trade = target_shares - shares_held[i-1]
            vega_option_to_trade = target_vega_option - vega_option_held[i-1]
            gamma_option_to_trade = target_gamma_option - gamma_option_held[i-1]

            cost = (abs(shares_to_trade) * transaction_cost_per_share +
                    abs(shares_to_trade) * RE[i] * transaction_cost_percentage +
                    abs(vega_option_to_trade) * transaction_cost_per_share +
                    abs(vega_option_to_trade) * OP_vega[i] * transaction_cost_percentage +
                    abs(gamma_option_to_trade) * transaction_cost_per_share +
                    abs(gamma_option_to_trade) * OP_gamma[i] * transaction_cost_percentage)

            cash_position[i] = (cash_position[i-1] - cost - shares_to_trade * RE[i] -
                                vega_option_to_trade * OP_vega[i] - gamma_option_to_trade * OP_gamma[i])
            shares_held[i] = target_shares
            vega_option_held[i] = target_vega_option
            gamma_option_held[i] = target_gamma_option
            cumulative_costs[i] = cumulative_costs[i-1] + cost
        else:
            shares_held[i] = shares_held[i-1]
            vega_option_held[i] = vega_option_held[i-1]
            gamma_option_held[i] = gamma_option_held[i-1]
            cash_position[i] = cash_position[i-1]
            cumulative_costs[i] = cumulative_costs[i-1]

        portfolio_values[i] = (OP_primary[i] + shares_held[i] * RE[i] + 
                               vega_option_held[i] * OP_vega[i] + gamma_option_held[i] * OP_gamma[i] + 
                               cash_position[i])
        pnl[i] = portfolio_values[i] - portfolio_values[0]

    # Calculate hedging errors
    A_errors = np.zeros(n - 1)
    for i in range(n - 1):
        idx = (i // freq) * freq
        
        A = np.array([
            [vegas[idx, 1], vegas[idx, 2]],
            [gammas[idx, 1], gammas[idx, 2]]
        ])
        b = np.array([-vegas[idx, 0], -gammas[idx, 0]])
        try:
            hedge_ratios = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            hedge_ratios = np.zeros(2)

        current_vega_pos = hedge_ratios[0]
        current_gamma_pos = hedge_ratios[1]
        current_net_delta = deltas[idx, 0] + current_vega_pos * deltas[idx, 1] + current_gamma_pos * deltas[idx, 2]

        dC_primary = OP_primary[i+1] - OP_primary[i]
        dC_vega = OP_vega[i+1] - OP_vega[i]
        dC_gamma = OP_gamma[i+1] - OP_gamma[i]
        dR = RE[i+1] - RE[i]
        
        A_errors[i] = (dC_primary - current_net_delta * dR - 
                       current_vega_pos * dC_vega - current_gamma_pos * dC_gamma)

    return (df_hedge, OP_primary, OP_vega, OP_gamma, RE, A_errors,
            shares_held, vega_option_held, gamma_option_held,
            cash_position, portfolio_values, cumulative_costs, pnl)
