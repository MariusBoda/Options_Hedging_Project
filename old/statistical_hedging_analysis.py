import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
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

def load_simulation_data(data_folder="simulation_data"):
    """Load all simulation datasets from the specified folder."""
    datasets = {}
    maturity_dates = []

    # List all feather files in the simulation_data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.feather') and filename.startswith('spy_'):
            filepath = os.path.join(data_folder, filename)
            try:
                df = pd.read_feather(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()

                # Extract maturity date from filename (format: spy_YYMMDD_...)
                parts = filename.split('_')
                if len(parts) >= 2:
                    date_str = parts[1]  # YYMMDD format
                    if len(date_str) == 6:
                        maturity_date = pd.to_datetime(f'20{date_str}', format='%Y%m%d')
                        datasets[filename] = {
                            'data': df,
                            'maturity': maturity_date,
                            'filename': filename
                        }
                        maturity_dates.append(maturity_date)
                        print(f"Loaded {filename}: {len(df)} days, maturity {maturity_date.date()}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # Sort by maturity date
    maturity_dates.sort()
    sorted_datasets = {}
    for maturity in maturity_dates:
        for filename, data_dict in datasets.items():
            if data_dict['maturity'] == maturity:
                sorted_datasets[filename] = data_dict
                break

    return sorted_datasets

def run_single_hedging_simulation(df, start_idx, option_col, K, r, maturity, freq=1,
                                transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                                initial_portfolio_value=100):
    """Run a single delta hedging simulation."""
    df_hedge = df.iloc[start_idx:]
    OP = df[option_col].values[start_idx:]
    RE = df['Close'].values[start_idx:]
    n = len(df_hedge)

    # Check for NA values in option prices
    na_count = pd.isna(OP).sum()
    na_percentage = na_count / len(OP) * 100

    if na_percentage > 10:  # More than 10% NA values
        raise ValueError(f"Too many NA values in {option_col}: {na_count}/{len(OP)} ({na_percentage:.1f}%)")

    # Fill any remaining NA values with forward fill, then backward fill
    OP = pd.Series(OP).ffill().bfill().values

    # If still have NA values at the ends, skip this strike
    if pd.isna(OP).any():
        raise ValueError(f"Unable to fill all NA values in {option_col}")

    deltas = np.zeros(n)
    iv_values = np.zeros(n)
    shares_held = np.zeros(n)
    cash_position = np.zeros(n)
    portfolio_values = np.zeros(n)
    cumulative_costs = np.zeros(n)
    pnl = np.zeros(n)

    # Calculate deltas and IV
    for i in range(n):
        T = (maturity - df_hedge.index[i]).days / 365
        iv = implied_volatility(OP[i], RE[i], K, T, r)
        iv_values[i] = iv
        deltas[i] = black_scholes_delta(RE[i], K, T, r, iv)

    # Initialize portfolio
    shares_held[0] = -deltas[0]
    cash_position[0] = deltas[0] * RE[0] - OP[0]
    position_value = OP[0] + shares_held[0] * RE[0] + cash_position[0]
    portfolio_values[0] = initial_portfolio_value + position_value  # Absolute portfolio value
    pnl[0] = 0.0

    # Simulate hedging
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

        position_value = OP[i] + shares_held[i] * RE[i] + cash_position[i]
        portfolio_values[i] = initial_portfolio_value + position_value  # Absolute portfolio value
        pnl[i] = portfolio_values[i] - portfolio_values[0]

    # Calculate hedging errors
    A_errors = np.zeros(n - 1)
    for i in range(n-1):
        delta_idx = (i // freq) * freq
        current_delta = deltas[delta_idx]
        dC = OP[i+1] - OP[i]
        dR = RE[i+1] - RE[i]
        A_errors[i] = dC - current_delta * dR

    E = np.mean(A_errors**2)

    # Analyze market conditions during hedging period
    market_conditions = analyze_market_conditions(df, start_idx)

    return {
        'mse': E,
        'total_costs': cumulative_costs[-1],
        'final_pnl': pnl[-1],
        'deltas': deltas,
        'iv_values': iv_values,
        'a_errors': A_errors,
        'portfolio_values': portfolio_values,
        'cumulative_costs': cumulative_costs,
        'pnl': pnl,
        'df_hedge': df_hedge,
        'market_conditions': market_conditions
    }

def run_single_vega_hedging_simulation(df, start_idx, option_col_main, option_col_hedge, K, r,
                                     maturity_main, maturity_hedge, freq=1,
                                     transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                                     initial_portfolio_value=100):
    """Run a single delta-vega hedging simulation."""
    df_hedge = df.iloc[start_idx:]
    OP_main = df[option_col_main].values[start_idx:]  # Main option (shorter maturity)
    OP_hedge = df[option_col_hedge].values[start_idx:]  # Vega hedge option (longer maturity)
    RE = df['Close'].values[start_idx:]
    n = len(df_hedge)

    # Check for NA values in option prices
    for col_name, prices in [(option_col_main, OP_main), (option_col_hedge, OP_hedge)]:
        na_count = pd.isna(prices).sum()
        na_percentage = na_count / len(prices) * 100
        if na_percentage > 10:
            raise ValueError(f"Too many NA values in {col_name}: {na_count}/{len(prices)} ({na_percentage:.1f}%)")

    # Fill any remaining NA values
    OP_main = pd.Series(OP_main).ffill().bfill().values
    OP_hedge = pd.Series(OP_hedge).ffill().bfill().values

    # Check for remaining NA values
    if pd.isna(OP_main).any() or pd.isna(OP_hedge).any():
        raise ValueError("Unable to fill all NA values in option prices")

    # Initialize arrays for Greeks and positions
    deltas_main = np.zeros(n)
    deltas_hedge = np.zeros(n)
    vegas_main = np.zeros(n)
    vegas_hedge = np.zeros(n)
    iv_values_main = np.zeros(n)
    iv_values_hedge = np.zeros(n)
    vega_ratios = np.zeros(n)  # α = vega_main / vega_hedge
    net_deltas = np.zeros(n)   # α = delta_main - α * delta_hedge

    shares_held = np.zeros(n)      # Underlying shares
    hedge_options_held = np.zeros(n)  # Vega hedge options
    cash_position = np.zeros(n)
    portfolio_values = np.zeros(n)
    cumulative_costs = np.zeros(n)
    pnl = np.zeros(n)

    # Calculate Greeks for both options
    for i in range(n):
        T_main = (maturity_main - df_hedge.index[i]).days / 365
        T_hedge = (maturity_hedge - df_hedge.index[i]).days / 365

        iv_main = implied_volatility(OP_main[i], RE[i], K, T_main, r)
        iv_hedge = implied_volatility(OP_hedge[i], RE[i], K, T_hedge, r)

        iv_values_main[i] = iv_main
        iv_values_hedge[i] = iv_hedge

        deltas_main[i] = black_scholes_delta(RE[i], K, T_main, r, iv_main)
        deltas_hedge[i] = black_scholes_delta(RE[i], K, T_hedge, r, iv_hedge)
        vegas_main[i] = black_scholes_vega(RE[i], K, T_main, r, iv_main)
        vegas_hedge[i] = black_scholes_vega(RE[i], K, T_hedge, r, iv_hedge)

        # Vega hedge ratio: α = vega_main / vega_hedge
        vega_ratios[i] = vegas_main[i] / vegas_hedge[i] if vegas_hedge[i] != 0 else 0

        # Net delta exposure: Δ = delta_main - α * delta_hedge
        net_deltas[i] = deltas_main[i] - vega_ratios[i] * deltas_hedge[i]

    # Initialize portfolio
    shares_held[0] = -net_deltas[0]  # Hedge net delta with underlying
    hedge_options_held[0] = -vega_ratios[0]  # Short vega hedge options
    cash_position[0] = (net_deltas[0] * RE[0] + vega_ratios[0] * OP_hedge[0]) - OP_main[0]

    position_value = (OP_main[0] + shares_held[0] * RE[0] +
                     hedge_options_held[0] * OP_hedge[0] + cash_position[0])
    portfolio_values[0] = initial_portfolio_value + position_value
    pnl[0] = 0.0

    # Simulate hedging
    for i in range(1, n):
        if i % freq == 0 or i == n-1:
            target_shares = -net_deltas[i]
            target_hedge_options = -vega_ratios[i]

            shares_to_trade = target_shares - shares_held[i-1]
            hedge_options_to_trade = target_hedge_options - hedge_options_held[i-1]

            # Calculate transaction costs
            trade_value_shares = abs(shares_to_trade) * RE[i]
            trade_value_hedge = abs(hedge_options_to_trade) * OP_hedge[i]

            cost = (abs(shares_to_trade) * transaction_cost_per_share +
                   trade_value_shares * transaction_cost_percentage +
                   abs(hedge_options_to_trade) * transaction_cost_per_share +
                   trade_value_hedge * transaction_cost_percentage)

            cash_position[i] = cash_position[i-1] - cost - shares_to_trade * RE[i] - hedge_options_to_trade * OP_hedge[i]
            shares_held[i] = target_shares
            hedge_options_held[i] = target_hedge_options
            cumulative_costs[i] = cumulative_costs[i-1] + cost
        else:
            shares_held[i] = shares_held[i-1]
            hedge_options_held[i] = hedge_options_held[i-1]
            cash_position[i] = cash_position[i-1]
            cumulative_costs[i] = cumulative_costs[i-1]

        position_value = (OP_main[i] + shares_held[i] * RE[i] +
                         hedge_options_held[i] * OP_hedge[i] + cash_position[i])
        portfolio_values[i] = initial_portfolio_value + position_value
        pnl[i] = portfolio_values[i] - portfolio_values[0]

    # Calculate hedging errors
    A_errors = np.zeros(n - 1)
    for i in range(n-1):
        delta_idx = (i // freq) * freq
        current_net_delta = net_deltas[delta_idx]
        dC = OP_main[i+1] - OP_main[i]
        dR = RE[i+1] - RE[i]
        A_errors[i] = dC - current_net_delta * dR

    E = np.mean(A_errors**2)

    # Analyze market conditions during hedging period
    market_conditions = analyze_market_conditions(df, start_idx)

    return {
        'mse': E,
        'total_costs': cumulative_costs[-1],
        'final_pnl': pnl[-1],
        'deltas_main': deltas_main,
        'deltas_hedge': deltas_hedge,
        'vegas_main': vegas_main,
        'vegas_hedge': vegas_hedge,
        'vega_ratios': vega_ratios,
        'net_deltas': net_deltas,
        'iv_values_main': iv_values_main,
        'iv_values_hedge': iv_values_hedge,
        'a_errors': A_errors,
        'portfolio_values': portfolio_values,
        'cumulative_costs': cumulative_costs,
        'pnl': pnl,
        'shares_held': shares_held,
        'hedge_options_held': hedge_options_held,
        'cash_position': cash_position,
        'df_hedge': df_hedge,
        'market_conditions': market_conditions
    }

def run_statistical_vega_hedging_analysis(option_col_main='C400', option_col_hedge='C400', K=400, r=0.02, freq=1,
                                        transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                                        data_folder="simulation_data"):
    """Run comprehensive statistical delta-vega hedging analysis across all maturity and strike combinations."""

    print("Loading simulation datasets...")
    datasets = load_simulation_data(data_folder)

    if not datasets:
        print("No datasets found!")
        return None

    print(f"\nRunning comprehensive delta-vega hedging analysis...")
    print(f"Found {len(datasets)} maturity files")

    # Get all available strikes from the first dataset (assuming all datasets have the same strikes)
    first_dataset = list(datasets.values())[0]['data']
    available_strikes = [col for col in first_dataset.columns if col.startswith('C') and col[1:].isdigit()]

    if not available_strikes:
        print("No call option columns found in the data!")
        return None

    print(f"Found {len(available_strikes)} strike prices: {available_strikes}")
    print(f"Testing all combinations: {len(datasets)} maturities × {len(available_strikes)} strikes × {(len(datasets)-1)} hedge options")
    print(f"Total simulations: {len(datasets) * len(available_strikes) * (len(datasets) - 1)}")

    results = {}
    all_mse = []
    all_costs = []
    all_pnl = []
    simulation_count = 0
    successful_simulations = 0

    # For each main maturity file
    for main_filename, main_data_dict in datasets.items():
        main_maturity = main_data_dict['maturity']
        df_main = main_data_dict['data']

        print(f"\n{'='*80}")
        print(f"MAIN MATURITY: {main_filename} ({main_maturity.date()})")
        print(f"{'='*80}")

        # For each strike in the main maturity
        for strike_col in available_strikes:
            strike_price = int(strike_col[1:])

            print(f"\n  STRIKE: {strike_price} ({strike_col})")

            # For each possible hedge maturity (all other maturities)
            for hedge_filename, hedge_data_dict in datasets.items():
                if hedge_filename == main_filename:
                    continue  # Skip same maturity

                hedge_maturity = hedge_data_dict['maturity']
                df_hedge = hedge_data_dict['data']

                simulation_count += 1
                print(f"    Hedge: {hedge_filename} ({hedge_maturity.date()}) - Simulation {simulation_count}")

                try:
                    # Merge datasets on date index
                    merged_df = pd.merge(df_main, df_hedge, left_index=True, right_index=True,
                                       suffixes=('_main', '_hedge'))

                    if len(merged_df) < 30:  # Need minimum data for hedging
                        print(f"      Skipping - insufficient overlapping data ({len(merged_df)} days)")
                        continue

                    # Rename columns to match expected format
                    merged_df['Close'] = merged_df['Close_main']  # Underlying price
                    merged_df[strike_col] = merged_df[f'{strike_col}_main']  # Main option
                    merged_df[strike_col] = merged_df[f'{strike_col}_hedge']  # Hedge option (same strike)

                    # Find start index (last 45 trading days before main maturity)
                    total_days = len(merged_df)
                    start_idx = max(0, total_days - 45)

                    result = run_single_vega_hedging_simulation(
                        merged_df, start_idx, strike_col, strike_col, strike_price, r,
                        main_maturity, hedge_maturity, freq,
                        transaction_cost_per_share, transaction_cost_percentage
                    )

                    # Create unique key for this combination
                    combo_key = f"{main_filename}_{strike_col}_{hedge_filename}"

                    results[combo_key] = {
                        'result': result,
                        'main_maturity': main_maturity,
                        'hedge_maturity': hedge_maturity,
                        'strike': strike_price,
                        'strike_col': strike_col,
                        'main_filename': main_filename,
                        'hedge_filename': hedge_filename,
                        'days': total_days - start_idx
                    }

                    all_mse.append(result['mse'])
                    all_costs.append(result['total_costs'])
                    all_pnl.append(result['final_pnl'])
                    successful_simulations += 1

                    print(f"      ✓ MSE: {result['mse']:.6f}, Costs: ${result['total_costs']:.2f}, PnL: ${result['final_pnl']:.2f}")

                except Exception as e:
                    print(f"      ✗ Error: {str(e)[:100]}...")
                    continue

    # Calculate comprehensive statistics
    if all_mse:
        stats = {
            'total_simulations_attempted': simulation_count,
            'successful_simulations': successful_simulations,
            'success_rate': successful_simulations / simulation_count * 100,
            'maturities_tested': len(datasets),
            'strikes_tested': len(available_strikes),
            'mse_mean': np.mean(all_mse),
            'mse_std': np.std(all_mse),
            'mse_min': min(all_mse),
            'mse_max': max(all_mse),
            'costs_mean': np.mean(all_costs),
            'costs_std': np.std(all_costs),
            'pnl_mean': np.mean(all_pnl),
            'pnl_std': np.std(all_pnl),
            'individual_results': results,
            'hedging_type': 'delta-vega-comprehensive'
        }

        print(f"\n{'='*100}")
        print("COMPREHENSIVE DELTA-VEGA HEDGING ANALYSIS SUMMARY")
        print(f"{'='*100}")
        print(f"Simulations Attempted: {stats['total_simulations_attempted']}")
        print(f"Successful Simulations: {stats['successful_simulations']} ({stats['success_rate']:.1f}%)")
        print(f"Maturities Tested: {stats['maturities_tested']}")
        print(f"Strikes Tested: {stats['strikes_tested']}")
        print(f"Total Combinations: {stats['maturities_tested'] * stats['strikes_tested'] * (stats['maturities_tested'] - 1)}")
        print()

        print("MSE Statistics:")
        print(f"  Mean ± Std: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
        print(f"  Range: {stats['mse_min']:.6f} - {stats['mse_max']:.6f}")
        print()

        print("Transaction Costs:")
        print(f"  Mean ± Std: ${stats['costs_mean']:.2f} ± ${stats['costs_std']:.2f}")
        print()

        print("Portfolio Performance:")
        print(f"  Final PnL: ${stats['pnl_mean']:.2f} ± ${stats['pnl_std']:.2f}")
        print()

        return stats
    else:
        print("No successful delta-vega simulations!")
        return None

def compare_delta_vs_vega_hedging(option_col='C400', K=400, r=0.02, freq=1,
                                transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                                data_folder="simulation_data"):
    """Compare delta hedging vs delta-vega hedging performance."""

    print("Comparing Delta Hedging vs Delta-Vega Hedging")
    print("="*60)

    # Run delta hedging analysis
    print("\n1. Running Delta Hedging Analysis...")
    delta_stats = run_statistical_hedging_analysis(
        option_col, K, r, freq, transaction_cost_per_share, transaction_cost_percentage, data_folder
    )

    # Run delta-vega hedging analysis
    print("\n2. Running Delta-Vega Hedging Analysis...")
    vega_stats = run_statistical_vega_hedging_analysis(
        option_col, option_col, K, r, freq, transaction_cost_per_share, transaction_cost_percentage, data_folder
    )

    if not delta_stats or not vega_stats:
        print("Unable to compare - missing results from one or both strategies")
        return None

    # Create comparison results
    comparison = {
        'delta_hedging': delta_stats,
        'vega_hedging': vega_stats,
        'comparison': {
            'mse_improvement': delta_stats['mse_mean'] - vega_stats['mse_mean'],
            'mse_improvement_pct': ((delta_stats['mse_mean'] - vega_stats['mse_mean']) / delta_stats['mse_mean']) * 100,
            'cost_difference': vega_stats['costs_mean'] - delta_stats['costs_mean'],
            'pnl_difference': vega_stats['pnl_mean'] - delta_stats['pnl_mean'],
            'delta_experiments': len(delta_stats['individual_results']),
            'vega_experiments': len(vega_stats['individual_results'])
        }
    }

    # Print comparison summary
    print("\n" + "="*80)
    print("DELTA VS DELTA-VEGA HEDGING COMPARISON")
    print("="*80)

    print("Delta Hedging Performance:")
    print(f"  MSE: {delta_stats['mse_mean']:.6f} ± {delta_stats['mse_std']:.6f}")
    print(f"  Costs: ${delta_stats['costs_mean']:.2f} ± ${delta_stats['costs_std']:.2f}")
    print(f"  PnL: ${delta_stats['pnl_mean']:.2f} ± ${delta_stats['pnl_std']:.2f}")
    print(f"  Experiments: {len(delta_stats['individual_results'])}")

    print("\nDelta-Vega Hedging Performance:")
    print(f"  MSE: {vega_stats['mse_mean']:.6f} ± {vega_stats['mse_std']:.6f}")
    print(f"  Costs: ${vega_stats['costs_mean']:.2f} ± ${vega_stats['costs_std']:.2f}")
    print(f"  PnL: ${vega_stats['pnl_mean']:.2f} ± ${vega_stats['pnl_std']:.2f}")
    print(f"  Experiments: {len(vega_stats['individual_results'])}")

    print("\nPerformance Comparison:")
    mse_imp = comparison['comparison']['mse_improvement']
    mse_imp_pct = comparison['comparison']['mse_improvement_pct']
    cost_diff = comparison['comparison']['cost_difference']
    pnl_diff = comparison['comparison']['pnl_difference']

    print(f"  MSE Improvement: {mse_imp:.6f} ({mse_imp_pct:+.1f}%)")
    print(f"  Cost Difference: ${cost_diff:+.2f}")
    print(f"  PnL Difference: ${pnl_diff:+.2f}")

    if mse_imp > 0:
        print("  → Delta-vega hedging reduces hedging error!")
    else:
        print("  → Delta hedging has lower hedging error.")

    if cost_diff > 0:
        print("  → Delta-vega hedging costs more in transactions.")
    else:
        print("  → Delta-vega hedging costs less in transactions.")

    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Delta Hedging vs Delta-Vega Hedging Comparison', fontsize=16)

    strategies = ['Delta\nHedging', 'Delta-Vega\nHedging']
    mse_values = [delta_stats['mse_mean'], vega_stats['mse_mean']]
    mse_stds = [delta_stats['mse_std'], vega_stats['mse_std']]

    cost_values = [delta_stats['costs_mean'], vega_stats['costs_mean']]
    cost_stds = [delta_stats['costs_std'], vega_stats['costs_std']]

    pnl_values = [delta_stats['pnl_mean'], vega_stats['pnl_mean']]
    pnl_stds = [delta_stats['pnl_std'], vega_stats['pnl_std']]

    # MSE comparison
    axes[0].bar(range(len(strategies)), mse_values, yerr=mse_stds, capsize=5,
                color=['lightblue', 'lightgreen'], alpha=0.7)
    axes[0].set_title('Mean Squared Error Comparison')
    axes[0].set_ylabel('MSE')
    axes[0].set_xticks(range(len(strategies)))
    axes[0].set_xticklabels(strategies)

    # Cost comparison
    axes[1].bar(range(len(strategies)), cost_values, yerr=cost_stds, capsize=5,
                color=['lightblue', 'lightgreen'], alpha=0.7)
    axes[1].set_title('Transaction Costs Comparison')
    axes[1].set_ylabel('Total Costs ($)')
    axes[1].set_xticks(range(len(strategies)))
    axes[1].set_xticklabels(strategies)

    # PnL comparison
    colors = ['green' if x >= 0 else 'red' for x in pnl_values]
    axes[2].bar(range(len(strategies)), pnl_values, yerr=pnl_stds, capsize=5,
                color=colors, alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Final PnL Comparison')
    axes[2].set_ylabel('PnL ($)')
    axes[2].set_xticks(range(len(strategies)))
    axes[2].set_xticklabels(strategies)

    plt.tight_layout()
    plt.show()

    return comparison

def plot_vega_hedging_results(stats, freq=1, initial_portfolio_value=100):
    """Plot delta-vega hedging results across all experiments."""
    if not stats or stats.get('hedging_type') != 'delta-vega':
        print("No delta-vega hedging results to plot")
        return

    results = stats['individual_results']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Delta-Vega Hedging Analysis (Rehedge Frequency: {freq})', fontsize=16)

    # MSE comparison
    maturities = [pd.to_datetime(r['maturity_main']).strftime('%Y-%m-%d') for r in results.values()]
    mse_values = [r['result']['mse'] for r in results.values()]

    axes[0,0].bar(range(len(maturities)), mse_values)
    axes[0,0].set_title('Mean Squared Hedging Error by Maturity')
    axes[0,0].set_xlabel('Main Maturity Date')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].set_xticks(range(len(maturities)))
    axes[0,0].set_xticklabels(maturities, rotation=45)

    # Transaction costs comparison
    cost_values = [r['result']['total_costs'] for r in results.values()]
    axes[0,1].bar(range(len(maturities)), cost_values)
    axes[0,1].set_title('Total Transaction Costs by Maturity')
    axes[0,1].set_xlabel('Main Maturity Date')
    axes[0,1].set_ylabel('Total Costs ($)')
    axes[0,1].set_xticks(range(len(maturities)))
    axes[0,1].set_xticklabels(maturities, rotation=45)

    # PnL comparison
    pnl_values = [r['result']['final_pnl'] for r in results.values()]
    colors = ['green' if x >= 0 else 'red' for x in pnl_values]
    axes[0,2].bar(range(len(maturities)), pnl_values, color=colors)
    axes[0,2].set_title('Final PnL by Maturity')
    axes[0,2].set_xlabel('Main Maturity Date')
    axes[0,2].set_ylabel('Final PnL ($)')
    axes[0,2].set_xticks(range(len(maturities)))
    axes[0,2].set_xticklabels(maturities, rotation=45)
    axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Vega ratios over time (first experiment)
    first_result = list(results.values())[0]['result']
    df_hedge = first_result['df_hedge']
    vega_ratios = first_result['vega_ratios']

    axes[1,0].plot(df_hedge.index, vega_ratios, color='purple', linestyle='-', linewidth=2, label='Vega Hedge Ratio (α)')
    axes[1,0].set_title('Vega Hedge Ratio Evolution (First Experiment)')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Vega Ratio (α)')
    axes[1,0].grid(True, alpha=0.3)

    # Net delta exposure (first experiment)
    net_deltas = first_result['net_deltas']
    axes[1,1].plot(df_hedge.index, net_deltas, color='blue', linestyle='-', linewidth=2, label='Net Delta Exposure')
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Delta Neutral')
    axes[1,1].set_title('Net Delta Exposure (First Experiment)')
    axes[1,1].set_xlabel('Date')
    axes[1,1].set_ylabel('Net Delta')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # Hedging error distribution
    all_errors = []
    for r in results.values():
        all_errors.extend(r['result']['a_errors'])

    axes[1,2].hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1,2].set_title('Distribution of Daily Hedging Errors')
    axes[1,2].set_xlabel('Hedging Error ($)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Hedge')
    axes[1,2].legend()

    plt.tight_layout()
    plt.show()

def plot_vega_portfolio_evolution(stats, initial_portfolio_value=100):
    """Plot portfolio value evolution for delta-vega hedging."""
    if not stats or stats.get('hedging_type') != 'delta-vega':
        print("No delta-vega hedging results to plot")
        return

    results = stats['individual_results']

    fig, axes = plt.subplots(len(results), 1, figsize=(15, 4*len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    fig.suptitle(f'Delta-Vega Hedging Portfolio Evolution (Initial Balance: ${initial_portfolio_value:,.0f})', fontsize=16)

    for i, (filename, data_dict) in enumerate(results.items()):
        result = data_dict['result']
        maturity_main = data_dict['maturity_main']
        maturity_hedge = data_dict['maturity_hedge']
        df_hedge = result['df_hedge']
        portfolio_values = result['portfolio_values']

        axes[i].plot(df_hedge.index, portfolio_values, color='purple', linestyle='-', linewidth=2, label='Portfolio Value')
        axes[i].axhline(y=initial_portfolio_value, color='red', linestyle='--', alpha=0.7,
                       label=f'Initial Balance (${initial_portfolio_value:,.0f})')
        axes[i].fill_between(df_hedge.index, initial_portfolio_value, portfolio_values,
                           where=(portfolio_values >= initial_portfolio_value),
                           color='green', alpha=0.3, label='Profit')
        axes[i].fill_between(df_hedge.index, initial_portfolio_value, portfolio_values,
                           where=(portfolio_values < initial_portfolio_value),
                           color='red', alpha=0.3, label='Loss')

        axes[i].set_title(f'Main: {maturity_main.date()}, Hedge: {maturity_hedge.date()}')
        axes[i].set_ylabel('Portfolio Value ($)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

def run_vega_hedging_analysis(option_col='C400', K=400, r=0.02, freq=1,
                            transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                            data_folder="simulation_data"):
    """Run delta-vega hedging analysis with plotting."""
    stats = run_statistical_vega_hedging_analysis(
        option_col, option_col, K, r, freq, transaction_cost_per_share, transaction_cost_percentage, data_folder
    )
    if stats:
        plot_vega_hedging_results(stats, freq=freq)
        plot_vega_portfolio_evolution(stats)
    return stats

def run_delta_vs_vega_comparison(option_col='C400', K=400, r=0.02, freq=1,
                               transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                               data_folder="simulation_data"):
    """Run and display comparison between delta and delta-vega hedging."""
    return compare_delta_vs_vega_hedging(
        option_col, K, r, freq, transaction_cost_per_share, transaction_cost_percentage, data_folder
    )

def run_statistical_hedging_analysis(option_col='C400', K=400, r=0.02, freq=1,
                                   transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                                   data_folder="simulation_data"):
    """Run statistical analysis across all available datasets."""

    print("Loading simulation datasets...")
    datasets = load_simulation_data(data_folder)

    if not datasets:
        print("No datasets found!")
        return None

    print(f"\nRunning statistical analysis on {len(datasets)} datasets...")
    print(f"Parameters: Strike={K}, Rehedge freq={freq}, Transaction costs=({transaction_cost_per_share}, {transaction_cost_percentage})")

    results = {}
    all_mse = []
    all_costs = []
    all_pnl = []

    for filename, data_dict in datasets.items():
        df = data_dict['data']
        maturity = data_dict['maturity']

        print(f"\nProcessing {filename} (maturity: {maturity.date()})")

        # Find start index (last 45 trading days before maturity, or all available data)
        total_days = len(df)
        start_idx = max(0, total_days - 45)

        print(f"Using {total_days - start_idx} trading days for hedging")

        try:
            result = run_single_hedging_simulation(
                df, start_idx, option_col, K, r, maturity, freq,
                transaction_cost_per_share, transaction_cost_percentage
            )

            results[filename] = {
                'result': result,
                'maturity': maturity,
                'days': total_days - start_idx
            }

            all_mse.append(result['mse'])
            all_costs.append(result['total_costs'])
            all_pnl.append(result['final_pnl'])

            print(f"  MSE: {result['mse']:.6f}")
            print(f"  Total Costs: ${result['total_costs']:.2f}")
            print(f"  Final PnL: ${result['final_pnl']:.2f}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Calculate statistics
    if all_mse:
        stats = {
            'mse_mean': np.mean(all_mse),
            'mse_std': np.std(all_mse),
            'costs_mean': np.mean(all_costs),
            'costs_std': np.std(all_costs),
            'pnl_mean': np.mean(all_pnl),
            'pnl_std': np.std(all_pnl),
            'individual_results': results
        }

        print("\n" + "="*60)
        print("STATISTICAL SUMMARY ACROSS ALL EXPERIMENTS")
        print("="*60)
        print(f"Number of experiments: {len(all_mse)}")
        print(f"MSE: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
        print(f"Transaction Costs: ${stats['costs_mean']:.2f} ± ${stats['costs_std']:.2f}")
        print(f"Final PnL: ${stats['pnl_mean']:.2f} ± ${stats['pnl_std']:.2f}")

        return stats
    else:
        print("No successful simulations!")
        return None

def plot_statistical_results(stats, freq=1):
    """Plot statistical results across all experiments."""
    if not stats:
        return

    results = stats['individual_results']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Statistical Hedging Analysis (Rehedge Frequency: {freq})', fontsize=16)

    # MSE comparison
    maturities = [pd.to_datetime(r['maturity']).strftime('%Y-%m-%d') for r in results.values()]
    mse_values = [r['result']['mse'] for r in results.values()]

    axes[0,0].bar(range(len(maturities)), mse_values)
    axes[0,0].set_title('Mean Squared Hedging Error by Maturity')
    axes[0,0].set_xlabel('Maturity Date')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].set_xticks(range(len(maturities)))
    axes[0,0].set_xticklabels(maturities, rotation=45)

    # Transaction costs comparison
    cost_values = [r['result']['total_costs'] for r in results.values()]
    axes[0,1].bar(range(len(maturities)), cost_values)
    axes[0,1].set_title('Total Transaction Costs by Maturity')
    axes[0,1].set_xlabel('Maturity Date')
    axes[0,1].set_ylabel('Total Costs ($)')
    axes[0,1].set_xticks(range(len(maturities)))
    axes[0,1].set_xticklabels(maturities, rotation=45)

    # PnL comparison
    pnl_values = [r['result']['final_pnl'] for r in results.values()]
    colors = ['green' if x >= 0 else 'red' for x in pnl_values]
    axes[1,0].bar(range(len(maturities)), pnl_values, color=colors)
    axes[1,0].set_title('Final PnL by Maturity')
    axes[1,0].set_xlabel('Maturity Date')
    axes[1,0].set_ylabel('Final PnL ($)')
    axes[1,0].set_xticks(range(len(maturities)))
    axes[1,0].set_xticklabels(maturities, rotation=45)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Hedging error distribution
    all_errors = []
    for r in results.values():
        all_errors.extend(r['result']['a_errors'])

    axes[1,1].hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Daily Hedging Errors')
    axes[1,1].set_xlabel('Hedging Error ($)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Hedge')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

def plot_portfolio_evolution(stats, initial_portfolio_value=100):
    """Plot portfolio value evolution over time for all experiments."""
    if not stats:
        return

    results = stats['individual_results']

    fig, axes = plt.subplots(len(results), 1, figsize=(15, 4*len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]  # Make it iterable

    fig.suptitle(f'Portfolio Value Evolution (Initial Balance: ${initial_portfolio_value:,.0f})', fontsize=16)

    for i, (filename, data_dict) in enumerate(results.items()):
        result = data_dict['result']
        maturity = data_dict['maturity']
        df_hedge = result['df_hedge']
        portfolio_values = result['portfolio_values']

        axes[i].plot(df_hedge.index, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
        axes[i].axhline(y=initial_portfolio_value, color='red', linestyle='--', alpha=0.7, label=f'Initial Balance (${initial_portfolio_value:,.0f})')
        axes[i].fill_between(df_hedge.index, initial_portfolio_value, portfolio_values,
                           where=(portfolio_values >= initial_portfolio_value),
                           color='green', alpha=0.3, label='Profit')
        axes[i].fill_between(df_hedge.index, initial_portfolio_value, portfolio_values,
                           where=(portfolio_values < initial_portfolio_value),
                           color='red', alpha=0.3, label='Loss')

        axes[i].set_title(f'Maturity: {maturity.date()}')
        axes[i].set_ylabel('Portfolio Value ($)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

def plot_statistical_results_with_portfolio(stats, freq=1, initial_portfolio_value=100):
    """Enhanced plot function that includes portfolio value and market direction analysis."""
    if not stats:
        return

    results = stats['individual_results']

    # Calculate portfolio value statistics
    final_portfolio_values = []
    max_portfolio_values = []
    min_portfolio_values = []
    market_directions = []

    for data_dict in results.values():
        portfolio_values = data_dict['result']['portfolio_values']
        final_portfolio_values.append(portfolio_values[-1])
        max_portfolio_values.append(np.max(portfolio_values))
        min_portfolio_values.append(np.min(portfolio_values))

        # Get market direction if available
        market_conditions = data_dict['result'].get('market_conditions', {})
        market_directions.append(market_conditions.get('trend_direction', 'unknown'))

    portfolio_stats = {
        'final_mean': np.mean(final_portfolio_values),
        'final_std': np.std(final_portfolio_values),
        'max_mean': np.mean(max_portfolio_values),
        'max_std': np.std(max_portfolio_values),
        'min_mean': np.mean(min_portfolio_values),
        'min_std': np.std(min_portfolio_values)
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Enhanced Statistical Hedging Analysis with Market Direction (Rehedge Frequency: {freq})', fontsize=16)

    # MSE comparison
    maturities = [pd.to_datetime(r['maturity']).strftime('%Y-%m-%d') for r in results.values()]
    mse_values = [r['result']['mse'] for r in results.values()]

    axes[0,0].bar(range(len(maturities)), mse_values)
    axes[0,0].set_title('Mean Squared Hedging Error by Maturity')
    axes[0,0].set_xlabel('Maturity Date')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].set_xticks(range(len(maturities)))
    axes[0,0].set_xticklabels(maturities, rotation=45)

    # Transaction costs comparison
    cost_values = [r['result']['total_costs'] for r in results.values()]
    axes[0,1].bar(range(len(maturities)), cost_values)
    axes[0,1].set_title('Total Transaction Costs by Maturity')
    axes[0,1].set_xlabel('Maturity Date')
    axes[0,1].set_ylabel('Total Costs ($)')
    axes[0,1].set_xticks(range(len(maturities)))
    axes[0,1].set_xticklabels(maturities, rotation=45)

    # Final portfolio value comparison
    colors = ['green' if x >= initial_portfolio_value else 'red' for x in final_portfolio_values]
    bars = axes[0,2].bar(range(len(maturities)), final_portfolio_values, color=colors)
    axes[0,2].axhline(y=initial_portfolio_value, color='black', linestyle='--', alpha=0.7,
                     label=f'Initial (${initial_portfolio_value:,.0f})')
    axes[0,2].set_title('Final Portfolio Value by Maturity')
    axes[0,2].set_xlabel('Maturity Date')
    axes[0,2].set_ylabel('Final Portfolio Value ($)')
    axes[0,2].set_xticks(range(len(maturities)))
    axes[0,2].set_xticklabels(maturities, rotation=45)
    axes[0,2].legend()
    axes[0,2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # PnL comparison
    pnl_values = [r['result']['final_pnl'] for r in results.values()]
    colors = ['green' if x >= 0 else 'red' for x in pnl_values]
    axes[1,0].bar(range(len(maturities)), pnl_values, color=colors)
    axes[1,0].set_title('Final PnL by Maturity')
    axes[1,0].set_xlabel('Maturity Date')
    axes[1,0].set_ylabel('Final PnL ($)')
    axes[1,0].set_xticks(range(len(maturities)))
    axes[1,0].set_xticklabels(maturities, rotation=45)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Portfolio value distribution
    axes[1,1].hist(final_portfolio_values, bins=10, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=initial_portfolio_value, color='red', linestyle='--', alpha=0.7,
                     label=f'Initial (${initial_portfolio_value:,.0f})')
    axes[1,1].axvline(x=portfolio_stats['final_mean'], color='blue', linestyle='-', alpha=0.7,
                     label=f'Mean (${portfolio_stats["final_mean"]:,.0f})')
    axes[1,1].set_title('Distribution of Final Portfolio Values')
    axes[1,1].set_xlabel('Final Portfolio Value ($)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Hedging error distribution
    all_errors = []
    for r in results.values():
        all_errors.extend(r['result']['a_errors'])

    axes[1,2].hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1,2].set_title('Distribution of Daily Hedging Errors')
    axes[1,2].set_xlabel('Hedging Error ($)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Hedge')
    axes[1,2].legend()

    # Market direction analysis plots
    if market_directions and 'unknown' not in market_directions:
        # Group data by market direction
        direction_data = {}
        for i, direction in enumerate(market_directions):
            if direction not in direction_data:
                direction_data[direction] = {'pnl': [], 'mse': [], 'costs': []}
            direction_data[direction]['pnl'].append(pnl_values[i])
            direction_data[direction]['mse'].append(mse_values[i])
            direction_data[direction]['costs'].append(cost_values[i])

        # Average PnL by market direction
        directions = list(direction_data.keys())
        avg_pnl_by_direction = [np.mean(direction_data[d]['pnl']) for d in directions]
        colors = ['green' if x >= 0 else 'red' for x in avg_pnl_by_direction]
        axes[0,3].bar(directions, avg_pnl_by_direction, color=colors, alpha=0.7)
        axes[0,3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0,3].set_title('Average PnL by Market Direction')
        axes[0,3].set_xlabel('Market Direction')
        axes[0,3].set_ylabel('Average PnL ($)')
        axes[0,3].tick_params(axis='x', rotation=45)

        # Market direction distribution (pie chart)
        direction_counts = [len(direction_data[d]['pnl']) for d in directions]
        colors_pie = ['lightgreen' if d == 'bullish' else 'lightcoral' if d == 'bearish' else 'lightblue' for d in directions]
        axes[1,3].pie(direction_counts, labels=directions, autopct='%1.1f%%', colors=colors_pie)
        axes[1,3].set_title('Market Direction Distribution')
    else:
        # Fallback plots if market direction data not available
        axes[0,3].text(0.5, 0.5, 'Market Direction\nData Not Available',
                      transform=axes[0,3].transAxes, ha='center', va='center', fontsize=12)
        axes[0,3].set_title('Market Direction Analysis')
        axes[0,3].set_xlabel('Market Direction')
        axes[0,3].set_ylabel('Average PnL ($)')

        axes[1,3].text(0.5, 0.5, 'Market Direction\nData Not Available',
                      transform=axes[1,3].transAxes, ha='center', va='center', fontsize=12)
        axes[1,3].set_title('Market Direction Distribution')

    plt.tight_layout()
    plt.show()

    # Print portfolio value statistics
    print(f"\n{'='*60}")
    print("PORTFOLIO VALUE STATISTICS")
    print(f"{'='*60}")
    print(f"Initial Portfolio Value: ${initial_portfolio_value:,.0f}")
    print(f"Final Portfolio Value: ${portfolio_stats['final_mean']:,.0f} ± ${portfolio_stats['final_std']:,.0f}")
    print(f"Maximum Portfolio Value: ${portfolio_stats['max_mean']:,.0f} ± ${portfolio_stats['max_std']:,.0f}")
    print(f"Minimum Portfolio Value: ${portfolio_stats['min_mean']:,.0f} ± ${portfolio_stats['min_std']:,.0f}")
    print(f"Best Performance: ${max(final_portfolio_values):,.0f}")
    print(f"Worst Performance: ${min(final_portfolio_values):,.0f}")

def compare_rehedging_frequencies(option_col='C400', K=400, r=0.02,
                                transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                                data_folder="simulation_data"):
    """Compare different rehedging frequencies across all datasets."""

    frequencies = [1, 2, 5, 10]  # Daily, every 2 days, weekly, bi-weekly
    freq_results = {}

    print("Comparing rehedging frequencies...")

    for freq in frequencies:
        print(f"\nTesting rehedging every {freq} day{'s' if freq > 1 else ''}...")
        stats = run_statistical_hedging_analysis(
            option_col, K, r, freq, transaction_cost_per_share, transaction_cost_percentage, data_folder
        )

        if stats:
            freq_results[freq] = stats
            print(f"  MSE: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
            print(f"  Costs: ${stats['costs_mean']:.2f} ± ${stats['costs_std']:.2f}")

    # Plot comparison
    if freq_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Comparison of Rehedging Frequencies', fontsize=16)

        freq_labels = [f'Every {f} day{"s" if f > 1 else ""}' for f in frequencies]

        # MSE comparison
        mse_means = [freq_results[f]['mse_mean'] for f in frequencies]
        mse_stds = [freq_results[f]['mse_std'] for f in frequencies]
        axes[0].bar(range(len(frequencies)), mse_means, yerr=mse_stds, capsize=5)
        axes[0].set_title('Mean Squared Error')
        axes[0].set_xlabel('Rehedging Frequency')
        axes[0].set_ylabel('MSE')
        axes[0].set_xticks(range(len(frequencies)))
        axes[0].set_xticklabels(freq_labels, rotation=45)

        # Cost comparison
        cost_means = [freq_results[f]['costs_mean'] for f in frequencies]
        cost_stds = [freq_results[f]['costs_std'] for f in frequencies]
        axes[1].bar(range(len(frequencies)), cost_means, yerr=cost_stds, capsize=5)
        axes[1].set_title('Transaction Costs')
        axes[1].set_xlabel('Rehedging Frequency')
        axes[1].set_ylabel('Total Costs ($)')
        axes[1].set_xticks(range(len(frequencies)))
        axes[1].set_xticklabels(freq_labels, rotation=45)

        # PnL comparison
        pnl_means = [freq_results[f]['pnl_mean'] for f in frequencies]
        pnl_stds = [freq_results[f]['pnl_std'] for f in frequencies]
        colors = ['green' if x >= 0 else 'red' for x in pnl_means]
        axes[2].bar(range(len(frequencies)), pnl_means, yerr=pnl_stds, capsize=5, color=colors)
        axes[2].set_title('Final PnL')
        axes[2].set_xlabel('Rehedging Frequency')
        axes[2].set_ylabel('PnL ($)')
        axes[2].set_xticks(range(len(frequencies)))
        axes[2].set_xticklabels(freq_labels, rotation=45)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    return freq_results

# Example usage functions for the notebook
def run_basic_statistical_analysis():
    """Run basic statistical analysis with default parameters."""
    stats = run_statistical_hedging_analysis()
    if stats:
        plot_statistical_results_with_portfolio(stats)
        plot_portfolio_evolution(stats)
    return stats

def run_frequency_comparison():
    """Compare different rehedging frequencies."""
    return compare_rehedging_frequencies()

def run_custom_analysis(option_col='C400', K=400, freq=1, transaction_cost_per_share=0.01):
    """Run custom analysis with specified parameters."""
    stats = run_statistical_hedging_analysis(
        option_col=option_col, K=K, freq=freq,
        transaction_cost_per_share=transaction_cost_per_share
    )
    if stats:
        plot_statistical_results_with_portfolio(stats)
        plot_portfolio_evolution(stats)
    return stats

def run_portfolio_analysis(option_col='C400', K=400, freq=1, initial_balance=100):
    """Run analysis with portfolio value tracking and visualization."""
    stats = run_statistical_hedging_analysis(
        option_col=option_col, K=K, freq=freq
    )
    if stats:
        plot_portfolio_evolution(stats, initial_portfolio_value=initial_balance)
        plot_statistical_results_with_portfolio(stats, freq=freq, initial_portfolio_value=initial_balance)
    return stats

def compare_strikes(freq=1, r=0.02, transaction_cost_per_share=0.01, transaction_cost_percentage=0.0005,
                   data_folder="simulation_data"):
    """Compare hedging performance across all available strikes in the datasets."""

    print("Loading simulation datasets...")
    datasets = load_simulation_data(data_folder)

    if not datasets:
        print("No datasets found!")
        return None

    # Get the first dataset to determine available strikes
    first_dataset = list(datasets.values())[0]['data']
    strike_cols = [col for col in first_dataset.columns if col.startswith('C') and col[1:].isdigit()]

    if not strike_cols:
        print("No call option columns found in the data!")
        return None

    print(f"Found {len(strike_cols)} strike prices: {strike_cols}")
    print(f"Testing rehedging every {freq} day{'s' if freq > 1 else ''}...")

    # Run analysis for each strike
    strike_results = {}

    for strike_col in strike_cols:
        strike_price = int(strike_col[1:])  # Extract strike price from column name

        print(f"\n{'='*50}")
        print(f"Analyzing strike {strike_price} ({strike_col})")
        print(f"{'='*50}")

        try:
            stats = run_statistical_hedging_analysis(
                option_col=strike_col, K=strike_price, r=r, freq=freq,
                transaction_cost_per_share=transaction_cost_per_share,
                transaction_cost_percentage=transaction_cost_percentage,
                data_folder=data_folder
            )

            if stats:
                strike_results[strike_price] = stats
                print(f"Strike {strike_price} - MSE: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")

        except Exception as e:
            print(f"Error analyzing strike {strike_price}: {e}")

    # Create comparative analysis
    if strike_results:
        print(f"\n{'='*70}")
        print("STRIKE PRICE COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"Rehedging frequency: Every {freq} day{'s' if freq > 1 else ''}")
        print(f"Number of maturities tested: {len(list(datasets.keys()))}")
        print(f"Strike prices analyzed: {sorted(strike_results.keys())}")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparison of Hedging Performance by Strike Price', fontsize=16)

        strikes = sorted(strike_results.keys())
        strike_labels = [f'{s}' for s in strikes]

        # MSE comparison
        mse_means = [strike_results[s]['mse_mean'] for s in strikes]
        mse_stds = [strike_results[s]['mse_std'] for s in strikes]
        axes[0,0].bar(range(len(strikes)), mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
        axes[0,0].set_title('Mean Squared Error by Strike Price')
        axes[0,0].set_xlabel('Strike Price')
        axes[0,0].set_ylabel('MSE')
        axes[0,0].set_xticks(range(len(strikes)))
        axes[0,0].set_xticklabels(strike_labels, rotation=45)

        # Transaction costs comparison
        cost_means = [strike_results[s]['costs_mean'] for s in strikes]
        cost_stds = [strike_results[s]['costs_std'] for s in strikes]
        axes[0,1].bar(range(len(strikes)), cost_means, yerr=cost_stds, capsize=5, alpha=0.7)
        axes[0,1].set_title('Transaction Costs by Strike Price')
        axes[0,1].set_xlabel('Strike Price')
        axes[0,1].set_ylabel('Total Costs ($)')
        axes[0,1].set_xticks(range(len(strikes)))
        axes[0,1].set_xticklabels(strike_labels, rotation=45)

        # PnL comparison
        pnl_means = [strike_results[s]['pnl_mean'] for s in strikes]
        pnl_stds = [strike_results[s]['pnl_std'] for s in strikes]
        colors = ['green' if x >= 0 else 'red' for x in pnl_means]
        axes[1,0].bar(range(len(strikes)), pnl_means, yerr=pnl_stds, capsize=5, alpha=0.7, color=colors)
        axes[1,0].set_title('Final PnL by Strike Price')
        axes[1,0].set_xlabel('Strike Price')
        axes[1,0].set_ylabel('PnL ($)')
        axes[1,0].set_xticks(range(len(strikes)))
        axes[1,0].set_xticklabels(strike_labels, rotation=45)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Strike price vs performance scatter
        axes[1,1].scatter(strikes, mse_means, s=50, alpha=0.7, label='MSE')
        axes[1,1].scatter(strikes, cost_means, s=50, alpha=0.7, label='Costs')
        axes[1,1].set_title('Strike Price vs Performance Metrics')
        axes[1,1].set_xlabel('Strike Price')
        axes[1,1].set_ylabel('Performance Metric')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print detailed results table
        print(f"\n{'Strike':<8} {'MSE (mean±std)':<20} {'Costs (mean±std)':<20} {'PnL (mean±std)':<20}")
        print("-" * 70)
        for strike in strikes:
            mse_str = f"{strike_results[strike]['mse_mean']:.4f}±{strike_results[strike]['mse_std']:.4f}"
            cost_str = f"${strike_results[strike]['costs_mean']:.2f}±${strike_results[strike]['costs_std']:.2f}"
            pnl_str = f"${strike_results[strike]['pnl_mean']:.2f}±${strike_results[strike]['pnl_std']:.2f}"
            print(f"{strike:<8} {mse_str:<20} {cost_str:<20} {pnl_str:<20}")

        return strike_results
    else:
        print("No successful strike analyses!")
        return None

def plot_all_strikes_by_maturity(freq=1, r=0.02, transaction_cost_per_share=0.01,
                               transaction_cost_percentage=0.0005, data_folder="simulation_data",
                               initial_portfolio_value=100):
    """Plot all strikes for each maturity on separate plots."""

    print("Loading simulation datasets...")
    datasets = load_simulation_data(data_folder)

    if not datasets:
        print("No datasets found!")
        return None

    # Get available strikes
    first_dataset = list(datasets.values())[0]['data']
    strike_cols = [col for col in first_dataset.columns if col.startswith('C') and col[1:].isdigit()]

    if not strike_cols:
        print("No call option columns found in the data!")
        return None

    print(f"Found {len(strike_cols)} strike prices: {strike_cols}")
    print(f"Found {len(datasets)} maturities")
    print(f"Total simulations: {len(datasets) * len(strike_cols)}")

    # Create plots for each maturity
    fig, axes = plt.subplots(len(datasets), 1, figsize=(15, 5*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle(f'All Strikes by Maturity (Rehedge: {freq} day{"s" if freq > 1 else ""}, Initial: ${initial_portfolio_value})', fontsize=16)

    maturity_results = {}

    for i, (filename, data_dict) in enumerate(datasets.items()):
        df = data_dict['data']
        maturity = data_dict['maturity']

        print(f"\nProcessing {filename} (maturity: {maturity.date()})")

        maturity_strikes = []
        maturity_portfolio_values = []

        for strike_col in strike_cols:
            strike_price = int(strike_col[1:])

            try:
                # Find start index (last 45 trading days before maturity)
                total_days = len(df)
                start_idx = max(0, total_days - 45)

                result = run_single_hedging_simulation(
                    df, start_idx, strike_col, strike_price, r, maturity, freq,
                    transaction_cost_per_share, transaction_cost_percentage,
                    initial_portfolio_value
                )

                maturity_strikes.append(strike_price)
                maturity_portfolio_values.append(result['portfolio_values'][-1])

                # Plot portfolio evolution for this strike
                portfolio_values = result['portfolio_values']
                df_hedge = result['df_hedge']

                axes[i].plot(df_hedge.index, portfolio_values,
                           label=f'K={strike_price}', alpha=0.7, linewidth=1.5)

            except Exception as e:
                print(f"Error with strike {strike_price}: {e}")
                continue

        # Final portfolio value line
        axes[i].axhline(y=initial_portfolio_value, color='red', linestyle='--', alpha=0.7,
                       label=f'Initial (${initial_portfolio_value})')

        axes[i].set_title(f'Maturity: {maturity.date()} - All Strikes Portfolio Evolution')
        axes[i].set_ylabel('Portfolio Value ($)')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        maturity_results[filename] = {
            'maturity': maturity,
            'strikes': maturity_strikes,
            'final_values': maturity_portfolio_values
        }

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

    return maturity_results

def classify_moneyness(strike_price, underlying_price, tolerance_pct=0.02):
    """Classify option moneyness based on strike vs underlying price."""
    diff_pct = abs(strike_price - underlying_price) / underlying_price

    if diff_pct <= tolerance_pct:
        return 'ATM'  # At-The-Money
    elif strike_price < underlying_price:
        return 'ITM'  # In-The-Money (for calls)
    else:
        return 'OTM'  # Out-of-The-Money (for calls)

def analyze_market_conditions(df, start_idx):
    """Analyze market conditions during the hedging period."""
    prices = df['Close'].iloc[start_idx:].values

    if len(prices) < 2:
        return {
            'total_return': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'trend_direction': 'neutral',
            'avg_daily_return': 0.0,
            'price_range_pct': 0.0
        }

    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    total_return = (prices[-1] - prices[0]) / prices[0] * 100

    # Calculate volatility (annualized)
    daily_volatility = np.std(returns)
    volatility = daily_volatility * np.sqrt(252) * 100  # Annualized volatility

    # Calculate maximum drawdown
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)

    # Classify trend direction
    if total_return > 5:
        trend_direction = 'bullish'
    elif total_return < -5:
        trend_direction = 'bearish'
    else:
        trend_direction = 'sideways'

    # Additional metrics
    avg_daily_return = np.mean(returns) * 100
    price_range_pct = (np.max(prices) - np.min(prices)) / prices[0] * 100

    return {
        'total_return': total_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'trend_direction': trend_direction,
        'avg_daily_return': avg_daily_return,
        'price_range_pct': price_range_pct,
        'start_price': prices[0],
        'end_price': prices[-1],
        'high_price': np.max(prices),
        'low_price': np.min(prices)
    }

def run_comprehensive_strike_analysis(freq=1, r=0.02, transaction_cost_per_share=0.01,
                                    transaction_cost_percentage=0.0005, data_folder="simulation_data",
                                    initial_portfolio_value=100):
    """Run comprehensive analysis across all strikes and maturities with full statistics."""

    print("Loading simulation datasets...")
    datasets = load_simulation_data(data_folder)

    if not datasets:
        print("No datasets found!")
        return None

    # Get available strikes
    first_dataset = list(datasets.values())[0]['data']
    strike_cols = [col for col in first_dataset.columns if col.startswith('C') and col[1:].isdigit()]

    if not strike_cols:
        print("No call option columns found in the data!")
        return None

    total_simulations = len(datasets) * len(strike_cols)
    print(f"Found {len(strike_cols)} strike prices: {strike_cols}")
    print(f"Found {len(datasets)} maturities")
    print(f"Total simulations: {total_simulations}")
    print(f"Rehedging frequency: Every {freq} day{'s' if freq > 1 else ''}")
    print(f"Initial portfolio value: ${initial_portfolio_value}")

    # Run all simulations
    all_results = []
    simulation_count = 0

    for filename, data_dict in datasets.items():
        df = data_dict['data']
        maturity = data_dict['maturity']

        print(f"\n{'='*60}")
        print(f"Processing {filename} (maturity: {maturity.date()})")
        print(f"{'='*60}")

        for strike_col in strike_cols:
            strike_price = int(strike_col[1:])

            try:
                # Find start index (last 45 trading days before maturity)
                total_days = len(df)
                start_idx = max(0, total_days - 45)

                # Get underlying price at start of hedging period
                underlying_price_raw = df['Close'].iloc[start_idx]

                # Convert to float and handle potential string/NaN values
                try:
                    underlying_price = float(underlying_price_raw)
                    if pd.isna(underlying_price) or underlying_price <= 0:
                        raise ValueError(f"Invalid underlying price: {underlying_price_raw}")
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert underlying price '{underlying_price_raw}' to float: {e}")
                    raise ValueError(f"Invalid underlying price data for strike {strike_price}")

                result = run_single_hedging_simulation(
                    df, start_idx, strike_col, strike_price, r, maturity, freq,
                    transaction_cost_per_share, transaction_cost_percentage,
                    initial_portfolio_value
                )

                # Classify moneyness
                moneyness = classify_moneyness(strike_price, underlying_price)

                # Analyze market conditions during hedging period
                market_conditions = analyze_market_conditions(df, start_idx)

                simulation_result = {
                    'maturity': maturity,
                    'maturity_date': maturity.date(),
                    'strike': strike_price,
                    'underlying_price': underlying_price,
                    'moneyness': moneyness,
                    'filename': filename,
                    'mse': result['mse'],
                    'total_costs': result['total_costs'],
                    'final_pnl': result['final_pnl'],
                    'final_portfolio_value': result['portfolio_values'][-1],
                    'initial_portfolio_value': initial_portfolio_value,
                    'return_pct': (result['portfolio_values'][-1] - initial_portfolio_value) / initial_portfolio_value * 100,
                    'days_hedged': total_days - start_idx,
                    # Market condition metrics
                    'market_total_return': market_conditions['total_return'],
                    'market_volatility': market_conditions['volatility'],
                    'market_max_drawdown': market_conditions['max_drawdown'],
                    'market_trend_direction': market_conditions['trend_direction'],
                    'market_avg_daily_return': market_conditions['avg_daily_return'],
                    'market_price_range_pct': market_conditions['price_range_pct']
                }

                all_results.append(simulation_result)
                simulation_count += 1

                print(f"Strike {strike_price:3d} ({moneyness}): MSE={result['mse']:.6f}, "
                      f"Costs=${result['total_costs']:6.2f}, "
                      f"Final=${result['portfolio_values'][-1]:7.2f}, "
                      f"Return={simulation_result['return_pct']:6.2f}%")

            except Exception as e:
                print(f"Error with strike {strike_price}: {e}")
                continue

    if not all_results:
        print("No successful simulations!")
        return None

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)

    # Calculate comprehensive statistics
    stats = {
        'total_simulations': len(all_results),
        'completed_simulations': len(all_results),
        'maturities_analyzed': len(datasets),
        'strikes_analyzed': len(strike_cols),
        'strike_range': f"{min(results_df['strike'])}-{max(results_df['strike'])}",
        'rehedging_frequency': freq,
        'initial_portfolio_value': initial_portfolio_value,

        # MSE statistics
        'mse_mean': results_df['mse'].mean(),
        'mse_std': results_df['mse'].std(),
        'mse_min': results_df['mse'].min(),
        'mse_max': results_df['mse'].max(),

        # Costs statistics
        'costs_mean': results_df['total_costs'].mean(),
        'costs_std': results_df['total_costs'].std(),
        'costs_min': results_df['total_costs'].min(),
        'costs_max': results_df['total_costs'].max(),

        # PnL statistics
        'pnl_mean': results_df['final_pnl'].mean(),
        'pnl_std': results_df['final_pnl'].std(),
        'pnl_min': results_df['final_pnl'].min(),
        'pnl_max': results_df['final_pnl'].max(),

        # Portfolio value statistics
        'portfolio_final_mean': results_df['final_portfolio_value'].mean(),
        'portfolio_final_std': results_df['final_portfolio_value'].std(),
        'portfolio_final_min': results_df['final_portfolio_value'].min(),
        'portfolio_final_max': results_df['final_portfolio_value'].max(),

        # Return statistics
        'return_pct_mean': results_df['return_pct'].mean(),
        'return_pct_std': results_df['return_pct'].std(),
        'return_pct_min': results_df['return_pct'].min(),
        'return_pct_max': results_df['return_pct'].max(),

        # Best and worst performers
        'best_return_idx': results_df['return_pct'].idxmax(),
        'worst_return_idx': results_df['return_pct'].idxmin(),

        'results_df': results_df
    }

    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE STRIKE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Simulations: {stats['total_simulations']}")
    print(f"Maturities: {stats['maturities_analyzed']}")
    print(f"Strikes: {stats['strikes_analyzed']} ({stats['strike_range']})")
    print(f"Rehedging: Every {freq} day{'s' if freq > 1 else ''}")
    print(f"Initial Portfolio: ${initial_portfolio_value}")
    print()

    print("MSE Statistics:")
    print(f"  Mean ± Std: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
    print(f"  Range: {stats['mse_min']:.6f} - {stats['mse_max']:.6f}")
    print()

    print("Transaction Costs:")
    print(f"  Mean ± Std: ${stats['costs_mean']:.2f} ± ${stats['costs_std']:.2f}")
    print(f"  Range: ${stats['costs_min']:.2f} - ${stats['costs_max']:.2f}")
    print()

    print("Portfolio Performance:")
    print(f"  Final Value: ${stats['portfolio_final_mean']:.2f} ± ${stats['portfolio_final_std']:.2f}")
    print(f"  Return: {stats['return_pct_mean']:.2f}% ± {stats['return_pct_std']:.2f}%")
    print(f"  Best: {stats['return_pct_max']:.2f}%")
    print(f"  Worst: {stats['return_pct_min']:.2f}%")
    print()

    # Best and worst performers
    best_row = results_df.loc[stats['best_return_idx']]
    worst_row = results_df.loc[stats['worst_return_idx']]

    print("Best Performing Combination:")
    print(f"  Maturity: {best_row['maturity_date']}, Strike: {best_row['strike']}")
    print(f"  Return: {best_row['return_pct']:.2f}%, Final: ${best_row['final_portfolio_value']:.2f}")
    print()

    print("Worst Performing Combination:")
    print(f"  Maturity: {worst_row['maturity_date']}, Strike: {worst_row['strike']}")
    print(f"  Return: {worst_row['return_pct']:.2f}%, Final: ${worst_row['final_portfolio_value']:.2f}")
    print()

    # Add moneyness statistics
    moneyness_stats = results_df.groupby('moneyness').agg({
        'mse': ['mean', 'std', 'count'],
        'total_costs': ['mean', 'std'],
        'return_pct': ['mean', 'std', 'min', 'max']
    }).round(4)

    print("Performance by Moneyness:")
    print(moneyness_stats)
    print()

    # Create summary plots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Comprehensive Strike Analysis Summary ({total_simulations} Simulations)', fontsize=16)

    # MSE distribution
    axes[0,0].hist(results_df['mse'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('MSE Distribution')
    axes[0,0].set_xlabel('Mean Squared Error')
    axes[0,0].set_ylabel('Frequency')

    # Transaction costs distribution
    axes[0,1].hist(results_df['total_costs'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Transaction Costs Distribution')
    axes[0,1].set_xlabel('Total Costs ($)')
    axes[0,1].set_ylabel('Frequency')

    # Return % distribution
    axes[0,2].hist(results_df['return_pct'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    axes[0,2].set_title('Return Distribution')
    axes[0,2].set_xlabel('Return (%)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].legend()

    # Moneyness comparison - Return by category
    moneyness_return = results_df.groupby('moneyness')['return_pct'].mean()
    moneyness_order = ['OTM', 'ATM', 'ITM']
    moneyness_return = moneyness_return.reindex(moneyness_order)
    colors = ['green' if x >= 0 else 'red' for x in moneyness_return.values]
    axes[0,3].bar(moneyness_return.index, moneyness_return.values, color=colors, alpha=0.7)
    axes[0,3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,3].set_title('Average Return by Moneyness')
    axes[0,3].set_xlabel('Moneyness')
    axes[0,3].set_ylabel('Average Return (%)')

    # Strike vs MSE
    strike_mse = results_df.groupby('strike')['mse'].mean()
    axes[1,0].plot(strike_mse.index, strike_mse.values, 'o-', alpha=0.7)
    axes[1,0].set_title('Average MSE by Strike Price')
    axes[1,0].set_xlabel('Strike Price')
    axes[1,0].set_ylabel('Average MSE')
    axes[1,0].grid(True, alpha=0.3)

    # Strike vs Return
    strike_return = results_df.groupby('strike')['return_pct'].mean()
    colors = ['green' if x >= 0 else 'red' for x in strike_return.values]
    axes[1,1].bar(strike_return.index, strike_return.values, color=colors, alpha=0.7)
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Average Return by Strike Price')
    axes[1,1].set_xlabel('Strike Price')
    axes[1,1].set_ylabel('Average Return (%)')

    # Maturity vs Return
    maturity_return = results_df.groupby('maturity_date')['return_pct'].mean()
    maturity_return.index = pd.to_datetime(maturity_return.index)
    maturity_return = maturity_return.sort_index()
    colors = ['green' if x >= 0 else 'red' for x in maturity_return.values]
    axes[1,2].bar(range(len(maturity_return)), maturity_return.values, color=colors, alpha=0.7)
    axes[1,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,2].set_title('Average Return by Maturity')
    axes[1,2].set_xlabel('Maturity Date')
    axes[1,2].set_ylabel('Average Return (%)')
    axes[1,2].set_xticks(range(len(maturity_return)))
    axes[1,2].set_xticklabels([d.strftime('%Y-%m-%d') for d in maturity_return.index], rotation=45)

    # Moneyness distribution
    moneyness_counts = results_df['moneyness'].value_counts()
    moneyness_counts = moneyness_counts.reindex(moneyness_order)
    axes[1,3].pie(moneyness_counts.values, labels=moneyness_counts.index, autopct='%1.1f%%',
                  colors=['lightcoral', 'lightblue', 'lightgreen'])
    axes[1,3].set_title('Distribution of Moneyness Categories')

    plt.tight_layout()
    plt.show()

    # Additional moneyness analysis plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Moneyness Analysis Details', fontsize=16)

    # Box plot of returns by moneyness
    results_df.boxplot(column='return_pct', by='moneyness', ax=axes[0])
    axes[0].set_title('Return Distribution by Moneyness')
    axes[0].set_xlabel('Moneyness')
    axes[0].set_ylabel('Return (%)')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)

    # MSE by moneyness
    mse_by_moneyness = results_df.groupby('moneyness')['mse'].mean()
    mse_by_moneyness = mse_by_moneyness.reindex(moneyness_order)
    axes[1].bar(mse_by_moneyness.index, mse_by_moneyness.values, alpha=0.7)
    axes[1].set_title('Average MSE by Moneyness')
    axes[1].set_xlabel('Moneyness')
    axes[1].set_ylabel('Average MSE')

    # Costs by moneyness
    costs_by_moneyness = results_df.groupby('moneyness')['total_costs'].mean()
    costs_by_moneyness = costs_by_moneyness.reindex(moneyness_order)
    axes[2].bar(costs_by_moneyness.index, costs_by_moneyness.values, alpha=0.7)
    axes[2].set_title('Average Transaction Costs by Moneyness')
    axes[2].set_xlabel('Moneyness')
    axes[2].set_ylabel('Average Costs ($)')

    plt.tight_layout()
    plt.show()

    # Market Direction Analysis Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Market Direction Analysis', fontsize=16)

    # Average return by market direction
    market_return = results_df.groupby('market_trend_direction')['return_pct'].mean()
    market_order = ['bullish', 'sideways', 'bearish']
    market_return = market_return.reindex(market_order)
    colors = ['green' if x >= 0 else 'red' for x in market_return.values]
    axes[0,0].bar(market_return.index, market_return.values, color=colors, alpha=0.7)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Average Return by Market Direction')
    axes[0,0].set_xlabel('Market Direction')
    axes[0,0].set_ylabel('Average Return (%)')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Average MSE by market direction
    market_mse = results_df.groupby('market_trend_direction')['mse'].mean()
    market_mse = market_mse.reindex(market_order)
    axes[0,1].bar(market_mse.index, market_mse.values, alpha=0.7)
    axes[0,1].set_title('Average MSE by Market Direction')
    axes[0,1].set_xlabel('Market Direction')
    axes[0,1].set_ylabel('Average MSE')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Average costs by market direction
    market_costs = results_df.groupby('market_trend_direction')['total_costs'].mean()
    market_costs = market_costs.reindex(market_order)
    axes[0,2].bar(market_costs.index, market_costs.values, alpha=0.7)
    axes[0,2].set_title('Average Transaction Costs by Market Direction')
    axes[0,2].set_xlabel('Market Direction')
    axes[0,2].set_ylabel('Average Costs ($)')
    axes[0,2].tick_params(axis='x', rotation=45)

    # Return distribution by market direction (box plot)
    results_df.boxplot(column='return_pct', by='market_trend_direction', ax=axes[1,0])
    axes[1,0].set_title('Return Distribution by Market Direction')
    axes[1,0].set_xlabel('Market Direction')
    axes[1,0].set_ylabel('Return (%)')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)

    # Market direction distribution (pie chart)
    market_counts = results_df['market_trend_direction'].value_counts()
    market_counts = market_counts.reindex(market_order)
    colors_pie = ['lightgreen' if d == 'bullish' else 'lightcoral' if d == 'bearish' else 'lightblue' for d in market_counts.index]
    axes[1,1].pie(market_counts.values, labels=market_counts.index, autopct='%1.1f%%', colors=colors_pie)
    axes[1,1].set_title('Market Direction Distribution')

    # Market volatility vs performance scatter
    market_volatility = results_df.groupby('market_trend_direction')['market_volatility'].mean()
    market_volatility = market_volatility.reindex(market_order)
    axes[1,2].bar(market_volatility.index, market_volatility.values, alpha=0.7)
    axes[1,2].set_title('Average Market Volatility by Direction')
    axes[1,2].set_xlabel('Market Direction')
    axes[1,2].set_ylabel('Average Volatility (%)')
    axes[1,2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Print market direction statistics
    print(f"\n{'='*60}")
    print("MARKET DIRECTION ANALYSIS")
    print(f"{'='*60}")

    market_stats = results_df.groupby('market_trend_direction').agg({
        'return_pct': ['mean', 'std', 'count', 'min', 'max'],
        'mse': ['mean', 'std'],
        'total_costs': ['mean', 'std'],
        'market_volatility': ['mean', 'std'],
        'market_total_return': ['mean', 'std']
    }).round(4)

    print("Performance by Market Direction:")
    print(market_stats)
    print()

    # Best and worst market conditions
    best_market = market_return.idxmax()
    worst_market = market_return.idxmin()

    print(f"Best performing market direction: {best_market} (avg return: {market_return[best_market]:.2f}%)")
    print(f"Worst performing market direction: {worst_market} (avg return: {market_return[worst_market]:.2f}%)")

    return stats
