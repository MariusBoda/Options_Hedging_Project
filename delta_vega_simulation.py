import pandas as pd
import numpy as np
import os
from itertools import combinations
from options_lib.hedging import delta_vega_hedging
from options_lib.plots import plot_hedging_simulation_stats
from options_lib.data import data_load
from datetime import datetime

def run_delta_vega_hedging_intervals(df1, df2, maturity1, maturity2,
                                     option_primary, option_vega,
                                     K_primary, K_vega,
                                     interval_length=45, step_size=5, num_intervals=10,
                                     r=0.05, freq=1,
                                     transaction_cost_per_share=0.01,
                                     transaction_cost_percentage=0.0005):
    """
    Runs multiple delta-vega hedging simulations over rolling time intervals on
    pre-aligned dataframes with overlapping dates.
    """
    results = []
    
    # DataFrames are expected to be pre-aligned with a common index.
    if len(df1) < interval_length:
        # Not enough data to run even one simulation
        return pd.DataFrame()

    for i in range(num_intervals):
        start_idx = i * step_size
        end_idx = start_idx + interval_length
        
        if end_idx > len(df1):
            break
            
        interval_df1 = df1.iloc[start_idx:end_idx]
        interval_df2 = df2.iloc[start_idx:end_idx]
        
        # Check for missing data in the interval for the specific options
        if interval_df1[[option_primary, 'Close']].isna().any().any() or \
           interval_df2[[option_vega]].isna().any().any():
            print(f"Skipping interval {i} due to missing data for pair {option_primary}-{option_vega}.")
            continue
            
        start_date = interval_df1.index[0]
        end_date = interval_df1.index[-1]
        
        calendar_days = (end_date - start_date).days
        
        try:
            result = delta_vega_hedging(
                df1, df2, start_date, end_date,
                option_primary, option_vega,
                K_primary, K_vega, r,
                maturity1, maturity2, freq,
                transaction_cost_per_share,
                transaction_cost_percentage
            )
            
            op_primary_initial = result[3][0] if len(result[3]) > 0 else 0
            
            stats = {
                'interval': len(results),
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(interval_df1),
                'calendar_days': calendar_days,
                'mean_squared_error': np.mean(result[8]**2),
                'total_costs': result[13][-1],
                'final_pnl': result[14][-1],
                'portfolio_volatility': np.std(result[12]),
                'max_portfolio_value': np.max(result[12]),
                'min_portfolio_value': np.min(result[12]),
                'pnl_percentage': (result[14][-1] / op_primary_initial * 100) if op_primary_initial != 0 else 0
            }
            results.append(stats)
        except Exception as e:
            print(f"Error in simulation for interval {i} ({start_date.date()} to {end_date.date()}): {e}")

    return pd.DataFrame(results)


def run_full_delta_vega_analysis(data_folder, strike_tolerance=2, min_maturity_diff_days=30, **kwargs):
    """
    Orchestrates the entire delta-vega hedging analysis across multiple data files.

    1. Finds all data files in the given folder.
    2. Creates pairs of files to compare options with different maturities.
    3. For each pair of files, finds options with similar strikes.
    4. For each valid option pair, runs multiple hedging simulations on overlapping data intervals.
    5. Aggregates and returns all simulation statistics.
    """
    try:
        files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.feather')]
    except FileNotFoundError:
        print(f"Data folder '{data_folder}' not found.")
        return pd.DataFrame()

    if len(files) < 2:
        print("Need at least two data files for this analysis.")
        return pd.DataFrame()

    all_stats = []
    file_combinations = list(combinations(files, 2))
    print(f"Found {len(files)} files, creating {len(file_combinations)} pairs for analysis.")

    for file1_path, file2_path in file_combinations:
        df1 = data_load(file1_path)
        df2 = data_load(file2_path)

        maturity1 = df1.index[-1]
        maturity2 = df2.index[-1]

        if abs((maturity1 - maturity2).days) < min_maturity_diff_days:
            continue

        # Find overlapping data period
        common_index = df1.index.intersection(df2.index)
        if common_index.empty:
            continue
        
        df1_sync = df1.loc[common_index]
        df2_sync = df2.loc[common_index]

        call_cols1 = [col for col in df1.columns if col.startswith('C')]
        call_cols2 = [col for col in df2.columns if col.startswith('C')]

        for opt1_col in call_cols1:
            for opt2_col in call_cols2:
                try:
                    K1 = int(opt1_col[1:])
                    K2 = int(opt2_col[1:])
                except (ValueError, IndexError):
                    continue

                if abs(K1 - K2) <= strike_tolerance:
                    
                    # The primary option for hedging is typically the one with shorter maturity.
                    if maturity1 < maturity2:
                        df_primary, df_vega = df1_sync, df2_sync
                        mat_primary, mat_vega = maturity1, maturity2
                        opt_primary, opt_vega = opt1_col, opt2_col
                        K_primary, K_vega = K1, K2
                        f_primary, f_vega = file1_path, file2_path
                    else:
                        df_primary, df_vega = df2_sync, df1_sync
                        mat_primary, mat_vega = maturity2, maturity1
                        opt_primary, opt_vega = opt2_col, opt1_col
                        K_primary, K_vega = K2, K1
                        f_primary, f_vega = file2_path, file1_path

                    print(f"Found pair: {opt_primary} (K={K_primary}, T={mat_primary.date()}) and {opt_vega} (K={K_vega}, T={mat_vega.date()})")

                    # Pass through kwargs like interval_length, step_size, num_intervals
                    stats_df = run_delta_vega_hedging_intervals(
                        df1=df_primary,
                        df2=df_vega,
                        maturity1=mat_primary,
                        maturity2=mat_vega,
                        option_primary=opt_primary,
                        option_vega=opt_vega,
                        K_primary=K_primary,
                        K_vega=K_vega,
                        **kwargs
                    )

                    if not stats_df.empty:
                        stats_df['dataset_pair'] = f"{os.path.basename(f_primary)}-{os.path.basename(f_vega)}"
                        stats_df['option_pair'] = f"{opt_primary}-{opt_vega}"
                        stats_df['strike_primary'] = K_primary
                        stats_df['strike_vega'] = K_vega
                        stats_df['maturity_diff_days'] = abs((mat_primary - mat_vega).days)
                        all_stats.append(stats_df)

    if not all_stats:
        print("\nNo hedging simulations were successfully completed across any file pairs.")
        return pd.DataFrame()

    combined_stats = pd.concat(all_stats, ignore_index=True)
    return combined_stats

# --- Example Usage ---
# This block demonstrates how to use the new analysis function.
# You would typically run this from your Jupyter notebook.
if __name__ == '__main__':
    print("Running Delta-Vega Hedging Analysis Example...")
    
    # Define the folder where your data is located
    data_folder = "simulation_data"
    
    # Run the full analysis
    # You can customize parameters like interval_length, num_intervals, etc.
    # These are passed down to the `run_delta_vega_hedging_intervals` function.
    all_results = run_full_delta_vega_analysis(
        data_folder,
        strike_tolerance=2,
        min_maturity_diff_days=30,
        interval_length=45,
        step_size=5,
        num_intervals=10
    )
    
    if not all_results.empty:
        print(f"\nAnalysis complete. Generated {len(all_results)} simulation results.")
        print("\n--- Overall Delta-Vega Hedging Performance ---")
        # Display descriptive statistics
        # In a notebook, you might use display(all_results.describe())
        print(all_results.describe())
        
        # Plot the aggregated results
        plot_hedging_simulation_stats(
            all_results,
            title="Combined Delta-Vega Hedging Statistics (Cross-Maturity)"
        )
    else:
        print("\nNo results were generated from the delta-vega analysis.")