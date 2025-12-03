import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import matplotlib.dates as mdates
import numpy as np

def plot_spy_and_options(df, option_cols):    
    
    spy_df = df[["Open", "High", "Low", "Close"]]
    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="i", wick="i", volume="in")
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle="--", facecolor="#f8f9fa")
    
    mpf.plot(
        spy_df,
        type="candle",
        style=s,
        title="SPY Candlesticks",
        ylabel="SPY Price ($)",
        figsize=(14, 7),
        tight_layout=True,
        figratio=(16, 9),
    )
    
    for option_col in option_cols:
        if option_col in df.columns:
            opt_df = df[[option_col]]
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(opt_df.index, opt_df[option_col], color="#1f77b4", lw=2, label=f"{option_col} Price")
            ax.set_title(f"SPY {option_col} Option Price", fontsize=14, weight="bold")
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Option Price ($)", fontsize=12)
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Option column '{option_col}' not found in the data.")

    print(f"Data starts: {df.index.min()}")
    print(f"Data ends: {df.index.max()}")
    print(f"Number of days: {(df.index.max() - df.index.min()).days}")
    print(f"Number of data points: {len(df)}")

def plot_hedging_errors(df_hedge, A_errors):
    plt.figure(figsize=(8, 4))
    plt.plot(df_hedge.index[:-1], A_errors, 'b-', linewidth=2)
    plt.title('Daily Hedging Errors')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def plot_positions(df_hedge, OP, RE, deltas):
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(df_hedge.index, OP, 'b-', linewidth=2, label='Long Call Position')
    ax1.set_ylabel('Option Position Value ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df_hedge.index, -deltas * RE, 'r--', linewidth=2, label='Short Underlying Position')
    ax2.set_ylabel('Underlying Position Value ($)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Portfolio Positions (Dual Scale)')
    plt.xlabel('Date')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_delta_positions(df_hedge, deltas):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(df_hedge.index, deltas, 'b-', linewidth=2, label='Call Delta')
    
    ax.plot(df_hedge.index, -deltas, 'r--', linewidth=2, label='Hedge Delta (Short)')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
    
    ax.set_ylabel('Delta Contribution')
    ax.set_xlabel('Date')
    ax.set_title('Delta Positions - Symmetric Neutrality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

def plot_portfolio_and_pnl(dates, portfolio_values, pnl, title="Portfolio Value and PnL"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(dates, portfolio_values, 'g-', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=portfolio_values[0], color='black', linestyle='--', alpha=0.7)
    ax1.set_title('Portfolio Value')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dates, pnl, 'b-', linewidth=2, label='Cumulative PnL')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.fill_between(dates, pnl, 0, where=(pnl >= 0), color='green', alpha=0.3)
    ax2.fill_between(dates, pnl, 0, where=(pnl < 0), color='red', alpha=0.3)
    ax2.set_title('Cumulative PnL')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('PnL ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_hedging_simulation_stats(stats_df, title="Hedging Simulation Statistics"):
    """
    Plot key metrics from multiple hedging simulations with mean lines
    """
    if len(stats_df) == 0:
        print("No data to plot")
        return
    
    # Use unique sequential numbers for x-axis
    x_values = range(len(stats_df))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # MSE
    mse_data = stats_df['mean_squared_error']
    bar_width = max(0.5, 8.0 / len(x_values))
    bars1 = ax1.bar(x_values, mse_data, color='skyblue', alpha=0.7, width=bar_width)
    mse_mean = mse_data.mean()
    ax1.axhline(y=mse_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mse_mean:.3f}')
    ax1.set_title('Mean Squared Hedging Error')
    ax1.set_xlabel('Simulation Number')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(x_values) > 20:
        tick_spacing = max(1, len(x_values) // 10)
        ax1.set_xticks(x_values[::tick_spacing])
        ax1.set_xticklabels(x_values[::tick_spacing])
    
    # Transaction Costs
    cost_data = stats_df['total_costs']
    bars2 = ax2.bar(x_values, cost_data, color='lightgreen', alpha=0.7, width=bar_width)
    cost_mean = cost_data.mean()
    ax2.axhline(y=cost_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {cost_mean:.3f}')
    ax2.set_title('Total Transaction Costs')
    ax2.set_xlabel('Simulation Number')
    ax2.set_ylabel('Costs ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if len(x_values) > 20:
        ax2.set_xticks(x_values[::tick_spacing])
        ax2.set_xticklabels(x_values[::tick_spacing])
    
    # PnL Percentage
    pnl_data = stats_df['pnl_percentage']
    colors = ['red' if x < 0 else 'green' for x in pnl_data]
    bars3 = ax3.bar(x_values, pnl_data, color=colors, alpha=0.7, width=bar_width)
    pnl_mean = pnl_data.mean()
    ax3.axhline(y=pnl_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {pnl_mean:.1f}%')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Break-even')
    ax3.set_title('PnL Percentage')
    ax3.set_xlabel('Simulation Number')
    ax3.set_ylabel('PnL (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if len(x_values) > 20:
        ax3.set_xticks(x_values[::tick_spacing])
        ax3.set_xticklabels(x_values[::tick_spacing])
    
    # Portfolio Volatility
    vol_data = stats_df['portfolio_volatility']
    bars4 = ax4.bar(x_values, vol_data, color='orange', alpha=0.7, width=bar_width)
    vol_mean = vol_data.mean()
    ax4.axhline(y=vol_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {vol_mean:.3f}')
    ax4.set_title('Portfolio Volatility')
    ax4.set_xlabel('Simulation Number')
    ax4.set_ylabel('Volatility')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    if len(x_values) > 20:
        ax4.set_xticks(x_values[::tick_spacing])
        ax4.set_xticklabels(x_values[::tick_spacing])
    
    plt.tight_layout()
    plt.show()

def plot_hedging_summary_distributions(stats_df, title="Hedging Metrics Distributions"):
    """
    Plot distributions of hedging metrics using box plots
    """
    if len(stats_df) == 0:
        print("No data to plot")
        return
    
    metrics = ['mean_squared_error', 'total_costs', 'pnl_percentage', 'portfolio_volatility']
    labels = ['MSE', 'Costs ($)', 'PnL (%)', 'Volatility']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        data = stats_df[metric]
        axes[i].boxplot(data, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[i].set_title(f'{label} Distribution')
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = data.mean()
        if '%' in label:
            axes[i].axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}%')
        else:
            axes[i].axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def plot_performance_by_interval_length(combined_stats, title="Hedging Performance by Interval Length"):
    """
    Analyze how different interval lengths affect hedging performance
    """
    if 'interval_length' not in combined_stats.columns:
        print("No interval_length column found")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # MSE by interval length
    mse_by_length = [combined_stats[combined_stats['interval_length'] == length]['mean_squared_error'].values 
                    for length in sorted(combined_stats['interval_length'].unique())]
    labels = [f"{length}d" for length in sorted(combined_stats['interval_length'].unique())]
    
    ax1.boxplot(mse_by_length, labels=labels, patch_artist=True, 
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax1.set_title('MSE Distribution by Interval Length')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, alpha=0.3)
    
    # PnL % by interval length
    pnl_by_length = [combined_stats[combined_stats['interval_length'] == length]['pnl_percentage'].values 
                    for length in sorted(combined_stats['interval_length'].unique())]
    
    ax2.boxplot(pnl_by_length, labels=labels, patch_artist=True, 
               boxprops=dict(facecolor='lightgreen', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_title('PnL % Distribution by Interval Length')
    ax2.set_ylabel('PnL (%)')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Average metrics by length
    length_summary = combined_stats.groupby('interval_length')[['mean_squared_error', 'total_costs', 'pnl_percentage', 'portfolio_volatility']].mean()
    
    lengths = length_summary.index
    ax3.bar(lengths - 1, length_summary['mean_squared_error'], width=2, alpha=0.7, label='MSE', color='skyblue')
    ax3.set_xlabel('Interval Length (days)')
    ax3.set_ylabel('Mean Squared Error', color='skyblue')
    ax3.tick_params(axis='y', labelcolor='skyblue')
    ax3.grid(True, alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.bar(lengths + 1, length_summary['pnl_percentage'], width=2, alpha=0.7, label='PnL %', color='green')
    ax3_twin.set_ylabel('PnL (%)', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_title('Average MSE and PnL % by Interval Length')
    ax3.set_xticks(lengths)
    ax3.set_xticklabels([f"{int(x)}d" for x in lengths])
    
    # Count by length
    length_counts = combined_stats['interval_length'].value_counts().sort_index()
    ax4.bar(length_counts.index, length_counts.values, color='orange', alpha=0.7)
    ax4.set_title('Number of Simulations by Interval Length')
    ax4.set_xlabel('Interval Length (days)')
    ax4.set_ylabel('Number of Simulations')
    ax4.grid(True, alpha=0.3)
    for i, v in enumerate(length_counts.values):
        ax4.text(length_counts.index[i], v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_underlying_for_simulation(combined_stats, simulation_idx, data_dict, title=None):
    """
    Plot the SPY underlying price for a specific simulation
    
    Parameters:
    - combined_stats: DataFrame with simulation results
    - simulation_idx: Index of the simulation (0 to N-1)
    - data_dict: Dictionary with dataset names as keys and DataFrames as values
    - title: Optional custom title
    """
    if simulation_idx >= len(combined_stats):
        print(f"Simulation index {simulation_idx} out of range (0-{len(combined_stats)-1})")
        return
    
    # Get simulation details
    sim = combined_stats.iloc[simulation_idx]
    dataset = sim['dataset']
    start_date = sim['start_date']
    end_date = sim['end_date']
    option = sim['option']
    pnl_pct = sim['pnl_percentage']
    
    # Get the data
    if dataset not in data_dict:
        print(f"Dataset {dataset} not found in data_dict")
        return
    
    df = data_dict[dataset]
    
    # Slice the data for this simulation period
    mask = (df.index >= start_date) & (df.index <= end_date)
    period_data = df.loc[mask]
    
    if len(period_data) == 0:
        print("No data found for this simulation period")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(period_data.index, period_data['Close'], 'b-', linewidth=2, label='SPY Close')
    
    # Add start/end markers
    plt.axvline(x=start_date, color='green', linestyle='--', alpha=0.7, label='Start')
    plt.axvline(x=end_date, color='red', linestyle='--', alpha=0.7, label='End')
    
    # Title
    if title is None:
        title = f'SPY Underlying Price - Simulation {simulation_idx}\n{option} | {start_date.date()} to {end_date.date()}\nPnL: {pnl_pct:.1f}%'
    plt.title(title)
    
    plt.xlabel('Date')
    plt.ylabel('SPY Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format dates
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_delta_vega_hedging(df_hedge, net_deltas, alphas, OP_primary, OP_vega, RE, 
                           A_errors, shares_held, vega_option_held, portfolio_values, 
                           cumulative_costs, pnl, title="Delta-Vega Hedging Results"):
    """
    Plot comprehensive delta-vega hedging results.
    
    Parameters from delta_vega_hedging return tuple:
    - df_hedge: DataFrame with hedging dates
    - net_deltas: Net delta exposure after vega hedging
    - alphas: Vega hedge ratios
    - OP_primary, OP_vega: Option prices
    - RE: Underlying prices
    - A_errors: Hedging errors
    - shares_held: Underlying shares held
    - vega_option_held: Vega hedge options held
    - portfolio_values: Portfolio values over time
    - cumulative_costs: Transaction costs
    - pnl: Profit and loss
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Portfolio Value and PnL
    ax1 = axes[0, 0]
    ax1.plot(df_hedge.index, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
    ax1.set_ylabel('Portfolio Value ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Portfolio Value')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_hedge.index, pnl, 'r--', alpha=0.7, label='PnL')
    ax1_twin.set_ylabel('PnL ($)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Hedging Errors
    ax2 = axes[0, 1]
    ax2.plot(df_hedge.index[:-1], A_errors, 'g-', alpha=0.8, label='Hedging Errors')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Hedging Error ($)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Hedging Errors (MSE: {np.mean(A_errors**2):.4f})')
    
    # Positions Held
    ax3 = axes[1, 0]
    ax3.plot(df_hedge.index, shares_held, 'b-', linewidth=2, label='Underlying Shares')
    ax3.set_ylabel('Shares Held', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Underlying Shares Position')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df_hedge.index, vega_option_held, 'r-', alpha=0.7, label='Vega Options')
    ax3_twin.set_ylabel('Vega Options Held', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    # Vega Hedge Ratios
    ax4 = axes[1, 1]
    ax4.plot(df_hedge.index, alphas, 'purple', linewidth=2, label='Vega Hedge Ratio')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Vega Hedge Ratio')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Vega Hedge Ratios Over Time')
    
    # Net Delta Exposure
    ax5 = axes[2, 0]
    ax5.plot(df_hedge.index, net_deltas, 'orange', linewidth=2, label='Net Delta')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_ylabel('Net Delta')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Net Delta Exposure After Vega Hedging')
    
    # Transaction Costs
    ax6 = axes[2, 1]
    ax6.plot(df_hedge.index, cumulative_costs, 'brown', linewidth=2, label='Cumulative Costs')
    ax6.set_ylabel('Transaction Costs ($)')
    ax6.grid(True, alpha=0.3)
    ax6.set_title(f'Total Costs: ${cumulative_costs[-1]:.2f}')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=== DELTA-VEGA HEDGING SUMMARY ===")
    print(f"MSE: {np.mean(A_errors**2):.4f}")
    print(f"RMSE: {np.sqrt(np.mean(A_errors**2)):.4f}")
    print(f"Mean Error: {np.mean(A_errors):.4f}")
    print(f"Final PnL: ${pnl[-1]:.2f}")
    print(f"Total Costs: ${cumulative_costs[-1]:.2f}")
    print(f"Average Vega Ratio: {np.mean(alphas):.4f}")
    print(f"Final Net Delta: {net_deltas[-1]:.4f}")

def plot_delta_vega_gamma_hedging(df_hedge, OP_primary, OP_vega, OP_gamma, RE, A_errors,
                                   shares_held, vega_option_held, gamma_option_held,
                                   cash_position, portfolio_values, cumulative_costs, pnl,
                                   title="Delta-Vega-Gamma Hedging Results"):
    """
    Plot comprehensive delta-vega-gamma hedging results.
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Portfolio Value and PnL
    ax1 = axes[0, 0]
    ax1.plot(df_hedge.index, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
    ax1.set_ylabel('Portfolio Value ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Portfolio Value')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_hedge.index, pnl, 'r--', alpha=0.7, label='PnL')
    ax1_twin.set_ylabel('PnL ($)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Hedging Errors
    ax2 = axes[0, 1]
    ax2.plot(df_hedge.index[:-1], A_errors, 'g-', alpha=0.8, label='Hedging Errors')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Hedging Error ($)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Hedging Errors (MSE: {np.mean(A_errors**2):.4f})')
    
    # Positions Held
    ax3 = axes[1, 0]
    ax3.plot(df_hedge.index, shares_held, 'b-', linewidth=2, label='Underlying Shares')
    ax3.set_ylabel('Shares Held', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Underlying Shares Position')
    
    # Vega and Gamma Positions
    ax4 = axes[1, 1]
    ax4.plot(df_hedge.index, vega_option_held, 'r-', alpha=0.7, label='Vega Options')
    ax4.set_ylabel('Vega Options Held', color='r')
    ax4.tick_params(axis='y', labelcolor='r')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df_hedge.index, gamma_option_held, 'purple', alpha=0.7, label='Gamma Options')
    ax4_twin.set_ylabel('Gamma Options Held', color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    ax4.set_title('Vega and Gamma Option Positions')
    
    # Transaction Costs
    ax5 = axes[2, 0]
    ax5.plot(df_hedge.index, cumulative_costs, 'brown', linewidth=2, label='Cumulative Costs')
    ax5.set_ylabel('Transaction Costs ($)')
    ax5.grid(True, alpha=0.3)
    ax5.set_title(f'Total Costs: ${cumulative_costs[-1]:.2f}')

    # Remove empty subplot
    fig.delaxes(axes[2,1])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=== DELTA-VEGA-GAMMA HEDGING SUMMARY ===")
    print(f"MSE: {np.mean(A_errors**2):.4f}")
    print(f"RMSE: {np.sqrt(np.mean(A_errors**2)):.4f}")
    print(f"Mean Error: {np.mean(A_errors):.4f}")
    print(f"Final PnL: ${pnl[-1]:.2f}")
    print(f"Total Costs: ${cumulative_costs[-1]:.2f}")

