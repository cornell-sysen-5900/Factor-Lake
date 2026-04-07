import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import random
from pathlib import Path

random.seed(42)

# --- SETTINGS & CONFIG ---
TARGET_VOL = 0.1
BURN_IN = 60
MAX_LEVERAGE = 5
START_YEAR = 2010
END_YEAR = 2024
# Options: 'D' (Daily), 'W' (Weekly), 'M' (Monthly), 'Q' (Quarterly)
REBALANCE_FREQ = 'D' 
CHART_FILE = "vol_target_report.png"

def load_data(file_name="r2000_price_crsp.csv"):
    path = Path(__file__).resolve().parent / file_name
    df = pd.read_csv(path, usecols=['date', 'PERMNO', 'RET'], low_memory=False)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    return df.dropna().sort_values('date')

def solve_weights(cov_matrix, target_vol, max_leverage):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    portfolio_variance = cp.quad_form(w, cov_matrix) * 252
    
    # CONSTRAINT: Force the variance to be EXACTLY the target squared
    prob = cp.Problem(cp.Minimize(cp.sum(w)), [
        portfolio_variance == target_vol**2, 
        cp.sum(w) <= max_leverage,
        w >= 0
    ])
    
    try:
        # Using ECOS or OSQP; ensure the problem is feasible
        prob.solve(solver=cp.ECOS)
        if w.value is None:
            # If 15% is impossible with 6x leverage, fallback to max leverage equal-weight
            return np.ones(n) / n * max_leverage 
        return w.value
    except:
        return np.ones(n) / n * (target_vol / 0.20)

def build_yearly_metrics(results):
    yearly = results.groupby(results.index.year).agg(['mean', 'std'])
    yearly_ann_return = yearly['mean'] * 252
    yearly_vol = yearly['std'] * np.sqrt(252)
    metrics = pd.DataFrame({
        'ann_return': yearly_ann_return,
        'yearly_vol': yearly_vol,
    })
    metrics.index.name = 'year'
    return metrics


def plot_final_report(metrics, target_vol, file_name=CHART_FILE):
    years = metrics.index.astype(int)

    plt.figure(figsize=(11, 6))
    plt.plot(years, metrics['ann_return'] * 100, marker='o', linewidth=2, label='Ann. Return (%)')
    plt.plot(years, metrics['yearly_vol'] * 100, marker='s', linewidth=2, label='Yearly Vol (%)')
    plt.axhline(target_vol * 100, color='black', linestyle='--', linewidth=2, label=f'Target Vol ({target_vol*100:.1f}%)')

    plt.title('Volatility Targeting Report by Year')
    plt.xlabel('Year')
    plt.ylabel('Percent')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    chart_path = Path(__file__).resolve().parent / file_name
    plt.savefig(chart_path, dpi=150)
    print(f"Saved chart: {chart_path}")
    plt.close()

def main():
    df = load_data()
    all_rets, all_dates = [], []

    for year in range(START_YEAR, END_YEAR + 1):
        year_mask = df['date'].dt.year == year
        if not year_mask.any(): continue
        
        first_day = pd.to_datetime(df[year_mask]['date'].min())
        available = df[df['date'] == first_day]['permno'].unique()
        selected_permnos = random.sample(list(available), 100) if len(available) >= 100 else list(available)

        # Matrix setup
        lookback = first_day - pd.Timedelta(days=450)
        subset = df[(df['permno'].isin(selected_permnos)) & (df['date'] >= lookback) & (df['date'].dt.year <= year)]
        matrix = subset.pivot(index='date', columns='permno', values='ret').fillna(0)
        
        current_year_days = matrix.index[matrix.index.year == year]
        
        # Tracking weights for rebalancing frequency
        current_weights = None
        
        print(f"--- Processing {year} (Freq: {REBALANCE_FREQ}) ---")
        
        for i, d in enumerate(current_year_days):
            t_idx = matrix.index.get_loc(d)
            if t_idx < BURN_IN: continue

            # REBALANCE LOGIC:
            # Check if it's the first day of the period (Month/Week/Quarter)
            is_rebalance_day = False
            if REBALANCE_FREQ == 'D':
                is_rebalance_day = True
            elif REBALANCE_FREQ == 'W' and d.weekday() == 0: # Monday
                is_rebalance_day = True
            elif REBALANCE_FREQ == 'M' and d.is_month_start:
                is_rebalance_day = True
            elif REBALANCE_FREQ == 'Q' and d.is_quarter_start:
                is_rebalance_day = True
            elif current_weights is None: # Initial setup
                is_rebalance_day = True

            if is_rebalance_day:
                window = matrix.iloc[t_idx - BURN_IN : t_idx].values
                cov_matrix = np.cov(window, rowvar=False)
                # Add a small value to diagonal for solver stability (Regularization)
                cov_matrix += np.eye(len(selected_permnos)) * 1e-6 
                current_weights = solve_weights(cov_matrix, TARGET_VOL, MAX_LEVERAGE)

            # Daily Performance
            daily_asset_rets = matrix.iloc[t_idx].values
            port_ret = np.dot(current_weights, daily_asset_rets)
            all_rets.append(port_ret)
            all_dates.append(d)

    # FINAL REPORT
    results = pd.Series(all_rets, index=all_dates)
    if results.empty:
        print("No portfolio returns were generated. Check data/timeframe settings.")
        return

    ann_vol = results.std() * np.sqrt(252)
    ann_ret = results.mean() * 252
    yearly_metrics = build_yearly_metrics(results)

    print("\n" + "="*45)
    print(f"OPTIMIZED RESULTS (Freq: {REBALANCE_FREQ})")
    print(f"Realized Vol: {ann_vol*100:.2f}% | Target: {TARGET_VOL*100:.2f}%")
    print(f"Ann. Return:  {ann_ret*100:.2f}%")
    print(f"Sharpe Ratio: {ann_ret / ann_vol:.2f}")
    print("="*45)


    plot_final_report(yearly_metrics, TARGET_VOL)

if __name__ == "__main__":
    main()