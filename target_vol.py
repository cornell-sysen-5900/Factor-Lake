import pandas as pd
import numpy as np
import time

def main():
    print("Loading data from Factor-Lake/data/russell2000_cleaned.csv...")
    try:
        # permno is unique per asset, whereas tickers can be reused
        df = pd.read_csv("Factor-Lake/data/russell2000_cleaned.csv", usecols=['date', 'permno', 'ret'], low_memory=False)
    except FileNotFoundError:
        print("Data file not found. Ensure you are running this from the workspace root.")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    
    # Coerce returns to numeric (CRSP uses 'C' or 'B' sometimes for missing/bad data)
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df = df.dropna(subset=['ret'])
    
    print("Selecting 100 assets with the longest history...")
    counts = df['permno'].value_counts()
    top_100_permnos = counts.head(100).index
    
    df = df[df['permno'].isin(top_100_permnos)]
    df = df.drop_duplicates(subset=['date', 'permno'])
    
    print("Pivoting data to create a returns matrix...")
    # Reshape long-form data into a 2D matrix (dates x assets) and sort chronologically
    ret_matrix = df.pivot(index='date', columns='permno', values='ret').sort_index()
    
    # Replace missing returns with 0 so matrix math (covariance, dot products) doesn't break
    ret_matrix = ret_matrix.fillna(0)
    
    TARGET_VOLS = [0.10, 0.15, 0.20, 0.25]
    BURN_IN = 100
    
    if len(ret_matrix) <= BURN_IN:
        print("Not enough data to support the burn-in period!")
        return

    n_assets = len(ret_matrix.columns)
    dates = ret_matrix.index
    
    # Store daily tracking items for each target
    results = {tv: {'returns': [], 'k': [], 'est_vol': []} for tv in TARGET_VOLS}
    
    print(f"Beginning Volatility Targeting over {len(dates) - BURN_IN} days...")
    start_time = time.time()
    
    for t in range(BURN_IN, len(dates)):
        cov_window = ret_matrix.iloc[t-BURN_IN:t]
        cov_matrix = cov_window.cov().values
        w_baseline = np.ones(n_assets) / n_assets
        
        var = w_baseline.T @ cov_matrix @ w_baseline
        est_vol = np.sqrt(var) * np.sqrt(252)
        daily_ret = ret_matrix.iloc[t].values
        
        for tv in TARGET_VOLS:
            if est_vol == 0:
                k = 1.0 
            else:
                k = tv / est_vol
                
            w_target = w_baseline * k
            portfolio_return = np.dot(w_target, daily_ret)
            
            results[tv]['returns'].append(portfolio_return)
            results[tv]['k'].append(k)
            results[tv]['est_vol'].append(est_vol)
            
        if t % 500 == 0:
            print(f"Processed {t}/{len(dates)} days...")

    print(f"Finished processing in {time.time() - start_time:.2f} seconds.")

    out_df = pd.DataFrame({'date': dates[BURN_IN:]})
    
    print("\n--- Strategy Performance Summary ---")
    print(f"Universe Size:           {n_assets} equal-weighted stocks")
    print("-" * 50)
    
    for tv in TARGET_VOLS:
        strat_returns = pd.Series(results[tv]['returns'])
        k_series = pd.Series(results[tv]['k'])
        
        realized_vol = strat_returns.std() * np.sqrt(252)
        ann_ret = strat_returns.mean() * 252
        avg_k = k_series.mean()
        
        # Add to the output dataframe
        out_df[f'strat_ret_{int(tv*100)}'] = strat_returns
        out_df[f'scale_k_{int(tv*100)}'] = k_series
        out_df[f'est_vol_annual'] = results[tv]['est_vol'] # identical across TVs but stored once
        
        print(f"Target Annual Vol:       {tv*100:.2f}%")
        print(f"Realized Annual Vol:     {realized_vol*100:.2f}%")
        print(f"Annualized Return:       {ann_ret*100:.2f}%")
        print(f"Average Leverage (k):    {avg_k:.2f}x")
        print(f"  --- Yearly Volatility Breakdown ---")
        
        strat_ts = pd.Series(strat_returns.values, index=out_df['date'])
        yearly_vol = strat_ts.groupby(strat_ts.index.year).std() * np.sqrt(252)
        for year, vol in yearly_vol.items():
            print(f"    {year}: {vol*100:.2f}%")
            
        print("-" * 50)
    
    out_csv = "vol_target_results.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nDetails saved to {out_csv}")

if __name__ == "__main__":
    main()

