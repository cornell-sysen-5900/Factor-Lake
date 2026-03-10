from scipy import stats
from .market_object import MarketObject
from .portfolio import Portfolio
import numpy as np
import pandas as pd
from .factors_doc import FACTOR_DOCS
from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list

def calculate_holdings(factor, aum, market, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False, weighting_method='Equal Weight', target_volatility=None, volatility_metric=None):
    # Apply sector restrictions if enabled
    if restrict_fossil_fuels:
        industry_col = 'FactSet Industry'
        if industry_col in market.stocks.columns:
            fossil_keywords = ['oil', 'gas', 'coal', 'energy', 'fossil']
            series = market.stocks[industry_col].astype(str).str.lower()
            mask = series.apply(
                lambda x: not any(kw in x for kw in fossil_keywords) if pd.notna(x) else True)
            # Report which tickers are being removed in this step
            try:
                removed_tickers = list(market.stocks.loc[~mask].index)
                if removed_tickers:
                    print(f"Fossil filter (holdings) removed {len(removed_tickers)} tickers: {', '.join(removed_tickers[:25])}{' ...' if len(removed_tickers) > 25 else ''}")
            except Exception:
                pass
            market.stocks = market.stocks[mask].copy()

    # Get eligible stocks for factor calculation
    # Prefer vectorized series from market.stocks when available so we can normalize
    factor_col = getattr(factor, 'column_name', str(factor))
    factor_values = {}

    if factor_col in market.stocks.columns:
        raw_series = pd.to_numeric(market.stocks[factor_col], errors='coerce')
        # Determine direction from FACTOR_DOCS if available
        meta = FACTOR_DOCS.get(factor_col, {})
        higher_is_better = meta.get('higher_is_better', True)
        # Normalize series (winsorize + zscore) and invert if needed so higher == better
        normed = normalize_series(raw_series, higher_is_better=higher_is_better)
        factor_values = normed.dropna().to_dict()
    else:
        # Fallback to original per-ticker get() when column not present
        factor_values = {
            ticker: factor.get(ticker, market)
            for ticker in market.stocks.index
            if isinstance(factor.get(ticker, market), (int, float))
        }
    
    # ...existing code...
    
    if len(factor_values) == 0:
        # Return empty portfolio instead of crashing
        return Portfolio(name=f"Portfolio_{market.t}")
    
    sorted_securities = sorted(factor_values.items(), key=lambda x: x[1], reverse=True)

    # Select the top or bottom `top_pct`% of securities (default 10%)
    import math
    n_select = max(1, math.floor(len(sorted_securities) * (top_pct / 100.0))) if sorted_securities else 0
    if n_select == 0:
        selected = []
    else:
        if which == 'top':
            selected = sorted_securities[:n_select]
        else:
            # bottom: take the weakest n_select securities
            selected = sorted_securities[-n_select:]

    # Calculate number of shares for each selected security
    portfolio_new = Portfolio(name=f"Portfolio_{market.t}")
    
    if weighting_method == 'Volatility Weighted' and volatility_metric:
        # Inverse-Volatility Weighting (Risk Parity):
        # Weights are inversely proportional to each stock's expected volatility.
        # Low-vol stocks get higher allocations. Weights sum to 1.0 (fully invested, no leverage).
        vols = {}
        prices = {}
        for ticker, _ in selected:
            if volatility_metric in market.stocks.columns:
                try:
                    vol = market.stocks.loc[ticker, volatility_metric]
                    if isinstance(vol, (pd.Series, np.ndarray)):
                        vol = vol.iloc[0] if len(vol) > 0 else None
                    if pd.notna(vol) and vol > 0:
                        vols[ticker] = float(vol)
                except (KeyError, IndexError):
                    pass
            
            price = market.get_price(ticker)
            if price is not None and price > 0:
                prices[ticker] = price
                
        valid_vols = {t: v for t, v in vols.items() if t in prices}
        
        if valid_vols:
            # Inverse volatility weights, normalized to sum to 1.0
            inv_vols = {t: 1.0 / v for t, v in valid_vols.items()}
            sum_inv_vols = sum(inv_vols.values())
            weights = {t: iv / sum_inv_vols for t, iv in inv_vols.items()}
            
            for ticker, w in weights.items():
                dollar_investment = w * aum
                shares = dollar_investment / prices[ticker]
                portfolio_new.add_investment(ticker, shares)
        else:
            print(f"Warning: No valid volatility data for year {market.t}; falling back to Equal Weight.")
            _apply_equal_weight(portfolio_new, selected, aum, market)

    elif use_market_cap_weight or weighting_method == 'Market Cap Weight':
        # Market capitalization-based weighting (similar to Russell 2000)
        # Collect market cap and price for each selected ticker, then allocate
        market_caps = {}
        prices = {}
        for ticker, _ in selected:
            # Try to get market cap from the data
            if 'Market Capitalization' in market.stocks.columns:
                try:
                    market_cap = market.stocks.loc[ticker, 'Market Capitalization']
                    if isinstance(market_cap, (pd.Series, np.ndarray)):
                        market_cap = market_cap.iloc[0] if len(market_cap) > 0 else None
                    if pd.notna(market_cap) and market_cap > 0:
                        market_caps[ticker] = float(market_cap)
                except (KeyError, IndexError):
                    pass
            # also capture entry price availability
            price = market.get_price(ticker)
            if price is not None and price > 0:
                prices[ticker] = price

        # If we have market caps and at least one valid price, use them for weighting
        valid_caps = {t: c for t, c in market_caps.items() if t in prices}
        if valid_caps:
            total_market_cap = sum(valid_caps.values())
            for ticker, cap in valid_caps.items():
                # Weight by market cap: (ticker_market_cap / total_market_cap) * AUM
                weight = cap / total_market_cap if total_market_cap > 0 else 0
                dollar_investment = weight * aum
                price = prices.get(ticker)
                if price is not None and price > 0 and dollar_investment > 0:
                    shares = dollar_investment / price
                    portfolio_new.add_investment(ticker, shares)
            # If for some reason no shares were added (e.g., rounding), fallback to equal among priced tickers
            if not portfolio_new.investments and prices:
                valid_tickers = list(prices.keys())
                equal_investment = aum / len(valid_tickers)
                for t in valid_tickers:
                    shares = equal_investment / prices[t]
                    portfolio_new.add_investment(t, shares)
        else:
            # Fallback to equal weighting among tickers that have valid prices
            print(f"Warning: No valid market caps for year {market.t}; falling back to Equal Weight.")
            _apply_equal_weight(portfolio_new, selected, aum, market)
    else:
        # Equal dollar weighting
        _apply_equal_weight(portfolio_new, selected, aum, market)

    return portfolio_new

def _apply_equal_weight(portfolio_new, selected, aum, market):
    valid_tickers = [t for t, _ in selected if market.get_price(t) is not None and market.get_price(t) > 0]
    if not valid_tickers and selected:
        print(f"Warning: No valid priced tickers for equal-weighting in year {market.t}; returning empty portfolio.")
    else:
        equal_investment = aum / len(valid_tickers) if valid_tickers else 0
        for ticker in valid_tickers:
            price = market.get_price(ticker)
            if price is not None and price > 0:
                shares = equal_investment / price
                portfolio_new.add_investment(ticker, shares)

    return portfolio_new

def calculate_growth(portfolio, current_market, verbosity=0):
    """
    Calculates portfolio growth using the forward-looking 'Next_Year_Return' column.
    """
    total_weighted_rtn = 0
    total_investment_value = 0

    for factor_portfolio in portfolio:
        for inv in factor_portfolio.investments:
            ticker = inv["ticker"]
            
            # Retrieve the pre-calculated total return from the current market object
            try:
                # Dividing by 100 because the data is in percentage format (e.g., 20.0 for 20%)
                raw_rtn = current_market.stocks.loc[ticker, "Next_Year_Return"]
                # Handle duplicate index entries: extract scalar first
                if isinstance(raw_rtn, (pd.Series, pd.DataFrame)):
                    raw_rtn = raw_rtn.iloc[0]
                stock_rtn = float(raw_rtn) / 100
                
                # Weight the return by the dollar value of the position at the start of the year
                entry_price = current_market.get_price(ticker)
                if entry_price:
                    position_value = inv["number_of_shares"] * entry_price
                    total_weighted_rtn += stock_rtn * position_value
                    total_investment_value += position_value
                    
            except KeyError:
                if verbosity >= 2:
                    print(f"Warning: {ticker} return data missing, treating as 0%.")
                continue

    # Calculate aggregate growth for the year
    growth = total_weighted_rtn / total_investment_value if total_investment_value > 0 else 0
    
    # Calculate end value to maintain compatibility with the rebalancing loop
    total_end_value = total_investment_value * (1 + growth)
    
    return growth, total_investment_value, total_end_value

def rebalance_portfolio(data, factors, start_year, end_year, initial_aum, benchmark_index=1, verbosity=0, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False, factor_directions=None, weighting_method='Equal Weight', target_volatility=None, volatility_metric=None, data_frequency='Yearly', rebalance_frequency=None):
    """
    Executes a multi-period backtest. 
    Calculates Sharpe and Beta using period-specific excess returns.
    
    For 'Yearly' data_frequency, rebalances annually (legacy behavior).
    For 'Monthly'/'Daily', rebalances each period using the Period column.
    
    rebalance_frequency controls how often the portfolio is reconstructed.
    Between rebalances, existing holdings compound returns at data granularity.

    Args:
        factor_directions: optional dict mapping factor column_name -> 'top'/'bottom'.
        data_frequency: 'Yearly', 'Monthly', or 'Daily'
        rebalance_frequency: 'Daily', 'Monthly', 'Quarterly', 'Yearly', or None (same as data)
    """
    
    # 100% Legacy fallback for Yearly frequency
    if data_frequency == 'Yearly':
        return _rebalance_portfolio_yearly(
            data=data, factors=factors, start_year=start_year, end_year=end_year, 
            initial_aum=initial_aum, benchmark_index=benchmark_index, verbosity=verbosity, 
            restrict_fossil_fuels=restrict_fossil_fuels, top_pct=top_pct, which=which, 
            use_market_cap_weight=use_market_cap_weight, factor_directions=factor_directions
        )

    aum = initial_aum
    portfolio_returns = []  
    portfolio_values = [aum]  
    periods = []
    
    verbosity = 0 if verbosity is None else verbosity
    risk_free_rate_source = "FRED 4 Week T-Bill (Oct 1)"
    
    is_sub_annual = data_frequency in ['Monthly', 'Daily']
    
    # Default rebalance_frequency to match data_frequency
    if rebalance_frequency is None:
        rebalance_frequency = data_frequency

    def _build_portfolio(period_data, period_label):
        """Construct portfolio from factor scores (no growth applied)."""
        market = MarketObject(period_data, period_label)
        period_portfolio = []

        for factor in factors:
            factor_which = which
            if factor_directions:
                col_name = getattr(factor, 'column_name', str(factor))
                factor_which = factor_directions.get(col_name, which)

            factor_portfolio = calculate_holdings(
                factor=factor,
                aum=aum / len(factors),
                market=market,
                restrict_fossil_fuels=restrict_fossil_fuels,
                top_pct=top_pct,
                which=factor_which,
                use_market_cap_weight=use_market_cap_weight,
                weighting_method=weighting_method,
                target_volatility=target_volatility,
                volatility_metric=volatility_metric
            )
            period_portfolio.append(factor_portfolio)

        return period_portfolio
    
    def _apply_growth(portfolio_list, period_data, period_label):
        """Apply one period's returns to an existing portfolio."""
        nonlocal aum
        market = MarketObject(period_data, period_label)
        growth, total_start_value, total_end_value = calculate_growth(portfolio_list, market, verbosity)

        if verbosity >= 2:
            print(f"Period {period_label}: Growth: {growth:.2%}, Start: ${total_start_value:.2f}, End: ${total_end_value:.2f}")

        aum = total_end_value if total_end_value > 0 else aum
        portfolio_returns.append(growth)
        portfolio_values.append(aum)

    def _build_and_grow(period_data, period_label):
        """Run portfolio construction and growth for one period (legacy single-step)."""
        portfolio_list = _build_portfolio(period_data, period_label)
        _apply_growth(portfolio_list, period_data, period_label)

    def _get_rebalance_key(period_str, rebal_freq):
        """Map a period label to its rebalance window key."""
        if rebal_freq == 'Daily':
            return period_str  # rebalance every period
        elif rebal_freq == 'Monthly':
            # Daily "2020-01-15" -> "2020-01", Monthly "2020-01" -> "2020-01"
            return period_str[:7]
        elif rebal_freq == 'Quarterly':
            # Map month to quarter: Jan-Mar -> Q1, Apr-Jun -> Q2, etc.
            month = int(period_str[5:7])
            quarter = (month - 1) // 3 + 1
            return f"{period_str[:4]}-Q{quarter}"
        elif rebal_freq == 'Yearly':
            return period_str[:4]
        return period_str
    
    # --- 1. Rebalancing Loop ---
    if is_sub_annual and 'Period' in data.columns:
        sorted_periods = sorted(data['Period'].unique())
        if not sorted_periods:
            periods = [f"{start_year}-01"]
        else:
            periods = [sorted_periods[0]]  # Starting label
            
            current_portfolio = None
            last_rebal_key = None
            
            for period in sorted_periods:
                period_data = data.loc[data['Period'] == period]
                if period_data.empty:
                    portfolio_returns.append(0.0)
                    portfolio_values.append(aum)
                    periods.append(period)
                    continue
                
                rebal_key = _get_rebalance_key(period, rebalance_frequency)
                
                if rebal_key != last_rebal_key:
                    # --- REBALANCE POINT: reconstruct portfolio AND apply growth ---
                    current_portfolio = _build_portfolio(period_data, period)
                    _apply_growth(current_portfolio, period_data, period)
                    last_rebal_key = rebal_key
                else:
                    # --- HOLD: apply growth to existing portfolio ---
                    _apply_growth(current_portfolio, period_data, period)
                
                periods.append(period)
    else:
        # Annual: legacy behavior
        periods = [start_year]
        for year in range(start_year, end_year):
            year_data = data.loc[data['Year'] == year]
            
            if year_data.empty:
                portfolio_returns.append(0.0)
                portfolio_values.append(aum)
                periods.append(year + 1)
                if verbosity >= 1:
                    print(f"Year {year}: No data available, carrying forward AUM ${aum:,.2f}")
                continue
            
            _build_and_grow(year_data, year)
            periods.append(year + 1)

    # Backward-compatible alias
    years = periods
    
    # --- 2. Data Alignment (Benchmarks & Risk-Free) ---
    # Always fetch yearly benchmark data for stats and chart overlays
    if is_sub_annual:
        rf_list = get_benchmark_list(4, start_year, end_year + 1)
        bench_list = get_benchmark_list(benchmark_index, start_year, end_year + 1)
        growth_bench_list = get_benchmark_list(2, start_year, end_year + 1)
        value_bench_list = get_benchmark_list(3, start_year, end_year + 1)
    else:
        # Annual rebalancing is forward-looking (realizes return in year+1)
        rf_list = get_benchmark_list(4, start_year, end_year)
        bench_list = get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)
        growth_bench_list = get_benchmark_list(2, start_year + 1, end_year + 1)
        value_bench_list = get_benchmark_list(3, start_year + 1, end_year + 1)

    portfolio_returns_np = np.array(portfolio_returns)
    benchmark_returns_np = np.array(bench_list) / 100
    growth_returns_np = np.array(growth_bench_list) / 100
    value_returns_np = np.array(value_bench_list) / 100
    rf_rates_yearly_np = np.array(rf_list)

    # Annualize Volatility if sub-annual (for apples-to-apples comparison with yearly benchmarks)
    annual_factor = 1
    if is_sub_annual:
        annual_factor = 12 if data_frequency == 'Monthly' else 252

    vol_raw_portfolio = np.std(portfolio_returns_np, ddof=1) * np.sqrt(annual_factor) if len(portfolio_returns_np) > 1 else 0
    vol_raw_benchmark = np.std(benchmark_returns_np, ddof=1) if len(benchmark_returns_np) > 1 else 0
    vol_raw_growth = np.std(growth_returns_np, ddof=1) if len(growth_returns_np) > 1 else 0
    vol_raw_value = np.std(value_returns_np, ddof=1) if len(value_returns_np) > 1 else 0
    
# --- 3. Excess Return Calculations ---
    # Portfolio excess returns: use rf=0 for sub-annual (no monthly rf available)
    if is_sub_annual:
        excess_portfolio_returns = portfolio_returns_np  # rf=0
    else:
        excess_portfolio_returns = portfolio_returns_np - rf_rates_yearly_np
    
    # Benchmark excess returns: always yearly
    excess_benchmark_returns = benchmark_returns_np - rf_rates_yearly_np
    excess_growth_returns = growth_returns_np - rf_rates_yearly_np
    excess_value_returns = value_returns_np - rf_rates_yearly_np
    
    vol_excess_p = np.std(excess_portfolio_returns, ddof=1) * np.sqrt(annual_factor) if len(excess_portfolio_returns) > 1 else 0
    vol_excess_b = np.std(excess_benchmark_returns, ddof=1) if len(excess_benchmark_returns) > 1 else 0
    vol_excess_growth = np.std(excess_growth_returns, ddof=1) if len(excess_growth_returns) > 1 else 0
    vol_excess_value = np.std(excess_value_returns, ddof=1) if len(excess_value_returns) > 1 else 0

    std_excess_p_unscaled = vol_excess_p / np.sqrt(annual_factor) if vol_excess_p > 0 else 0
    # Use portfolio value-based CAGR for Sharpe (matches the displayed CAGR exactly)
    # This is robust against data loading issues that cause period count mismatches
    if is_sub_annual and vol_excess_p > 0 and len(portfolio_values) > 1:
        calendar_years = max(end_year - start_year + 1, 1)  # Inclusive: 2020-2024 = 5 years
        total_growth = portfolio_values[-1] / portfolio_values[0]
        cagr_return = total_growth ** (1.0 / calendar_years) - 1 if total_growth > 0 else 0
        sharpe_portfolio = cagr_return / vol_excess_p if vol_excess_p > 0 else 0
    else:
        sharpe_portfolio = (np.mean(excess_portfolio_returns) / std_excess_p_unscaled) * np.sqrt(annual_factor) if std_excess_p_unscaled > 0 else 0
    sharpe_benchmark = np.mean(excess_benchmark_returns) / vol_excess_b if vol_excess_b > 0 else 0
    sharpe_growth = np.mean(excess_growth_returns) / vol_excess_growth if vol_excess_growth > 0 else 0
    sharpe_value = np.mean(excess_value_returns) / vol_excess_value if vol_excess_value > 0 else 0
    
    portfolio_beta = None
    portfolio_beta_growth = None
    portfolio_beta_value = None
    information_ratio = None
    information_ratio_growth = None
    information_ratio_value = None
    
    if is_sub_annual and len(portfolio_returns_np) >= 3:
        # Aggregate daily/monthly portfolio returns to yearly for beta/IR calculation
        # periods list has the period labels (e.g. "2020-01-02", "2020-03-15")
        period_labels = periods[1:] if len(periods) > len(portfolio_returns_np) else periods
        yearly_pf_returns = {}
        for i, ret in enumerate(portfolio_returns_np):
            if i < len(period_labels):
                year_key = str(period_labels[i])[:4]
                if year_key not in yearly_pf_returns:
                    yearly_pf_returns[year_key] = 1.0
                yearly_pf_returns[year_key] *= (1 + ret)
        
        # Convert compounded values to returns
        yearly_pf_list = [(y, v - 1) for y, v in sorted(yearly_pf_returns.items())]
        yearly_pf_np = np.array([r for _, r in yearly_pf_list])
        
        # Align with benchmark returns (both should be yearly now)
        n_common = min(len(yearly_pf_np), len(excess_benchmark_returns))
        if n_common >= 3:
            yr_pf = yearly_pf_np[:n_common]
            yr_bench = excess_benchmark_returns[:n_common]
            yr_growth = excess_growth_returns[:n_common]
            yr_value = excess_value_returns[:n_common]
            yr_rf = rf_rates_yearly_np[:n_common] if len(rf_rates_yearly_np) >= n_common else np.zeros(n_common)
            yr_pf_excess = yr_pf - yr_rf
            
            coeffs = np.polyfit(yr_bench, yr_pf_excess, 1)
            portfolio_beta = float(coeffs[0])
            growth_coeffs = np.polyfit(yr_growth, yr_pf_excess, 1)
            portfolio_beta_growth = float(growth_coeffs[0])
            value_coeffs = np.polyfit(yr_value, yr_pf_excess, 1)
            portfolio_beta_value = float(value_coeffs[0])
            
            information_ratio = (np.mean(yr_pf_excess - yr_bench) /
                                 np.std(yr_pf_excess - yr_bench, ddof=1)) if np.std(yr_pf_excess - yr_bench, ddof=1) > 0 else 0
            information_ratio_growth = (np.mean(yr_pf_excess - yr_growth) /
                                       np.std(yr_pf_excess - yr_growth, ddof=1)) if np.std(yr_pf_excess - yr_growth, ddof=1) > 0 else 0
            information_ratio_value = (np.mean(yr_pf_excess - yr_value) /
                                      np.std(yr_pf_excess - yr_value, ddof=1)) if np.std(yr_pf_excess - yr_value, ddof=1) > 0 else 0
    
    elif not is_sub_annual and len(portfolio_returns_np) >= 3:
        coeffs = np.polyfit(excess_benchmark_returns, excess_portfolio_returns, 1)
        portfolio_beta = float(coeffs[0])
        growth_coeffs = np.polyfit(excess_growth_returns, excess_portfolio_returns, 1)
        portfolio_beta_growth = float(growth_coeffs[0])
        value_coeffs = np.polyfit(excess_value_returns, excess_portfolio_returns, 1)
        portfolio_beta_value = float(value_coeffs[0])
        
        information_ratio = (np.mean(excess_portfolio_returns - excess_benchmark_returns) / 
                             np.std(excess_portfolio_returns - excess_benchmark_returns, ddof=1)) if np.std(excess_portfolio_returns - excess_benchmark_returns, ddof=1) > 0 else 0
        information_ratio_growth = (np.mean(excess_portfolio_returns - excess_growth_returns) / 
                                   np.std(excess_portfolio_returns - excess_growth_returns, ddof=1)) if np.std(excess_portfolio_returns - excess_growth_returns, ddof=1) > 0 else 0
        information_ratio_value = (np.mean(excess_portfolio_returns - excess_value_returns) / 
                                  np.std(excess_portfolio_returns - excess_value_returns, ddof=1)) if np.std(excess_portfolio_returns - excess_value_returns, ddof=1) > 0 else 0
    
    cumulative_values = np.array(portfolio_values)
    running_peak = np.maximum.accumulate(cumulative_values)
    max_drawdown_portfolio = np.min((cumulative_values - running_peak) / running_peak) if len(running_peak) > 0 else 0
    
    # Always compute benchmark drawdowns from yearly data
    benchmark_cumulative = [initial_aum]
    growth_cumulative = [initial_aum]
    value_cumulative = [initial_aum]
    for r in benchmark_returns_np:
        benchmark_cumulative.append(benchmark_cumulative[-1] * (1 + r))
    for r in growth_returns_np:
        growth_cumulative.append(growth_cumulative[-1] * (1 + r))
    for r in value_returns_np:
        value_cumulative.append(value_cumulative[-1] * (1 + r))
    
    benchmark_cumulative_np = np.array(benchmark_cumulative)
    benchmark_peak = np.maximum.accumulate(benchmark_cumulative_np)
    max_drawdown_benchmark = np.min((benchmark_cumulative_np - benchmark_peak) / benchmark_peak)

    growth_cumulative_np = np.array(growth_cumulative)
    growth_peak = np.maximum.accumulate(growth_cumulative_np)
    max_drawdown_growth = np.min((growth_cumulative_np - growth_peak) / growth_peak)

    value_cumulative_np = np.array(value_cumulative)
    value_peak = np.maximum.accumulate(value_cumulative_np)
    max_drawdown_value = np.min((value_cumulative_np - value_peak) / value_peak)

    # Compute actual realized yearly portfolio returns for realistic win-rate comparisons
    yearly_portfolio_returns = []
    if is_sub_annual:
        last_val = initial_aum
        for yr in range(start_year, end_year + 1):
            yr_str = str(yr)
            yr_indices = [i for i, p in enumerate(periods) if str(p).startswith(yr_str)]
            if yr_indices:
                end_idx = yr_indices[-1]
                val = portfolio_values[end_idx]
                yearly_ret = (val / last_val) - 1
                yearly_portfolio_returns.append(yearly_ret)
                last_val = val
            elif yr <= end_year:
                yearly_portfolio_returns.append(0.0)
    else:
        yearly_portfolio_returns = portfolio_returns_np.tolist()

    wins = 0
    yearly_comparisons = []
    for i, (p_ret, b_ret, g_ret, v_ret) in enumerate(zip(yearly_portfolio_returns, benchmark_returns_np, growth_returns_np, value_returns_np)):
        win = p_ret > b_ret
        if win: wins += 1
        yr_label = start_year + i
        yearly_comparisons.append({
            'year': yr_label,
            'portfolio_return': p_ret * 100,
            'benchmark_return': b_ret * 100,
            'win': win,
            'growth_return': g_ret * 100,
            'growth_win': p_ret > g_ret,
            'value_return': v_ret * 100,
            'value_win': p_ret > v_ret
        })
    win_rate = wins / len(yearly_portfolio_returns) if len(yearly_portfolio_returns) > 0 else 0

    wins_growth = 0
    wins_value = 0
    for p_ret, g_ret, v_ret in zip(yearly_portfolio_returns, growth_returns_np, value_returns_np):
        if p_ret > g_ret: wins_growth += 1
        if p_ret > v_ret: wins_value += 1
    win_rate_growth = wins_growth / len(yearly_portfolio_returns) if len(yearly_portfolio_returns) > 0 else 0
    win_rate_value = wins_value / len(yearly_portfolio_returns) if len(yearly_portfolio_returns) > 0 else 0

    if verbosity >= 1:
        print(f"\n==== Performance Metrics ({start_year}-{end_year}) ====")
        print(f"Final AUM: ${aum:.2f}")
        print(f"Sharpe Ratio (Portfolio): {sharpe_portfolio:.4f}")
        print(f"Max Drawdown (Portfolio): {max_drawdown_portfolio:.2%}")

    return {
        'final_value': aum,
        'yearly_returns': portfolio_returns,
        'benchmark_returns': bench_list,
        'growth_benchmark_returns': growth_bench_list,
        'value_benchmark_returns': value_bench_list,
        'years': years,
        'portfolio_values': portfolio_values,
        'portfolio_beta': portfolio_beta,
        'portfolio_beta_growth': portfolio_beta_growth,
        'portfolio_beta_value': portfolio_beta_value,
        'max_drawdown_portfolio': max_drawdown_portfolio,
        'max_drawdown_benchmark': max_drawdown_benchmark,
        'max_drawdown_growth': max_drawdown_growth,
        'max_drawdown_value': max_drawdown_value,
        'sharpe_portfolio': sharpe_portfolio,
        'sharpe_benchmark': sharpe_benchmark,
        'sharpe_growth': sharpe_growth,
        'sharpe_value': sharpe_value,
        'vol_raw_portfolio': vol_raw_portfolio,
        'vol_raw_benchmark': vol_raw_benchmark,
        'vol_raw_growth': vol_raw_growth,
        'vol_raw_value': vol_raw_value,
        'vol_excess_portfolio': vol_excess_p,
        'vol_excess_benchmark': vol_excess_b,
        'vol_excess_growth': vol_excess_growth,
        'vol_excess_value': vol_excess_value,
        'win_rate': win_rate,
        'win_rate_growth': win_rate_growth,
        'win_rate_value': win_rate_value,
        'risk_free_rate_source': risk_free_rate_source,
        'yearly_comparisons': yearly_comparisons,
        'information_ratio': information_ratio,
        'information_ratio_growth': information_ratio_growth,
        'information_ratio_value': information_ratio_value,
        'data_frequency': data_frequency
    }
def calculate_information_ratio(portfolio_returns, benchmark_returns, verbosity=0):
    """
    Calculates the Information Ratio (IR) for a given set of portfolio returns and benchmark returns.
    
    Parameters:
        portfolio_returns (list or np.array): List of portfolio returns over time.
        benchmark_returns (list or np.array): List of benchmark returns over time.
    
    Returns:
        float: The Information Ratio value.
    """
    # Ensure inputs are numpy arrays for mathematical operations
    verbosity = 0 if verbosity is None else verbosity
    portfolio_returns = np.array(portfolio_returns)
    #benchmark_returns = np.array(benchmark_returns)
    benchmark_returns = np.array(benchmark_returns) / 100

    # Calculate active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Calculate the mean active return (numerator)
    mean_active_return = np.mean(active_returns)
    
    # Calculate tracking error (denominator)
    tracking_error = np.std(active_returns, ddof=1) if len(active_returns) > 1 else 0  # Use sample std deviation

    # Prevent division by zero
    if tracking_error == 0:
        return 0.0  # Or return float('nan') to indicate undefined IR
    
    # Compute Information Ratio
    information_ratio = mean_active_return / tracking_error
    if verbosity >=1:
        print(f"Information Ratio: {information_ratio:.4f}")
    return information_ratio


def _calculate_holdings_yearly(factor, aum, market, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False):
    # Apply sector restrictions if enabled
    if restrict_fossil_fuels:
        industry_col = 'FactSet Industry'
        if industry_col in market.stocks.columns:
            fossil_keywords = ['oil', 'gas', 'coal', 'energy', 'fossil']
            series = market.stocks[industry_col].astype(str).str.lower()
            mask = series.apply(
                lambda x: not any(kw in x for kw in fossil_keywords) if pd.notna(x) else True)
            # Report which tickers are being removed in this step
            try:
                removed_tickers = list(market.stocks.loc[~mask].index)
                if removed_tickers:
                    print(f"Fossil filter (holdings) removed {len(removed_tickers)} tickers: {', '.join(removed_tickers[:25])}{' ...' if len(removed_tickers) > 25 else ''}")
            except Exception:
                pass
            market.stocks = market.stocks[mask].copy()

    # Get eligible stocks for factor calculation
    # Prefer vectorized series from market.stocks when available so we can normalize
    factor_col = getattr(factor, 'column_name', str(factor))
    factor_values = {}

    if factor_col in market.stocks.columns:
        raw_series = pd.to_numeric(market.stocks[factor_col], errors='coerce')
        # Determine direction from FACTOR_DOCS if available
        meta = FACTOR_DOCS.get(factor_col, {})
        higher_is_better = meta.get('higher_is_better', True)
        # Normalize series (winsorize + zscore) and invert if needed so higher == better
        normed = normalize_series(raw_series, higher_is_better=higher_is_better)
        factor_values = normed.dropna().to_dict()
    else:
        # Fallback to original per-ticker get() when column not present
        factor_values = {
            ticker: factor.get(ticker, market)
            for ticker in market.stocks.index
            if isinstance(factor.get(ticker, market), (int, float))
        }
    
    # ...existing code...
    
    if len(factor_values) == 0:
        # Return empty portfolio instead of crashing
        return Portfolio(name=f"Portfolio_{market.t}")
    
    sorted_securities = sorted(factor_values.items(), key=lambda x: x[1], reverse=True)

    # Select the top or bottom `top_pct`% of securities (default 10%)
    import math
    n_select = max(1, math.floor(len(sorted_securities) * (top_pct / 100.0))) if sorted_securities else 0
    if n_select == 0:
        selected = []
    else:
        if which == 'top':
            selected = sorted_securities[:n_select]
        else:
            # bottom: take the weakest n_select securities
            selected = sorted_securities[-n_select:]

    # Calculate number of shares for each selected security
    portfolio_new = Portfolio(name=f"Portfolio_{market.t}")
    
    if use_market_cap_weight:
        # Market capitalization-based weighting (similar to Russell 2000)
        # Collect market cap and price for each selected ticker, then allocate
        market_caps = {}
        prices = {}
        for ticker, _ in selected:
            # Try to get market cap from the data
            if 'Market Capitalization' in market.stocks.columns:
                try:
                    market_cap = market.stocks.loc[ticker, 'Market Capitalization']
                    if isinstance(market_cap, (pd.Series, np.ndarray)):
                        market_cap = market_cap.iloc[0] if len(market_cap) > 0 else None
                    if pd.notna(market_cap) and market_cap > 0:
                        market_caps[ticker] = float(market_cap)
                except (KeyError, IndexError):
                    pass
            # also capture entry price availability
            price = market.get_price(ticker)
            if price is not None and price > 0:
                prices[ticker] = price

        # If we have market caps and at least one valid price, use them for weighting
        valid_caps = {t: c for t, c in market_caps.items() if t in prices}
        if valid_caps:
            total_market_cap = sum(valid_caps.values())
            for ticker, cap in valid_caps.items():
                # Weight by market cap: (ticker_market_cap / total_market_cap) * AUM
                weight = cap / total_market_cap if total_market_cap > 0 else 0
                dollar_investment = weight * aum
                price = prices.get(ticker)
                if price is not None and price > 0 and dollar_investment > 0:
                    shares = dollar_investment / price
                    portfolio_new.add_investment(ticker, shares)
            # If for some reason no shares were added (e.g., rounding), fallback to equal among priced tickers
            if not portfolio_new.investments and prices:
                valid_tickers = list(prices.keys())
                equal_investment = aum / len(valid_tickers)
                for t in valid_tickers:
                    shares = equal_investment / prices[t]
                    portfolio_new.add_investment(t, shares)
        else:
            # Fallback to equal weighting among tickers that have valid prices
            valid_tickers = [t for t, _ in selected if market.get_price(t) is not None and market.get_price(t) > 0]
            if not valid_tickers:
                print(f"Warning: No valid priced tickers for year {market.t}; returning empty portfolio.")
            else:
                equal_investment = aum / len(valid_tickers)
                for ticker in valid_tickers:
                    price = market.get_price(ticker)
                    if price is not None and price > 0:
                        shares = equal_investment / price
                        portfolio_new.add_investment(ticker, shares)
    else:
        # Equal dollar weighting (allocate only to tickers with valid entry prices)
        valid_tickers = [t for t, _ in selected if market.get_price(t) is not None and market.get_price(t) > 0]
        if not valid_tickers and selected:
            # nothing priced; warn and return empty portfolio
            print(f"Warning: No valid priced tickers for equal-weighting in year {market.t}; returning empty portfolio.")
        else:
            equal_investment = aum / len(valid_tickers) if valid_tickers else 0
            for ticker in valid_tickers:
                price = market.get_price(ticker)
                if price is not None and price > 0:
                    shares = equal_investment / price
                    portfolio_new.add_investment(ticker, shares)

    return portfolio_new

def _calculate_growth_yearly(portfolio, current_market, verbosity=0):
    """
    Calculates portfolio growth using the forward-looking 'Next_Year_Return' column.
    """
    total_weighted_rtn = 0
    total_investment_value = 0

    for factor_portfolio in portfolio:
        for inv in factor_portfolio.investments:
            ticker = inv["ticker"]
            
            # Retrieve the pre-calculated total return from the current market object
            try:
                # Dividing by 100 because the data is in percentage format (e.g., 20.0 for 20%)
                stock_rtn = current_market.stocks.loc[ticker, "Next_Year_Return"] / 100
                
                # Weight the return by the dollar value of the position at the start of the year
                entry_price = current_market.get_price(ticker)
                if entry_price:
                    position_value = inv["number_of_shares"] * entry_price
                    total_weighted_rtn += stock_rtn * position_value
                    total_investment_value += position_value
                    
            except KeyError:
                if verbosity >= 2:
                    print(f"Warning: {ticker} return data missing, treating as 0%.")
                continue

    # Calculate aggregate growth for the year
    growth = total_weighted_rtn / total_investment_value if total_investment_value > 0 else 0
    
    # Calculate end value to maintain compatibility with the rebalancing loop
    total_end_value = total_investment_value * (1 + growth)
    
    return growth, total_investment_value, total_end_value

def _rebalance_portfolio_yearly(data, factors, start_year, end_year, initial_aum, benchmark_index=1, verbosity=0, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False, factor_directions=None):
    """
    Executes a multi-year backtest. 
    Calculates Sharpe and Beta using year-specific excess returns.

    Args:
        factor_directions: optional dict mapping factor column_name -> 'top'/'bottom'.
            When provided, overrides the global ``which`` on a per-factor basis.
    """
    aum = initial_aum
    years = [start_year]
    portfolio_returns = []  
    portfolio_values = [aum]  
    
    verbosity = 0 if verbosity is None else verbosity
    risk_free_rate_source = "FRED 4 Week T-Bill (Oct 1)"

    # --- 1. Annual Rebalancing Loop ---
    for year in range(start_year, end_year):
        market = MarketObject(data.loc[data['Year'] == year], year)
        yearly_portfolio = []

        for factor in factors:
            factor_which = which
            if factor_directions:
                col_name = getattr(factor, 'column_name', str(factor))
                factor_which = factor_directions.get(col_name, which)

            factor_portfolio = _calculate_holdings_yearly(
                factor=factor,
                aum=aum / len(factors),
                market=market,
                restrict_fossil_fuels=restrict_fossil_fuels,
                top_pct=top_pct,
                which=factor_which,
                use_market_cap_weight=use_market_cap_weight
            )
            yearly_portfolio.append(factor_portfolio)

        # Calculate annual growth
        growth, total_start_value, total_end_value = _calculate_growth_yearly(yearly_portfolio, market, verbosity)

        if verbosity >= 2:
            print(f"Year {year} to {year + 1}: Growth: {growth:.2%}, Start: ${total_start_value:.2f}, End: ${total_end_value:.2f}")

        aum = total_end_value  
        portfolio_returns.append(growth)
        portfolio_values.append(aum)
        years.append(year + 1)

    # --- 2. Data Alignment (Benchmarks & Risk-Free) ---
    # Fetched from benchmarks.py to ensure data integrity across the range
    rf_list = get_benchmark_list(4, start_year, end_year)
    bench_list = get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)
    growth_bench_list = get_benchmark_list(2, start_year + 1, end_year + 1)
    value_bench_list = get_benchmark_list(3, start_year + 1, end_year + 1)

    # Convert to NumPy for performance math
    portfolio_returns_np = np.array(portfolio_returns)
    rf_rates_np = np.array(rf_list)
    benchmark_returns_np = np.array(bench_list) / 100 
    growth_returns_np = np.array(growth_bench_list) / 100
    value_returns_np = np.array(value_bench_list) / 100

    # Full-sample volatility of RAW returns
    vol_raw_portfolio = np.std(portfolio_returns_np, ddof=1) if len(portfolio_returns_np) > 1 else 0
    vol_raw_benchmark = np.std(benchmark_returns_np, ddof=1) if len(benchmark_returns_np) > 1 else 0
    vol_raw_growth = np.std(growth_returns_np, ddof=1) if len(growth_returns_np) > 1 else 0
    vol_raw_value = np.std(value_returns_np, ddof=1) if len(value_returns_np) > 1 else 0
    
# --- 3. Excess Return Calculations ---
    excess_portfolio_returns = portfolio_returns_np - rf_rates_np
    excess_benchmark_returns = benchmark_returns_np - rf_rates_np
    excess_growth_returns = growth_returns_np - rf_rates_np
    excess_value_returns = value_returns_np - rf_rates_np
    
    # --- 4. Sharpe Ratio ---
    # Volatility of the EXCESS returns (Institutional Standard)
    vol_excess_p = np.std(excess_portfolio_returns, ddof=1) if len(excess_portfolio_returns) > 1 else 0
    vol_excess_b = np.std(excess_benchmark_returns, ddof=1) if len(excess_benchmark_returns) > 1 else 0
    vol_excess_growth = np.std(excess_growth_returns, ddof=1) if len(excess_growth_returns) > 1 else 0
    vol_excess_value = np.std(excess_value_returns, ddof=1) if len(excess_value_returns) > 1 else 0

    # Formula: Mean(Yearly Excess Returns) / Volatility(Yearly Excess Returns)
    sharpe_portfolio = np.mean(excess_portfolio_returns) / vol_excess_p if vol_excess_p > 0 else 0
    sharpe_benchmark = np.mean(excess_benchmark_returns) / vol_excess_b if vol_excess_b > 0 else 0
    sharpe_growth = np.mean(excess_growth_returns) / vol_excess_growth if vol_excess_growth > 0 else 0
    sharpe_value = np.mean(excess_value_returns) / vol_excess_value if vol_excess_value > 0 else 0
    
    # --- 5. Portfolio Beta ---
    portfolio_beta = None
    portfolio_beta_growth = None
    portfolio_beta_value = None
    if len(portfolio_returns_np) >= 3:
        # CAPM Beta using excess returns
        coeffs = np.polyfit(excess_benchmark_returns, excess_portfolio_returns, 1)
        portfolio_beta = float(coeffs[0])
        growth_coeffs = np.polyfit(excess_growth_returns, excess_portfolio_returns, 1)
        portfolio_beta_growth = float(growth_coeffs[0])
        value_coeffs = np.polyfit(excess_value_returns, excess_portfolio_returns, 1)
        portfolio_beta_value = float(value_coeffs[0])

    # --- 6. Additional Risk Metrics ---
    # Information Ratio
    information_ratio = calculate_information_ratio(portfolio_returns, bench_list, verbosity)
    information_ratio_growth = calculate_information_ratio(portfolio_returns, growth_bench_list, verbosity)
    information_ratio_value = calculate_information_ratio(portfolio_returns, value_bench_list, verbosity)
    
    # Portfolio Max Drawdown
    cumulative_values = np.array(portfolio_values)
    running_peak = np.maximum.accumulate(cumulative_values)
    max_drawdown_portfolio = np.min((cumulative_values - running_peak) / running_peak)
    
    # Benchmark Max Drawdown
    benchmark_cumulative = [initial_aum]
    growth_cumulative = [initial_aum]
    value_cumulative = [initial_aum]
    for r in benchmark_returns_np:
        benchmark_cumulative.append(benchmark_cumulative[-1] * (1 + r))
    for r in growth_returns_np:
        growth_cumulative.append(growth_cumulative[-1] * (1 + r))
    for r in value_returns_np:
        value_cumulative.append(value_cumulative[-1] * (1 + r))
    
    benchmark_cumulative_np = np.array(benchmark_cumulative)
    benchmark_peak = np.maximum.accumulate(benchmark_cumulative_np)
    max_drawdown_benchmark = np.min((benchmark_cumulative_np - benchmark_peak) / benchmark_peak)

    growth_cumulative_np = np.array(growth_cumulative)
    growth_peak = np.maximum.accumulate(growth_cumulative_np)
    max_drawdown_growth = np.min((growth_cumulative_np - growth_peak) / growth_peak)

    value_cumulative_np = np.array(value_cumulative)
    value_peak = np.maximum.accumulate(value_cumulative_np)
    max_drawdown_value = np.min((value_cumulative_np - value_peak) / value_peak)

    # Yearly Win Rate
    wins = 0
    yearly_comparisons = []
    for i, (p_ret, b_ret, g_ret, v_ret) in enumerate(zip(portfolio_returns_np, benchmark_returns_np, growth_returns_np, value_returns_np)):
        win = p_ret > b_ret
        if win: wins += 1
        yearly_comparisons.append({
            'year': years[i+1],
            'portfolio_return': p_ret * 100,
            'benchmark_return': b_ret * 100,
            'win': win,
            'growth_return': g_ret * 100,
            'growth_win': p_ret > g_ret,
            'value_return': v_ret * 100,
            'value_win': p_ret > v_ret
        })
    win_rate = wins / len(portfolio_returns_np) if len(portfolio_returns_np) > 0 else 0

    wins_growth = 0
    wins_value = 0
    for p_ret, g_ret, v_ret in zip(portfolio_returns_np, growth_returns_np, value_returns_np):
        if p_ret > g_ret:
            wins_growth += 1
        if p_ret > v_ret:
            wins_value += 1
    win_rate_growth = wins_growth / len(portfolio_returns_np) if len(portfolio_returns_np) > 0 else 0
    win_rate_value = wins_value / len(portfolio_returns_np) if len(portfolio_returns_np) > 0 else 0

    # --- 7. Summary Output ---
    if verbosity >= 1:
        print(f"\n==== Performance Metrics ({start_year}-{end_year}) ====")
        print(f"Final AUM: ${aum:.2f}")
        print(f"Volatility (Raw Returns, Portfolio): {vol_raw_portfolio:.4%}")
        print(f"Volatility (Raw Returns, Benchmark): {vol_raw_benchmark:.4%}")
        print(f"Volatility (Raw Returns, Growth Index): {vol_raw_growth:.4%}")
        print(f"Volatility (Raw Returns, Value Index): {vol_raw_value:.4%}")
        print(f"Volatility (Excess Returns, Portfolio): {vol_excess_p:.4%}")
        print(f"Volatility (Excess Returns, Benchmark): {vol_excess_b:.4%}")
        print(f"Volatility (Excess Returns, Growth Index): {vol_excess_growth:.4%}")
        print(f"Volatility (Excess Returns, Value Index): {vol_excess_value:.4%}")
        print(f"Sharpe Ratio (Portfolio): {sharpe_portfolio:.4f}")
        print(f"Sharpe Ratio (Benchmark): {sharpe_benchmark:.4f}")
        print(f"Sharpe Ratio (Growth Index): {sharpe_growth:.4f}")
        print(f"Sharpe Ratio (Value Index): {sharpe_value:.4f}")
        if portfolio_beta is not None:
            print(f"Portfolio Beta: {portfolio_beta:.4f}")
        if portfolio_beta_growth is not None:
            print(f"Beta (Portfolio vs Growth Index): {portfolio_beta_growth:.4f}")
        if portfolio_beta_value is not None:
            print(f"Beta (Portfolio vs Value Index): {portfolio_beta_value:.4f}")
        print(f"Max Drawdown (Portfolio): {max_drawdown_portfolio:.2%}")
        print(f"Max Drawdown (Benchmark): {max_drawdown_benchmark:.2%}")
        print(f"Max Drawdown (Growth Index): {max_drawdown_growth:.2%}")
        print(f"Max Drawdown (Value Index): {max_drawdown_value:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Win Rate (vs Growth Index): {win_rate_growth:.2%}")
        print(f"Win Rate (vs Value Index): {win_rate_value:.2%}")

    return {
        'final_value': aum,
        'yearly_returns': portfolio_returns,
        'benchmark_returns': bench_list,
        'growth_benchmark_returns': growth_bench_list,
        'value_benchmark_returns': value_bench_list,
        'years': years,
        'portfolio_values': portfolio_values,
        'portfolio_beta': portfolio_beta,
        'portfolio_beta_growth': portfolio_beta_growth,
        'portfolio_beta_value': portfolio_beta_value,
        'max_drawdown_portfolio': max_drawdown_portfolio,
        'max_drawdown_benchmark': max_drawdown_benchmark,
        'max_drawdown_growth': max_drawdown_growth,
        'max_drawdown_value': max_drawdown_value,
        'sharpe_portfolio': sharpe_portfolio,
        'sharpe_benchmark': sharpe_benchmark,
        'sharpe_growth': sharpe_growth,
        'sharpe_value': sharpe_value,
        'vol_raw_portfolio': vol_raw_portfolio,
        'vol_raw_benchmark': vol_raw_benchmark,
        'vol_raw_growth': vol_raw_growth,
        'vol_raw_value': vol_raw_value,
        'vol_excess_portfolio': vol_excess_p,
        'vol_excess_benchmark': vol_excess_b,
        'vol_excess_growth': vol_excess_growth,
        'vol_excess_value': vol_excess_value,
        'win_rate': win_rate,
        'win_rate_growth': win_rate_growth,
        'win_rate_value': win_rate_value,
        'risk_free_rate_source': risk_free_rate_source,
        'yearly_comparisons': yearly_comparisons,
        'information_ratio': information_ratio,
        'information_ratio_growth': information_ratio_growth,
        'information_ratio_value': information_ratio_value,
        'data_frequency': 'Yearly'
    }
