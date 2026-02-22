from scipy import stats
from .market_object import MarketObject
from .portfolio import Portfolio
import numpy as np
import pandas as pd
from .factors_doc import FACTOR_DOCS
from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list

def calculate_holdings(factor, aum, market, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False):
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

def rebalance_portfolio(data, factors, start_year, end_year, initial_aum, benchmark_index=1, verbosity=0, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False):
    """
    Executes a multi-year backtest. 
    Calculates Sharpe and Beta using year-specific excess returns.
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
            factor_portfolio = calculate_holdings(
                factor=factor,
                aum=aum / len(factors),
                market=market,
                restrict_fossil_fuels=restrict_fossil_fuels,
                top_pct=top_pct,
                which=which,
                use_market_cap_weight=use_market_cap_weight
            )
            yearly_portfolio.append(factor_portfolio)

        # Calculate annual growth
        growth, total_start_value, total_end_value = calculate_growth(yearly_portfolio, market, verbosity)

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

    # Convert to NumPy for performance math
    portfolio_returns_np = np.array(portfolio_returns)
    rf_rates_np = np.array(rf_list)
    benchmark_returns_np = np.array(bench_list) / 100 
    
# --- 3. Excess Return Calculations ---
    excess_portfolio_returns = portfolio_returns_np - rf_rates_np
    excess_benchmark_returns = benchmark_returns_np - rf_rates_np
    
    # --- 4. Sharpe Ratio ---
    # Volatility of the EXCESS returns (Institutional Standard)
    vol_excess_p = np.std(excess_portfolio_returns, ddof=1)
    vol_excess_b = np.std(excess_benchmark_returns, ddof=1)

    # Formula: Mean(Yearly Excess Returns) / Volatility(Yearly Excess Returns)
    sharpe_portfolio = np.mean(excess_portfolio_returns) / vol_excess_p if vol_excess_p > 0 else 0
    sharpe_benchmark = np.mean(excess_benchmark_returns) / vol_excess_b if vol_excess_b > 0 else 0
    
    # --- 5. Portfolio Beta ---
    portfolio_beta = None
    if len(portfolio_returns_np) >= 3:
        # CAPM Beta using excess returns
        coeffs = np.polyfit(excess_benchmark_returns, excess_portfolio_returns, 1)
        portfolio_beta = float(coeffs[0])

    # --- 6. Additional Risk Metrics ---
    # Information Ratio
    information_ratio = calculate_information_ratio(portfolio_returns, bench_list, verbosity)
    
    # Portfolio Max Drawdown
    cumulative_values = np.array(portfolio_values)
    running_peak = np.maximum.accumulate(cumulative_values)
    max_drawdown_portfolio = np.min((cumulative_values - running_peak) / running_peak)
    
    # Benchmark Max Drawdown
    benchmark_cumulative = [initial_aum]
    for r in benchmark_returns_np:
        benchmark_cumulative.append(benchmark_cumulative[-1] * (1 + r))
    
    benchmark_cumulative_np = np.array(benchmark_cumulative)
    benchmark_peak = np.maximum.accumulate(benchmark_cumulative_np)
    max_drawdown_benchmark = np.min((benchmark_cumulative_np - benchmark_peak) / benchmark_peak)

    # Yearly Win Rate
    wins = 0
    yearly_comparisons = []
    for i, (p_ret, b_ret) in enumerate(zip(portfolio_returns_np, benchmark_returns_np)):
        win = p_ret > b_ret
        if win: wins += 1
        yearly_comparisons.append({
            'year': years[i+1],
            'portfolio_return': p_ret * 100,
            'benchmark_return': b_ret * 100,
            'win': win
        })
    win_rate = wins / len(portfolio_returns_np) if len(portfolio_returns_np) > 0 else 0

    # --- 7. Summary Output ---
    if verbosity >= 1:
        print(f"\n==== Performance Metrics ({start_year}-{end_year}) ====")
        print(f"Final AUM: ${aum:.2f}")
        print(f"Sharpe Ratio (Portfolio): {sharpe_portfolio:.4f}")
        print(f"Sharpe Ratio (Benchmark): {sharpe_benchmark:.4f}")
        if portfolio_beta is not None:
            print(f"Portfolio Beta: {portfolio_beta:.4f}")
        print(f"Max Drawdown (Portfolio): {max_drawdown_portfolio:.2%}")
        print(f"Max Drawdown (Benchmark): {max_drawdown_benchmark:.2%}")
        print(f"Win Rate: {win_rate:.2%}")

    return {
        'final_value': aum,
        'yearly_returns': portfolio_returns,
        'benchmark_returns': bench_list,
        'years': years,
        'portfolio_values': portfolio_values,
        'portfolio_beta': portfolio_beta,
        'max_drawdown_portfolio': max_drawdown_portfolio,
        'max_drawdown_benchmark': max_drawdown_benchmark,
        'sharpe_portfolio': sharpe_portfolio,
        'sharpe_benchmark': sharpe_benchmark,
        'win_rate': win_rate,
        'risk_free_rate_source': risk_free_rate_source,
        'yearly_comparisons': yearly_comparisons,
        'information_ratio': information_ratio
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
    tracking_error = np.std(active_returns, ddof=1)  # Use sample std deviation

    # Prevent division by zero
    if tracking_error == 0:
        return None  # Or return float('nan') to indicate undefined IR
    
    # Compute Information Ratio
    information_ratio = mean_active_return / tracking_error
    if verbosity >=1:
        print(f"Information Ratio: {information_ratio:.4f}")
    return information_ratio
