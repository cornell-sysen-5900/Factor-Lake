from scipy import stats
from .market_object import MarketObject
from .portfolio import Portfolio
import numpy as np
import pandas as pd
from .factors_doc import FACTOR_DOCS
from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list

def calculate_holdings(factor, aum, market, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False, enforce_liquidity=False, liquidity_volume_col='vol', liquidity_participation_rate=1.0):
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
    
    def _cap_shares_by_liquidity(ticker, requested_shares):
        if not enforce_liquidity:
            return requested_shares
        if liquidity_volume_col not in market.stocks.columns:
            return requested_shares
        try:
            daily_volume = market.stocks.loc[ticker, liquidity_volume_col]
            if isinstance(daily_volume, (pd.Series, np.ndarray)):
                if hasattr(daily_volume, 'dropna') and not daily_volume.dropna().empty:
                    daily_volume = daily_volume.dropna().iloc[0]
                else:
                    daily_volume = daily_volume.iloc[0] if hasattr(daily_volume, 'iloc') and len(daily_volume) > 0 else None
            daily_volume = pd.to_numeric(daily_volume, errors='coerce')
            if pd.isna(daily_volume) or daily_volume <= 0:
                return 0.0
            max_trade_shares = float(daily_volume) * max(0.0, min(float(liquidity_participation_rate), 1.0))
            return max(0.0, min(float(requested_shares), max_trade_shares))
        except Exception:
            return requested_shares

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
                    shares = _cap_shares_by_liquidity(ticker, shares)
                    if shares > 0:
                         portfolio_new.add_investment(ticker, shares)
            # If for some reason no shares were added (e.g., rounding), fallback to equal among priced tickers
            if not portfolio_new.investments and prices:
                valid_tickers = list(prices.keys())
                equal_investment = aum / len(valid_tickers)
                for t in valid_tickers:
                    shares = equal_investment / prices[t]
                    shares = _cap_shares_by_liquidity(t, shares)
                    if shares > 0:
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
                        shares = _cap_shares_by_liquidity(ticker, shares)
                        if shares > 0:
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
                    shares = _cap_shares_by_liquidity(ticker, shares)
                    if shares > 0:
                        portfolio_new.add_investment(ticker, shares)

    return portfolio_new

def calculate_growth(portfolio, current_market, next_market=None, verbosity=0):
    """
    Calculates portfolio growth for a single period.
    
    When next_market is provided (POC mode): uses price-to-price returns.
    When next_market is None (Legacy mode): uses pre-computed Next_Year_Return column.
    """
    total_weighted_rtn = 0
    total_investment_value = 0

    for factor_portfolio in portfolio:
        for inv in factor_portfolio.investments:
            ticker = inv["ticker"]
            
            entry_price = current_market.get_price(ticker)
            
            if next_market is not None:
                # --- POC mode: price-to-price returns ---
                exit_price = next_market.get_price(ticker)
                
                if entry_price and entry_price > 0:
                    if exit_price and exit_price > 0:
                        stock_rtn = (exit_price / entry_price) - 1
                    else:
                        stock_rtn = 0.0  # Assume capital returned near last active price
                    
                    position_value = inv["number_of_shares"] * entry_price
                    total_weighted_rtn += stock_rtn * position_value
                    total_investment_value += position_value
            else:
                # --- Legacy mode: use Next_Year_Return column ---
                if entry_price and entry_price > 0:
                    try:
                        stock_rtn = current_market.stocks.loc[ticker, "Next_Year_Return"] / 100
                    except KeyError:
                        stock_rtn = 0.0
                    
                    position_value = inv["number_of_shares"] * entry_price
                    total_weighted_rtn += stock_rtn * position_value
                    total_investment_value += position_value

    growth = total_weighted_rtn / total_investment_value if total_investment_value > 0 else 0
    total_end_value = total_investment_value * (1 + growth)
    
    return growth, total_investment_value, total_end_value

def rebalance_portfolio(
    data,
    factors,
    start_year,
    end_year,
    initial_aum,
    benchmark_index=1,
    verbosity=0,
    restrict_fossil_fuels=False,
    top_pct=10,
    which='top',
    use_market_cap_weight=False,
    factor_directions=None,
    frequency='Yearly',
    data_mode='poc',
    enforce_liquidity=False,
    liquidity_volume_col='vol',
    liquidity_participation_rate=1.0
):
    """
    Executes a multi-year backtest. 
    Calculates Sharpe and Beta using year-specific excess returns.

    Args:
        factor_directions: optional dict mapping factor column_name -> 'top'/'bottom'.
            When provided, overrides the global ``which`` on a per-factor basis.
    """
    aum = initial_aum
    
    verbosity = 0 if verbosity is None else verbosity
    risk_free_rate_source = "FRED 4 Week T-Bill (Oct 1)"

    if data_mode == 'legacy':
        # ====================================================================
        # LEGACY MODE: Year-based loop using Next_Year_Return column
        # ====================================================================
        years = [str(start_year)]
        portfolio_returns = []
        portfolio_values = [aum]

        for year in range(start_year, end_year):
            market = MarketObject(data.loc[data['Year'] == year], year)
            yearly_portfolio = []

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
                    enforce_liquidity=enforce_liquidity,
                    liquidity_volume_col=liquidity_volume_col,
                    liquidity_participation_rate=liquidity_participation_rate
                )
                yearly_portfolio.append(factor_portfolio)

            # Legacy: calculate_growth with next_market=None uses Next_Year_Return
            growth, total_start_value, total_end_value = calculate_growth(yearly_portfolio, market, None, verbosity)

            if verbosity >= 2:
                print(f"Year {year} to {year + 1}: Growth: {growth:.2%}, Start: ${total_start_value:.2f}, End: ${total_end_value:.2f}")

            aum = total_end_value
            portfolio_returns.append(growth)
            portfolio_values.append(aum)
            years.append(str(year + 1))

        # Legacy benchmarks from benchmarks.py (returns are in %, convert to decimals)
        rf_list = get_benchmark_list(4, start_year, end_year)
        bench_list = [r / 100 for r in get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)]

        portfolio_returns_np = np.array(portfolio_returns)
        rf_rates_np = np.array(rf_list)
        benchmark_returns_np = np.array(bench_list)

    else:
        # ====================================================================
        # POC MODE: Date-based loop using price-to-price returns (UNCHANGED)
        # ====================================================================
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])

        if frequency == 'Monthly':
            dates = data.groupby([data['Date'].dt.year, data['Date'].dt.month])['Date'].max().values
        elif frequency == 'Quarterly':
            dates = data.groupby([data['Date'].dt.year, data['Date'].dt.quarter])['Date'].max().values
        else:  # Yearly
            dates = data.groupby(data['Date'].dt.year)['Date'].max().values
            
        dates = pd.to_datetime(dates)
        start_dt = pd.Timestamp(year=start_year, month=1, day=1)
        end_dt = pd.Timestamp(year=end_year, month=12, day=31)
        
        dates = dates[(dates >= start_dt) & (dates <= end_dt)]
        dates = sorted(dates.unique())
        
        if len(dates) < 2:
            raise ValueError("Not enough overlapping dates to run a backtest. Please widen your date bounds.")
            
        time_labels = [d.strftime('%Y-%m-%d') for d in dates]
        years = time_labels
        portfolio_returns = []  
        portfolio_values = [aum]  
        benchmark_returns = []

        # --- 1. Rebalancing Loop ---
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i+1]
            
            market = MarketObject(data[data['Date'] == current_date], current_date)
            next_market = MarketObject(data[data['Date'] == next_date], next_date)
            
            yearly_portfolio = []

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
                    enforce_liquidity=enforce_liquidity,
                    liquidity_volume_col=liquidity_volume_col,
                    liquidity_participation_rate=liquidity_participation_rate
                )
                yearly_portfolio.append(factor_portfolio)

            # Calculate period growth for the invested portion
            _, total_start_value, total_end_value = calculate_growth(yearly_portfolio, market, next_market, verbosity)

            # Retain cash that was not allocated to any valid stocks
            uninvested_cash = aum - total_start_value
            new_aum = total_end_value + uninvested_cash
            
            actual_growth = (new_aum / aum) - 1 if aum > 0 else 0

            # POC benchmark return (Market-Cap Weighted Universe)
            try:
                entry_prices = market.stocks['Ending Price']
                entry_prices = entry_prices[~entry_prices.index.duplicated(keep='first')]
                
                mcaps = market.stocks.get('Market Capitalization', None)
                if mcaps is not None:
                    mcaps = mcaps[~mcaps.index.duplicated(keep='first')]
                
                exit_prices = next_market.stocks['Ending Price']
                exit_prices = exit_prices[~exit_prices.index.duplicated(keep='first')]
                
                exit_prices = exit_prices.reindex(entry_prices.index)
                returns = (exit_prices / entry_prices) - 1
                returns = returns.fillna(0.0)
                valid_returns = returns[entry_prices > 0]
                
                if mcaps is not None:
                    valid_mcaps = mcaps.reindex(valid_returns.index).fillna(0.0)
                    total_mcap = valid_mcaps.sum()
                    if total_mcap > 0:
                        bench_ret = (valid_returns * valid_mcaps).sum() / total_mcap
                    else:
                        bench_ret = valid_returns.mean()
                else:
                    bench_ret = valid_returns.mean() if not valid_returns.empty else 0.0
            except Exception as e:
                if verbosity >= 2:
                    print(f"Benchmark eval exception: {e}")
                bench_ret = 0.0
                
            benchmark_returns.append(bench_ret)

            if verbosity >= 2:
                print(f"{current_date.date()} to {next_date.date()}: Growth: {actual_growth:.2%}, Start: ${aum:.2f}, End: ${new_aum:.2f} (Uninvested: ${uninvested_cash:.2f}) | Bench: {bench_ret:.2%}")

            aum = new_aum  
            portfolio_returns.append(actual_growth)
            portfolio_values.append(aum)

        # --- 2. Data Alignment (Benchmarks & Risk-Free) ---
        rf_list = get_benchmark_list(4, start_year, end_year)
        if len(rf_list) > len(portfolio_returns):
            rf_list = rf_list[:len(portfolio_returns)]
        elif len(rf_list) < len(portfolio_returns):
            rf_list = rf_list + [rf_list[-1] if rf_list else 0] * (len(portfolio_returns) - len(rf_list))

        bench_list = benchmark_returns

        portfolio_returns_np = np.array(portfolio_returns)
        rf_rates_np = np.array(rf_list) / 100 
        benchmark_returns_np = np.array(bench_list)
    
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
    benchmark_returns = np.array(benchmark_returns)

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
