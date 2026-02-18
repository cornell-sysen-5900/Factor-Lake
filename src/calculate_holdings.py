from scipy import stats
from .market_object import MarketObject
from .portfolio import Portfolio
import numpy as np
import pandas as pd
from .factors_doc import FACTOR_DOCS
from .factor_utils import normalize_series

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

def rebalance_portfolio(data, factors, start_year, end_year, initial_aum, verbosity=0, restrict_fossil_fuels=False, top_pct=10, which='top', use_market_cap_weight=False):
    aum = initial_aum
    years = [start_year] # Start with the initial year
    portfolio_returns = []  # Store yearly returns for Information Ratio
    benchmark_returns = []  # Store benchmark returns for comparison
    portfolio_values = [aum]  # track total AUM over time
    
    # Ensure verbosity is not None
    verbosity = 0 if verbosity is None else verbosity
    
    # Risk-free rate lookup from FRED (October 1)
    risk_free_rate_lookup = {
        2002: 0.0156, 2003: 0.0102, 2004: 0.0182, 2005: 0.0349, 2006: 0.0473,
        2007: 0.0462, 2008: 0.0148, 2009: 0.0014, 2010: 0.0014, 2011: 0.0010,
        2012: 0.0015, 2013: 0.0005, 2014: 0.0002, 2015: 0.0001, 2016: 0.0033,
        2017: 0.0110, 2018: 0.0220, 2019: 0.0179, 2020: 0.0009, 2021: 0.0003, 2022: 0.0306
    }
    risk_free_rate_source = "FRED (Oct 1)"

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

 
        # No need for next_market anymore; all data is in the current 'market' object
        growth, total_start_value, total_end_value = calculate_growth(yearly_portfolio, market, verbosity)

        if verbosity is not None and verbosity >= 2:
            print(f"Year {year} to {year + 1}: Growth: {growth:.2%}, "
                    f"Start Value: ${total_start_value:.2f}, End Value: ${total_end_value:.2f}")

        aum = total_end_value  # Liquidate and reinvest

        # Append annual return (growth) to portfolio_returns
        portfolio_returns.append(growth)

        # Get benchmark return for the year (replace it as needed)
        benchmark_return = get_benchmark_return(year)  # Define this function based on benchmark data
        benchmark_returns.append(benchmark_return)
        portfolio_values.append(aum)  # add current AUM to the list

        years.append(year+1) #adding next year to match portfolio_values

    
    if verbosity is not None and verbosity >= 1:

        print("\n==== Final Summary ====")
        print(f"Initial Portfolio Value: ${initial_aum:.2f}")
        # Calculate overall growth
        overall_growth = (aum - initial_aum) / initial_aum if initial_aum else 0
        print(f"Final Portfolio Value after {end_year}: ${aum:.2f}")
        print(f"Overall Growth from {start_year} to {end_year}: {overall_growth * 100:.2f}%")
        print(f"\n==== Performance Metrics ====")

    #backtest stats 
    portfolio_returns_np = np.array(portfolio_returns)
    benchmark_returns_np = np.array(benchmark_returns) / 100
    active_returns = portfolio_returns_np - benchmark_returns_np

    annualized_return = (np.prod(1 + portfolio_returns_np))**(1 / len(portfolio_returns_np)) - 1
    annualized_volatility = np.std(portfolio_returns_np, ddof=1) * np.sqrt(1)  # yearly data
    active_volatility = np.std(active_returns, ddof=1)

    if verbosity is not None and verbosity >= 1:
        print(f"Annualized Return (Portfolio): {annualized_return:.2%}")
        print(f"Annualized Volatility (Portfolio): {annualized_volatility:.2%}")
        print(f"Active Volatility (Portfolio vs Benchmark): {active_volatility:.2%}")

    # Calculate Information Ratio
    information_ratio = calculate_information_ratio(portfolio_returns, benchmark_returns, verbosity)
    if information_ratio is None and verbosity is not None and verbosity >= 1:
        print("Information Ratio could not be calculated due to zero tracking error.")
    
    # Calculate max drawdown for portfolio
    cumulative_values = np.array(portfolio_values)
    running_peak = np.maximum.accumulate(cumulative_values)
    drawdowns = (cumulative_values - running_peak) / running_peak
    max_drawdown_portfolio = np.min(drawdowns)
    
    # Calculate max drawdown for benchmark
    benchmark_values = [initial_aum]
    for ret in benchmark_returns:
        benchmark_values.append(benchmark_values[-1] * (1 + ret / 100))
    benchmark_values = np.array(benchmark_values)
    benchmark_peak = np.maximum.accumulate(benchmark_values)
    benchmark_drawdowns = (benchmark_values - benchmark_peak) / benchmark_peak
    max_drawdown_benchmark = np.min(benchmark_drawdowns)
    
    # Calculate Sharpe ratios
    sharpe_portfolio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    benchmark_annualized = (np.prod(1 + benchmark_returns_np))**(1 / len(benchmark_returns_np)) - 1
    benchmark_volatility = np.std(benchmark_returns_np, ddof=1)
    sharpe_benchmark = benchmark_annualized / benchmark_volatility if benchmark_volatility > 0 else 0

    # ── Portfolio Beta vs Russell 2000 ──
    portfolio_beta = None
    if len(portfolio_returns_np) >= 3:
        coeffs = np.polyfit(benchmark_returns_np, portfolio_returns_np, 1)
        portfolio_beta = float(coeffs[0])

    # Calculate yearly win rate
    yearly_wins = 0
    yearly_comparisons = []
    for i, (p_ret, b_ret) in enumerate(zip(portfolio_returns, benchmark_returns)):
        p_pct = p_ret * 100
        b_pct = b_ret
        win = p_pct > b_pct
        if win:
            yearly_wins += 1
        yearly_comparisons.append({
            'year': years[i+1],
            'portfolio_return': p_pct,
            'benchmark_return': b_pct,
            'win': win
        })
    win_rate = yearly_wins / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
    
    if verbosity is not None and verbosity >= 1:
        print(f"\n==== Advanced Backtest Stats ====")
        print(f"Risk-Free Rate Source: {risk_free_rate_source}")
        print(f"Portfolio Beta (vs Russell 2000): {portfolio_beta:.4f}" if portfolio_beta is not None else "Portfolio Beta: N/A")
        print(f"Max Drawdown (Portfolio): {max_drawdown_portfolio * 100:.2f}%")
        print(f"Max Drawdown (Benchmark): {max_drawdown_benchmark * 100:.2f}%")
        print(f"Sharpe Ratio (Portfolio): {sharpe_portfolio:.4f}")
        print(f"Sharpe Ratio (Benchmark): {sharpe_benchmark:.4f}")
        print(f"Yearly Win Rate vs Benchmark: {win_rate * 100:.2f}%")
        print(f"\n==== Yearly Win/Loss Comparison ====")
        for comp in yearly_comparisons:
            print(f"Year {comp['year']}: Portfolio={comp['portfolio_return']:.2f}%, Benchmark={comp['benchmark_return']:.2f}%, Win={comp['win']}")
        
    return {
        'final_value': aum,
        'yearly_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
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
    
def get_benchmark_return(year):
    """
    This function should return the benchmark return for the given year.
    """
    # Data from Factset (September)
    benchmark_data = {
        2002: 34.62, 2003: 17.48, 2004: 16.56, 2005: 8.65, 2006: 11.01,
        2007: -15.63, 2008: -11.08, 2009: 11.89, 2010: -4.73, 2011: 30.01,
        2012: 28.22, 2013: 2.6, 2014: -0.09, 2015: 13.71, 2016: 19.11,
        2017: 13.8, 2018: -10.21, 2019: -1.03, 2020: 46.21, 2021: -24.48, 2022: 7.23
    }
    return benchmark_data.get(year, 0)

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
