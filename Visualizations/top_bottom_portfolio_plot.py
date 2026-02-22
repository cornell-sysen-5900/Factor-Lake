import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.market_object import MarketObject
import math
import pandas as pd

# reuse normalization used elsewhere to match selection logic
from src.factor_utils import normalize_series

# Respect per-factor direction (higher_is_better) from the docs
from src.factors_doc import FACTOR_DOCS

# Optionally compute baseline portfolio values using calculate_holdings.rebalance_portfolio
try:
    from src.calculate_holdings import rebalance_portfolio
except Exception:
    rebalance_portfolio = None


def plot_top_bottom_percent(rdata,
                            factors,
                            years,
                            percent=10,
                            show_bottom=True,
                            restrict_fossil_fuels=False,
                            benchmark_returns=None,
                            benchmark_label='Russell 2000',
                            initial_investment=None,
                            require_all_factors=True,
                            verbose=True,
                            drop_missing_next_price=True,
                            selection_mode='by_factor',
                            return_details=False,
                            weight_mode='equal',
                            show_percent_guides=True,
                            baseline_portfolio_values=None,
                            baseline_pct=10,
                            use_rebalance_for_selection=True):
    """
    Plot dollar-invested growth for the top-N% and optionally bottom-N% portfolios
    constructed from a list of factors each year, alongside a benchmark.

        Notes:
            - Ranks are normalized per-factor to [0..1] where 1 is most attractive.
                The factor direction is read from `factors_doc.FACTOR_DOCS` via the
                factor's `column_name` attribute (`higher_is_better`). If a factor has
                `higher_is_better == False` it will be inverted so lower raw values are
                treated as more attractive.
            - Defaults are chosen to be deterministic and robust: `selection_mode`
                defaults to 'by_factor' (this mirrors `calculate_holdings`' equal-
                allocation-across-factors behavior) and `drop_missing_next_price`
                defaults to True to avoid zero-return substitution bias caused by
                carrying forward entry prices for missing exits.
            - When `drop_missing_next_price=True` tickers without a next-year price
                are excluded from that year's realized returns. The allocation logic
                below will reallocate the factor's dollars equally among the remaining
                valid tickers so the portfolio is always fully invested (subject to
                available valid tickers).
    """

    percent = int(percent)
    if percent < 1:
        percent = 1
    if percent > 100:
        percent = 100

    if percent == 100:
        show_bottom = False

    # default to a realistic AUM so plots show dollar scales (matches upstream n_graph)
    if initial_investment is None:
        initial_investment = 1000000.0

    # thresholds for helpful runtime warnings (small cohorts)
    MIN_COHORT_WARNING = 3

    # Inline selection only supports 'by_factor' mode now. The canonical
    # rebalance-driven selection (use_rebalance_for_selection=True) is the
    # default and will call `calculate_holdings.rebalance_portfolio` which
    # applies normalize_series(...) and the factor direction consistently.

    # compute_raw_combined_scores removed: inline selection uses by_factor only

    top_values = [initial_investment]
    bottom_values = [initial_investment] if show_bottom else None

    # Detailed diagnostics collected per-year when requested. Structure:
    # { 'years': [..], 'per_year': [ { year, combined_scores, top: {positions, avg_return,start,end}, bottom: {...} } ] }
    details = {'years': [], 'per_year': []} if return_details else None

    # Optionally compute top/bottom series using the full rebalance backtest logic
    skip_inline_selection = False
    if use_rebalance_for_selection and rebalance_portfolio is not None:
        try:
            start_year = years[0]
            end_year = years[-1]
            # Top series using rebalance logic with user-selected percent
            res_top = rebalance_portfolio(rdata, factors, start_year, end_year, initial_investment, verbosity=0, restrict_fossil_fuels=restrict_fossil_fuels, top_pct=percent, which='top', use_market_cap_weight=st.session_state.use_cap_weight)
            top_values = res_top.get('portfolio_values', [initial_investment])
            # Bottom series using rebalance logic with user-selected percent
            if show_bottom:
                res_bot = rebalance_portfolio(rdata, factors, start_year, end_year, initial_investment, verbosity=0, restrict_fossil_fuels=restrict_fossil_fuels, top_pct=percent, which='bottom', use_market_cap_weight=st.session_state.use_cap_weight)
                bottom_values = res_bot.get('portfolio_values', [initial_investment])
            skip_inline_selection = True
        except Exception:
            # fallback to inline selection logic below
            skip_inline_selection = False
        # If caller requested diagnostics, build a concise final-year diagnostics
        # when using the rebalance-driven selection. The rebalance result
        # contains portfolio dollar series but not per-year selection details,
        # so inspect the market for the final rebalancing year to compute
        # n_selected and dropped counts while using the rebalance dollar
        # series for start/end values.
        if skip_inline_selection and return_details:
            try:
                if len(years) >= 2:
                    diag_year = years[-2]
                    diag_next = years[-1]
                else:
                    diag_year = years[0]
                    diag_next = years[0]

                market = MarketObject(rdata.loc[rdata['Year'] == diag_year], diag_year)
                next_market = MarketObject(rdata.loc[rdata['Year'] == diag_next], diag_next)

                def compute_cohort_stats(is_top: bool):
                    universe = 0
                    n_selected = 0
                    dropped = 0
                    for factor in factors:
                        col = getattr(factor, 'column_name', str(factor))
                        if col in market.stocks.columns:
                            series = pd.to_numeric(market.stocks[col], errors='coerce')
                            grouped = series.groupby(series.index).mean().dropna()
                            higher_is_better = FACTOR_DOCS.get(col, {}).get('higher_is_better', True)
                            items = []
                            for t, v in grouped.items():
                                try:
                                    val = float(v)
                                except Exception:
                                    continue
                                score = val if higher_is_better else -val
                                items.append((t, score))
                        else:
                            items = []
                            for t in market.stocks.index:
                                try:
                                    v = factor.get(t, market)
                                except Exception:
                                    v = None
                                if v is None:
                                    continue
                                try:
                                    items.append((t, float(v)))
                                except Exception:
                                    continue
                        items = sorted(items, key=lambda x: (x[1], x[0]))
                        universe += len(items)
                        n = max(1, math.floor(len(items) * (percent / 100.0))) if items else 0
                        n_selected += n
                        if n:
                            sel = [t for t, _ in (items[-n:] if is_top else items[:n])]
                            # count dropped due to missing next-year prices (if configured)
                            valid = []
                            for t in sel:
                                entry = market.get_price(t)
                                if entry is None or entry <= 0:
                                    continue
                                exit_price = next_market.get_price(t)
                                if exit_price is None and drop_missing_next_price:
                                    continue
                                valid.append(t)
                            dropped += (len(sel) - len(valid))
                    return {'n_selected': n_selected, 'dropped': dropped}

                top_stats = compute_cohort_stats(True)
                bot_stats = compute_cohort_stats(False) if show_bottom else None

                top_start = top_end = None
                bot_start = bot_end = None
                try:
                    if isinstance(res_top, dict) and 'portfolio_values' in res_top:
                        tv = list(res_top.get('portfolio_values', []))
                        if len(tv) >= 2:
                            top_start = tv[-2]
                            top_end = tv[-1]
                        elif len(tv) == 1:
                            top_start = tv[0]
                            top_end = tv[0]
                except Exception:
                    pass
                try:
                    if show_bottom and isinstance(res_bot, dict) and 'portfolio_values' in res_bot:
                        bv = list(res_bot.get('portfolio_values', []))
                        if len(bv) >= 2:
                            bot_start = bv[-2]
                            bot_end = bv[-1]
                        elif len(bv) == 1:
                            bot_start = bv[0]
                            bot_end = bv[0]
                except Exception:
                    pass

                details = {'years': [diag_year], 'per_year': []}
                per_year = {
                    'year': diag_year,
                    'combined_scores': [],
                    'top': {
                        'positions': [],
                        'avg_return': None,
                        'start': top_start,
                        'end': top_end,
                        'dropped': top_stats['dropped'],
                        'n_selected': top_stats['n_selected'],
                    },
                    'bottom': {
                        'positions': [],
                        'avg_return': None,
                        'start': bot_start,
                        'end': bot_end,
                        'dropped': bot_stats['dropped'] if bot_stats else 0,
                        'n_selected': bot_stats['n_selected'] if bot_stats else 0,
                    }
                }
                details['per_year'].append(per_year)
            except Exception:
                # keep details as empty dict in case of failure
                details = {'years': [], 'per_year': []}

    # initialize variables used by both inline and rebalance-driven selection paths
    sorted_combined_scores = []
    top_positions = []
    bottom_positions = []
    start_top = end_top = start_bottom = end_bottom = 0.0
    top_returns = []
    bottom_returns = []
    top_dropped = bot_dropped = 0
    top_tickers = []
    bottom_tickers = []
    universe_size_top = None
    universe_size_bot = None
    n_top = 0
    n_bot = 0
    top_factor_stats = []
    bottom_factor_stats = []
    year = None

    if not skip_inline_selection:
        for i in range(len(years) - 1):
            year = years[i]
            next_year = years[i + 1]

            market = MarketObject(rdata.loc[rdata['Year'] == year], year)
            next_market = MarketObject(rdata.loc[rdata['Year'] == next_year], next_year)

            # initialize per-year per-factor stats containers so verbose printing is safe
            top_factor_stats = []
            bottom_factor_stats = []

            # Inline path supports only 'by_factor' selection; rebalance path is
            # the canonical choice and is used by default. We do not compute
            # combined raw scores here.

            # Top cohort
            start_top = 0.0
            end_top = 0.0
            top_returns = []
            top_positions = []
            top_dropped = 0
            # aggregate list for verbose printing (populated in both selection modes)
            top_tickers = []
            universe_size_top = 0
            n_top = 0
            

            top_values.append(end_top)

            # Bottom cohort
            universe_size_bot = None
            n_bot = 0
            bottom_tickers = []
            bot_dropped = 0
            # initialize these so verbose diagnostics don't reference possibly-unbound names
            start_bottom = 0.0
            end_bottom = 0.0
            bottom_returns = []
            bottom_positions = []
            if show_bottom:
                assert bottom_values is not None
                if selection_mode != 'by_factor':
                    raise ValueError(f"Unknown selection_mode: {selection_mode}. Inline selection only supports 'by_factor'.")

                n_factors = max(1, len(factors))
                per_factor_alloc_b = bottom_values[-1] / n_factors
                universe_size_bot = 0
                n_bot = 0
                # per-factor diagnostics for bottom
                bottom_factor_stats = []
                for factor in factors:
                    col = getattr(factor, 'column_name', str(factor))
                    if col in market.stocks.columns:
                        # aggregate multiple samples per ticker by averaging non-null samples
                        series = pd.to_numeric(market.stocks[col], errors='coerce')
                        grouped = series.groupby(series.index).mean()
                        grouped = grouped.dropna()
                        higher_is_better = FACTOR_DOCS.get(col, {}).get('higher_is_better', True)
                        items = []
                        for t, v in grouped.items():
                            try:
                                val = float(v)
                            except Exception:
                                continue
                            score = val if higher_is_better else -val
                            items.append((t, score))
                    else:
                        items = []
                        for t in market.stocks.index:
                            try:
                                v = factor.get(t, market)
                            except Exception:
                                v = None
                            if v is None:
                                continue
                            try:
                                items.append((t, float(v)))
                            except Exception:
                                continue
                    # stable sort: worst->best with deterministic tie-breaker
                    items = sorted(items, key=lambda x: (x[1], x[0]))  # worst->best
                    universe_size_bot += len(items)
                    n = max(1, math.floor(len(items) * (percent / 100.0))) if items else 0
                    n_bot += n
                    bot_list = [t for t, _ in items[:n]]
                    # collect tickers for verbose diagnostics
                    bottom_tickers.extend(bot_list)
                    if bot_list:
                        # pre-filter valid tickers so per-factor allocation is fully invested
                        valid = []
                        for t in bot_list:
                            entry = market.get_price(t)
                            if entry is None or entry <= 0:
                                continue
                            exit_price = next_market.get_price(t)
                            if exit_price is None:
                                if drop_missing_next_price:
                                    continue
                                exit_price = entry
                            valid.append((t, entry, exit_price))
                        dropped = len(bot_list) - len(valid)
                        bot_dropped += dropped
                        bottom_factor_stats.append((col, len(bot_list), len(valid), dropped))
                        if valid:
                            equal = per_factor_alloc_b / len(valid)
                            for t, entry, exit_price in valid:
                                shares = equal / entry
                                start_bottom += shares * entry
                                end_bottom += shares * exit_price
                                try:
                                    r = (exit_price / entry) - 1.0
                                except Exception:
                                    r = 0.0
                                bottom_returns.append(r)
                                bottom_positions.append({'ticker': t, 'entry': entry, 'exit': exit_price, 'shares': shares, 'weight': equal, 'return': r})
                if end_bottom == 0:
                    end_bottom = bottom_values[-1]
                bottom_values.append(end_bottom)

            # Verbose diagnostics if requested + some deterministic warnings (small cohorts / dropped tickers)
            if verbose:
                print(f"Year {year}: universe_size={universe_size_top}, top_n={n_top}")
            else:
                # still print small-cohort warnings even when not verbose
                pass

            if show_bottom and verbose:
                print(f"Year {year}: universe_size={universe_size_bot}, bottom_n={n_bot}")

            if top_tickers:
                if verbose:
                    print("  Top sample:", top_tickers[:10])
                # per-factor stats if available
                try:
                    if verbose and 'top_factor_stats' in locals() and top_factor_stats:
                        print("  Top per-factor (factor, selected, valid, dropped):")
                        for fcol, sel, valid_c, dropped in top_factor_stats:
                            print(f"    {fcol}: selected={sel}, valid={valid_c}, dropped={dropped}")
                except Exception:
                    pass
                try:
                    avg_top_r = sum(top_returns) / len(top_returns) if top_returns else 0.0
                except Exception:
                    avg_top_r = 0.0
                if verbose:
                    print(f"  Top avg next-year return: {avg_top_r*100:.2f}% | start ${start_top:.2f} end ${end_top:.2f}")
                    # report how many tickers were dropped due to missing prices (helpful diagnostic)
                    if top_dropped:
                        print(f"  Top dropped tickers this year (missing prices): {top_dropped}")
                    if n_top and n_top < MIN_COHORT_WARNING:
                        print(f"  Warning: Top cohort size is small ({n_top}); results will be noisy.")

            if show_bottom and bottom_tickers:
                if verbose:
                    print("  Bottom sample:", bottom_tickers[:10])
                # per-factor stats for bottom if available
                try:
                    if verbose and 'bottom_factor_stats' in locals() and bottom_factor_stats:
                        print("  Bottom per-factor (factor, selected, valid, dropped):")
                        for fcol, sel, valid_c, dropped in bottom_factor_stats:
                            print(f"    {fcol}: selected={sel}, valid={valid_c}, dropped={dropped}")
                except Exception:
                    pass
            try:
                avg_bot_r = sum(bottom_returns) / len(bottom_returns) if bottom_returns else 0.0
            except Exception:
                avg_bot_r = 0.0
            if verbose:
                print(f"  Bottom avg next-year return: {avg_bot_r*100:.2f}% | start ${start_bottom:.2f} end ${end_bottom:.2f}")
                if bot_dropped:
                    print(f"  Bottom dropped tickers this year (missing prices): {bot_dropped}")
                if n_bot and n_bot < MIN_COHORT_WARNING:
                    print(f"  Warning: Bottom cohort size is small ({n_bot}); results will be noisy.")
        # Collect structured details for this year if requested
        try:
            avg_top_r = sum(top_returns) / len(top_returns) if top_returns else 0.0
        except Exception:
            avg_top_r = 0.0
        try:
            avg_bot_r = sum(bottom_returns) / len(bottom_returns) if bottom_returns else 0.0
        except Exception:
            avg_bot_r = 0.0

        if details is not None:
            per_year = {
                'year': year,
                'combined_scores': sorted_combined_scores,
                'top': {
                    'positions': top_positions,
                    'avg_return': avg_top_r,
                    'start': start_top,
                    'end': end_top,
                    'dropped': top_dropped,
                    'n_selected': n_top,
                },
                'bottom': {
                    'positions': bottom_positions,
                    'avg_return': avg_bot_r,
                    'start': start_bottom,
                    'end': end_bottom,
                    'dropped': bot_dropped,
                    'n_selected': n_bot,
                }
            }
            details['years'].append(year)
            details['per_year'].append(per_year)

    # === ROBUST BENCHMARK ALIGNMENT ===
    benchmark_values = None
    if benchmark_returns is not None:
        br = list(benchmark_returns)
        benchmark_values = [initial_investment]
        
        # Years: [2000, 2001, ..., 2022] (Length N)
        # Returns: [36.5, 18.76, ...] (Length N-1)
        for i in range(len(years) - 1):
            if i < len(br):
                # We know these are percentages (e.g. 36.5)
                # Divide by 100 to get the multiplier (1.365)
                ret_decimal = float(br[i]) / 100.0
                next_val = benchmark_values[-1] * (1 + ret_decimal)
                benchmark_values.append(next_val)
            else:
                # Fallback to keep array lengths equal
                benchmark_values.append(benchmark_values[-1])

    # Plotting check
    if benchmark_values is not None:
        plt.plot(years, benchmark_values, marker='s', linestyle='--', color='r', 
                 label=benchmark_label, linewidth=1.2)

    # === DEBUG PRINT (Check terminal/console) ===
    # print(f"DEBUG: Years count: {len(years)}, Bench values count: {len(benchmark_values)}")
                
            
    # Plot
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    # Plot top cohort
    plt.plot(years, top_values, marker='o', linestyle='-', color='g', label=f'Top {percent}%', linewidth=1.6, markersize=6)
    if show_bottom and bottom_values is not None:
        # Plot bottom cohort normally so both series start at the same initial investment
        # and the bottom line will go down when the cohort loses value (e.g., < initial_investment).
        plt.plot(years, bottom_values, marker='o', linestyle='-', color='m', label=f'Bottom {percent}%', linewidth=1.6, markersize=6)
    if benchmark_values is not None and len(benchmark_values) == len(years):
        plt.plot(years, benchmark_values, marker='s', linestyle='--', color='r', label=benchmark_label, linewidth=1.2)

    # Optionally plot a baseline portfolio growth series (from portfolio_growth_plot)
    if baseline_portfolio_values is None and rebalance_portfolio is not None:
        try:
            # compute baseline portfolio values using the same factors and rdata
            start_year = years[0]
            end_year = years[-1]
            res = rebalance_portfolio(rdata, factors, start_year, end_year, initial_investment, verbosity=0, restrict_fossil_fuels=restrict_fossil_fuels, top_pct=baseline_pct)
            candidate = res.get('portfolio_values') if isinstance(res, dict) else None
            if candidate:
                baseline_portfolio_values = candidate
        except Exception:
            baseline_portfolio_values = None

    if baseline_portfolio_values is not None:
        try:
            bp = list(baseline_portfolio_values)
            if len(bp) == len(years):
                plt.plot(years, bp, marker='o', linestyle='-', color='b', label='Portfolio', linewidth=1.6, markersize=6)
            else:
                common_len = min(len(years), len(bp))
                plt.plot(years[:common_len], bp[:common_len], marker='o', linestyle='-', color='b', label='Portfolio', linewidth=1.6, markersize=6)
        except Exception:
            pass

    try:
        factor_set_name = ", ".join([str(f) for f in factors])
    except Exception:
        factor_set_name = "Selected Factors"

    plt.title(f"Top/Bottom {percent}% Portfolios ({factor_set_name}) vs {benchmark_label}")
    plt.xlabel('Year')
    plt.ylabel('Dollar Invested ($)')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    # ensure x-ticks are the years provided
    try:
        ax.set_xticks(years)
    except Exception:
        plt.xticks(years)
    # format y-axis as dollars
    try:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))
    except Exception:
        pass
    # show initial investment baseline for reference
    try:
        plt.axhline(initial_investment, color='k', linestyle=':', linewidth=0.8)
    except Exception:
        pass
    # Optionally draw horizontal guide lines at the cohorts' final percent returns (converted to dollars)
    # if show_percent_guides:
    #     try:
    #         # final values
    #         top_final = top_values[-1]
    #         top_pct = (top_final / initial_investment - 1.0) * 100.0
    #         plt.axhline(top_final, color='g', linestyle='--', linewidth=0.8)
    #         # annotate percentage on the right
    #         plt.text(years[-1], top_final, f"  {top_pct:+.1f}%", va='center', color='g')
    #     except Exception:
    #         pass
    #     if show_bottom and bottom_values is not None:
    #         try:
    #             bot_final = bottom_values[-1]
    #             bot_pct = (bot_final / initial_investment - 1.0) * 100.0
    #             plt.axhline(bot_final, color='m', linestyle='--', linewidth=0.8)
    #             plt.text(years[-1], bot_final, f"  {bot_pct:+.1f}%", va='center', color='m')
    #         except Exception:
    #             pass

    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Return the figure for Streamlit instead of calling plt.show()
    fig = plt.gcf()  # Get current figure
    
    # end of function
    if details is not None:
        return details
    return fig
