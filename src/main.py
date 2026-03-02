from .market_object import load_data
from .calculate_holdings import rebalance_portfolio
from .user_input import get_factors
from .verbosity_options import get_verbosity_level
from .fossil_fuel_restriction import get_fossil_fuel_restriction
from .supabase_input import get_supabase_preference, get_data_loading_verbosity
from .sector_selection import get_sector_selection
from Visualizations.portfolio_growth_plot import plot_portfolio_growth
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ### Ask about fossil fuel restriction first ###
    restrict_fossil_fuels = get_fossil_fuel_restriction()  # Prompt user (Yes/No)

    ### Ask about data source ###
    use_supabase = get_supabase_preference()
    
    # Ask about data loading verbosity
    show_loading = get_data_loading_verbosity()
    
    # Sector selection
    selected_sectors = get_sector_selection()

    # Load market data
    rdata = load_data(
        restrict_fossil_fuels=restrict_fossil_fuels, 
        use_supabase=use_supabase,
        show_loading_progress=show_loading,
        sectors=selected_sectors
    )

    ### Data preprocessing ###
    # Note: Fossil fuel filtering is applied later in calculate_holdings() for each year
    rdata['Ticker'] = rdata['Ticker-Region'].dropna().apply(lambda x: x.split('-')[0].strip())
    rdata['Year'] = pd.to_datetime(rdata['Date']).dt.year

    available_factors = [
        'ROE using 9/30 Data', 'ROA using 9/30 Data', '12-Mo Momentum %',
        '6-Mo Momentum %', '1-Mo Momentum %', 'Price to Book Using 9/30 Data',
        'Next FY Earns/P', '1-Yr Price Vol %', 'Accruals/Assets', 'ROA %',
        '1-Yr Asset Growth %', '1-Yr CapEX Growth %', 'Book/Price'
    ]
    
    # Only select columns that actually exist
    cols_to_keep = ['Ticker', 'Year']
    if 'Ending Price' in rdata.columns:
        cols_to_keep.append('Ending Price')
    elif 'Ending_Price' in rdata.columns:
        rdata['Ending Price'] = rdata['Ending_Price']
        cols_to_keep.append('Ending Price')
    
    for factor in available_factors:
        if factor in rdata.columns:
            cols_to_keep.append(factor)
    
    rdata = rdata[cols_to_keep]

    ### Get user selections ###
    factors = get_factors(available_factors)
    verbosity_level = get_verbosity_level()

    # Separate factor objects from their names for use downstream
    factor_objects, factor_names = (zip(*factors) if factors else ([], []))

    ### Rebalancing portfolio across years ###
    # ...existing code...
    results = rebalance_portfolio(
        rdata, list(factor_objects),
        start_year=2002, end_year=2023,
        initial_aum=1,
        verbosity=verbosity_level,
        restrict_fossil_fuels=restrict_fossil_fuels
    )
    
    # Plot portfolio growth
    plot_portfolio_growth(
        years=results['years'],
        portfolio_values=results['portfolio_values'],
        selected_factors=list(factor_names),
        restrict_fossil_fuels=restrict_fossil_fuels,
        benchmark_returns=results.get('benchmark_returns'),
        benchmark_label='Russell 2000',
        initial_investment=results.get('portfolio_values', [None])[0]
    )


if __name__ == "__main__":
    main()
