"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/portfolio.py
PURPOSE: Object-oriented representation of a managed investment portfolio.
VERSION: 2.0.0
"""

import pandas as pd
from typing import List, Dict, Optional, Any

class Portfolio:
    """
    Manages a collection of equity investments and calculates aggregate performance metrics.
    
    This class serves as a container for holding positions and provides methods 
    for valuation and return attribution. It is designed to interface with 
    standardized DataFrames for high-speed backtesting.
    """

    def __init__(self, name: str, investments: Optional[List[Dict[str, Any]]] = None):
        """
        Initializes the portfolio with a unique identifier and an optional set of assets.

        Args:
            name (str): The label for the portfolio (e.g., 'Factor_Strategy').
            investments (list, optional): A list of dictionaries containing 'ticker' 
                                          and 'number_of_shares'.
        """
        self.name = name
        self.investments = investments if investments is not None else []

    def add_investment(self, ticker: str, n_shares: float):
        """
        Appends a new position to the portfolio.

        Args:
            ticker (str): The symbol of the security.
            n_shares (float): The quantity of shares acquired.
        """
        self.investments.append({
            'ticker': ticker, 
            'number_of_shares': n_shares
        })

    def remove_investment(self, ticker: str):
        """
        Liquidates all holdings of a specific security from the portfolio.

        Args:
            ticker (str): The symbol of the security to be removed.
        """
        self.investments = [
            inv for inv in self.investments if inv['ticker'] != ticker
        ]

    def present_value(self, market_data: pd.DataFrame) -> float:
        """
        Calculates the current market value of all held positions using a DataFrame.

        This method expects a DataFrame indexed by Ticker, containing an 'Ending Price' 
        column. This is significantly faster than the previous row-by-row lookup.

        Args:
            market_data (pd.DataFrame): Dataframe with ticker symbols as index.

        Returns:
            float: The aggregate dollar value of the portfolio.
        """
        total_value = 0.0
        for inv in self.investments:
            ticker = inv['ticker']
            if ticker in market_data.index:
                price = market_data.loc[ticker, 'Ending Price']
                if pd.notna(price):
                    total_value += price * inv['number_of_shares']
        return total_value

    def calculate_return(self, t1_value: float, t2_value: float) -> float:
        """
        Computes the percentage return between two valuation points.

        Args:
            t1_value (float): The beginning period value (Basis).
            t2_value (float): The ending period value (Market Value).

        Returns:
            float: The percentage change in value.

        Raises:
            ValueError: If the initial value is zero.
        """
        if t1_value == 0:
            raise ValueError('Initial portfolio value (t1_value) cannot be zero.')
        
        return ((t2_value - t1_value) / t1_value) * 100

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the current holdings into a Pandas DataFrame for UI display.

        Returns:
            pd.DataFrame: A table of tickers and their respective share counts.
        """
        if not self.investments:
            return pd.DataFrame(columns=['ticker', 'number_of_shares'])
        return pd.DataFrame(self.investments)