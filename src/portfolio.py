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
    Manages a collection of equity investments and handles position state.
    
    This class serves as the primary container for portfolio constituents during 
    the rebalancing process. It provides a structured interface for the backtest 
    engine to track share counts and ticker allocations across factor cohorts.
    """

    def __init__(self, name: str, investments: Optional[List[Dict[str, Any]]] = None):
        """
        Initializes the portfolio state.
        """
        self.name = name
        self.investments = investments if investments is not None else []

    def add_investment(self, ticker: str, n_shares: float) -> None:
        """
        Appends a new position to the portfolio constituent list.
        """
        self.investments.append({
            'ticker': ticker, 
            'number_of_shares': n_shares
        })

    def remove_investment(self, ticker: str) -> None:
        """
        Liquidates all holdings of a specific security.
        """
        self.investments = [
            inv for inv in self.investments if inv['ticker'] != ticker
        ]

    def present_value(self, market_data: pd.DataFrame) -> float:
        """
        Calculates the aggregate market value of the portfolio.

        Note: Uses 'Ending_Price' to align with the 2026 Supabase schema.
        """
        if not self.investments or market_data.empty:
            return 0.0

        # Create a temporary series for vectorized valuation
        holdings = pd.DataFrame(self.investments).set_index('ticker')['number_of_shares']
        
        # Determine correct price column (handle potential underscores)
        price_col = 'Ending_Price' if 'Ending_Price' in market_data.columns else 'Ending Price'
        
        if price_col not in market_data.columns:
            return 0.0

        # Align prices with holdings and compute dot product
        prices = market_data[price_col].reindex(holdings.index).fillna(0)
        return float((holdings * prices).sum())

    def calculate_return(self, t1_value: float, t2_value: float) -> float:
        """
        Computes the arithmetic return between two valuation points.
        """
        if t1_value == 0:
            return 0.0
        
        return ((t2_value - t1_value) / t1_value) * 100

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports the constituent list to a DataFrame for UI visualization.
        """
        if not self.investments:
            return pd.DataFrame(columns=['ticker', 'number_of_shares'])
        return pd.DataFrame(self.investments)