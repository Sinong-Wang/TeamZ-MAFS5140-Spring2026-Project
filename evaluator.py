import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        """
        periods_per_year: 252 for daily data, 252*390 for minute data, etc.
        """
        # Drop NaNs to ensure accurate period counting and calculations
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year

    def cumulative_return(self) -> float:
        return (1 + self.returns).prod() - 1.0

    def cagr(self) -> float:
        """
        Calculates the Geometric Annualized Return (CAGR).
        This is the standard metric for reporting overall portfolio growth.
        """
        cum_ret = self.cumulative_return()
        # Use .count() instead of len() to safely ignore any missing data
        num_periods = self.returns.count() 
        
        if num_periods == 0:
            return 0.0
            
        return (1 + cum_ret) ** (self.periods_per_year / num_periods) - 1.0

    def annualized_volatility(self) -> float:
        # pandas .std() uses Bessel's correction (ddof=1) by default
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the annualized Sharpe Ratio.
        Mathematically, this must use the Arithmetic Mean, not the CAGR.
        """
        ann_vol = self.annualized_volatility()
        if ann_vol == 0:
            return 0.0
            
        # Calculate Arithmetic Annualized Return for the Sharpe numerator
        arithmetic_ann_ret = self.returns.mean() * self.periods_per_year
        
        return (arithmetic_ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(self) -> float:
        """
        Calculates the maximum drawdown.
        Prepends 1.0 to cumulative wealth to capture drawdowns starting from the very first period.
        """
        cumulative_wealth = (1 + self.returns).cumprod()
        
        # Prepend 1.0 to represent the initial portfolio value before any trading
        wealth_with_initial = pd.concat([pd.Series([1.0]), cumulative_wealth], ignore_index=True)
        
        rolling_max = wealth_with_initial.cummax()
        drawdowns = (wealth_with_initial - rolling_max) / rolling_max
        
        return drawdowns.min()

    def generate_report(self):
        """
        Prints and returns a dictionary of all computed metrics.
        """
        metrics = {
            "Cumulative Return": f"{self.cumulative_return() * 100:.4f}%",
            "Annualized Volatility": f"{self.annualized_volatility() * 100:.4f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio():.4f}",
            "Max Drawdown": f"{self.max_drawdown() * 100:.4f}%"
        }
        
        print("\n--- Strategy Performance Report ---")
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        print("-----------------------------------")
        
        return metrics
