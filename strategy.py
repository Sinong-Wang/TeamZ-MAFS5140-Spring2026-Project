import pandas as pd

"""
STUDENT INSTRUCTIONS:
1. This Strategy class is where you will implement your own trading strategy.
2. The current implementation is just a SIMPLE EXAMPLE (Moving Average Trend Following) 
   provided for your reference. Please modify this class to build your own strategy.
3. You may create new Python scripts and import them into this file if you 
   want to organize your code. 
4. IMPORTANT: Do NOT modify any other existing scripts in the backtest 
   framework. Changing core engine files may break the backtester and cause 
   evaluation errors.
"""

class Strategy:
    def __init__(self):
        """
        Initialize state for a simple intraday cross-sectional strategy.

        IMPORTANT LIMITATION:
        - The framework passes only `current_market_data` into `step()`.
        - The timestamp is not available inside this class.
        - We therefore approximate day boundaries with a fixed 78 bars/day
          counter. This works for normal full trading days.
        - On abnormal short days in the dataset, exact end-of-day flattening is
          not possible from `strategy.py` alone.
        """
        self.prev_close = None
        self.close_history = []
        self.held_weights = None
        self.bars_left_in_hold = 0
        self.bar_in_day = 0
        self.bars_per_day = 78
        self.top_k = 1
        self.skip_open_bars = 0
        self.lookback_bars = 1
        self.hold_bars = 1
        self.min_volume_rank = 0.0
        self.mode = "reversal"

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Trade a cross-sectional reversal or momentum signal.

        - `reversal`: buy the `top_k` names with the worst last-bar returns.
        - `momentum`: buy the `top_k` names with the best last-bar returns.
        - `skip_open_bars`: hold cash for the first N assumed bars each day.
        - `lookback_bars`: signal horizon used to rank assets.
        - `hold_bars`: number of bars to keep a selected basket.
        - `min_volume_rank`: optional same-bar liquidity filter in [0, 1].
        - The final assumed bar of each normal day is held in cash.
        """

        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_close = current_market_data["close"]
        weights = pd.Series(0.0, index=current_close.index)

        self.bar_in_day += 1
        if self.bar_in_day > self.bars_per_day:
            self.bar_in_day = 1
            self.prev_close = None
            self.close_history = []
            self.held_weights = None
            self.bars_left_in_hold = 0

        # Go flat on the final assumed bar of a normal trading day.
        if self.bar_in_day == self.bars_per_day:
            self.prev_close = current_close
            self.close_history.append(current_close)
            if len(self.close_history) > self.lookback_bars + 1:
                self.close_history.pop(0)
            self.held_weights = weights
            self.bars_left_in_hold = 0
            return weights

        if self.prev_close is None:
            self.prev_close = current_close
            self.close_history = [current_close]
            self.held_weights = weights
            self.bars_left_in_hold = 0
            return weights

        if self.top_k <= 0:
            self.prev_close = current_close
            self.held_weights = weights
            self.bars_left_in_hold = 0
            return weights

        if self.skip_open_bars < 0:
            raise ValueError("skip_open_bars must be >= 0.")

        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be >= 1.")

        if self.hold_bars <= 0:
            raise ValueError("hold_bars must be >= 1.")

        if not 0.0 <= self.min_volume_rank <= 1.0:
            raise ValueError("min_volume_rank must be between 0.0 and 1.0.")

        self.close_history.append(current_close)
        if len(self.close_history) > self.lookback_bars + 1:
            self.close_history.pop(0)

        # Bar 1 stores the opening reference price. Skipping N opening bars means
        # we also stay in cash for bars 2 through N+1.
        if self.bar_in_day <= self.skip_open_bars + 1:
            self.prev_close = current_close
            self.held_weights = weights
            self.bars_left_in_hold = 0
            return weights

        if self.bars_left_in_hold > 0 and self.held_weights is not None:
            self.bars_left_in_hold -= 1
            self.prev_close = current_close
            return self.held_weights.copy()

        if len(self.close_history) < self.lookback_bars + 1:
            self.prev_close = current_close
            self.held_weights = weights
            self.bars_left_in_hold = 0
            return weights

        signal_return = self.close_history[-1] / self.close_history[-1 - self.lookback_bars] - 1.0

        eligible_mask = pd.Series(True, index=current_close.index)
        if self.min_volume_rank > 0.0:
            volume_rank = current_market_data["volume"].rank(pct=True)
            eligible_mask = volume_rank >= self.min_volume_rank

        eligible_returns = signal_return[eligible_mask]
        if eligible_returns.empty:
            self.prev_close = current_close
            self.held_weights = weights
            self.bars_left_in_hold = 0
            return weights

        if self.mode == "reversal":
            selected = eligible_returns.nsmallest(self.top_k).index
        elif self.mode == "momentum":
            selected = eligible_returns.nlargest(self.top_k).index
        else:
            raise ValueError("mode must be either 'reversal' or 'momentum'.")

        if len(selected) > 0:
            weights.loc[selected] = 1.0 / len(selected)

        self.held_weights = weights.copy()
        self.bars_left_in_hold = self.hold_bars - 1
        self.prev_close = current_close
        return weights
