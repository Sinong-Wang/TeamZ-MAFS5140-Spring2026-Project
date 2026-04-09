# MAFS5140 Project Framework

Welcome to the final project for the Quantitative Finance course! This repository contains the event-driven backtesting framework you will use to develop, test, and evaluate your trading strategies. 

## 📂 Project Structure

The framework is divided into five main Python scripts. **You only need to modify one of them.**

*   **`strategy.py`**: **(YOUR WORKSPACE)** This is where you will implement your trading logic. It contains the `Strategy` base class.
*   **`data_feed.py`**: Handles loading the historical market data (from a `.parquet` file) and feeding it to the engine one timestamp at a time.
*   **`engine.py`**: The core backtest loop. It simulates the passage of time, calls your strategy to get target weights, calculates portfolio returns, and strictly enforces trading rules.
*   **`evaluator.py`**: Computes standard performance metrics (Cumulative Return, Annualized Return, Volatility, Sharpe Ratio, Max Drawdown) based on your strategy's return history.
*   **`main.py`**: The execution script. Run this file to test your strategy locally and view your performance report and any error messages.

## 🛠️ How to Build Your Strategy

You will write your code entirely within the `Strategy` class inside `strategy.py`. 

### The `step` Function
The engine will call your `step(self, current_market_data)` function at every timestamp. 
*   **Input (`current_market_data`)**: A Pandas DataFrame containing the market snapshot of all assets for the current timestamp. 
   * Index = Tickers
   * Columns = fields (e.g. `close`, `volume`)
*   **Output**: You must return a Pandas Series of target portfolio weights. (Index = Tickers, Values = Weights).

### State Management
Because the `step` function only receives a snapshot of the *current* market data, you must use the `__init__(self)` method to initialize any variables (like lists or dataframes) if you need to store historical prices/volumes or compute rolling indicators (e.g., moving averages).

### ⚠️ Trading Rules & Constraints
The `engine.py` will strictly validate your output at every step. If you violate these rules, the backtest will instantly fail and throw an error:
1.  **No Short Selling**: Every individual weight must be $\ge 0\$.
2.  **No Leverage**: The sum of your weights must be $\le 1.0$. 
3.  **Cash Handling**: If the sum of your weights is less than 1.0, the engine assumes the remaining portion is held in cash. Cash earns a 0% return.

## 🚀 How to Run and Test

1. Ensure you have the provided dataset in your project directory.
2. Open your terminal or command prompt.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python main.py
