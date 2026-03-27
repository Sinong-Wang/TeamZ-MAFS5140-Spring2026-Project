from data_feed import DataFeed
from engine import BacktestEngine
from evaluator import Evaluator
from strategy import Strategy 

def main():
    # 1. Define the path to the dataset
    data_path = "D:/coding_experiment/TeamZ-MAFS5140-Spring2026-Project/validation.parquet" 
    
    try:
        # 2. Initialize components
        print("Loading data...")
        feed = DataFeed(data_path)
        
        print("Initializing strategy...")
        strategy = Strategy()
        
        engine = BacktestEngine(data_feed=feed, strategy=strategy)
        
        # 3. Run the backtest
        portfolio_returns = engine.run()
        
        # 4. Evaluate the results
        evaluator = Evaluator(portfolio_returns, periods_per_year=252*78)
        evaluator.generate_report()

    except Exception as e:
        print(f"\n[BACKTEST FAILED] {type(e).__name__}: {e}")
        print("Please fix the error in your strategy and try again.")

if __name__ == "__main__":
    main()