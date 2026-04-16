from pathlib import Path

from data_feed import DataFeed
from engine import BacktestEngine
from evaluator import Evaluator
from strategy import Strategy


def _resolve_data_path() -> Path:
    """Resolve validation parquet path relative to project root."""
    project_root = Path(__file__).resolve().parent

    fixed_candidates = [
        project_root / "test.parquet",
        project_root / "validation.parquet",
        project_root / "testparquet" / "test.parquet",
        project_root / "data" / "test.parquet",
    ]
    for candidate in fixed_candidates:
        if candidate.exists():
            return candidate

    # If TA places files under testparquet/, accept any parquet file there.
    test_dir = project_root / "testparquet"
    if test_dir.exists() and test_dir.is_dir():
        parquet_files = sorted(test_dir.glob("*.parquet"))
        if parquet_files:
            return parquet_files[0]

    expected = "\n".join(str(p) for p in fixed_candidates)
    raise FileNotFoundError(
        "Cannot find validation parquet file. Please put it at one of:\n"
        f"{expected}\n"
        "or under <project_root>/testparquet/*.parquet"
    )

def main():
    # 1. Resolve dataset path in a machine-independent way
    data_path = _resolve_data_path()
    try:
        # 2. Initialize components
        print("Loading data...")
        print(f"Using data file: {data_path}")
        feed = DataFeed(str(data_path))
        
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