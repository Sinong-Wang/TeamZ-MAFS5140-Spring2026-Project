from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from lgb_data_pipeline import build_feat_df, wide_to_long_with_mo
from project_paths import project_root

"""
Way2: 使用 `lgb_exp_way2_init_model.ipynb` 保存的单一原生 Booster
`artifacts/lgb_way2_booster.pkl`（init_model 续训）。

这里严格复用 notebook 同源的特征管线：每个 bar 基于“当日截至当前”的宽表，
调用 `wide_to_long_with_mo` + `build_feat_df` 生成特征，再按 notebook 方式清洗后预测。
"""


class Strategy:
    def __init__(self):
        """
        LGBM way2（单 Booster）+ 截面 Top-1 多头；日末最后一根 bar 平仓。
        """
        self.bars_per_day = 78
        self.bar_in_day = 0
        self._last_bar_date: Optional[date] = None
        self._synthetic_day = 0
        self.top_k = 1

        self.booster: Any = None
        self.feature_cols: List[str] = []
        self.best_iteration: Optional[int] = None
        self._model_ok = False

        self._session_rows: List[Dict[str, object]] = []
        self._session_tickers: List[str] = []

        self._load_model()
        if not self._model_ok:
            import sys

            print(
                "[Strategy way2] 模型未加载（将全程空仓，收益为 0）。"
                "请检查 artifacts/lgb_way2_booster.pkl；"
                "切换方式请设环境变量 LGB_STRATEGY_MODE=way2（仅终端，无需改 main）。",
                file=sys.stderr,
            )

    def _load_model(self) -> None:
        try:
            import joblib  # type: ignore
        except Exception:
            self._model_ok = False
            return

        path = project_root() / "artifacts" / "lgb_way2_booster.pkl"
        if not path.exists():
            self._model_ok = False
            return

        try:
            bundle = joblib.load(path)
            self.booster = bundle.get("booster", None)
            self.feature_cols = list(bundle.get("feature_cols", []))
            bi = getattr(self.booster, "best_iteration", None) if self.booster is not None else None
            if bi is None:
                bi = bundle.get("best_iteration", None)
            self.best_iteration = int(bi) if bi is not None else None
            self._model_ok = self.booster is not None and len(self.feature_cols) > 0
        except Exception:
            self._model_ok = False

    def _safe_float(self, x) -> Optional[float]:
        try:
            if x is None:
                return None
            v = float(x)
            if not np.isfinite(v):
                return None
            return v
        except Exception:
            return None

    def _reset_session(self) -> None:
        self._session_rows = []
        self._session_tickers = []

    def _advance_clock(self, timestamp) -> Tuple[pd.Timestamp, bool]:
        if timestamp is not None:
            ts = pd.Timestamp(timestamp)
            d = ts.date()
            is_new_session = self._last_bar_date is None or d != self._last_bar_date
            if is_new_session:
                self.bar_in_day = 1
                self._last_bar_date = d
            else:
                self.bar_in_day += 1
                if self.bar_in_day > self.bars_per_day:
                    self.bar_in_day = 1
            return ts, is_new_session

        self.bar_in_day += 1
        is_new_session = False
        if self.bar_in_day > self.bars_per_day:
            self.bar_in_day = 1
            self._synthetic_day += 1
            is_new_session = True
        ts = pd.Timestamp("2000-01-01") + pd.Timedelta(days=self._synthetic_day, minutes=5 * (self.bar_in_day - 1))
        return ts, is_new_session or not self._session_rows

    def _append_current_bar(self, current_market_data: pd.DataFrame, ts: pd.Timestamp) -> None:
        row: Dict[str, object] = {"datetime": ts.to_pydatetime()}
        for ticker in current_market_data.index:
            ticker_str = str(ticker)
            if ticker_str not in self._session_tickers:
                self._session_tickers.append(ticker_str)
            row[f"{ticker_str}_close"] = self._safe_float(current_market_data.loc[ticker, "close"])
            row[f"{ticker_str}_volume"] = self._safe_float(current_market_data.loc[ticker, "volume"])
        self._session_rows.append(row)

    def _prepare_feature_frame(self, tickers_now: Sequence[str], ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        if not self._session_rows or not self.feature_cols:
            return None

        tickers_use = [t for t in self._session_tickers if t in set(tickers_now)]
        if not tickers_use:
            return None

        pairs = {
            t: {
                "close": f"{t}_close",
                "volume": f"{t}_volume",
            }
            for t in tickers_use
        }

        try:
            df_w = pl.from_dicts(self._session_rows)
            long_df = wide_to_long_with_mo(df_w, pairs, tickers_use)
            feat_df = build_feat_df(long_df).sort(["ticker", "datetime"])
        except Exception:
            return None

        for c in self.feature_cols:
            if c not in feat_df.columns:
                feat_df = feat_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

        try:
            clean_df = feat_df.with_columns([
                pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
                for c in self.feature_cols
            ])
            clean_df = clean_df.with_columns([
                pl.col(c).fill_null(pl.col(c).median().over("ticker")).alias(c)
                for c in self.feature_cols
            ])
            clean_df = clean_df.with_columns([
                pl.col(c).fill_null(pl.col(c).median()).fill_null(0.0).cast(pl.Float32).alias(c)
                for c in self.feature_cols
            ])
            latest_df = (
                clean_df
                .filter(pl.col("datetime") == ts.to_pydatetime())
                .select(["ticker"] + self.feature_cols)
            )
        except Exception:
            return None

        if latest_df.height == 0:
            return None

        return latest_df.to_pandas().set_index("ticker")[self.feature_cols]

    def step(self, current_market_data: pd.DataFrame, timestamp=None) -> pd.Series:
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")
        if "volume" not in current_market_data.columns:
            current_market_data = current_market_data.copy()
            current_market_data["volume"] = 0.0

        weights = pd.Series(0.0, index=current_market_data.index, dtype=float)
        ticker_map = {str(t): t for t in current_market_data.index}

        ts, is_new_session = self._advance_clock(timestamp)
        if is_new_session:
            self._reset_session()
        self._append_current_bar(current_market_data, ts)

        if self.bar_in_day == self.bars_per_day:
            return weights

        if not self._model_ok or self.booster is None:
            return weights

        feat_df = self._prepare_feature_frame(list(ticker_map.keys()), ts)
        if feat_df is None or feat_df.empty:
            return weights

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

        try:
            if self.best_iteration is not None:
                preds = self.booster.predict(feat_df.values, num_iteration=self.best_iteration)
            else:
                preds = self.booster.predict(feat_df.values)
        except Exception:
            return weights

        pred_s = pd.Series(preds, index=feat_df.index).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
        selected = pred_s.nlargest(self.top_k).index.tolist()
        selected_labels = [ticker_map[t] for t in selected if t in ticker_map]

        if selected_labels:
            weights.loc[selected_labels] = 1.0 / float(len(selected_labels))

        return weights
