from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
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

class _TickerState:
    def __init__(self, maxlen: int):
        self.close: Deque[float] = deque(maxlen=maxlen)
        self.volume: Deque[float] = deque(maxlen=maxlen)
        self.ret_5m: Deque[float] = deque(maxlen=maxlen)
        self.prev_vol_chg: Optional[float] = None


class Strategy:
    def __init__(self):
        """
        LGBM-driven intraday cross-sectional strategy (5-minute bars).

        What it does each bar:
        - Build online features from rolling close/volume history.
        - Predict next-bar return for each ticker using a pre-trained LightGBM model.
        - Long the Top-K tickers (equal weight).
        - Go flat on the final assumed bar of each day to avoid overnight exposure.

        IMPORTANT LIMITATION (framework):
        - `step()` receives only `current_market_data` (no timestamp).
        - We approximate day boundaries with a fixed 78 bars/day counter.
          This works for normal full trading days; exact early-close handling is not
          possible from `strategy.py` alone.
        """

        # --- Intraday / risk controls ---
        self.bars_per_day = 78
        self.bar_in_day = 0
        self.top_k = 15

        # --- Rolling window config (5-min bars) ---
        self.bars_per_hour = 12
        self.w_1h = 1 * self.bars_per_hour       # 12
        self.w_4h = 4 * self.bars_per_hour       # 48
        self.w_8h = 8 * self.bars_per_hour       # 96
        self.w_24h = 24 * self.bars_per_hour     # 288
        self.w_3d = 3 * 24 * self.bars_per_hour  # 864
        self._maxlen = self.w_3d + 2
        self._eps = 1e-12

        # --- Per-ticker state ---
        self.state: Dict[str, _TickerState] = {}

        # --- Model bundle ---
        self.model = None
        self.feature_cols: List[str] = []
        self.best_iteration: Optional[int] = None
        self._model_ok = False

        self._load_model()

    def _load_model(self) -> None:
        """
        Load the LightGBM model saved from `lgb_exp.ipynb`.
        Expected bundle keys: model, feature_cols, best_iteration
        """
        try:
            import joblib  # type: ignore
        except Exception:
            self._model_ok = False
            return

        model_path = Path(__file__).resolve().parent / "artifacts" / "lgb_model_5m.pkl"
        if not model_path.exists():
            self._model_ok = False
            return

        try:
            bundle = joblib.load(model_path)
            self.model = bundle.get("model", None)
            self.feature_cols = list(bundle.get("feature_cols", []))
            bi = bundle.get("best_iteration", None)
            self.best_iteration = int(bi) if bi is not None else None
            self._model_ok = self.model is not None and len(self.feature_cols) > 0
        except Exception:
            self._model_ok = False

    def _get_state(self, ticker: str) -> _TickerState:
        st = self.state.get(ticker)
        if st is None:
            st = _TickerState(maxlen=self._maxlen)
            self.state[ticker] = st
        return st

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

    def _rolling_mean_std(self, arr: Deque[float], window: int) -> Tuple[Optional[float], Optional[float]]:
        if len(arr) < window:
            return None, None
        x = np.asarray(list(arr)[-window:], dtype=float)
        if x.size == 0:
            return None, None
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=0))
        if not np.isfinite(mu):
            mu = None
        if not np.isfinite(sd):
            sd = None
        return mu, sd

    def _rolling_max(self, arr: Deque[float], window: int) -> Optional[float]:
        if len(arr) < window:
            return None
        x = np.asarray(list(arr)[-window:], dtype=float)
        if x.size == 0:
            return None
        m = float(np.nanmax(x))
        return m if np.isfinite(m) else None

    def _rolling_corr(self, a: Deque[float], b: Deque[float], window: int) -> Optional[float]:
        if len(a) < window or len(b) < window:
            return None
        x = np.asarray(list(a)[-window:], dtype=float)
        y = np.asarray(list(b)[-window:], dtype=float)
        if x.size == 0 or y.size == 0:
            return None
        if np.nanstd(x) < self._eps or np.nanstd(y) < self._eps:
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        return float(c) if np.isfinite(c) else None

    def _compute_features_for_ticker(self, ticker: str) -> Dict[str, float]:
        """
        Compute a subset of notebook features online.
        Any missing feature required by the model will be filled with 0 later.
        """
        st = self.state[ticker]
        feats: Dict[str, float] = {}

        if len(st.close) < 2 or len(st.volume) < 2:
            return feats

        close_now = st.close[-1]
        close_prev = st.close[-2]
        vol_now = st.volume[-1]
        vol_prev = st.volume[-2]

        # --- returns ---
        ret_5m = close_now / (close_prev + self._eps) - 1.0
        feats["return_5m"] = float(ret_5m)

        if len(st.close) > self.w_1h:
            feats["return_1h"] = float(close_now / (st.close[-1 - self.w_1h] + self._eps) - 1.0)
        if len(st.close) > self.w_4h:
            feats["return_4h"] = float(close_now / (st.close[-1 - self.w_4h] + self._eps) - 1.0)
        if len(st.close) > self.w_24h:
            feats["return_24h"] = float(close_now / (st.close[-1 - self.w_24h] + self._eps) - 1.0)

        # --- volume change ---
        vol_chg = (vol_now - vol_prev) / (vol_prev + self._eps) if vol_prev is not None else 0.0
        feats["volume_change_rate"] = float(vol_chg)
        if st.prev_vol_chg is None or abs(vol_chg) < self._eps:
            feats["volume_change_rate_change_rate"] = 0.0
        else:
            feats["volume_change_rate_change_rate"] = float((vol_chg - st.prev_vol_chg) / (vol_chg + self._eps))
        st.prev_vol_chg = float(vol_chg)

        # --- price change change rate ---
        if len(st.ret_5m) >= 2 and abs(ret_5m) > self._eps:
            feats["price_change_change_rate"] = float((ret_5m - st.ret_5m[-1]) / (ret_5m + self._eps))
        else:
            feats["price_change_change_rate"] = 0.0

        feats["price_volume_direction"] = 1.0 if (ret_5m * vol_chg) >= 0 else 0.0

        # --- rolling stats ---
        mu_v_8h, sd_v_8h = self._rolling_mean_std(st.volume, self.w_8h)
        feats["volume_volatility_8h"] = float(sd_v_8h) if sd_v_8h is not None else 0.0

        mu_v_4h, _ = self._rolling_mean_std(st.volume, self.w_4h)
        feats["volume_mean_4h"] = float(mu_v_4h) if mu_v_4h is not None else 0.0

        # momentum_volatility_factor
        feats["momentum_volatility_factor"] = float(ret_5m) * feats["volume_volatility_8h"]

        # breakout confirmations (using close/volume rolling max over 24h)
        max_close_24h = self._rolling_max(st.close, self.w_24h)
        max_vol_24h = self._rolling_max(st.volume, self.w_24h)
        feats["price_breakout_confirmation"] = 1.0 if (max_close_24h is not None and close_now > max_close_24h) else 0.0
        feats["volume_breakout_confirmation"] = 1.0 if (max_vol_24h is not None and vol_now > max_vol_24h) else 0.0

        # --- 24h structure features on return_5m ---
        mu_r_24h, sd_r_24h = self._rolling_mean_std(st.ret_5m, self.w_24h)
        feats["return_5m_24h_mean"] = float(mu_r_24h) if mu_r_24h is not None else 0.0
        feats["return_5m_24h_vol"] = float(sd_r_24h) if sd_r_24h is not None else 0.0

        # --- MA ratio ---
        mu_c_8h, _ = self._rolling_mean_std(st.close, self.w_8h)
        mu_c_4h, _ = self._rolling_mean_std(st.close, self.w_4h)
        feats["ma_close_past_8h"] = float(mu_c_8h) if mu_c_8h is not None else 0.0
        feats["ma_close_past_4h"] = float(mu_c_4h) if mu_c_4h is not None else 0.0
        if mu_c_4h is None or abs(mu_c_4h) < self._eps:
            feats["ma_ratio_fast_slow_past"] = 0.0
        else:
            feats["ma_ratio_fast_slow_past"] = float((feats["ma_close_past_8h"] / (mu_c_4h + self._eps)) - 1.0)

        # --- correlation-based features (approx subset) ---
        corr_cv_24h = self._rolling_corr(st.close, st.volume, self.w_24h)
        feats["close_volume_corr"] = float(corr_cv_24h) if corr_cv_24h is not None else 0.0

        # Keep compatibility with some notebook names
        feats["corr_mean"] = feats["close_volume_corr"]
        feats["corr_std"] = 0.0
        feats["corr_mean_ma"] = feats["close_volume_corr"]
        feats["corr_std_ma"] = 0.0

        return feats

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Return target weights for the next bar.
        """

        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")
        if "volume" not in current_market_data.columns:
            current_market_data = current_market_data.copy()
            current_market_data["volume"] = 0.0

        tickers = current_market_data.index
        # Final output weights (must match tickers exactly)
        weights = pd.Series(0.0, index=tickers, dtype=float)
        # Two sub-strategies: LGBM Top-K and cross-sectional reversal (1 name)
        w_lgb = pd.Series(0.0, index=tickers, dtype=float)
        w_rev = pd.Series(0.0, index=tickers, dtype=float)

        # --- Update bar counter / day boundary approximation ---
        self.bar_in_day += 1
        if self.bar_in_day > self.bars_per_day:
            self.bar_in_day = 1

        # Go flat on the final assumed bar to avoid overnight exposure
        if self.bar_in_day == self.bars_per_day:
            for t in tickers:
                st = self._get_state(str(t))
                c = self._safe_float(current_market_data.loc[t, "close"])
                v = self._safe_float(current_market_data.loc[t, "volume"])
                if c is None or v is None:
                    continue
                if len(st.close) > 0:
                    st.ret_5m.append(c / (st.close[-1] + self._eps) - 1.0)
                st.close.append(c)
                st.volume.append(v)
            return weights

        # --- Update rolling states for each ticker ---
        for t in tickers:
            st = self._get_state(str(t))
            c = self._safe_float(current_market_data.loc[t, "close"])
            v = self._safe_float(current_market_data.loc[t, "volume"])
            if c is None or v is None:
                continue
            if len(st.close) > 0:
                st.ret_5m.append(c / (st.close[-1] + self._eps) - 1.0)
            st.close.append(c)
            st.volume.append(v)

        # --- Reversal sub-strategy (hold 1 name for 1 bar) ---
        # Buy the worst last-bar return name (cross-sectional reversal), equal weight within this sleeve.
        # Uses the latest computed 5-minute return stored in st.ret_5m (prev->current).
        rev_scores: Dict[str, float] = {}
        for t in tickers:
            tt = str(t)
            st = self.state.get(tt)
            if st is None or len(st.ret_5m) == 0:
                continue
            r = st.ret_5m[-1]
            if r is None or not np.isfinite(r):
                continue
            rev_scores[tt] = float(r)

        if rev_scores:
            # smallest return => strongest reversal candidate
            rev_pick = min(rev_scores, key=rev_scores.get)
            w_rev.loc[rev_pick] = 1.0

        # --- LGBM sub-strategy (Top-K) ---
        if self._model_ok and self.model is not None:
            rows: List[Dict[str, float]] = []
            row_tickers: List[str] = []
            for t in tickers:
                tt = str(t)
                feats = self._compute_features_for_ticker(tt)
                if len(feats) == 0:
                    continue
                rows.append(feats)
                row_tickers.append(tt)

            if rows:
                feat_df = pd.DataFrame(rows, index=row_tickers)
                for c in self.feature_cols:
                    if c not in feat_df.columns:
                        feat_df[c] = 0.0
                feat_df = (
                    feat_df[self.feature_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .astype(np.float32)
                )

                try:
                    if self.best_iteration is not None:
                        preds = self.model.predict(feat_df.values, num_iteration=self.best_iteration)
                    else:
                        preds = self.model.predict(feat_df.values)
                    pred_s = pd.Series(preds, index=feat_df.index).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
                    selected = pred_s.nlargest(self.top_k).index.tolist()
                    if selected:
                        w = 1.0 / len(selected)
                        w_lgb.loc[selected] = w
                except Exception:
                    # If prediction fails, keep this sleeve in cash
                    pass

        # --- Combine sleeves (50% / 50%) ---
        weights = 0.5 * w_lgb + 0.5 * w_rev
        # Safety: avoid float overshoots
        s = float(weights.sum())
        if s > 1.0 + 1e-9:
            weights = weights / s
        return weights
