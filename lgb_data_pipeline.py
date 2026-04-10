"""Shared Polars feature pipeline for lgb_exp (方式1) and lgb_exp_way2 (方式2).

逻辑与原先 notebook 中 wide→long 基础因子 + 5m 特征/EWMA 一致。
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl


def discover_pairs(columns: list[str]) -> dict[str, dict[str, str]]:
    pat_tuple = re.compile(r"^\('([^']+)'\s*,\s*'([^']+)'\)$")
    pat_flat = re.compile(r"^(.*?)_([^_]+)$")
    pairs: dict[str, dict[str, str]] = {}
    for c in columns:
        if c == "datetime":
            continue
        m1 = pat_tuple.match(c)
        if m1:
            t, f = m1.group(1), m1.group(2)
        else:
            m2 = pat_flat.match(c)
            if not m2:
                continue
            t, f = m2.group(1), m2.group(2)
        if f in ("close", "volume"):
            pairs.setdefault(t, {})[f] = c
    return pairs


def wide_columns_for_tickers(pairs: dict, tickers: list[str]) -> list[str]:
    cols = ["datetime"]
    for t in tickers:
        cols.append(pairs[t]["close"])
        cols.append(pairs[t]["volume"])
    return cols


def global_split_time_from_parquet(data_path: Path, frac: float = 0.8):
    """与 notebook 一致：全体样本按 datetime 排序后，取第 int(n*frac) 行的时刻。"""
    dt = pl.read_parquet(data_path, columns=["datetime"])
    s = dt.sort("datetime")
    n = s.height
    ix = int(n * frac)
    ix = min(max(ix, 0), n - 1)
    return s["datetime"][ix]


def wide_to_long_with_mo(df_pl: pl.DataFrame, pairs: dict, tickers: list[str]) -> pl.DataFrame:
    frames = []
    for t in tickers:
        c_close = pairs[t]["close"]
        c_vol = pairs[t]["volume"]
        frames.append(
            df_pl.select([
                pl.col("datetime"),
                pl.lit(t).alias("ticker"),
                pl.col(c_close).cast(pl.Float64).alias("close"),
                pl.col(c_vol).cast(pl.Float64).alias("volume"),
            ])
        )
    long_df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "datetime"])
    window = 24
    eps = 1e-9
    long_df = long_df.with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over("ticker") + eps) - 1).alias("return")
    )
    long_df = long_df.with_columns(
        pl.col("close").rolling_mean(window_size=window).over("ticker").alias("close_ma")
    ).with_columns(
        ((pl.col("close") - pl.col("close_ma")) / (pl.col("close_ma") + eps)).alias("close_change_rate")
    ).with_columns(
        pl.col("close_change_rate").rolling_mean(window_size=window).over("ticker").alias("factor_mo_04")
    )
    long_df = long_df.with_columns([
        pl.col("volume").diff().over("ticker").alias("volume_diff"),
        pl.col("volume").rolling_mean(window_size=window).over("ticker").alias("volume_ma"),
    ]).with_columns([
        ((pl.col("volume_diff") / (pl.col("volume_ma") + eps)).abs())
        .rolling_mean(window_size=window)
        .over("ticker")
        .alias("factor_mo_05_01"),
        pl.col("volume_diff").rolling_std(window_size=window).over("ticker").alias("volume_diff_std"),
    ]).with_columns(
        pl.col("volume_diff_std").rolling_mean(window_size=window).over("ticker").alias("factor_mo_05_02")
    )
    long_df = long_df.with_columns(
        pl.col("return").shift(1).over("ticker").alias("return_lag1")
    ).with_columns(
        pl.rolling_corr(pl.col("return"), pl.col("return_lag1"), window_size=window)
        .over("ticker")
        .alias("factor_mo_07")
    )
    long_df = long_df.with_columns(
        pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=window)
        .over("ticker")
        .alias("close_volume_corr")
    ).with_columns([
        pl.col("close_volume_corr").rolling_mean(window_size=window).over("ticker").alias("corr_mean"),
        pl.col("close_volume_corr").rolling_std(window_size=window).over("ticker").alias("corr_std"),
    ]).with_columns([
        pl.col("corr_mean").rolling_mean(window_size=window).over("ticker").alias("corr_mean_ma"),
        pl.col("corr_std").rolling_mean(window_size=window).over("ticker").alias("corr_std_ma"),
    ]).with_columns(
        (
            (pl.col("corr_mean") - pl.col("corr_mean_ma")) / (pl.col("corr_std") + eps)
            + (pl.col("corr_std") - pl.col("corr_std_ma")) / (pl.col("corr_std_ma") + eps)
        ).alias("factor_mo_03")
    )
    return long_df


def build_feat_df(long_df: pl.DataFrame) -> pl.DataFrame:
    bars_per_hour = 12
    w_1h = 1 * bars_per_hour
    w_4h = 4 * bars_per_hour
    w_8h = 8 * bars_per_hour
    w_24h = 24 * bars_per_hour
    w_3d = 3 * 24 * bars_per_hour
    eps = 1e-9

    feat_df = long_df.sort(["ticker", "datetime"])

    # ---------- 价格/成交量基础变化 ----------
    feat_df = feat_df.with_columns([
        (pl.col("close") / (pl.col("close").shift(1).over("ticker") + eps) - 1).alias("return_5m"),
        (pl.col("close") / (pl.col("close").shift(w_1h).over("ticker") + eps) - 1).alias("return_1h"),
        (pl.col("close") / (pl.col("close").shift(w_4h).over("ticker") + eps) - 1).alias("return_4h"),
        (pl.col("close") / (pl.col("close").shift(w_24h).over("ticker") + eps) - 1).alias("return_24h"),
    ])

    feat_df = feat_df.with_columns([
        ((pl.col("volume") - pl.col("volume").shift(1).over("ticker")) /
         (pl.col("volume").shift(1).over("ticker") + eps)).fill_null(0.0).alias("volume_change_rate"),
    ])

    feat_df = feat_df.with_columns([
        pl.when(pl.col("volume_change_rate") != 0)
        .then(pl.col("volume_change_rate").diff().over("ticker") / (pl.col("volume_change_rate") + eps))
        .otherwise(0.0)
        .alias("volume_change_rate_change_rate"),
    ])

    # ---------- 波动/分位/均值 ----------
    feat_df = feat_df.with_columns([
        pl.col("volume").rank("dense").over("ticker").rolling_mean(window_size=w_24h).alias("volume_quantile_rank_24h"),
        pl.col("volume").rolling_std(window_size=w_8h).over("ticker").alias("volume_volatility_8h"),
        pl.col("volume").rolling_mean(window_size=w_4h).over("ticker").alias("volume_mean_4h"),
    ])

    feat_df = feat_df.with_columns([
        pl.when(pl.col("return_5m") != 0)
        .then(pl.col("return_5m").diff().over("ticker") / (pl.col("return_5m") + eps))
        .otherwise(0.0)
        .alias("price_change_change_rate"),
        ((pl.col("return_5m") * pl.col("volume_change_rate")) >= 0).cast(pl.Int8).alias("price_volume_direction"),
        (pl.col("return_5m") * pl.col("volume_volatility_8h")).alias("momentum_volatility_factor"),
    ])

    # ---------- 突破确认（close 近似 high/low） ----------
    feat_df = feat_df.with_columns([
        (pl.col("close") > pl.col("close").rolling_max(window_size=w_24h).shift(1).over("ticker")).cast(pl.Int8).alias("price_breakout_confirmation"),
        (pl.col("volume") > pl.col("volume").rolling_max(window_size=w_24h).shift(1).over("ticker")).cast(pl.Int8).alias("volume_breakout_confirmation"),
    ])

    # ---------- 结构动量 ----------
    feat_df = feat_df.with_columns([
        pl.col("return_5m").rolling_mean(window_size=w_24h).over("ticker").shift(1).alias("return_5m_24h_mean"),
        pl.col("return_1h").rolling_mean(window_size=w_24h).over("ticker").shift(1).alias("return_1h_24h_mean"),
        pl.col("return_5m").rolling_std(window_size=w_24h).over("ticker").shift(1).alias("return_5m_24h_vol"),
    ])

    feat_df = feat_df.with_columns([
        pl.col("close").rolling_mean(window_size=w_8h).over("ticker").shift(1).alias("ma_close_past_8h"),
        pl.col("close").rolling_mean(window_size=w_4h).over("ticker").shift(1).alias("ma_close_past_4h"),
    ])

    feat_df = feat_df.with_columns([
        pl.when(pl.col("ma_close_past_4h") != 0)
        .then(pl.col("ma_close_past_8h") / (pl.col("ma_close_past_4h") + eps) - 1)
        .otherwise(0.0)
        .alias("ma_ratio_fast_slow_past"),
    ])

    # ---------- RSI 相关（24h窗口） ----------
    feat_df = feat_df.with_columns([
        pl.when(pl.col("return_5m") > 0)
        .then(pl.col("return_5m").rolling_mean(window_size=w_24h).over("ticker").shift(1))
        .otherwise(None)
        .alias("upward_mean_past_24h"),
        pl.when(pl.col("return_5m") <= 0)
        .then(pl.col("return_5m").rolling_mean(window_size=w_24h).over("ticker").shift(1))
        .otherwise(None)
        .alias("downward_mean_past_24h"),
    ])

    feat_df = feat_df.with_columns([
        pl.when(
            pl.col("upward_mean_past_24h").is_not_null()
            & pl.col("downward_mean_past_24h").is_not_null()
            & (pl.col("downward_mean_past_24h") != 0)
        )
        .then(pl.col("upward_mean_past_24h") / (pl.col("downward_mean_past_24h") + eps))
        .otherwise(0.0)
        .alias("rs_past_24h"),
    ])

    # 注意：Polars 同一个 with_columns 中不能稳定引用“本次新建列”，拆成两步
    feat_df = feat_df.with_columns([
        pl.when(pl.col("rs_past_24h") != 0)
        .then(100.0 - (100.0 / (pl.col("rs_past_24h") + eps)))
        .otherwise(0.0)
        .alias("rsi_past_24h"),
    ])

    feat_df = feat_df.with_columns([
        (pl.col("rsi_past_24h") > 70).cast(pl.Int8).alias("rsi_overbought_past"),
        (pl.col("rsi_past_24h") < 30).cast(pl.Int8).alias("rsi_oversold_past"),
    ])

    # ---------- 从 factor_calculation_okx 复用的 mo 因子（5m窗口） ----------
    feat_df = feat_df.with_columns([
        pl.col("volume").diff().over("ticker").alias("volume_diff"),
        pl.col("volume").rolling_mean(window_size=w_24h).over("ticker").alias("volume_ma_24h"),
        pl.col("close").rolling_mean(window_size=w_24h).over("ticker").alias("close_ma_24h"),
    ])

    feat_df = feat_df.with_columns([
        ((pl.col("volume_diff") / (pl.col("volume_ma_24h") + eps)).abs())
        .rolling_mean(window_size=w_24h)
        .over("ticker")
        .alias("factor_mo_05_01"),
        pl.col("volume_diff").rolling_std(window_size=w_24h).over("ticker").alias("volume_diff_std"),
        ((pl.col("close") - pl.col("close_ma_24h")) / (pl.col("close_ma_24h") + eps)).alias("close_change_rate"),
    ])

    feat_df = feat_df.with_columns([
        pl.col("volume_diff_std").rolling_mean(window_size=w_24h).over("ticker").alias("factor_mo_05_02"),
        pl.col("close_change_rate").rolling_mean(window_size=w_24h).over("ticker").alias("factor_mo_04"),
        pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=w_24h).over("ticker").alias("close_volume_corr"),
    ])

    feat_df = feat_df.with_columns([
        pl.col("close_volume_corr").rolling_mean(window_size=w_24h).over("ticker").alias("close_volume_corr_mean"),
        pl.col("close_volume_corr").rolling_std(window_size=w_24h).over("ticker").alias("close_volume_corr_std"),
    ])

    feat_df = feat_df.with_columns([
        (
            (pl.col("close_volume_corr_mean") - pl.col("close_volume_corr_mean").rolling_mean(window_size=w_24h).over("ticker"))
            / (pl.col("close_volume_corr_std") + eps)
            + (pl.col("close_volume_corr_std") - pl.col("close_volume_corr_std").rolling_mean(window_size=w_24h).over("ticker"))
            / (pl.col("close_volume_corr_std").rolling_mean(window_size=w_24h).over("ticker") + eps)
        ).alias("factor_mo_03"),
    ])

    # 注意：rolling_corr 内不要再嵌套 window expression（shift().over）
    feat_df = feat_df.with_columns([
        pl.col("return_5m").shift(1).over("ticker").alias("return_5m_lag1"),
    ])

    feat_df = feat_df.with_columns([
        pl.rolling_corr(pl.col("return_5m"), pl.col("return_5m_lag1"), window_size=w_24h)
        .over("ticker")
        .alias("factor_mo_07"),
    ])

    # ---------- 3天 EWMA 降频（除高频因子外） ----------
    high_freq_cols = {
        "return_5m", "volume_change_rate", "volume_change_rate_change_rate",
        "price_change_change_rate", "price_volume_direction",
        "price_breakout_confirmation", "volume_breakout_confirmation"
    }

    candidate_factor_cols = [
        c for c in feat_df.columns
        if c not in {"datetime", "ticker", "close", "volume", "return_5m_lag1", "close_ma", "volume_diff", "volume_ma", "volume_diff_std", "close_change_rate", "close_ma_24h", "volume_ma_24h", "close_volume_corr", "close_volume_corr_mean", "close_volume_corr_std", "upward_mean_past_24h", "downward_mean_past_24h", "rs_past_24h"}
    ]

    numeric_cols = [
        c for c in candidate_factor_cols
        if feat_df.schema[c] in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ]

    ewm_base_cols = [c for c in numeric_cols if c not in high_freq_cols]

    feat_df = feat_df.with_columns([
        pl.col(c).cast(pl.Float64).ewm_mean(span=w_3d, adjust=False, min_samples=1).over("ticker").alias(f"{c}_ewm3d")
        for c in ewm_base_cols
    ])

    return feat_df
