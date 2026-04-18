"""Shared Polars feature pipeline for lgb_exp (方式1) and lgb_exp_way2 (方式2).

逻辑与原先 notebook 中 wide→long 基础因子 + 5m 特征/EWMA 一致。

时间序列运算按「交易日」分块（`_session_date` = `datetime` 的日期部分），避免休盘/隔夜后
相邻两行被当成连续 5m bar，从而出现「次日开盘相对前日收盘」的假收益进入特征或标签。
"""
from __future__ import annotations

import gc
import re
import tempfile
from pathlib import Path

import polars as pl

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore

# 与 shift/rolling/ewm 等配合：同一标的内按自然日分区（不含时区换算；若需交易所日历可再扩展）
_BY_SESSION = ["ticker", "_session_date"]


def _downcast_float64_to_float32(df: pl.DataFrame) -> pl.DataFrame:
    """降低特征表内存占用（Float64→Float32），对 LGBM 通常足够。"""
    casts = [pl.col(c).cast(pl.Float32).alias(c) for c, dt in zip(df.columns, df.dtypes) if dt == pl.Float64]
    return df.with_columns(casts) if casts else df


def _pyarrow_merge_parquet_files(paths: list[Path], destination: Path) -> None:
    """按顺序把多个 parquet 行组写入一个文件，避免在 Polars 里 pl.concat 多张大表。"""
    if pq is None:
        raise RuntimeError("需要安装 pyarrow 才能合并 parquet（见 requirements.txt）")
    writer: pq.ParquetWriter | None = None
    try:
        for p in paths:
            table = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(str(destination), table.schema, compression="zstd")
            writer.write_table(table)
            del table
    finally:
        if writer is not None:
            writer.close()


def _sort_feat_parquet_to_parquet(src: Path, dst: Path, verbose: bool) -> None:
    """尽量用 lazy sink 外排；失败则回退为一次性内存 sort（小表才安全）。"""
    try:
        pl.scan_parquet(str(src)).sort(["ticker", "datetime"]).sink_parquet(str(dst))
        return
    except Exception as e:
        if verbose:
            print(f"  [feat] sink_parquet 排序不可用，回退内存 sort: {e!s}", flush=True)
    pl.read_parquet(str(src)).sort(["ticker", "datetime"]).write_parquet(str(dst), compression="zstd")


def _read_feat_merged_parquet(path: Path) -> pl.DataFrame:
    """尽量 memory_map，减少一次完整拷贝。"""
    try:
        return pl.read_parquet(str(path), memory_map=True)
    except TypeError:
        return pl.read_parquet(str(path))


def _finalize_feat_part_parquets(
    paths: list[Path],
    tmp: Path,
    verbose: bool,
    *,
    force_global_sort: bool = False,
) -> pl.DataFrame:
    """
    分片 parquet → PyArrow 顺序合并 →（可选）全局排序 → mmap 读入 → Float32。

    默认 **不做** 对整表 merge 后的 `sort(ticker, datetime)`：`build_feat_*_chunked` 按
    `tickers_use` 的**连续切片**分批，各分片标的互不重叠；且 `wide_to_long_with_mo` 已按
    `(ticker, datetime)` 排序。按 `part_0000, part_0001, …` 顺序合并后，在「标的列表已排序」
    的前提下，全局行序已满足 `shift().over([ticker, _session_date])` 所需，**再整表排序会
    OOM**。若你改动了标的顺序或合批逻辑，可设 `force_global_sort=True`（仅小数据可承受）。
    """
    if not paths:
        raise ValueError("_finalize_feat_part_parquets: paths 为空")
    merged = tmp / "merged.parquet"
    if verbose:
        print("  [feat] PyArrow 顺序合并分片 parquet …", flush=True)
    _pyarrow_merge_parquet_files(paths, merged)
    if force_global_sort:
        sorted_p = tmp / "sorted.parquet"
        if verbose:
            print("  [feat] 按 ticker, datetime 全局排序落盘（force_global_sort=True）…", flush=True)
        _sort_feat_parquet_to_parquet(merged, sorted_p, verbose)
        read_from = sorted_p
    else:
        if verbose:
            print(
                "  [feat] 跳过全局 sort（分片为连续标的批、片内已按 ticker,datetime 有序）…",
                flush=True,
            )
        read_from = merged
    if verbose:
        print("  [feat] mmap 读入 + Float64→Float32 …", flush=True)
    out = _read_feat_merged_parquet(read_from)
    return _downcast_float64_to_float32(out)


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
    long_df = long_df.with_columns(pl.col("datetime").dt.date().alias("_session_date"))
    window = 24
    eps = 1e-9
    long_df = long_df.with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over(_BY_SESSION) + eps) - 1).alias("return")
    )
    long_df = long_df.with_columns(
        pl.col("close").rolling_mean(window_size=window).over(_BY_SESSION).alias("close_ma")
    ).with_columns(
        ((pl.col("close") - pl.col("close_ma")) / (pl.col("close_ma") + eps)).alias("close_change_rate")
    ).with_columns(
        pl.col("close_change_rate").rolling_mean(window_size=window).over(_BY_SESSION).alias("factor_mo_04")
    )
    long_df = long_df.with_columns([
        pl.col("volume").diff().over(_BY_SESSION).alias("volume_diff"),
        pl.col("volume").rolling_mean(window_size=window).over(_BY_SESSION).alias("volume_ma"),
    ]).with_columns([
        ((pl.col("volume_diff") / (pl.col("volume_ma") + eps)).abs())
        .rolling_mean(window_size=window)
        .over(_BY_SESSION)
        .alias("factor_mo_05_01"),
        pl.col("volume_diff").rolling_std(window_size=window).over(_BY_SESSION).alias("volume_diff_std"),
    ]).with_columns(
        pl.col("volume_diff_std").rolling_mean(window_size=window).over(_BY_SESSION).alias("factor_mo_05_02")
    )
    long_df = long_df.with_columns(
        pl.col("return").shift(1).over(_BY_SESSION).alias("return_lag1")
    ).with_columns(
        pl.rolling_corr(pl.col("return"), pl.col("return_lag1"), window_size=window)
        .over(_BY_SESSION)
        .alias("factor_mo_07")
    )
    long_df = long_df.with_columns(
        pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=window)
        .over(_BY_SESSION)
        .alias("close_volume_corr")
    ).with_columns([
        pl.col("close_volume_corr").rolling_mean(window_size=window).over(_BY_SESSION).alias("corr_mean"),
        pl.col("close_volume_corr").rolling_std(window_size=window).over(_BY_SESSION).alias("corr_std"),
    ]).with_columns([
        pl.col("corr_mean").rolling_mean(window_size=window).over(_BY_SESSION).alias("corr_mean_ma"),
        pl.col("corr_std").rolling_mean(window_size=window).over(_BY_SESSION).alias("corr_std_ma"),
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
    feat_df = feat_df.with_columns(pl.col("datetime").dt.date().alias("_session_date"))

    # ---------- 价格/成交量基础变化 ----------
    feat_df = feat_df.with_columns([
        (pl.col("close") / (pl.col("close").shift(1).over(_BY_SESSION) + eps) - 1).alias("return_5m"),
        (pl.col("close") / (pl.col("close").shift(w_1h).over(_BY_SESSION) + eps) - 1).alias("return_1h"),
        (pl.col("close") / (pl.col("close").shift(w_4h).over(_BY_SESSION) + eps) - 1).alias("return_4h"),
        (pl.col("close") / (pl.col("close").shift(w_24h).over(_BY_SESSION) + eps) - 1).alias("return_24h"),
    ])

    feat_df = feat_df.with_columns([
        ((pl.col("volume") - pl.col("volume").shift(1).over(_BY_SESSION)) /
         (pl.col("volume").shift(1).over(_BY_SESSION) + eps)).fill_null(0.0).alias("volume_change_rate"),
    ])

    feat_df = feat_df.with_columns([
        pl.when(pl.col("volume_change_rate") != 0)
        .then(pl.col("volume_change_rate").diff().over(_BY_SESSION) / (pl.col("volume_change_rate") + eps))
        .otherwise(0.0)
        .alias("volume_change_rate_change_rate"),
    ])

    # ---------- 波动/分位/均值 ----------
    feat_df = feat_df.with_columns([
        pl.col("volume")
        .rank("dense")
        .over(_BY_SESSION)
        .rolling_mean(window_size=w_24h)
        .over(_BY_SESSION)
        .alias("volume_quantile_rank_24h"),
        pl.col("volume").rolling_std(window_size=w_8h).over(_BY_SESSION).alias("volume_volatility_8h"),
        pl.col("volume").rolling_mean(window_size=w_4h).over(_BY_SESSION).alias("volume_mean_4h"),
    ])

    feat_df = feat_df.with_columns([
        pl.when(pl.col("return_5m") != 0)
        .then(pl.col("return_5m").diff().over(_BY_SESSION) / (pl.col("return_5m") + eps))
        .otherwise(0.0)
        .alias("price_change_change_rate"),
        ((pl.col("return_5m") * pl.col("volume_change_rate")) >= 0).cast(pl.Int8).alias("price_volume_direction"),
        (pl.col("return_5m") * pl.col("volume_volatility_8h")).alias("momentum_volatility_factor"),
    ])

    # ---------- 突破确认（close 近似 high/low） ----------
    feat_df = feat_df.with_columns([
        (pl.col("close") > pl.col("close").rolling_max(window_size=w_24h).shift(1).over(_BY_SESSION)).cast(pl.Int8).alias("price_breakout_confirmation"),
        (pl.col("volume") > pl.col("volume").rolling_max(window_size=w_24h).shift(1).over(_BY_SESSION)).cast(pl.Int8).alias("volume_breakout_confirmation"),
    ])

    # ---------- 结构动量 ----------
    feat_df = feat_df.with_columns([
        pl.col("return_5m").rolling_mean(window_size=w_24h).over(_BY_SESSION).shift(1).alias("return_5m_24h_mean"),
        pl.col("return_1h").rolling_mean(window_size=w_24h).over(_BY_SESSION).shift(1).alias("return_1h_24h_mean"),
        pl.col("return_5m").rolling_std(window_size=w_24h).over(_BY_SESSION).shift(1).alias("return_5m_24h_vol"),
    ])

    feat_df = feat_df.with_columns([
        pl.col("close").rolling_mean(window_size=w_8h).over(_BY_SESSION).shift(1).alias("ma_close_past_8h"),
        pl.col("close").rolling_mean(window_size=w_4h).over(_BY_SESSION).shift(1).alias("ma_close_past_4h"),
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
        .then(pl.col("return_5m").rolling_mean(window_size=w_24h).over(_BY_SESSION).shift(1))
        .otherwise(None)
        .alias("upward_mean_past_24h"),
        pl.when(pl.col("return_5m") <= 0)
        .then(pl.col("return_5m").rolling_mean(window_size=w_24h).over(_BY_SESSION).shift(1))
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
        pl.col("volume").diff().over(_BY_SESSION).alias("volume_diff"),
        pl.col("volume").rolling_mean(window_size=w_24h).over(_BY_SESSION).alias("volume_ma_24h"),
        pl.col("close").rolling_mean(window_size=w_24h).over(_BY_SESSION).alias("close_ma_24h"),
    ])

    feat_df = feat_df.with_columns([
        ((pl.col("volume_diff") / (pl.col("volume_ma_24h") + eps)).abs())
        .rolling_mean(window_size=w_24h)
        .over(_BY_SESSION)
        .alias("factor_mo_05_01"),
        pl.col("volume_diff").rolling_std(window_size=w_24h).over(_BY_SESSION).alias("volume_diff_std"),
        ((pl.col("close") - pl.col("close_ma_24h")) / (pl.col("close_ma_24h") + eps)).alias("close_change_rate"),
    ])

    feat_df = feat_df.with_columns([
        pl.col("volume_diff_std").rolling_mean(window_size=w_24h).over(_BY_SESSION).alias("factor_mo_05_02"),
        pl.col("close_change_rate").rolling_mean(window_size=w_24h).over(_BY_SESSION).alias("factor_mo_04"),
        pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size=w_24h).over(_BY_SESSION).alias("close_volume_corr"),
    ])

    feat_df = feat_df.with_columns([
        pl.col("close_volume_corr").rolling_mean(window_size=w_24h).over(_BY_SESSION).alias("close_volume_corr_mean"),
        pl.col("close_volume_corr").rolling_std(window_size=w_24h).over(_BY_SESSION).alias("close_volume_corr_std"),
    ])

    feat_df = feat_df.with_columns([
        (
            (pl.col("close_volume_corr_mean") - pl.col("close_volume_corr_mean").rolling_mean(window_size=w_24h).over(_BY_SESSION))
            / (pl.col("close_volume_corr_std") + eps)
            + (pl.col("close_volume_corr_std") - pl.col("close_volume_corr_std").rolling_mean(window_size=w_24h).over(_BY_SESSION))
            / (pl.col("close_volume_corr_std").rolling_mean(window_size=w_24h).over(_BY_SESSION) + eps)
        ).alias("factor_mo_03"),
    ])

    # 注意：rolling_corr 内不要再嵌套 window expression（shift().over）
    feat_df = feat_df.with_columns([
        pl.col("return_5m").shift(1).over(_BY_SESSION).alias("return_5m_lag1"),
    ])

    feat_df = feat_df.with_columns([
        pl.rolling_corr(pl.col("return_5m"), pl.col("return_5m_lag1"), window_size=w_24h)
        .over(_BY_SESSION)
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
        if c not in {
            "datetime",
            "ticker",
            "_session_date",
            "close",
            "volume",
            "return_5m_lag1",
            "close_ma",
            "volume_diff",
            "volume_ma",
            "volume_diff_std",
            "close_change_rate",
            "close_ma_24h",
            "volume_ma_24h",
            "close_volume_corr",
            "close_volume_corr_mean",
            "close_volume_corr_std",
            "upward_mean_past_24h",
            "downward_mean_past_24h",
            "rs_past_24h",
        }
    ]

    numeric_cols = [
        c for c in candidate_factor_cols
        if feat_df.schema[c] in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    ]

    ewm_base_cols = [c for c in numeric_cols if c not in high_freq_cols]

    feat_df = feat_df.with_columns([
        pl.col(c).cast(pl.Float64).ewm_mean(span=w_3d, adjust=False, min_samples=1).over(_BY_SESSION).alias(f"{c}_ewm3d")
        for c in ewm_base_cols
    ])

    return feat_df


def build_feat_df_from_wide_chunked(
    df_wide: pl.DataFrame,
    pairs: dict,
    tickers: list[str],
    chunk_size: int = 10,
) -> pl.DataFrame:
    """
    对已载入的宽表按标的分批：列子集 → wide_to_long_with_mo → build_feat_df，再纵向合并并排序。

    用于避免「一次展开全市场 long 表（千万行）再 build_feat_df」导致的内存峰值过高、内核 OOM。
    数学上与一次性全量等价（各 ticker 独立拼接）。
    """
    if chunk_size < 1:
        chunk_size = 1
    if not tickers:
        raise ValueError("build_feat_df_from_wide_chunked: tickers 为空")
    n = len(tickers)
    with tempfile.TemporaryDirectory(prefix="lgb_feat_wide_") as tmpdir:
        tmp = Path(tmpdir)
        paths: list[Path] = []
        for j, i in enumerate(range(0, n, chunk_size)):
            batch = tickers[i : i + chunk_size]
            subcols = wide_columns_for_tickers(pairs, batch)
            present = [c for c in subcols if c in df_wide.columns]
            df_b = df_wide.select(present)
            long_b = wide_to_long_with_mo(df_b, pairs, batch)
            del df_b
            gc.collect()
            feat_b = build_feat_df(long_b)
            del long_b
            gc.collect()
            pth = tmp / f"part_{j:04d}.parquet"
            feat_b.write_parquet(pth, compression="zstd")
            del feat_b
            gc.collect()
            paths.append(pth)
        out = _finalize_feat_part_parquets(paths, tmp, verbose=True)
    return out


def build_feat_df_from_parquet_path_chunked(
    parquet_path: str | Path,
    pairs: dict,
    tickers: list[str],
    chunk_size: int = 5,
    *,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    **推荐**：从 Parquet 路径按标的分批，每批只 `read_parquet(columns=该批宽表列)`，再 long→build_feat_df，最后纵向合并。

    避免在内存中同时持有「全市场宽表」+ 千万行 long（此前即便分批特征，先整表 read 仍会 OOM）。
    """
    pq = Path(parquet_path)
    if chunk_size < 1:
        chunk_size = 1
    if not tickers:
        raise ValueError("build_feat_df_from_parquet_path_chunked: tickers 为空")
    n = len(tickers)
    n_chunks = (n + chunk_size - 1) // chunk_size
    # 每批写完即落盘并释放，避免最后 pl.concat(parts) 时内存里同时驻留多份大块表导致 OOM
    with tempfile.TemporaryDirectory(prefix="lgb_feat_pq_") as tmpdir:
        tmp = Path(tmpdir)
        paths: list[Path] = []
        for j, i in enumerate(range(0, n, chunk_size)):
            batch = tickers[i : i + chunk_size]
            subcols = wide_columns_for_tickers(pairs, batch)
            if verbose:
                print(
                    f"  [feat chunk {j + 1}/{n_chunks}] {len(batch)} tickers, wide cols={len(subcols)}",
                    flush=True,
                )
            df_b = pl.read_parquet(pq, columns=subcols)
            long_b = wide_to_long_with_mo(df_b, pairs, batch)
            del df_b
            gc.collect()
            feat_b = build_feat_df(long_b)
            del long_b
            gc.collect()
            pth = tmp / f"part_{j:04d}.parquet"
            feat_b.write_parquet(pth, compression="zstd")
            del feat_b
            gc.collect()
            paths.append(pth)
        out = _finalize_feat_part_parquets(paths, tmp, verbose=verbose)
    return out


def compute_time_bucket_edges(
    train_df: pl.DataFrame,
    n_buckets: int,
    time_col: str = "datetime",
) -> list:
    """
    在训练集上按 `time_col` 排序后做**等频**时间切分，返回长度 `n_buckets + 1` 的边界时刻（含首尾）。
    用于「按时间段训练多个模型」：验证集用同一套边界打 `_time_bucket`。
    """
    n = train_df.height
    if n == 0:
        raise ValueError("compute_time_bucket_edges: 训练表为空")
    if n_buckets < 2:
        raise ValueError("n_buckets 至少为 2")
    ts = train_df.sort(time_col).select(time_col).to_series()
    edges: list = []
    for i in range(n_buckets + 1):
        idx = min(int(i * (n - 1) / n_buckets), n - 1)
        edges.append(ts[idx])
    return edges


def add_time_bucket_column(
    df: pl.DataFrame,
    edges: list,
    time_col: str = "datetime",
) -> pl.DataFrame:
    """
    根据 `compute_time_bucket_edges` 得到的边界，为每行分配 `_time_bucket` ∈ {0..len(edges)-2}。
    桶 i 对应时间区间 [edges[i], edges[i+1])，最后一桶右端包含 edges[-1]（与 searchsorted 一致）。
    """
    import numpy as np

    t = df[time_col].to_numpy()
    t64 = t.astype("datetime64[ns]").astype(np.int64)
    e64 = np.array(
        [np.datetime64(e, "ns").astype(np.int64) for e in edges],
        dtype=np.int64,
    )
    b = np.searchsorted(e64, t64, side="right") - 1
    nb = len(edges) - 1
    b = np.clip(b, 0, nb - 1).astype(np.int32)
    return df.with_columns(pl.Series("_time_bucket", b))


def add_target_fwd_5m(df: pl.DataFrame, horizon: int = 1) -> pl.DataFrame:
    """
    在特征表上追加标签 `target_fwd_5m`：未来 `horizon` 根 K 线收益，且仅在同一 `_session_date` 内移位，
    避免「当日最后一根」的标签指向「次日开盘」。
    """
    out = df
    if "_session_date" not in out.columns:
        out = out.with_columns(pl.col("datetime").dt.date().alias("_session_date"))
    return out.with_columns(
        (pl.col("close").shift(-horizon).over(_BY_SESSION) / (pl.col("close") + 1e-12) - 1).alias("target_fwd_5m")
    )
