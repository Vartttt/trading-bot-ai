# -*- coding: utf-8 -*-
import os
import json
import time
from typing import List, Dict, Optional, Union, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta


def _get_session(total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """HTTP session with retries & timeouts."""
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _infer_kline_columns(sample_row: List[Union[str, float, int]]) -> List[str]:
    """
    Map MEXC kline array to DataFrame columns.
    Known formats:
      - 12 cols: open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_base, taker_quote, ignore
      - 8  cols: open_time, open, high, low, close, volume, close_time, quote_asset_volume
    """
    if len(sample_row) == 12:
        return [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades", "taker_base",
            "taker_quote", "ignore"
        ]
    if len(sample_row) == 8:
        return [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume"
        ]
    # Fallback to first 6 OHLCV + times if unknown
    base = ["open_time", "open", "high", "low", "close", "volume"]
    return base + [f"col_{i}" for i in range(len(sample_row) - len(base))]


def _build_df(raw: List[List[Union[str, float]]]) -> pd.DataFrame:
    cols = _infer_kline_columns(raw[0])
    df = pd.DataFrame(raw, columns=cols)

    # Ensure numeric types
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamps to datetime (ms)
    for col in ("open_time", "close_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit="ms")

    # Sort & deduplicate
    if "open_time" in df.columns:
        df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")

    return df


def _compute_indicators(
    df: pd.DataFrame,
    ema_short: int,
    ema_long: int,
    rsi_period: int,
    atr_period: int,
    vol_window: int,
    add_extra_indicators: bool,
) -> pd.DataFrame:
    # EMA short/long & their diff
    ema_s = ta.trend.EMAIndicator(close=df["close"], window=ema_short).ema_indicator()
    ema_l = ta.trend.EMAIndicator(close=df["close"], window=ema_long).ema_indicator()
    df["ema_short"] = ema_s
    df["ema_long"] = ema_l
    df["ema_diff"] = df["ema_short"] - df["ema_long"]

    # RSI / ATR
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=rsi_period).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=atr_period
    ).average_true_range()

    # Volume z-score
    vol_mean = df["volume"].rolling(vol_window).mean()
    vol_std = df["volume"].rolling(vol_window).std()
    df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    # Trend acceleration (1st derivative of ema_diff)
    df["trend_accel"] = df["ema_diff"].diff()

    if add_extra_indicators:
        # MACD (12,26,9)
        macd_i = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_i.macd()
        df["macd_signal"] = macd_i.macd_signal()
        df["macd_hist"] = macd_i.macd_diff()

        # ADX (14)
        adx_i = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["adx"] = adx_i.adx()

        # Bollinger (20, 2)
        bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["close"] + 1e-9)

    return df


def _ensure_min_rows(df: pd.DataFrame, min_required: int) -> None:
    if len(df) < min_required:
        raise ValueError(f"Недостатньо даних після обчислення індикаторів: {len(df)} < {min_required}")


def load_training_data(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    limit: int = 20_000,
    *,
    ema_short: int = 9,
    ema_long: int = 21,
    rsi_period: int = 14,
    atr_period: int = 14,
    vol_window: int = 20,
    add_extra_indicators: bool = False,
    add_target: bool = False,
    target_horizon: int = 1,
    out_dir: str = "models",
    out_basename: str = "train_data",
    return_df: bool = False,
) -> Union[List[Dict[str, float]], pd.DataFrame]:
    """
    Завантажує kline з MEXC, обчислює фічі та зберігає тренувальний датасет.

    Parameters
    ----------
    symbol : str
        Напр., "BTCUSDT".
    interval : str
        Напр., "15m".
    limit : int
        Скільки останніх рядків повернути *після* дропа NaN.
    ema_short, ema_long, rsi_period, atr_period, vol_window : int
        Параметри індикаторів.
    add_extra_indicators : bool
        Додати MACD/ADX/Bollinger.
    add_target : bool
        Додати таргет: майбутня доходність close на `target_horizon` свічок.
    target_horizon : int
        Горизонт таргету в свічках.
    out_dir : str
        Куди зберігати файли.
    out_basename : str
        Базова назва файлів без розширення.
    return_df : bool
        Повернути DataFrame (True) або список dict для JSON (False).

    Returns
    -------
    list|DataFrame
        Або список фіч-слівників, або DataFrame (за `return_df=True`).
    """
    warmup = max(ema_long, rsi_period, atr_period, vol_window, 26) + 50
    requested = limit + warmup

    print(f"📊 Завантажую ~{requested} свічок з MEXC для {symbol} ({interval})...")
    url = "https://api.mexc.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": requested}

    session = _get_session()

    try:
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"HTTP {resp.status_code}: {err}")

        raw = resp.json()
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"Некоректна відповідь API MEXC: {raw}")

        df = _build_df(raw)

        # Базова перевірка
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Відсутні необхідні колонки: {missing}")

        # Обчислюємо індикатори
        df = _compute_indicators(
            df, ema_short, ema_long, rsi_period, atr_period, vol_window, add_extra_indicators
        )

        # Опційний таргет: наступна доходність close
        if add_target:
            df["next_return"] = df["close"].pct_change(periods=target_horizon).shift(-target_horizon)

        # Дроп NaN після індикаторів
        df = df.dropna().copy()

        # Залишаємо рівно `limit` останніх рядків
        _ensure_min_rows(df, limit)
        df = df.tail(limit).copy()

        # Стиснення типів
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")

        # Підготуємо вихідні колонки (основні)
        df_out = pd.DataFrame(index=df.index)
        df_out["ema_diff"] = df["ema_diff"]
        df_out["rsi"] = df["rsi"]
        df_out["atr"] = df["atr"]
        df_out["vol_z"] = df["vol_z"]
        df_out["trend_accel"] = df["trend_accel"]

        # Back-compat ключі (щоб не ламати існуючий код):
        df_out["ema_diff5"] = df_out["ema_diff"]
        df_out["rsi5"] = df_out["rsi"]
        df_out["volz5"] = df_out["vol_z"]

        # Додаткові індикатори, якщо увімкнено
        if add_extra_indicators:
            for col in ("macd", "macd_signal", "macd_hist", "adx", "bb_high", "bb_low", "bb_width"):
                if col in df.columns:
                    df_out[col] = df[col]

        # Додаємо таргет, якщо треба
        if add_target and "next_return" in df.columns:
            df_out["next_return"] = df["next_return"].astype("float32")

        # Створюємо директорію та зберігаємо
        os.makedirs(out_dir, exist_ok=True)

        # JSON (records)
        json_path = os.path.join(out_dir, f"{out_basename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(df_out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        # Parquet з тайм-індексом та базовими OHLCV (зручно для перегляду/дебагу)
        parquet_cols = ["open_time", "open", "high", "low", "close", "volume"] + list(df_out.columns)
        parquet_cols = [c for c in parquet_cols if c in df.columns or c in df_out.columns]
        df_parquet = pd.concat([df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
                                df_out.reset_index(drop=True)], axis=1)
        pq_path = os.path.join(out_dir, f"{out_basename}.parquet")
        df_parquet.to_parquet(pq_path, index=False)

        print(f"✅ Дані збережено:\n   • {json_path} ({len(df_out)} рядків)\n   • {pq_path}")

        return df_out if not return_df else df_out.copy()

    except Exception as e:
        print(f"❌ Помилка при завантаженні або обробці історії: {e}")
        return []


if __name__ == "__main__":
    # Приклад запуску за замовчуванням
    load_training_data(
        symbol="BTCUSDT",
        interval="15m",
        limit=20_000,
        ema_short=9,
        ema_long=21,
        rsi_period=14,
        atr_period=14,
        vol_window=20,
        add_extra_indicators=False,  # True — додасть MACD/ADX/BB
        add_target=False,            # True — додасть next_return
        out_dir="models",
        out_basename="train_data",
        return_df=False,
    )

