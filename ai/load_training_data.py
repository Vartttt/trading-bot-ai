# -*- coding: utf-8 -*-
import os
import json
from typing import List, Dict, Union, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta


def _get_session(total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    a = HTTPAdapter(max_retries=r)
    s.mount("https://", a)
    s.mount("http://", a)
    return s


def _infer_kline_columns(sample_row):
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
    base = ["open_time", "open", "high", "low", "close", "volume"]
    return base + [f"col_{i}" for i in range(len(sample_row) - len(base))]


def _build_df(raw: List[List[Union[str, float]]]) -> pd.DataFrame:
    cols = _infer_kline_columns(raw[0])
    df = pd.DataFrame(raw, columns=cols)

    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("open_time", "close_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(pd.to_numeric(df[c], errors="coerce"), unit="ms")

    if "open_time" in df.columns:
        df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")

    return df


def _compute_indicators(df: pd.DataFrame,
                        ema_short: int, ema_long: int,
                        rsi_period: int, atr_period: int, vol_window: int,
                        add_extra: bool) -> pd.DataFrame:
    ema_s = ta.trend.EMAIndicator(df["close"], window=ema_short).ema_indicator()
    ema_l = ta.trend.EMAIndicator(df["close"], window=ema_long).ema_indicator()
    df["ema_short"] = ema_s
    df["ema_long"] = ema_l
    df["ema_diff"] = df["ema_short"] - df["ema_long"]

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_period).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=atr_period)\
        .average_true_range()

    vol_mean = df["volume"].rolling(vol_window).mean()
    vol_std = df["volume"].rolling(vol_window).std()
    df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    df["trend_accel"] = df["ema_diff"].diff()

    if add_extra:
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.adx()

        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["close"] + 1e-9)

    return df


def _fetch_klines_paginated(session: requests.Session,
                            symbol: str, interval: str,
                            needed: int, batch_limit: int = 1000) -> List[List[Union[str, float]]]:
    """
    Тягне свічки батчами назад у часі через endTime, поки не наберемо 'needed'
    або поки біржа перестане віддавати дані.
    """
    url = "https://api.mexc.com/api/v3/klines"
    out: List[List[Union[str, float]]] = []
    end_time: Optional[int] = None  # ms

    while len(out) < needed:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(batch_limit, needed - len(out)),
        }
        if end_time is not None:
            params["endTime"] = end_time

        resp = session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"MEXC HTTP {resp.status_code}: {err}")

        batch = resp.json()
        if not isinstance(batch, list) or len(batch) == 0:
            # кінець історії
            break

        # захист від дублів
        if out and batch[-1][0] == out[0][0]:
            batch = [row for row in batch if row[0] != out[0][0]]

        out = batch + out  # додаємо старі свічки ліворуч
        # наступний крок: ще старіше, тобто перед найстарішою в батчі
        end_time = batch[0][0] - 1  # open_time(ms) - 1

        # якщо біржа віддала дуже мало — теж виходимо, щоб не зациклитись
        if len(batch) < 2:
            break

    return out


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
    allow_partial: bool = True,   # 👈 не падати, якщо даних менше, ніж просили
) -> Union[List[Dict[str, float]], pd.DataFrame]:
    """
    Завантажує kline з MEXC з пагінацією, рахує фічі та зберігає датасет.
    """
    warmup = max(ema_long, rsi_period, atr_period, vol_window, 26) + 50
    requested = limit + warmup

    print(f"📊 Завантажую ≤{requested} свічок з MEXC для {symbol} ({interval}) батчами...")
    session = _get_session()

    try:
        raw_all = _fetch_klines_paginated(session, symbol, interval, requested, batch_limit=1000)
        if not raw_all:
            raise ValueError("MEXC повернув порожню відповідь.")

        df = _build_df(raw_all)

        required = {"open", "high", "low", "close", "volume"}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"Відсутні необхідні колонки: {miss}")

        df = _compute_indicators(
            df, ema_short, ema_long, rsi_period, atr_period, vol_window, add_extra_indicators
        )

        if add_target:
            df["next_return"] = df["close"].pct_change(periods=target_horizon).shift(-target_horizon)

        df = df.dropna().copy()

        # Якщо даних не вистачає — або зрізати по максимуму, або впасти (залежить від allow_partial)
        if len(df) < limit:
            msg = f"⚠️ Доступно лише {len(df)} рядків після обчислення індикаторів (< {limit})."
            if not allow_partial:
                raise ValueError(msg)
            else:
                print(msg)

        df = df.tail(min(limit, len(df))).copy()

        # Стиснення
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

        # Вихідні фічі (з back-compat назвами)
        out_df = pd.DataFrame(index=df.index)
        out_df["ema_diff"] = df["ema_diff"]
        out_df["rsi"] = df["rsi"]
        out_df["atr"] = df["atr"]
        out_df["vol_z"] = df["vol_z"]
        out_df["trend_accel"] = df["trend_accel"]

        out_df["ema_diff5"] = out_df["ema_diff"]
        out_df["rsi5"] = out_df["rsi"]
        out_df["volz5"] = out_df["vol_z"]

        if add_extra_indicators:
            for c in ("macd", "macd_signal", "macd_hist", "adx", "bb_high", "bb_low", "bb_width"):
                if c in df.columns:
                    out_df[c] = df[c]
        if add_target and "next_return" in df.columns:
            out_df["next_return"] = df["next_return"].astype("float32")

        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, f"{out_basename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        pq_path = os.path.join(out_dir, f"{out_basename}.parquet")
        base_cols = ["open_time", "open", "high", "low", "close", "volume"]
        base_cols = [c for c in base_cols if c in df.columns]
        pd.concat([df[base_cols].reset_index(drop=True),
                   out_df.reset_index(drop=True)], axis=1)\
          .to_parquet(pq_path, index=False)

        print(f"✅ Дані збережено:\n   • {json_path} ({len(out_df)} рядків)\n   • {pq_path}")

        return out_df if return_df else out_df.to_dict(orient="records")

    except Exception as e:
        print(f"❌ Помилка при завантаженні або обробці історії: {e}")
        return []


if __name__ == "__main__":
    load_training_data(
        symbol="BTCUSDT",
        interval="15m",
        limit=20_000,
        ema_short=9, ema_long=21,
        rsi_period=14, atr_period=14, vol_window=20,
        add_extra_indicators=False,
        add_target=False,
        out_dir="models",
        out_basename="train_data",
        return_df=False,
        allow_partial=True,  # 👈 тепер не впаде, якщо свічок менше
    )

