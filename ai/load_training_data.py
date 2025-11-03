# -*- coding: utf-8 -*-
import os
import json
from typing import List, Dict, Union, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta


# ============================================================
# 🌐 Сесія з повторними спробами
# ============================================================
def _get_session(total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Створює сесію з автоматичними повторними спробами."""
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


# ============================================================
# 🧱 Побудова DataFrame з сирих даних
# ============================================================
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
    """Конвертує сирі дані MEXC у структурований DataFrame."""
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

    df.dropna(subset=["close"], inplace=True)
    return df


# ============================================================
# 📈 Розрахунок індикаторів
# ============================================================
def _compute_indicators(df: pd.DataFrame,
                        ema_short: int, ema_long: int,
                        rsi_period: int, atr_period: int, vol_window: int,
                        add_extra: bool) -> pd.DataFrame:
    """Додає технічні індикатори у DataFrame."""
    df["ema_short"] = ta.trend.EMAIndicator(df["close"], window=ema_short).ema_indicator()
    df["ema_long"] = ta.trend.EMAIndicator(df["close"], window=ema_long).ema_indicator()
    df["ema_diff"] = df["ema_short"] - df["ema_long"]

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_period).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=atr_period
    ).average_true_range()

    df["vol_z"] = (df["volume"] - df["volume"].rolling(vol_window).mean()) / (
        df["volume"].rolling(vol_window).std() + 1e-9
    )
    df["trend_accel"] = df["ema_diff"].diff()

    if add_extra:
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()

        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["close"] + 1e-9)

    df.dropna(inplace=True)
    return df


# ============================================================
# ⏳ Пагінація для завантаження історії
# ============================================================
def _fetch_klines_paginated(session: requests.Session,
                            symbol: str, interval: str,
                            needed: int, batch_limit: int = 1000) -> List[List[Union[str, float]]]:
    """
    Тягне свічки батчами назад у часі через endTime, поки не наберемо 'needed'.
    """
    url = "https://api.mexc.com/api/v3/klines"
    out: List[List[Union[str, float]]] = []
    end_time: Optional[int] = None

    while len(out) < needed:
        params = {"symbol": symbol, "interval": interval, "limit": min(batch_limit, needed - len(out))}
        if end_time is not None:
            params["endTime"] = end_time

        resp = session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"MEXC HTTP {resp.status_code}: {resp.text}")

        batch = resp.json()
        if not isinstance(batch, list) or len(batch) == 0:
            break

        out = batch + out
        end_time = batch[0][0] - 1
        if len(batch) < 2:
            break

    return out


# ============================================================
# 🧠 Основна функція для побудови датасету
# ============================================================
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
    allow_partial: bool = True,
) -> Union[List[Dict[str, float]], pd.DataFrame]:
    """
    Завантажує історію з MEXC, додає індикатори і формує тренувальний датасет.
    """
    warmup = max(ema_long, rsi_period, atr_period, vol_window, 26) + 50
    requested = limit + warmup

    print(f"📊 Завантажую ≤{requested} свічок для {symbol} ({interval})...")
    session = _get_session()

    try:
        raw_all = _fetch_klines_paginated(session, symbol, interval, requested)
        if not raw_all:
            raise ValueError("MEXC повернув порожню відповідь.")

        df = _build_df(raw_all)
        df = _compute_indicators(df, ema_short, ema_long, rsi_period, atr_period, vol_window, add_extra_indicators)

        if add_target:
            df["next_return"] = df["close"].pct_change(periods=target_horizon).shift(-target_horizon)

        df.dropna(inplace=True)

        if len(df) < limit:
            msg = f"⚠️ Доступно лише {len(df)} рядків (< {limit})."
            if allow_partial:
                print(msg)
            else:
                raise ValueError(msg)

        df = df.tail(min(limit, len(df)))

        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

        # 🔹 Формуємо вихідні фічі
        out_df = df[[
            "ema_diff", "rsi", "atr", "vol_z", "trend_accel"
        ]].rename(columns={
            "ema_diff": "ema_diff5",
            "rsi": "rsi5",
            "vol_z": "volz5"
        })

        if add_target and "next_return" in df.columns:
            out_df["next_return"] = df["next_return"].astype("float32")

        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, f"{out_basename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        pq_path = os.path.join(out_dir, f"{out_basename}.parquet")
        out_df.to_parquet(pq_path, index=False)

        print(f"✅ Дані збережено:\n   • {json_path} ({len(out_df)} рядків)\n   • {pq_path}")

        return out_df if return_df else out_df.to_dict(orient="records")

    except Exception as e:
        print(f"❌ Помилка при завантаженні або обробці історії: {e}")
        return []


# ============================================================
# 🚀 Автозапуск
# ============================================================
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
        allow_partial=True,
    )
