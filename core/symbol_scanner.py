"""
Dynamic Symbols (як у v8.2), скоринг: ліквідність × волатильність, анти-кореляція.
"""
import os, json, time, math, ccxt, numpy as np, pandas as pd
from typing import List, Tuple

EXCHANGE = os.getenv("EXCHANGE","mexc3")
DYNSYM_TOPN = int(os.getenv("DYNSYM_TOPN","12"))
DYNSYM_CANDIDATES = int(os.getenv("DYNSYM_CANDIDATES","60"))
DYNSYM_LIQ_USDT = float(os.getenv("DYNSYM_LIQ_USDT","2000000"))
DYNSYM_MAX_SPREAD = float(os.getenv("DYNSYM_MAX_SPREAD","0.001"))
DYNSYM_MAX_CORR = float(os.getenv("DYNSYM_MAX_CORR","0.82"))
DYNSYM_REFRESH_MIN = int(os.getenv("DYNSYM_REFRESH_MIN","60"))
CACHE_PATH = os.getenv("DYNSYM_CACHE_PATH","state/dynamic_symbols.json")

def _client():
    cls = getattr(ccxt, EXCHANGE) if hasattr(ccxt, EXCHANGE) else ccxt.mexc3
    return cls({"enableRateLimit": True})

def _save_cache(symbols: List[str], meta: dict):
    os.makedirs("state", exist_ok=True)
    json.dump({"ts": int(time.time()), "symbols": symbols, "meta": meta}, open(CACHE_PATH, "w"), indent=2)

def _load_cache():
    if not os.path.exists(CACHE_PATH): return None
    try: return json.load(open(CACHE_PATH))
    except: return None

def _safe_spread(ex, symbol) -> float:
    try:
        ob = ex.fetch_order_book(symbol, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not bid or not ask:
            return 1e9
        return (ask - bid) / ((ask + bid) / 2.0)
    except Exception:
        return 1e9

def _returns_vol(df: pd.DataFrame) -> float:
    r = df["c"].pct_change().dropna()
    if len(r) < 10: return 0.0
    return float(r.std())

def _series_for_corr(df: pd.DataFrame) -> pd.Series:
    r = df["c"].pct_change().dropna()
    return r.tail(100).reset_index(drop=True) if len(r)>0 else pd.Series([0.0])

def _anti_correlate(symbols_scored: List[Tuple[str, float]], ex) -> List[str]:
    picked = []
    series_map = {}
    for sym, _ in symbols_scored:
        try:
            o = ex.fetch_ohlcv(sym, timeframe="1h", limit=200)
            df = pd.DataFrame(o, columns=["t","o","h","l","c","v"])
            series_map[sym] = _series_for_corr(df)
        except Exception:
            series_map[sym] = pd.Series([0.0])
    for sym, _ in symbols_scored:
        ok = True
        for psym in picked:
            s1, s2 = series_map.get(sym), series_map.get(psym)
            if s1 is None or s2 is None: continue
            n = min(len(s1), len(s2))
            if n < 5: continue
            c = float(np.corrcoef(s1.tail(n), s2.tail(n))[0,1])
            if math.isnan(c): c = 0.0
            if c >= DYNSYM_MAX_CORR:
                ok = False; break
        if ok: picked.append(sym)
        if len(picked) >= DYNSYM_TOPN: break
    return picked

def rank_symbols(ex=None, top_n=DYNSYM_TOPN, candidates=DYNSYM_CANDIDATES) -> List[str]:
    if ex is None: ex = _client()
    markets = ex.load_markets()
    usdt_pairs = [s for s in markets if s.endswith("/USDT")]
    tickers = ex.fetch_tickers(usdt_pairs)
    liq = []
    for sym, t in tickers.items():
        try:
            qv = t.get("quoteVolume") or ((t.get("baseVolume") or 0) * (t.get("last") or 0))
            if (qv or 0) < DYNSYM_LIQ_USDT: continue
            sp = _safe_spread(ex, sym)
            if sp > DYNSYM_MAX_SPREAD: continue
            liq.append((sym, float(qv)))
        except Exception:
            continue
    liq.sort(key=lambda x: x[1], reverse=True)
    cands = [s for s,_ in liq[:candidates]]

    scored = []
    for sym in cands:
        try:
            o = ex.fetch_ohlcv(sym, timeframe="1h", limit=120)
            if not o: continue
            df = pd.DataFrame(o, columns=["t","o","h","l","c","v"])
            vol = _returns_vol(df); vavg = float(df["v"].tail(60).mean() or 0.0)
            score = vol * math.log1p(vavg)
            if score > 0: scored.append((sym, score))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored: return []

    final_syms = _anti_correlate(scored, ex)
    return final_syms[:top_n]

def get_dynamic_symbols(top_n=DYNSYM_TOPN, force_refresh=False):
    cache = _load_cache()
    if cache and not force_refresh:
        age = time.time() - cache.get("ts",0)
        if age <= DYNSYM_REFRESH_MIN * 60:
            return cache.get("symbols", [])
    ex = _client()
    syms = rank_symbols(ex, top_n=top_n)
    _save_cache(syms, {"exchange": EXCHANGE, "top_n": top_n, "ts": int(time.time())})
    return syms or ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","TON/USDT"]
