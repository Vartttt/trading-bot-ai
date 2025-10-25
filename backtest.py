# backtest.py
import ccxt, numpy as np
from indicators import df_from_ohlcv, add_indicators, latest_row
from strategy import compute_signal_with_mtf, choose_leverage
from risk import position_size_usd

def backtest(symbol="BTC/USDT", timeframe="15m", candles=1500):
    ex = ccxt.mexc({"enableRateLimit": True, "options":{"defaultType":"swap"}})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles, params={"type":"swap"})
    eq = 1000.0
    peak = eq
    trades, wins, losses = 0, 0, 0
    for i in range(400, len(ohlcv)-1):
        df = add_indicators(df_from_ohlcv(ohlcv[:i]))
        side, conf, _ = compute_signal_with_mtf(ex, symbol, df)
        lev, _ = choose_leverage(conf)
        price = df["close"].iloc[-1]
        notional = position_size_usd(eq) * lev
        amount = notional / price
        # простий TP/SL
        tp = price * (1.006 if side=="LONG" else 0.994)
        sl = price * (0.996 if side=="LONG" else 1.004)
        nxt = ohlcv[i+1][4]
        # результат наступної свічки (демо)
        pnl = (nxt - price) * amount * (1 if side=="LONG" else -1)
        eq += pnl
        peak = max(peak, eq)
        trades += 1
        if pnl > 0: wins += 1
        else: losses += 1
    print(f"trades={trades}, wins={wins}, losses={losses}, eq={eq:.2f}, maxDD={(peak-eq)/peak:.3%}")

if __name__ == "__main__":
    backtest()
