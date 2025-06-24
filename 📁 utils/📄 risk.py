def calculate_order_size(balance, risk_pct=1.5):
    return round(balance * (risk_pct / 100), 2)
