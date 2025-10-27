from core.market_phase import compute_phase_from_df

def filter_symbol_phase(df1h, df4h, global_phase):
    if df1h is None or len(df1h)==0:
        return 1.0, "no_data", "UNKNOWN", "range"
    local = compute_phase_from_df(df1h, df4h)
    local_phase = local.get("phase","UNKNOWN")
    local_regime = local.get("regime","range")
    gphase = (global_phase or {}).get("phase","UNKNOWN")

    mult, comment = 1.0, "aligned"
    if gphase.startswith("BULL") and local_phase.startswith("BULL"): mult=1.1
    elif gphase.startswith("BEAR") and local_phase.startswith("BEAR"): mult=1.1
    elif gphase.startswith("RANGE") and "RANGE" in local_phase: mult=1.0
    elif gphase == "SPIKE_EVENT": mult,comment = 0.6,"market spike"
    else: mult,comment = 0.7,f"phase mismatch ({local_phase} vs {gphase})"
    return round(mult,2), comment, local_phase, local_regime
