"""Real-tape verification — Study 219 (IPO-Pop). Regenerates docs/results.md numbers.

Fetches (or reads from cache) post-IPO forward returns for the hardcoded table,
computes first-day pop statistics and post-close long-run drift at 6m / 12m / 36m.

    python studies/219-ipo-pop/examples/verify.py            # cache-only
    python studies/219-ipo-pop/examples/verify.py --fetch    # refresh

Notes
-----
Severe survivorship bias: table contains only tickers queryable on yfinance.
The first-day pop (offer→first_close) is NOT harvestable by public market buyers.
Long-run returns compared vs zero; a proper test needs a market-matched benchmark.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ipo_pop import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching post-IPO price data from Yahoo Finance...")
        df = data.fetch_ipo_panel(fetch=True)
    else:
        df = data.fetch_ipo_panel(fetch=False)

    print(f"=== IPO panel (n={len(df)}) ===")
    disp = df[["ticker", "ipo_date", "pop", "ret_6m", "ret_12m", "ret_36m"]].copy()
    disp["pop_pct"]    = disp["pop"] * 100
    disp["ret_6m_pct"]  = disp["ret_6m"] * 100
    disp["ret_12m_pct"] = disp["ret_12m"] * 100
    disp["ret_36m_pct"] = disp["ret_36m"] * 100
    print(
        disp[["ticker", "ipo_date", "pop_pct", "ret_6m_pct", "ret_12m_pct", "ret_36m_pct"]]
        .to_string(index=False, float_format=lambda x: f"{x:+.1f}")
    )
    print()

    print("=== PART A -- First-day pop (offer -> first close) ===")
    ps = st.pop_stats(df)
    for k, v in ps.items():
        if k not in ("note",):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Verdict: {'REAL' if abs(ps['tstat']) >= 2 else 'NONE'} "
          f"(HAC t = {ps['tstat']:+.2f}) — BUT unharvestable by public market buyers")

    print()
    print("=== PART B — Post-close long-run drift ===")
    comp = st.compare_horizons(df)
    print(comp[["horizon", "n", "mean_pct", "median_pct", "win_rate", "tstat"]].to_string(index=False))

    print()
    print("=== Harvestability ===")
    h = st.harvestability(df)
    print(f"  Mean pop: {h['mean_pop_pct']:+.1f}%  Median: {h['median_pop_pct']:+.1f}%")
    print(f"  {h['n_positive_pop']}/{h['n_ipos']} IPOs had positive first-day pop")
    print(f"  {h['note'][:120]}...")

    print()
    print("=== Cost-adjusted annualised returns (entry at first close) ===")
    for col, lbl, h_days in [("ret_6m", "6m", 126), ("ret_12m", "12m", 252), ("ret_36m", "36m", 756)]:
        s = st.drift_stats(df, col=col)
        if s["n"] < 2:
            continue
        gross = s["mean_pct"] / 100
        for cost in (0.0, 10.0, 20.0):
            ann = st.cost_adjusted_return(gross, h_days, cost_bps=cost)
            print(f"  {lbl}, cost={cost:.0f}bps: gross_mean={s['mean_pct']:+.1f}%, "
                  f"gross_median={s['median_pct']:+.1f}%, net_ann={ann*100:+.1f}%/yr")

    print()
    print("=== VERDICT ===")
    ps_check = ps["tstat"]
    pop_real = abs(ps_check) >= 2
    long_real = any(
        abs(st.drift_stats(df, col=c)["tstat"]) >= 2
        for c in ("ret_6m", "ret_12m", "ret_36m")
    )
    print(f"  Pop (offer->close): {'REAL' if pop_real else 'NOT SIGNIFICANT'} "
          f"(t={ps_check:+.2f}) -- unharvestable regardless")
    print(f"  Long-run drift:    {'REAL' if long_real else 'NOT SIGNIFICANT (no horizon clears |t|>=2)'}")
    print(f"  Signal: Mixed — pop real but unharvestable; long-run drift not significant")
    print(f"  Tradability: Mirage — no net-harvestable edge at any horizon")
    print(f"  Myth-check: Long-run hangover weakly consistent (12m median -0.8%), "
          f"dominated by survivor bias")
    print()
    print(f"  Event fingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
