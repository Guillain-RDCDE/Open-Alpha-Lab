"""Study 118 — Fed-Model.

The "Fed Model" (Yardeni 1997): stocks are cheap — and forward returns should be
high — when the equity earnings yield (E/P from Shiller Earnings/SP500) exceeds the
10-year Treasury yield (Shiller "Long Interest Rate"). We test whether the
*Fed-Model spread* (E/P minus 10y yield) forecasts forward 1-year and 10-year real
S&P returns, against an unconditional baseline, and also test E/P alone and the
spread over the REAL rate (Asness 2003's critique: the original model conflates real
and nominal).

Verdict (from docs/results.md):
  - Signal: WEAK — 10yr in-sample HAC t ≈ 5 (overlapping observations), OOS R² < 0 once
    the post-WWII interest-rate regime shifts; the 1yr signal is economically tiny.
  - Tradability: MIRAGE — the 10yr horizon is untradable (100% turnover every decade),
    and the 1yr signal is too weak and noisy to time the market.
"""
