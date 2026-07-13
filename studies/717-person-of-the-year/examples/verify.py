"""Reproducible headline run for Study 717 — Person-of-the-Year.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily closes under
``_cache/`` if present (the real-tape event study), and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from person_of_the_year import data, strategy as st

print("# Person-of-the-Year — long-horizon market-model abnormal returns after the cover")
if data.have_real():
    prices, events = data.load_real()
    print(f"honoree tickers cached : {sorted(c for c in prices.columns if c != 'SPY')}")
    print(f"honorees with price data: {len(events)} of {len(data.POY_EVENTS)} "
          f"(table fingerprint {data.fingerprint(data.POY_EVENTS)})")
    print(f"named-but-untradable    : {[d[0]+' '+d[1] for d in data._DROPPED]}")

    panel = st.car_panel(prices, events, window=(1, 252), with_runup=True)
    print("\n# Canonical 12-month post-coronation CAR[+1,+252] (market model, SPY, total-return)")
    print(panel[["ticker", "honoree", "direct", "car", "runup"]].to_string(
        index=False, formatters={"car": lambda v: f"{v*100:+.1f}%",
                                 "runup": lambda v: f"{v*100:+.1f}%"}))

    s = st.summarize(panel, prices=prices, tickers=data.TICKERS, window=(1, 252), n_draws=8000)
    b = s["all"]
    print(f"\n  POOLED: n={b['n']}  mean CAR={b['mean_pct']:+.1f}%  "
          f"curse-rate(CAR<0)={b['curse']*100:.0f}%  t(vs 0)={b['t']:+.2f}")
    print(f"  placebo p (random mid-Dec windows, one-sided-left) = {s['placebo_p_left']:.3f} "
          f"(two-sided {s['placebo_p_two']:.3f}); null mean = {s['null_mean_pct']:+.1f}%")

    print("\n# Horizon sweep — the 'curse' is a slow bleed, absent at 1 month")
    for w, lab in [((1, 21), "1m"), ((1, 63), "3m"), ((1, 126), "6m"), ((1, 252), "12m")]:
        pw = st.car_panel(prices, events, window=w)
        sw = st.summarize(pw, prices=prices, tickers=data.TICKERS, window=w, n_draws=4000)
        bb = sw["all"]
        print(f"  {lab:>3} {str(w):>9}: mean={bb['mean_pct']:+7.1f}%  t={bb['t']:+5.2f}  "
              f"placebo p_left={sw['placebo_p_left']:.3f}")

    print("\n# Leave-one-out (12m) — is the pooled t hostage to one name?")
    car = panel["car"].to_numpy()
    tk = panel["ticker"].tolist()
    for i, t in enumerate(tk):
        loo = np.delete(car, i)
        print(f"  drop {t:>5}: mean={loo.mean()*100:+7.1f}%  t={st.welch_t(loo):+5.2f}")

    print("\n# THE CONFOUND — regress 12m CAR on the prior-year run-up (selection)")
    run = panel["runup"].to_numpy()
    bcoef, acoef = np.polyfit(run, car, 1)
    resid = car - (acoef + bcoef * run)
    print(f"  corr(prior run-up, post CAR) = {np.corrcoef(run, car)[0, 1]:+.3f}")
    print(f"  CAR = {acoef:+.3f} + {bcoef:+.3f}*runup")
    print(f"  RESIDUAL cover-curse after removing run-up: mean={resid.mean()*100:+.1f}%  "
          f"t={st.welch_t(resid):+.2f}  <-- the magazine adds ~nothing")

    print("\n# TRADABILITY — realized short P&L (short at +1 day, hold 12m, pay borrow)")
    grosses, nets = [], []
    for e in events:
        srs = prices[e["ticker"]].dropna()
        pos = int(np.searchsorted(srs.index, e["announce_date"]))
        path = srs.values[pos + 1:pos + 1 + 252]
        if len(path) < 10:
            continue
        raw12 = path[-1] / srs.values[pos + 1] - 1.0
        max_adv = path.max() / srs.values[pos + 1] - 1.0     # worst squeeze vs a short
        nc = st.net_of_costs(raw12, borrow_ann=0.05)
        grosses.append(nc["gross_pct"]); nets.append(nc["net_pct"])
        print(f"  {e['ticker']:>5}: short gross={nc['gross_pct']:+6.1f}%  "
              f"net@5%borrow+10bps={nc['net_pct']:+6.1f}%  "
              f"max-adverse(squeeze)={max_adv*100:+5.1f}%")
    print(f"  average short: gross={np.mean(grosses):+.1f}%  net={np.mean(nets):+.1f}%  "
          f"(costs are NOT the killer — capacity, path-risk and it-being-momentum are)")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED post-coronation drift and must NOT manufacture a")
print("  'curse' from four events when the true edge is 0.")
for edge in (0.0, -3000.0, -12000.0):
    syn = data.synthetic_events(curse_bps=edge, seed=717)
    fc = st.summarize_bucket(syn["car"])
    print(f"  planted curse_bps={edge:+8.0f}: n={fc['n']}  mean CAR={fc['mean_pct']:+7.1f}%  "
          f"t={fc['t']:+5.2f}")
