"""Reproducible headline run for Study 852 - Movie-Sequel Fatigue.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached studio + SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from sequel_fatigue import data as dt, strategy as st  # noqa: E402


def fingerprint(prices) -> str:
    h = hashlib.sha256()
    for t in dt.ALL_TICKERS:
        s = prices[t]
        h.update(t.encode())
        h.update(np.ascontiguousarray(s.to_numpy()).tobytes())
        h.update(np.ascontiguousarray(s.index.view("int64")).tobytes())
    return h.hexdigest()[:12]


print("# Movie-Sequel Fatigue - do later franchise entries make the studio react worse?")

if not dt.have_real():
    print("(cache miss - fetching DIS + CMCSA + PARA + SPY once)")
    dt.fetch()

prices = dt.load_real()
print(f"calendar: {len(dt.EVENTS)} franchise entries across "
      f"{dt.events_frame()['franchise'].nunique()} sub-franchise lines, "
      f"{min(int(d[3][:4]) for d in dt.EVENTS)}->{max(int(d[3][:4]) for d in dt.EVENTS)}, hardcoded opening dates")
print(f"fingerprint {fingerprint(prices)}  (DIS+CMCSA+PARA+SPY total-return closes, "
      f"{len(prices['DIS'])} DIS rows {prices['DIS'].index.min().date()} -> "
      f"{prices['DIS'].index.max().date()}); as-of {dt.AS_OF}")

cars = st.build_event_cars(prices)
inc = cars[cars["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} "
      f"(3 pre-2021 Transformers drop - PARA continuity)")

print("\n# THE STUDIO REACTION - opening-reaction abnormal return (studio - SPY), "
      "[anchor-1 .. anchor+3]")
d0 = st.day0_stats(inc, "car")
print(f"  mean CAR {d0['mean']*100:+.3f}%  t={d0['t']:+.3f}  NW-t={d0['t_nw']:+.3f}  "
      f"up {d0['up_k']}/{d0['up_n']} (Wilson [{d0['up_lo']*100:.1f}%, {d0['up_hi']*100:.1f}%])")

print("\n# H1 - THE FATIGUE SLOPE (CAR regressed on sequel number). Claim: NEGATIVE.")
raw = st.fatigue_slope(inc, demean=False)
dem = st.fatigue_slope(inc, demean=True)
print(f"  raw            slope={raw['slope']*100:+.4f}%/seq  t={raw['t']:+.3f}  r={raw['r']:+.3f}")
print(f"  franchise-FE   slope={dem['slope']*100:+.4f}%/seq  t={dem['t']:+.3f}  r={dem['r']:+.3f}")

print("\n# H1 robustness - the franchise-FE slope by era (a Real stamp needs BOTH signs to hold)")
era = st.era_slopes(inc, demean=True)
print(f"  early (<2018) n={era['n_early']:2d}  slope={era['early']['slope']*100:+.4f}%  t={era['early']['t']:+.3f}")
print(f"  late  (>=2018) n={era['n_late']:2d}  slope={era['late']['slope']*100:+.4f}%  t={era['late']['t']:+.3f}")

print("\n# H2 - FATIGUE PERSISTENCE (does a down entry drag the next?). Claim: POSITIVE AR(1).")
per = st.fatigue_persistence(inc)
print(f"  pairs={per['n_pairs']}  AR(1) slope={per['ar1_slope']:+.4f}  t={per['ar1_t']:+.3f}")
print(f"  next-CAR | prev-down {per['down_mean']*100:+.3f}% (n={per['n_down']})  vs  "
      f"prev-up {per['up_mean']*100:+.3f}% (n={per['n_up']})  Welch t={per['welch_t']:+.3f}")

print("\n# PLACEBO 1 - permute the sequel-number labels (H1 falsification)")
pld = st.permute_slope_pvalue(inc, demean=True, n_perm=5000)
plr = st.permute_slope_pvalue(inc, demean=False, n_perm=5000)
print(f"  franchise-FE: obs {pld['obs_slope']*100:+.4f}%  vs null mean {pld['placebo_mean']*100:+.4f}% "
      f"(sd {pld['placebo_sd']*100:.4f}%)  p_left={pld['p_left']:.4f}  p_two={pld['p_two']:.4f}")
print(f"  raw:          obs {plr['obs_slope']*100:+.4f}%  p_left={plr['p_left']:.4f}  p_two={plr['p_two']:.4f}")

print("\n# PLACEBO 2 - random pseudo-event dates (are the per-event CARs ordinary noise?)")
rd = st.random_date_placebo(prices, inc)
print(f"  observed mean CAR {rd['obs']*100:+.3f}%  vs random-window mean {rd['placebo_mean']*100:+.3f}% "
      f"(sd {rd['placebo_sd']*100:.3f}%) over {rd['n_draws']:,} draws  p_two={rd['p_two']:.4f}")

print("\n# TRADABILITY - short the fatigued sequel (calendar-known entry), net of costs")
tm = st.fatigue_timer(inc, cost_bps=5.0)
tm10 = st.fatigue_timer(inc, cost_bps=10.0)
print(f"  n={tm['n']} fires  gross {tm['gross_mean']*100:+.3f}% (t={tm['t_gross']:+.2f})  "
      f"net@5bps {tm['net_mean']*100:+.3f}% (t={tm['t_net']:+.2f})  "
      f"net@10bps {tm10['net_mean']*100:+.3f}% (t={tm10['t_net']:+.2f})  "
      f"[cost {tm['cost_bps']:.1f} bps/round-trip]")

print("\n# SYNTHETIC POSITIVE CONTROL - deterministic, no network")
null_t = np.array([st.synthetic_detect(edge=0.0, seed=852 + s)["t"] for s in range(20)])
print(f"  null (edge=0), 20 seeds: mean slope-t = {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_t) >= 2).sum()}/20")
for e in (0.004, 0.008, 0.012):
    r = st.synthetic_detect(edge=e, seed=852)
    print(f"  planted edge={e:.3f}: slope {r['slope']*100:+.4f}%/seq  t = {r['t']:+.2f}")

print("\n# VERDICT")
print("  Signal: Weak | Tradability: Fragile  (see docs/results.md for the stamped table)")
