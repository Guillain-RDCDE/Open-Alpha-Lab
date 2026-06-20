"""Reproduce the headline numbers in docs/results.md for Study 310 (Platinum-Palladium).

Cache-first: reads the cached PL=F / PA=F parquets under ../_cache/ if present, else
fetches from Yahoo (``--fetch``). The reproducible headline run drops the partial current
month (as-of 2026-05-31). Run from the study root:

    python examples/verify.py            # use cache
    python examples/verify.py --fetch    # populate cache from Yahoo, then compute
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from platinum_palladium import data, strategy as st  # noqa: E402

AS_OF = pd.Timestamp("2026-05-31")  # drop the partial June 2026 month


def main(fetch: bool = False) -> None:
    plat = data.fetch_daily(data.PLATINUM_TICKER, fetch=fetch)
    pall = data.fetch_daily(data.PALLADIUM_TICKER, fetch=fetch)
    plat = plat[plat.index <= AS_OF]
    pall = pall[pall.index <= AS_OF]

    print(f"PL=F  n={len(plat):,}  {plat.index.min().date()} -> {plat.index.max().date()}"
          f"  fp={data.fingerprint(plat)}")
    print(f"PA=F  n={len(pall):,}  {pall.index.min().date()} -> {pall.index.max().date()}"
          f"  fp={data.fingerprint(pall)}")

    ratio = st.compute_ratio(plat, pall)
    print(f"\nratio range {ratio.min():.3f} - {ratio.max():.3f}  mean {ratio.mean():.3f}")

    print("\n-- structural tests --")
    print(f"ADF p-value      : {st.adf_pvalue(ratio):.4f}")
    print(f"OU half-life     : {st.half_life(ratio):.1f} trading days")
    print(f"Engle-Granger p  : {st.engle_granger_pvalue(plat['close'], pall['close']):.4f}")

    ent = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    led = st.run_trades(plat, pall, ent, cost_bps=0.0)
    rnd = st.run_trades(plat, pall, st.random_entries(ent, seed=310), cost_bps=0.0)
    s = st.summarize(led, "ret_gross")
    r = st.summarize(rnd, "ret_gross")
    lo, hi = st.block_bootstrap_ci(led, "ret_gross")
    print("\n-- strategy (enter |z|>1.5, exit |z|<0.5) --")
    print(f"STRAT n={s['n_trades']}  win={s['win_rate']:.3f}  mean={s['mean_bps']:+.1f} bps"
          f"  HAC t={s['tstat']:+.2f}  skew={s['skew']:+.2f}")
    print(f"RAND  n={r['n_trades']}  win={r['win_rate']:.3f}  mean={r['mean_bps']:+.1f} bps"
          f"  HAC t={r['tstat']:+.2f}")
    print(f"bootstrap 95% CI on mean: [{lo:+.1f}, {hi:+.1f}] bps")
    print(f"avg hold: {led['trading_days'].mean():.0f} trading days")

    print("\n-- cost sweep (net) --")
    for c in (0, 5, 10, 20):
        ss = st.summarize(st.run_trades(plat, pall, ent, cost_bps=c), "ret_net")
        print(f"cost {c:>2} bps: mean={ss['mean_bps']:+.1f}  HAC t={ss['tstat']:+.2f}")

    print("\n-- buy and hold --")
    for nm, b in (("PL=F", plat), ("PA=F", pall)):
        tot = st.bah_return(b)
        yrs = (b.index.max() - b.index.min()).days / 365.25
        ann = (np.exp(tot / yrs) - 1) * 100
        print(f"{nm}: total log {tot*100:+.1f}%  ann {ann:+.1f}%/yr")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
