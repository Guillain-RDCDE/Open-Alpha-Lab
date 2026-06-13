"""Headline run for Study 93 (Round-Numbers) on the real tape.

Runs the two tests the study is built on — (a) the chi-square clustering test on the
distance from each close to its nearest round level, and (b) the round-number fade as a
fixed-horizon event study — across a basket (^GSPC, SPY, AAPL, MSFT), and pins the fade
against a random-level control that fades the same spacing at random phases. Prints the
numbers the README front-card and the notebooks quote, and stamps the as-of date + content
fingerprint. Re-run to reproduce; match the fingerprint to confirm you hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from round_numbers import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
BAND = 0.004
HORIZON = 1
N_SEEDS = 200
COST_BPS = 5.0


def main() -> None:
    names = data.load_basket(as_of=AS_OF)
    fp = data.fingerprint([close.to_numpy() for _, _, close in names])
    print(f"[data] basket of {len(names)} names  as-of {AS_OF}  fingerprint={fp}")
    for ticker, grid, close in names:
        print(f"   {ticker:6} grid={grid:6.0f}  {len(close):,} rows  "
              f"{close.index[0].date()} -> {close.index[-1].date()}")

    # ---- (a) Clustering: chi-square per name and pooled ----
    print("\n--- (a) CLUSTERING: distance to nearest round level, chi-square uniformity ---")
    all_close = []
    for ticker, grid, close in names:
        c = close.to_numpy()
        all_close.append(c)
        r = strategy.clustering_test(c, grid)
        print(f"   {ticker:6} n={r['n']:5}  chi2 {r['chi2']:8.1f}  p {r['p']:.1e}  "
              f"frac within 5% of round = {r['frac_near']*100:4.1f}% (uniform 10.0%)")
    pooled = np.concatenate(all_close)
    rp = strategy.clustering_test(pooled, 1.0)  # pooled on the whole-dollar scale
    print(f"   POOLED n={rp['n']:,}  chi2 {rp['chi2']:.1f}  p {rp['p']:.1e}  "
          f"frac near = {rp['frac_near']*100:.1f}%")

    # ---- (b) Tradability: the round-number fade vs the random-level control ----
    print("\n--- (b) TRADABILITY: fade an approach to a round level (gross, demeaned, "
          f"H={HORIZON}d, band={BAND*100:.1f}%) ---")
    for ticker, grid, close in names:
        tr = strategy.fade_returns(close, grid, band=BAND, horizon=HORIZON)
        print(f"   {ticker:6} trades {len(tr):4}  mean {tr.mean()*1e4:+6.2f} bps  "
              f"HAC t {strategy.hac_tstat(tr):+.2f}")
    real = strategy.pooled_fade(names, band=BAND, horizon=HORIZON)
    t_real = strategy.hac_tstat(real)
    net = strategy.net_of_costs(real, cost_bps=COST_BPS)
    rand_means = strategy.random_level_pool_means(names, n_seeds=N_SEEDS, band=BAND, horizon=HORIZON)
    beats = float((real.mean() > rand_means).mean())

    print(f"\n   POOLED real fade  : trades {len(real):,}  mean {real.mean()*1e4:+.2f} bps  "
          f"HAC t {t_real:+.2f}")
    print(f"   Net of {COST_BPS:.0f} bps/leg : mean {net.mean()*1e4:+.2f} bps/trade")
    print(f"   Random-level fade : mean {rand_means.mean()*1e4:+.2f} bps  "
          f"(sd {rand_means.std()*1e4:.2f}, {N_SEEDS} seeds)")
    print(f"   Real beats random-level on mean : {beats*100:.0f}% of {N_SEEDS} seeds")


if __name__ == "__main__":
    main()
