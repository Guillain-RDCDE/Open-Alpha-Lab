"""Headline run for Study 88 (Dogs-of-the-Dow) on the real point-in-time tape.

Builds the annual Dogs portfolio — each January buy the ten highest trailing-yield stocks
in the **then-current** Dow 30, equal-weight, hold a year — against the same-basis Dow
total-return benchmark (DIA), net of 20 bps/turnover, and prints the numbers the README
front-card and the notebooks quote. Stamps the as-of date + content fingerprint; rerun and
match the fingerprint to confirm you hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from dogs_of_the_dow import data, strategy  # noqa: E402

AS_OF = "2026-06-13"
COST_BPS = 20.0
FIRST_YEAR, LAST_YEAR = 2000, 2025


def main() -> None:
    # Cache lives under the study root _cache/ regardless of cwd.
    cache = os.path.abspath(os.path.join(HERE, "..", "_cache"))
    panel = data.load_panel(end=AS_OF, cache_dir=cache)
    fp = data.fingerprint(panel)
    px = panel["prices"]
    print(f"[data] Dow panel: {px.shape[1]} tickers x {len(px):,} days  "
          f"bench {panel['bench'].index[0].date()} -> {panel['bench'].index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")
    print(f"[universe] point-in-time Dow-30 from 1999-11-01; sample {FIRST_YEAR}-{LAST_YEAR} "
          f"(26 January rebalances). Unrecoverable members left out and disclosed: "
          f"{', '.join(data.UNRECOVERABLE)} (EK 2000-04, WBA 2019-24) -- 29 of 30 priced each year.")

    ann = strategy.annual_returns_from_panel(panel, FIRST_YEAR, LAST_YEAR, cost_bps=COST_BPS)
    s = strategy.summarize(ann)
    exc = ann["excess"].to_numpy()
    t = strategy.hac_tstat(exc)
    lo, hi, p = strategy.block_bootstrap_ci(exc)
    ab = strategy.alpha_beta(ann)

    print(f"\n--- Dogs vs Dow (DIA total return), {s['n_years']} years, "
          f"net of {COST_BPS:.0f} bps/turnover ---")
    print(f"  Dogs   CAGR {s['dogs_cagr']*100:6.2f}%   grew $1 -> ${s['total_dogs']:.2f}")
    print(f"  Dow    CAGR {s['dow_cagr']*100:6.2f}%   grew $1 -> ${s['total_dow']:.2f}")
    print(f"  CAGR gap (Dogs - Dow)      = {s['cagr_gap']*100:+.2f} pts/yr")
    print(f"  Mean annual excess         = {s['mean_excess']*100:+.2f}%/yr")
    print(f"  Win rate (Dogs > Dow)      = {s['win_rate']*100:.0f}% "
          f"({int(round(s['win_rate']*s['n_years']))}/{s['n_years']} years)")
    print(f"\n  HAC t on annual excess     = {t:+.2f}   (|t| < 2 -> not certifiable)")
    print(f"  Block-bootstrap 95% CI     = [{lo*100:+.2f}%, {hi*100:+.2f}%]  "
          f"two-sided p = {p:.2f}")
    print(f"\n  alpha-vs-beta (Dogs ~ Dow):")
    print(f"    alpha = {ab['alpha']*100:+.2f}%/yr  (t = {ab['alpha_t']:+.2f})")
    print(f"    beta  = {ab['beta']:.2f}   R2 = {ab['r2']:.2f}")
    print(f"  -> beta < 1 + alpha t < 2: a defensive high-yield tilt, not a certified free lunch.")


if __name__ == "__main__":
    main()
