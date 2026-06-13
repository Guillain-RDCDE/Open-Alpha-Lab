"""Headline run for Study 101 (Slow-and-Steady) on the real total-return tape.

Runs the lump-sum vs dollar-cost-averaging (DCA) race over EVERY rolling 12-month
deployment window on SPY (total return), prints the numbers the README front-card and the
notebooks quote, and stamps the as-of date + content fingerprint. Re-run to reproduce;
match the fingerprint to confirm you hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from slow_and_steady import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
DEPLOY_MONTHS = 12


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    race = strategy.rolling_race(close, deploy_months=DEPLOY_MONTHS, step=1)
    s = strategy.win_stats(race)
    t = strategy.hac_tstat(race["ls_minus_dca"].to_numpy())

    print(f"\n--- Real tape (SPY total return), {DEPLOY_MONTHS}-month deployment, "
          f"cash earns 0% while waiting ---")
    print(f"  rolling windows                 : {s['n_windows']:,}")
    print(f"  lump-sum wins                   : {s['ls_wins']:,}/{s['n_windows']:,} "
          f"= {s['ls_win_rate']*100:.1f}%  "
          f"(Wilson 95% CI {s['ls_win_ci'][0]*100:.1f}-{s['ls_win_ci'][1]*100:.1f}%)")
    print(f"  mean terminal-wealth gap (LS-DCA): {s['mean_gap_pct']:+.2f} cents per $1  "
          f"({s['mean_gap']:+.4f})")
    print(f"  median terminal-wealth gap       : {s['median_gap_pct']:+.2f} cents per $1")
    print(f"  HAC t on the gap series          : {t:+.2f}")
    print(f"\n  -- terminal wealth of $1 over the window --")
    print(f"  lump-sum : mean {s['ls_mean']:.4f}  std {s['ls_std']:.4f}  worst {s['ls_worst']:.4f}")
    print(f"  DCA      : mean {s['dca_mean']:.4f}  std {s['dca_std']:.4f}  worst {s['dca_worst']:.4f}")
    print(f"  dispersion ratio (DCA std / LS std): {s['dispersion_ratio']:.3f}  "
          f"(< 1 means DCA outcomes are tighter -- DCA's one real benefit)")


if __name__ == "__main__":
    main()
