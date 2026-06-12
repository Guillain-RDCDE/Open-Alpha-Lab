"""Real-data run — the headline numbers behind Study 06's verdict.

Loads the cached ``^VIX`` (and ``^GSPC`` for the trade), works in **log-VIX**, and runs the
full gauntlet against the tweet's own claims:

    1. red-noise p-values at every period the tweet names (40d, 80d, 100d≈20wk, 250d, 1000d≈4y);
    2. which peaks, if any, clear the 95% / 99% AR(1) envelope;
    3. period stability — does the dominant in-band period hold, or wander (curve-fit tell);
    4. walk-forward direction skill at 10/20/40-session horizons vs a random-phase null;
    5. the tradeable expression — long the S&P when the VIX cycle is projected to fall —
       net Sharpe vs buy-and-hold and vs the null, with a bootstrap CI.

Cache-only (reuses Study 03's parquets). Prints every table and writes a reproducible
results doc (as-of date + a content fingerprint of the VIX close) to ``docs/results.md``.

    python examples/verify_real.py
"""

import os
import sys

import numpy as np
import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from vix_cycles import backtest, data, robustness, spectral
from quantlab.repro import DEFAULT_AS_OF, fingerprint

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

try:                                   # keep em-dashes/≈ printable on a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = os.path.join(_STUDY, "docs", "results.md")

BAND = (15.0, 400.0)            # the weeks-to-months cycle band the claim lives in
TARGETS = (40.0, 80.0, 100.0, 250.0, 1000.0)   # the tweet's own clocks (sessions)
N_SIM = 2000
N_NULL_SKILL = 999              # phase rotations for the walk-forward null (floor 1/1000)


def _round4(d: dict) -> dict:
    """Round the numeric values of a stats dict, passing strings (e.g. method tags) through."""
    return {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in d.items()}


def main():
    asof = DEFAULT_AS_OF
    vix = data.vix_series(asof=asof)
    spx = data.spx_series(asof=asof)
    logv = data.log_signal(vix)
    print(f"VIX: {len(vix)} sessions, {vix.index.min().date()} -> {vix.index.max().date()}")

    # 1 · red-noise p-values at the claimed periods
    pvals = robustness.red_noise_pvalues(logv, targets=TARGETS, n_sim=N_SIM, seed=0)
    print("\n[red-noise p-values at the claimed periods]\n", pvals.round(4).to_string())

    # 2 · what clears the envelope at all
    env = spectral.red_noise_envelope(logv, n_sim=N_SIM, seed=0)
    peaks95 = spectral.significant_peaks(env, q=0.95, band=BAND)
    peaks99 = spectral.significant_peaks(env, q=0.99, band=BAND)
    print(f"\n[significant peaks in {BAND} sessions]  rho={env['rho']:.3f}")
    print("  @95%:", "none" if len(peaks95) == 0 else
          peaks95.head(6).round(3).to_string(index=False))
    print("  @99%:", "none" if len(peaks99) == 0 else
          peaks99.head(6).round(3).to_string(index=False))

    # 3 · period stability
    stab = robustness.period_stability(logv, band=BAND, window=1000, step=125)
    print(f"\n[dominant in-band period over rolling 4y windows] "
          f"mean {stab['dominant_period'].mean():.0f}, std {stab['dominant_period'].std():.0f}, "
          f"range {stab['dominant_period'].min():.0f}-{stab['dominant_period'].max():.0f} sessions")

    # 4 · walk-forward direction skill at several horizons
    skill_rows = []
    for h in (10, 20, 40):
        r = backtest.oos_direction_skill(logv, band=BAND, horizon=h, min_train=750,
                                         step=10, n_null=N_NULL_SKILL, seed=0)
        skill_rows.append({"horizon": h, "skill": r["skill"], "null_mean": r["null_mean"],
                           "p_value": r["p_value"], "n_calls": r["n_calls"]})
    skill = pd.DataFrame(skill_rows).set_index("horizon")
    print("\n[walk-forward direction skill vs random-phase null]\n", skill.round(4).to_string())

    # 5 · the tradeable expression
    trade = backtest.phase_trade(logv, spx, band=BAND, min_train=750,
                                 refit=5, cost_bps=1.0, n_null=200, seed=0)
    boot = robustness.bootstrap_sharpe(trade.daily.iloc[750:])
    print("\n[cycle trade — long S&P when VIX cycle projected to fall]")
    print(_round4(trade.stats))
    print("  bootstrap Sharpe CI:", _round4(boot))

    # 6 · sub-period skill (does it ever work?)
    sub = robustness.subperiod_skill(logv, band=BAND, horizon=20, min_train=500,
                                     step=10, n_null=300, seed=0)
    print("\n[direction skill by decade]\n", sub.round(4).to_string())

    fp = fingerprint(vix.to_frame("Close"), round_dp=4)
    _write_results(OUT, vix, env, pvals, peaks95, peaks99, stab, skill, trade, boot, sub, asof, fp)
    print(f"\nwrote {OUT}")


def _write_results(path, vix, env, pvals, peaks95, peaks99, stab, skill, trade, boot, sub, asof, fp):
    def md(df):
        return "```\n" + df.round(4).to_string() + "\n```"

    peaks95_s = "none — nothing in the band clears the 95% red-noise envelope" if len(peaks95) == 0 \
        else md(peaks95.head(8))
    peaks99_s = "none" if len(peaks99) == 0 else md(peaks99.head(8))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Results — Study 06 (Clockwork-Vol) on the cached VIX

*Generated by [`examples/verify_real.py`](../examples/verify_real.py). Signal: **log-VIX**
(``^VIX`` daily close, taken raw — adjustment is meaningless for a level; see Study 03). The
S&P 500 spot (``^GSPC``) drives the tradeable expression. As-of **{asof}**. VIX close
fingerprint **`{fp}`** ({len(vix):,} sessions, {vix.index.min().date()} →
{vix.index.max().date()}). AR(1) red-noise null fitted at **rho = {env['rho']:.3f}**, with
**{env['n_sim']:,}** Monte-Carlo surrogates.*

> **The test, stated before the verdict.** A cycle is "real" here only if its periodogram
> peak **clears the AR(1) red-noise envelope** (a persistent level sprouts random peaks on
> its own), if its **period holds** rather than wandering, and if a **walk-forward** forecast
> built on it beats a **random-phase** null. All three are announced; none moves afterward.

## 1 · Red-noise p-values at the tweet's own clocks
*The claimed periods: VIX 40d & 80d, stocks' 20-week (~100d), the yearly (~250d) and 4-year
(~1000d). p < 0.05 would be a peak red noise struggles to fake.*
{md(pvals)}

## 2 · What clears the red-noise envelope (band 15–400 sessions)
- **@95%:** {peaks95_s}
- **@99%:** {peaks99_s}

## 3 · Does the dominant period hold, or wander?
Rolling 4-year (1000-session) windows, dominant in-band period:
**mean {stab['dominant_period'].mean():.0f}**, **std {stab['dominant_period'].std():.0f}**,
**range {stab['dominant_period'].min():.0f}–{stab['dominant_period'].max():.0f}** sessions.
{md(stab)}

## 4 · Walk-forward direction skill vs random-phase null
*Fit the dominant period + cosine on the past only, project the sign of the next-horizon
VIX move, score the hit rate. The null scrambles only the phase (period & amplitude kept),
**{N_NULL_SKILL} rotations**: p asks whether the **learned timing** beats an arbitrary one. One
honesty note on resolution: because the score is a hit rate over discrete calls, the null's
support is a step function of the rotation — ties with the observed skill count against it,
so p has an effective floor well above 1/{N_NULL_SKILL + 1:,} (see the fixed-80d discussion in
[extensions.md](extensions.md)).*
{md(skill)}

## 5 · The trade — long the S&P while the VIX cycle is projected to fall
*Positions read the fitted cycle's one-session projected slope — the alignment under which the
harness **banks a planted cycle** (the positive control in the notebooks and
`tests/test_backtest.py`), so its silence here is the market's, not the code's.*
`{_round4(trade.stats)}`
- bootstrap Sharpe CI: `{_round4(boot)}`

## 6 · Direction skill by decade (does it ever work?)
{md(sub)}

---
*Multiplicity, stated plainly: nothing above (or in [extensions.md](extensions.md)) is
corrected for the number of looks — ~5 target periods × 3 horizons × 4 rescue variants ×
4 sub-periods. At that count, a raw p in the few-percent range somewhere in the grid is
roughly what chance owes us.*
""")


if __name__ == "__main__":
    main()
