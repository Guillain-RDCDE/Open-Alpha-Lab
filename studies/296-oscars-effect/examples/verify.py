"""Real-data verification -- Study 296 (Oscars-Effect). Regenerates docs/results.md numbers.

Loads the cached ^GSPC daily tape (cache-first; if the study-local cache is missing it
stages a copy from the repo-level _cache), builds the event windows around each Oscar
ceremony, and runs the full event study: abnormal return on event day 0, the CAR
profile, the tradable rule net of costs, and the permutation/placebo control.

No network is required -- the tape ships in the repo-level _cache. Run from anywhere:

    python studies/296-oscars-effect/examples/verify.py
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oscars_effect import data, strategy as st  # noqa: E402


def _ensure_study_cache() -> None:
    """Stage the study-local cache from the repo-level ^GSPC tape if it is missing."""
    target = os.path.join(data.STUDY_CACHE, data._CACHE_FILE)
    if os.path.exists(target):
        return
    src = os.path.join(data.DEFAULT_CACHE, data._REPO_GSPC_FILE)
    if os.path.exists(src):
        os.makedirs(data.STUDY_CACHE, exist_ok=True)
        shutil.copyfile(src, target)


def main() -> None:
    print("Study 296 -- Oscars-Effect -- real-data verification\n")
    print("Data: ^GSPC daily (cache-first _cache/oscars_gspc_daily.parquet)")
    print("      Oscar ceremonies: hardcoded in data.py (1995-2025)\n")

    _ensure_study_cache()
    ret, cer = data.fetch_daily()
    fp = data.fingerprint(ret)

    print(f"Data stamp: {ret.index.min().date()} -> {ret.index.max().date()}, "
          f"n={len(ret)} sessions, fingerprint={fp}")
    print(f"Ceremonies in window: {len(cer)}\n")

    w = data.event_windows(ret, cer, pre=1, post=3)

    et = st.event_day_test(ret, w)
    print("=== Event day 0 (first session after the broadcast) ===")
    print(f"  n events:          {et['n_events']}")
    print(f"  Event-day mean:    {et['event_mean_bps']:+.1f} bps")
    print(f"  Baseline mean:     {et['baseline_mean_bps']:+.1f} bps")
    print(f"  Abnormal return:   {et['abnormal_bps']:+.1f} bps")
    print(f"  HAC t-stat:        {et['event_t']:+.2f}")
    print(f"  Up-rate:           {et['up_rate']:.1%}  (baseline {et['baseline_up_rate']:.1%})\n")

    print("=== CAR profile (relative days -1..+3) ===")
    print(st.car_profile(ret, w).to_string(index=False))
    print()

    tr = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=1.0)
    print("=== Tradable rule: buy day-0, hold 1 session ===")
    print(f"  n trades:          {tr['n_trades']}")
    print(f"  Gross mean/trade:  {tr['gross_mean_bps']:+.1f} bps")
    print(f"  Net mean/trade:    {tr['net_mean_bps']:+.1f} bps  (1.0 bps one-way)")
    print(f"  HAC t (net):       {tr['net_t']:+.2f}")
    print(f"  Buy-and-hold/day:  {tr['bah_mean_bps']:+.1f} bps\n")

    perm = st.permutation_null(ret, w, n_perm=5000, seed=296)
    print("=== Permutation / placebo control (5,000 random 31-day sets) ===")
    print(f"  Observed abnormal: {perm['abnormal_bps']:+.1f} bps")
    print(f"  Two-sided placebo p: {perm['placebo_p']:.3f}\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE "
          "-- the market does not care who wins Best Picture.")


if __name__ == "__main__":
    main()
