"""Real-tape verification — Study 322 (FOMC-Blackout). Regenerates docs/results.md.

Loads SPY daily total returns (cache-first from the shared repo cache), flags each
trading day as inside / outside the ~10-day pre-meeting FOMC communications blackout,
and reports the blackout-vs-rest mean, HAC t, Welch diff, the calendar-shift placebo,
the pre/post-2011 split, the volatility ("calm?") check, and the cost-swept timing book.
Network is touched only with --fetch.

    python studies/322-fomc-blackout/examples/verify.py            # cache-only
    python studies/322-fomc-blackout/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fomc_blackout import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    frame = data.load_real(fetch=fetch)
    print(f"SPY total-return tape: {frame.index[0].date()} -> {frame.index[-1].date()}, "
          f"{len(frame)} days   fingerprint {data.fingerprint(frame)}")
    print(f"blackout days: {int(frame['blackout'].sum())}  "
          f"non-blackout: {int((~frame['blackout']).sum())}\n")

    tbl = st.blackout_table(frame)
    print("=== Blackout vs non-blackout (daily total return) ===")
    for arm in ("blackout", "other", "all"):
        a = tbl[arm]
        print(f"  {arm:10s} n={a['n']:>5}  mean={a['mean_bps']:+6.2f} bps  "
              f"HAC t={a['tstat']:+5.2f}  Sharpe={a['sharpe_ann']:+5.2f}")
    print(f"  diff = {tbl['diff_bps']:+.2f} bps   Welch t = {tbl['welch_t']:+.2f}")

    vt = st.vol_table(frame)
    print(f"\n=== Calm? (annualised vol) ===\n  blackout={vt['blackout_vol_bps']:.0f} bps  "
          f"other={vt['other_vol_bps']:.0f} bps  ratio={vt['ratio']:.3f}")

    print("\n=== Calendar-shift placebo (diff vs window offset) ===")
    print(st.placebo_shift(frame, data.FOMC_DATES).round(2).to_string())

    era = st.split_by_era(frame)
    print("\n=== Pre/post-2011 ===")
    for k in ("pre", "post"):
        e = era[k]
        print(f"  {k:5s} diff={e['diff_bps']:+6.2f} bps  Welch t={e['welch_t']:+5.2f}  "
              f"n_blackout={e['n_blackout']}")

    print("\n=== Timing book (long market on blackout days, cash otherwise) ===")
    print(st.cost_sweep(frame).round(2).to_string())
    es = st.excess_sharpe(frame)
    print(f"  book excess Sharpe={es['book_excess_sharpe']:+.3f}  "
          f"buy&hold excess Sharpe={es['buyhold_excess_sharpe']:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
