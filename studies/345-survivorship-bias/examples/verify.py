"""Real-tape / synthetic verification — Study 345 (Survivorship-Bias). Regenerates results.md.

Runs the SAME contrarian "buy past losers" book on two views of a panel — FULL (with the
firms that delist) vs SURVIVORS (current-membership-only) — and reports the survivorship
delta: the gap in Sharpe, mean return, and HAC-t between the two reads. On the synthetic
tape it sweeps the planted death rate; on the real tape (if a cache is present) it
illustrates the bias on a live large-cap universe through the opt-in guard.

    python studies/345-survivorship-bias/examples/verify.py           # synthetic + cache-only real
    python studies/345-survivorship-bias/examples/verify.py --fetch   # refresh the real tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from survivorship_bias import data, strategy as st  # noqa: E402

LOOKBACK = 12
FRAC = 0.2
SEED = 345


def synthetic_sweep() -> None:
    print("=== Synthetic: contrarian long-only 'buy past losers', FULL vs SURVIVORS ===")
    print("death_rate  view       Sharpe   mean%/yr   HAC-t      delta_Sharpe  flip")
    for dr in (0.0, 0.15, 0.30, 0.45):
        rf, alive, truth = data.synthetic_panel(
            n_periods=240, n_assets=200, death_rate=dr, signal_strength=0.0, seed=SEED)
        surv = data.survivors_only(rf, alive)
        res = st.compare_views(rf, surv, lookback=LOOKBACK, frac=FRAC,
                               long_short=False, direction=-1, cost_bps=0.0, seed=SEED)
        sf, ss = res["summary_full"], res["summary_survivors"]
        tf, ts = res["test_full"], res["test_survivors"]
        print(f"  {dr:4.2f}      FULL       {sf['sharpe']:+5.2f}   {tf['mean_ann']*100:+6.2f}   "
              f"{tf['tstat']:+5.2f}")
        print(f"            SURVIVORS  {ss['sharpe']:+5.2f}   {ts['mean_ann']*100:+6.2f}   "
              f"{ts['tstat']:+5.2f}      {res['delta_sharpe']:+5.2f}      "
              f"{res['sign_flip']}")


def real_illustration(fetch: bool) -> None:
    print("\n=== Real illustration (large-cap convenience universe, opt-in guard) ===")
    try:
        rets = data.load_real(fetch=fetch, allow_survivorship_bias=True)
    except (FileNotFoundError, Exception) as e:  # noqa: BLE001
        print(f"  (no real cache / offline: {type(e).__name__}) — synthetic core stands alone.")
        return
    print(f"  panel: {rets.index[0].date()} .. {rets.index[-1].date()}  "
          f"({len(rets)} months, {rets.shape[1]} names)")
    print(f"  fingerprint: {data.fingerprint(rets)}")
    # The convenience universe IS survivors-only; we cannot recover the delisted names
    # here, so the real illustration is the *upper-bound* read this universe gives.
    res = backtest_real(rets)
    print(f"  contrarian buy-losers on survivors-only real tape: Sharpe={res['sharpe']:+.2f} "
          f"mean={res['mean_ann']*100:+.2f}%/yr HAC-t={res['t']:+.2f} (n={res['n']})")


def backtest_real(rets) -> dict:
    r = st.backtest(rets, lookback=LOOKBACK, frac=FRAC, long_short=False, direction=-1)
    s = st.summarize(r)
    t = st.hac_tstat(r)
    return {"sharpe": s["sharpe"], "mean_ann": t["mean_ann"], "t": t["tstat"], "n": t["n"]}


if __name__ == "__main__":
    synthetic_sweep()
    real_illustration(fetch="--fetch" in sys.argv)
