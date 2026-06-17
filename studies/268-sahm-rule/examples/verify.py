"""Real-data verification -- Study 268 (Sahm-Rule). Regenerates docs/results.md numbers.

Loads the hardcoded BLS unemployment table, computes the Sahm trigger, loads the
cached ^GSPC daily price (repo-level _cache, read-only -- no fetch required), and
runs the full analysis: trigger onsets, forward-return event study with a
one-month execution lag, Welch t, HAC t, block-permutation p, and the net-of-cost
timing overlay.

Cache-only by default. Run from anywhere:

    python studies/268-sahm-rule/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sahm_rule import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 268 -- Sahm-Rule -- real-data verification\n")
    print("Data: BLS U-3 SA unemployment, hardcoded in data.py (1959-2025)")
    print("      ^GSPC daily close (price-only) from repo-level _cache\n")

    u = data.unrate_series()
    g = data.fetch_gspc_daily()
    print(f"Unemployment: {u.index.min():%Y-%m}..{u.index.max():%Y-%m}  "
          f"n={len(u)}  fingerprint={data.fingerprint(u)}")
    print(f"^GSPC close : {g.index.min():%Y-%m-%d}..{g.index.max():%Y-%m-%d}  "
          f"n={len(g)}  fingerprint={data.fingerprint(g)}\n")

    onsets = st.trigger_onsets(u)
    print(f"Sahm trigger onsets: {len(onsets)}")
    for d in onsets:
        print("   ", d.strftime("%Y-%m"))
    print()

    es = st.event_study(u, g, horizon=12, exec_lag=1)
    hac_t = st.hac_tstat(es["event_fwd"] - es["uncond_mean"])
    perm = st.block_permutation_p(u, g, horizon=12, exec_lag=1, n_perm=5_000, seed=268)

    print("=== Event study (H=12m forward, +1m execution lag) ===")
    print(f"  n events:            {es['n_events']}")
    print(f"  Post-trigger mean:   {es['event_mean']*100:+.1f}%")
    print(f"  Unconditional mean:  {es['uncond_mean']*100:+.1f}%")
    print(f"  Difference:          {es['diff']*100:+.1f}pp")
    print(f"  Welch t (event/all): {es['welch_t']:+.3f}  (p={es['welch_p']:.3f})")
    print(f"  HAC/Newey-West t:    {hac_t:+.3f}")
    print(f"  Block-permutation p: {perm['perm_p']:.4f}  (one-sided, trigger -> low)\n")

    print("=== Horizon robustness ===")
    for H in (6, 12, 24):
        e = st.event_study(u, g, horizon=H, exec_lag=1)
        print(f"  H={H:>2}m: event {e['event_mean']*100:+.1f}%  "
              f"uncond {e['uncond_mean']*100:+.1f}%  "
              f"diff {e['diff']*100:+.1f}pp  welch_t {e['welch_t']:+.3f}")
    print()

    ov = st.timing_overlay(u, g, exec_lag=1, cost_bps=10.0, stay_out_months=12)
    o, b = ov["overlay"], ov["bah"]
    print("=== Timing overlay (out 12m on trigger, 10bps/switch) ===")
    print(f"  Overlay: CAGR {o['cagr']*100:+.1f}%  vol {o['vol']*100:.1f}%  "
          f"Sharpe {o['sharpe']:.2f}  MDD {o['mdd']*100:.1f}%")
    print(f"  B&H    : CAGR {b['cagr']*100:+.1f}%  vol {b['vol']*100:.1f}%  "
          f"Sharpe {b['sharpe']:.2f}  MDD {b['mdd']*100:.1f}%")
    print(f"  Switches: {ov['n_switches']}   active return HAC t: {ov['active_hac_t']:+.3f}\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a recession thermometer, not a sell button.")


if __name__ == "__main__":
    main()
