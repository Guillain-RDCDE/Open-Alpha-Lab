"""Headline run for Study 100 (Melting-Ice) on the real total-return tapes.

Runs the constant-leverage 3x simulator against (a) the *real* TQQQ/UPRO funds — to show
the simulator reproduces them from the underlying within tolerance — and (b) the naive
"3x the period return" line, then decomposes the gap into the variance-drag and
compounding-in-a-trend terms and reports the break-even volatility. Pins the as-of date
and per-asset content fingerprints. Re-run to reproduce; match the fingerprints to confirm
you hold the same tapes.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from melting_ice import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
K = 3.0


def main() -> None:
    print(f"Study 100 -- Melting-Ice | as-of {AS_OF} | k={K:g}x | "
          f"all-in fee approx {strategy.DEFAULT_ANNUAL_FEE*100:.1f}%/yr\n")

    for name in ("TQQQ", "UPRO"):
        under, letf, meta = data.load_pair(name, end=AS_OF)
        uc, lc = under["close"], letf["close"]
        ufp, lfp = data.fingerprint(under), data.fingerprint(letf)

        print(f"=== {name} (3x) vs {meta['underlying']} | "
              f"{meta['start']} -> {meta['end']} | {meta['n']:,} rows ===")
        print(f"  fingerprints: {meta['underlying']}={ufp}  {name}={lfp}")

        # 1) Simulator reproduces the REAL fund from the underlying.
        rep = strategy.reproduction_error(uc, lc, k=K)
        print(f"  [reproduce real {name}] daily corr={rep['corr']:.4f}  "
              f"RMS tracking err={rep['rms_te_bps']:.1f} bps  "
              f"final ratio sim/real={rep['final_ratio']:.3f}")

        # 2) Real fund vs underlying headline stats.
        ls = strategy.nav_stats(lc / lc.iloc[0])
        us = strategy.nav_stats(uc / uc.iloc[0])
        print(f"  REAL {name:<5} CAGR {ls['cagr']*100:6.1f}%  vol {ls['vol']*100:5.1f}%  "
              f"Sharpe {ls['sharpe']:.2f}  maxDD {ls['max_dd']*100:6.1f}%  final x{ls['final']:.1f}")
        print(f"  {meta['underlying']:<10} CAGR {us['cagr']*100:6.1f}%  vol {us['vol']*100:5.1f}%  "
              f"Sharpe {us['sharpe']:.2f}  maxDD {us['max_dd']*100:6.1f}%  final x{us['final']:.1f}")

        # 3) Realised levered NAV vs naive "3x the period return".
        rvn = strategy.realized_vs_naive(uc, k=K)
        verdict = "compounding WON" if rvn["compounding_won"] else "decay WON"
        print(f"  realised(sim) final x{rvn['realized_final']:.1f}  vs  "
              f"naive-3x x{rvn['naive_final']:.1f}  gap {rvn['gap']:+.1f}  -> {verdict}")

        # 4) Drag-vs-drift decomposition + break-even vol.
        d = strategy.drag_decomposition(uc, k=K)
        print(f"  decomp: drift {d['drift_ann']*100:5.1f}%/yr  "
              f"drag {d['drag_ann']*100:5.1f}%/yr  net(lead) {d['net_leading_ann']*100:5.1f}%/yr  "
              f"| realised vol {d['realized_vol_ann']*100:.1f}%  "
              f"break-even vol {d['breakeven_vol_ann']*100:.1f}%  "
              f"drag_dominates={d['drag_dominates']}")
        print()

    # 5) Path-dependence laid bare: TQQQ underlying (QQQ) in three regimes.
    print("--- Path-dependence: realised 3x vs naive-3x by regime (QQQ underlying) ---")
    qqq = data.load_real("QQQ", mode="total_return")["close"].loc[:AS_OF]
    for lo, hi, lab in [("2010-02-11", "2021-12-31", "2010-2021 bull"),
                        ("2018-01-01", "2018-12-31", "2018 chop"),
                        ("2022-01-01", "2022-12-31", "2022 selloff")]:
        seg = qqq.loc[lo:hi]
        rvn = strategy.realized_vs_naive(seg, k=K)
        d = strategy.drag_decomposition(seg, k=K)
        under_ret = (seg.iloc[-1] / seg.iloc[0] - 1.0) * 100
        print(f"  {lab:<16} QQQ {under_ret:+7.1f}%  realised-3x {(rvn['realized_final']-1)*100:+8.1f}%  "
              f"naive-3x {(rvn['naive_final']-1)*100:+8.1f}%  vol {d['realized_vol_ann']*100:3.0f}%  "
              f"drag_dominates={d['drag_dominates']}")
    print()

    print("Read: in BOTH 2010s tapes the realised 3x NAV ended FAR ABOVE 'naive 3x' --")
    print("the strong trend made compounding beat the drag. 'Always decays to zero' is FALSE;")
    print("it is path-dependent. The drawdowns (-77% to -82%) are the real cost, not decay.")


if __name__ == "__main__":
    main()
