"""Real-tape verification — Study 304 (Carbon-Credits). Regenerates docs/results.md numbers.

Loads (or fetches) KRBN + BIL daily total-return tapes, measures buy-and-hold KRBN against
cash (excess-of-cash HAC t — the inference-bar number), then races a time-series momentum
overlay against naive buy-and-hold (excess-vs-excess Sharpe, block-bootstrap CI), sweeps
the lookback and costs, and prints the synthetic positive controls. Network only on --fetch.

    python studies/304-carbon-credits/examples/verify.py            # cache-only
    python studies/304-carbon-credits/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carbon_credits import data, strategy as st  # noqa: E402

AS_OF_END = "2026-05-31"  # drop the partial June 2026 month


def main(fetch: bool) -> None:
    if fetch:
        frame = data.load_real(("KRBN", "BIL"), use_cache=False)
        for tk in ("KRBN", "BIL"):
            sub = frame[[tk]].rename(columns={tk: "Close"})
            sub.to_parquet(data._cache_path(tk, data.DEFAULT_CACHE))
        print("refreshed cache:", os.listdir(data.DEFAULT_CACHE), "\n")

    f = data.load_real(("KRBN", "BIL"), end=AS_OF_END)
    print(f"KRBN+BIL tape: {f.index[0].date()} to {f.index[-1].date()}, {len(f)} days")
    print(f"Fingerprint: {data.fingerprint(f)}\n")

    R = st.to_returns(f)
    rf = R["BIL"]
    bh = st.buy_and_hold(R)
    s = st.stats(bh, rf)
    print("=== Buy-and-hold KRBN (total return) ===")
    print(f"CAGR={s['cagr']*100:+.1f}%  vol={s['vol']*100:.1f}%  "
          f"Sharpe(ex-cash)={s['sharpe']:.2f}  maxDD={s['max_dd']*100:.1f}%")
    print(f"Excess-of-cash HAC t = {st.excess_tstat(bh, rf):+.2f}  "
          "(the inference-bar number)\n")

    print("=== Calendar-year KRBN total return ===")
    yr = R["KRBN"].groupby(R.index.year).apply(lambda x: (1 + x).prod() - 1)
    for y, v in yr.items():
        print(f"  {y}: {v*100:+6.1f}%")
    print()

    print("=== Momentum overlay vs buy-and-hold (excess-vs-excess Sharpe) ===")
    for lb in (63, 126, 252):
        for cost in (0.0, 5.0, 10.0):
            sig = st.momentum_signal(f, "KRBN", lookback=lb)
            ov = st.overlay_returns(R, sig, "KRBN", "BIL", cost_bps=cost)
            so = st.stats(ov, rf)
            d = st.bootstrap_sharpe_diff(ov, bh, rf, n_boot=2000)
            print(f"  lb={lb:3d} cost={cost:4.1f}bps  OV Sharpe={so['sharpe']:+.2f} "
                  f"CAGR={so['cagr']*100:+5.1f}% maxDD={so['max_dd']*100:5.1f}%  "
                  f"OV-BH Sharpe={d['point']:+.2f} CI[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] "
                  f"P(OV>BH)={d['frac_a_wins']:.2f}")
    print()

    print("=== Synthetic positive controls (machinery proof, NOT market evidence) ===")
    fp, _ = data.synthetic_carbon(n_days=3000, mu=0.20, momentum=0.0, seed=304)
    Rp = st.to_returns(fp)
    print(f"  drift control (mu=0.20):    BH excess-of-cash HAC t = "
          f"{st.excess_tstat(st.buy_and_hold(Rp), Rp['CASH']):+.2f} (engine banks a real reward)")
    ft, _ = data.synthetic_carbon(n_days=4000, mu=0.05, momentum=0.40,
                                  daily_vol=0.025, seed=304)
    Rt = st.to_returns(ft)
    sigt = st.momentum_signal(ft, "KRBN", lookback=63)
    ovt = st.overlay_returns(Rt, sigt, "KRBN", "CASH", cost_bps=1.0)
    dt = st.bootstrap_sharpe_diff(ovt, st.buy_and_hold(Rt), Rt["CASH"], n_boot=2000)
    print(f"  timing control (mom=0.40):  OV-BH Sharpe = {dt['point']:+.2f} "
          f"P(OV>BH)={dt['frac_a_wins']:.2f} (engine banks real persistence)")
    fn, _ = data.synthetic_carbon(n_days=3000, mu=0.0, momentum=0.0, seed=304)
    Rn = st.to_returns(fn)
    sign = st.momentum_signal(fn, "KRBN", lookback=63)
    ovn = st.overlay_returns(Rn, sign, "KRBN", "CASH", cost_bps=1.0)
    dn = st.bootstrap_sharpe_diff(ovn, st.buy_and_hold(Rn), Rn["CASH"], n_boot=2000)
    print(f"  null (mu=0, mom=0):         BH excess-of-cash HAC t = "
          f"{st.excess_tstat(st.buy_and_hold(Rn), Rn['CASH']):+.2f}; "
          f"OV-BH Sharpe = {dn['point']:+.2f} (no reward, timing is a coin)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
