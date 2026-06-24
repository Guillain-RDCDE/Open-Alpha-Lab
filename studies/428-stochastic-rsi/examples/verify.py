"""Real-tape verification — Study 428 (Stochastic RSI). Regenerates docs/results.md numbers.

Reads the cached daily tapes (offline) and reproduces every headline number: the SPY
long-flat Sharpe race vs buy-and-hold, plain RSI and SMA(50/200); the timing-alpha
regression; the same-exposure random-timing control; the long-short variant; the cost
sweep; the panel; and the synthetic positive control. Network is touched only with --fetch.

    python studies/428-stochastic-rsi/examples/verify.py            # cache-only
    python studies/428-stochastic-rsi/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stochastic_rsi import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = pd.Timestamp("2026-06-23")


def _spy(fetch: bool) -> pd.DataFrame:
    b = data.load_real("SPY", cache_dir=CACHE, force_fetch=fetch)
    return b[b.index <= ASOF]


def _rand_timing_sharpe(close, time_in_mkt, cost=1.0, n=50):
    n_bars = len(close)
    k = int(round(time_in_mkt * n_bars))
    out = []
    for seed in range(n):
        ix = np.random.default_rng(seed).choice(n_bars, size=k, replace=False)
        w = pd.Series(np.zeros(n_bars), index=close.index)
        w.iloc[ix] = 1.0
        out.append(st.sharpe(st.positions_to_returns(close, w, cost_bps=cost)["net"].to_numpy()))
    return np.array(out)


def main(fetch: bool) -> None:
    if fetch:
        for t in data.PANEL:
            b = data.load_real(t, cache_dir=CACHE, force_fetch=True)
            b = b[b.index <= ASOF]
            print(f"{t:4s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    b = _spy(False)
    close = b["close"]
    print(f"=== SPY long-flat, net 1bp, excess of cash (as-of {ASOF.date()}) ===")
    res = st.run_experiment(b, cost_bps=1.0, long_short=False)
    print(f"window {res['start']} .. {res['end']}  n_days={res['n_days']}")
    for k in ("srsi_net", "rsi_net", "sma_net", "bh"):
        s = res[k]
        print(f"  {k:9s} Sharpe {s['sharpe']:+.3f}  HAC t {s['tstat']:+.2f}  "
              f"ann {s.get('ann_ret', float('nan'))*100:+.2f}%  maxDD {s.get('maxdd', float('nan'))*100:.1f}%")
    print(f"  srsi gross Sharpe {res['srsi_gross']['sharpe']:+.3f}  HAC t {res['srsi_gross']['tstat']:+.2f}")
    print(f"  permutation p (net): {res['perm_p_srsi_net']:.3f}")

    print("\n=== Benchmark Sharpe-gap races (StochRSI - benchmark) ===")
    for name, kk in (("vs buy-hold", "vs_bh"), ("vs plain RSI", "vs_rsi"), ("vs SMA", "vs_sma")):
        d = res[kk]
        print(f"  {name:13s} dSharpe {d['diff']:+.3f}  95% CI [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  "
              f"P(not better)={d['frac_a_not_better']*100:.1f}%")

    print("\n=== Timing alpha (regress net on market excess) ===")
    led = st.positions_to_returns(close, st.stoch_rsi_position(close), cost_bps=1.0)
    y = led["net"].to_numpy(); x = led["mkt_excess"].to_numpy()
    m = np.isfinite(y) & np.isfinite(x); y, x = y[m], x[m]
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - beta * x.mean()
    resid = y - (a + beta * x)
    se = resid.std(ddof=2) / np.sqrt(len(y)) * np.sqrt(1 + x.mean() ** 2 / np.var(x, ddof=1))
    print(f"  beta={beta:.2f}  alpha={a*252*100:+.2f}%/yr  alpha t={a/se:+.2f}")

    print("\n=== Same-exposure random-timing control (net 1bp) ===")
    tim = np.nanmean(np.abs(led["pos"]))
    shs = _rand_timing_sharpe(close, tim)
    srsi_sh = st.sharpe(led["net"].to_numpy())
    print(f"  StochRSI {srsi_sh:.3f}  | coin mean {shs.mean():.3f} [{np.percentile(shs,2.5):.3f},{np.percentile(shs,97.5):.3f}]")
    print(f"  {(shs>=srsi_sh).mean()*100:.0f}% of random schedules beat StochRSI")

    print("\n=== Long-short variant (net 1bp) ===")
    ls = st.run_experiment(b, cost_bps=1.0, long_short=True)["srsi_net"]
    print(f"  Sharpe {ls['sharpe']:+.3f}  HAC t {ls['tstat']:+.2f}  maxDD {ls['maxdd']*100:.1f}%")

    print("\n=== Cost sweep (net Sharpe) ===")
    pos = st.stoch_rsi_position(close)
    for c in (0.0, 1.0, 2.0, 5.0):
        s = st.sharpe(st.positions_to_returns(close, pos, cost_bps=c)["net"].to_numpy())
        print(f"  cost {c:.0f}bp  net Sharpe {s:+.3f}")
    dpos = pos.fillna(0.0).diff().abs()
    print(f"  annual turnover (round-trips): {dpos.sum()/(len(close)/252):.1f}")

    print("\n=== Panel (long-flat net 1bp): StochRSI vs buy-and-hold ===")
    for t in data.PANEL:
        bb = data.load_real(t, cache_dir=CACHE)
        bb = bb[bb.index <= ASOF]
        r = st.run_experiment(bb, cost_bps=1.0)
        print(f"  {t:4s} StochRSI {r['srsi_net']['sharpe']:+.3f}  buy-hold {r['bh']['sharpe']:+.3f}  "
              f"vs_rsi {r['vs_rsi']['diff']:+.3f}")

    print("\n=== Synthetic positive control (edge over same-exposure coin) ===")
    for mr in (-0.20, -0.10, 0.0, 0.10, 0.25, 0.40):
        bb, _ = data.synthetic_panel(n_days=4000, mean_rev=mr, drift=0.0002, seed=428)
        c = bb["close"]
        ls_ = st.positions_to_returns(c, st.stoch_rsi_position(c), cost_bps=0.0)
        tim_ = np.nanmean(np.abs(ls_["pos"]))
        s_sh = st.sharpe(ls_["net"].to_numpy())
        coin = _rand_timing_sharpe(c, tim_, cost=0.0, n=25).mean()
        print(f"  mean_rev {mr:+.2f}  StochRSI {s_sh:+.3f}  coin {coin:+.3f}  edge {s_sh-coin:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
