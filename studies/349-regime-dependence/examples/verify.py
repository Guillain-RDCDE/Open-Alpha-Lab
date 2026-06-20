"""Real-tape verification — Study 349 (Regime-Dependence). Regenerates docs/results.md.

Loads (or fetches) monthly total-return series for the equity index and a bond proxy,
builds two canonical strategies — a 12-month trend overlay and a static 60/40 mix —
splits each by decade, and reports the regime-conditional Sharpe, the stability score,
and the danger number: the Sharpe and HAC t with the single best decade removed.

    python studies/349-regime-dependence/examples/verify.py           # cache-only
    python studies/349-regime-dependence/examples/verify.py --fetch   # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regime_dependence import data, strategy as st  # noqa: E402


def report(name: str, ret, regime) -> None:
    rs = st.regime_sharpes(ret, regime)
    ss = st.stability_score(ret, regime)
    lo = st.leave_hot_regime_out(ret, regime)
    gap = st.hot_minus_rest_ci(ret, regime, n_boot=2000)
    print(f"\n=== {name} ===")
    print(f"  full-sample Sharpe : {st.sharpe(ret):+.2f}  HAC t={st.hac_tstat(ret)['tstat']:+.2f} (n={len(ret)})")
    print("  decade Sharpes     : " + "  ".join(f"{int(k)}s:{v:+.2f}" for k, v in rs.items()))
    print(f"  stability CV       : {ss['cv']:.2f}   best-worst spread: {ss['best_minus_worst']:.2f}"
          f"   share in best decade: {ss['share_top_regime']*100:.0f}%")
    print(f"  DANGER (drop best {int(lo['hot_regime'])}s): Sharpe {lo['sharpe_full']:+.2f} -> "
          f"{lo['sharpe_ex_hot']:+.2f}   HAC t {lo['t_full']:+.2f} -> {lo['t_ex_hot']:+.2f}")
    print(f"  best-decade minus rest: {gap['gap_ann']*100:+.1f}%/yr  "
          f"95% CI [{gap['lo']*100:+.1f}, {gap['hi']*100:+.1f}]")


def main(fetch: bool) -> None:
    rets = data.load_real(fetch=fetch)
    print(f"tape: {rets.index[0].date()} .. {rets.index[-1].date()}  ({len(rets)} months)")
    print(f"fingerprint: {data.fingerprint(rets)}")

    trend = st.apply_trend(rets["equity"], lookback=12, cost_bps=0.0)
    mix = st.apply_6040(rets["equity"], rets["bond"], cost_bps=0.0)

    report("Trend overlay (12m TSMOM on the index)", trend, st.decade_label(trend.index))
    report("Static 60/40 stock/bond", mix, st.decade_label(mix.index))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
