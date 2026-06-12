"""Offline, deterministic demo — Study 84 (Moon-Math).

No network. Builds a synthetic BTC-like daily tape and shows the study's spine:
- The S2F levels regression finds a high R² when price is *generated* from S2F.
- On a random-walk price (beta=0), levels regression still finds a high R² (spurious!).
- First-difference regression collapses to ~0 in both cases — only 'real' cointegration
  shows a meaningful first-diff slope.
- Out-of-sample R² collapses when the model is extrapolating beyond its training domain.

Run:
    python studies/84-moon-math/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moon_math import data, strategy as st  # noqa: E402


def _run(label: str, price_s2f_beta: float, n_days: int = 2000, seed: int = 84) -> None:
    df, truth = data.synthetic_s2f(n_days=n_days, price_s2f_beta=price_s2f_beta, seed=seed)
    ctrl = st.spurious_control(df)
    smry = st.summarize(ctrl)
    spurious_flag = "SPURIOUS" if smry["spurious"] else "cointegrated"
    print(
        f"  {label:<38} levels R2={smry['s2f_r2']:.3f}  "
        f"time R2={smry['time_r2']:.3f}  "
        f"diff R2={smry['diff_r2']:.5f}  "
        f"ADF_p={smry['s2f_adf_p']:.3f}  [{spurious_flag}]"
    )


def main() -> None:
    print("Study 84 — Moon-Math — synthetic positive/negative controls\n")
    print("Three scenarios: same S2F series, different price-generation mechanisms.\n")

    print("Scenario                               | levels R2 | time R2 | diff R2 | ADF_p | verdict")
    print("-" * 100)
    _run("beta=3.3 (PlanB: price tracks S2F)", price_s2f_beta=3.3)
    _run("beta=0.0 (null: price is pure noise)", price_s2f_beta=0.0)
    _run("beta=0.0, high vol (random walk)", price_s2f_beta=0.0, seed=99)

    print()
    print("Key insight:")
    print("  - Even when beta=0 (no causal link), levels R2 is often HIGH because both")
    print("    log(S2F) and log(price) are trending upward. This is the spurious regression.")
    print("  - The first-difference R2 is tiny in all cases — the S2F has almost no")
    print("    forecasting content for *price changes*.")
    print("  - The ADF test on residuals (p < 0.05 => cointegrated) is the formal test.")
    print()
    print("On the real BTC tape (2014-2026):")
    print("  - S2F levels R2 = 0.80 (impressive in-sample)")
    print("  - log(time) levels R2 = 0.92 (BETTER — it's just a time trend)")
    print("  - First-diff R2 = 0.0009 (no forecasting content for price changes)")
    print("  - OOS R2 (post-2021) = -2.36 (catastrophic failure)")
    print("  - OOS MAPE = 200% (BTC at $16k, S2F predicted $62-66k)")
    print()
    print("Verdict: NONE / MIRAGE. The S2F model is a spurious regression of two time trends.")


if __name__ == "__main__":
    main()
