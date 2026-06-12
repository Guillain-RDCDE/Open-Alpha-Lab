"""Real-tape verification — Study 84 (Moon-Math). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC-USD daily tape from Yahoo Finance, runs the
Stock-to-Flow (S2F) levels regression, the competing log(time) regression, the
first-difference regression, and the out-of-sample evaluation post-2021. Prints all
the numbers that populate docs/results.md and the R dict in notebooks/build_notebooks.py.

    python studies/84-moon-math/examples/verify.py            # cache-only
    python studies/84-moon-math/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moon_math import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_btc(fetch=fetch)
    fp = data.fingerprint(df)
    print(f"BTC-USD daily: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"N rows: {len(df)}  |  Fingerprint: {fp}")
    print()

    # --- FULL SAMPLE (all available data) ---
    ctrl_full = st.spurious_control(df)
    smry = st.summarize(ctrl_full)
    r1 = ctrl_full["s2f_levels"]
    r2 = ctrl_full["time_levels"]
    r3 = ctrl_full["first_diff"]

    print("=== Full-sample regression (2014–2026) ===")
    print(f"  S2F levels:   R2={smry['s2f_r2']:.4f}  beta={r1['beta']:.3f}  "
          f"alpha={r1['alpha']:.3f}  ADF_p={smry['s2f_adf_p']:.4f}")
    print(f"  log(time):    R2={smry['time_r2']:.4f}  "
          f"ADF_p={smry['time_adf_p']:.4f}  (spurious control)")
    print(f"  first-diff:   R2={smry['diff_r2']:.6f}  "
          f"t={smry['diff_beta_t']:.3f}  (forecasting content test)")
    print(f"  spurious flag: {smry['spurious']}  "
          f"(True iff ADF_p > 0.05 on S2F residuals)")
    print()

    # --- IN-SAMPLE (original PlanB period: up to 2021-11-30) ---
    df_is = df[df.index <= "2021-11-30"]
    ctrl_is = st.spurious_control(df_is)
    smry_is = st.summarize(ctrl_is)
    r1_is = ctrl_is["s2f_levels"]
    r2_is = ctrl_is["time_levels"]

    print("=== In-sample only (<=2021-11-30, PlanB's training period) ===")
    print(f"  n_is={smry_is['n']}")
    print(f"  S2F levels:  R2={smry_is['s2f_r2']:.4f}  beta={r1_is['beta']:.3f}  "
          f"ADF_p={smry_is['s2f_adf_p']:.4f}")
    print(f"  log(time):   R2={smry_is['time_r2']:.4f}  ADF_p={r2_is['adf_pvalue']:.4f}")
    print(f"  first-diff:  R2={smry_is['diff_r2']:.6f}  "
          f"t={smry_is['diff_beta_t']:.3f}")
    print()

    # --- OUT-OF-SAMPLE (post-2021) ---
    oos = st.oos_eval(df, train_end="2021-11-30")
    pred = oos["oos_pred_series"]
    actual = oos["oos_actual_series"]
    print("=== Out-of-sample evaluation (train: <=2021-11-30, OOS: >2021-11-30) ===")
    print(f"  n_train={oos['n_train']}  n_oos={oos['n_oos']}")
    print(f"  Train R2:  {oos['train_r2']:.4f}")
    print(f"  OOS R2:    {oos['oos_r2']:.4f}  (< 0 means worse than predicting the mean)")
    print(f"  OOS MAPE:  {oos['oos_mape']:.1f}%")
    print()

    # 2022 trough: the smoking gun
    import numpy as np
    idx_2022 = actual.index[actual.index.year == 2022]
    if len(idx_2022):
        btc_min = actual[idx_2022].min()
        min_date = actual[idx_2022].idxmin()
        s2f_at_min = pred[min_date]
        err_pct = (s2f_at_min - btc_min) / btc_min * 100
        print(f"  2022 trough: BTC=${btc_min:,.0f} on {min_date.date()}")
        print(f"  S2F predicted:  ${s2f_at_min:,.0f}  (error: +{err_pct:.0f}%)")
        btc_2022_max = actual[idx_2022].max()
        s2f_2022_range = f"${pred[idx_2022].min():,.0f}–${pred[idx_2022].max():,.0f}"
        print(f"  2022 BTC range: ${btc_min:,.0f}–${btc_2022_max:,.0f}")
        print(f"  S2F range 2022: {s2f_2022_range}  (stable near a halving-epoch plateau)")

    print()
    print("=== R dict (for build_notebooks.py) ===")
    print("R = dict(")
    print(f"    n={smry['n']},")
    print(f"    n_is={smry_is['n']},")
    print(f"    s2f_r2_full={smry['s2f_r2']:.4f},")
    print(f"    time_r2_full={smry['time_r2']:.4f},")
    print(f"    s2f_r2_is={smry_is['s2f_r2']:.4f},")
    print(f"    time_r2_is={smry_is['time_r2']:.4f},")
    print(f"    s2f_adf_p={smry['s2f_adf_p']:.4f},")
    print(f"    time_adf_p={smry['time_adf_p']:.4f},")
    print(f"    diff_r2={smry['diff_r2']:.6f},")
    print(f"    diff_t={smry['diff_beta_t']:.3f},")
    print(f"    s2f_beta_is={r1_is['beta']:.3f},")
    print(f"    train_r2={oos['train_r2']:.4f},")
    print(f"    oos_r2={oos['oos_r2']:.4f},")
    print(f"    oos_mape={oos['oos_mape']:.1f},")
    print(f"    n_oos={oos['n_oos']},")
    if len(idx_2022):
        print(f"    btc_2022_min={btc_min:.0f},")
        print(f"    s2f_at_2022_min={s2f_at_min:.0f},")
        print(f"    err_2022_pct={err_pct:.0f},")
    print(")")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
