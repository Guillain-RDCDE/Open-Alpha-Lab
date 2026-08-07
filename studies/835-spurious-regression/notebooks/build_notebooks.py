"""Generate the two narrative notebooks for Study 835 (Spurious Regression).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The frozen headline numbers are quoted from
the ``R`` dict (mirroring docs/results.md); the live cells run only fast synthetic
controls (a small level-vs-difference experiment and a tiny cointegration demo), so
execution is quick and network-free.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen headline numbers — mirror of docs/results.md. Synthetic worlds, base seed 835,
# 5,000 pairs x 250 obs (driftless). Fingerprint 73e2821b184c.
R = dict(
    fingerprint="73e2821b184c", base_seed=835, n_pairs=5000, n_obs=250,
    # 1. the pitfall (levels) vs the fix (differences)
    lvl_reject=0.850, lvl_reject_x=17.0, lvl_wilson=(0.840, 0.860),
    lvl_mean_abs_t=8.99, lvl_median_abs_t=7.06, lvl_mean_r2=0.241, lvl_share_r2=0.398,
    dif_reject=0.053, dif_mean_abs_t=0.80, dif_mean_r2=0.004,
    # 2. trending makes it worse
    drift_reject=0.981, drift_mean_abs_t=28.48, drift_mean_r2=0.662, drift_share_r2=0.899,
    # 3. sample-size sweep  (n, level_reject, level_mean|t|, level_R2, diff_reject)
    sweep=[(50, 0.679, 3.99, 0.243, 0.059), (125, 0.787, 6.20, 0.236, 0.052),
           (250, 0.847, 8.99, 0.241, 0.052), (500, 0.895, 12.81, 0.241, 0.050),
           (1000, 0.926, 17.99, 0.240, 0.045)],
    # 4. specificity control (stationary)
    stat_reject=0.051, stat_mean_abs_t=0.80, stat_mean_r2=0.004,
    # 5. cointegration
    coint_indep_reject=0.050, coint_indep_p=0.495,
    coint_true_reject=1.000, coint_true_p=0.000,
    # 6. tradability
    timer=[(0.0, -27.46, -27.46, -1.23, -1.43, -69.2),
           (1.0, -27.46, -27.99, -1.25, -1.45, -70.5),
           (5.0, -27.46, -29.59, -1.33, -1.54, -74.6)],
)


HEADER = f"""# Study 835 — Spurious Regression 🎭

**Regress one random walk on another — and OLS hands you a "significant" relation that isn't there.**

Granger & Newbold (1974) showed the trap: take two **independent** random walks (each just
a cumulative sum of unrelated coin-flips, so there is *no* relationship between them),
regress one on the other, and the textbook *t*-statistic will call it "significant" the
vast majority of the time — with a high R² to match. It is all an artefact of the two
series **trending** (being non-stationary), not a signal. We simulate {R['n_pairs']:,}
such pairs and watch the trap spring.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`,
as-of 2026-06-30); the live cells run the fast synthetic controls. Synthetic-only method
demo — no real tape, so it can never earn `REAL` (capped at `NONE` on Signal).*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Two coin-flip paths that have nothing to do with each other\n\n"
           "A *random walk* is just a running total of independent random steps — like a "
           "drunkard's path. Build **two** of them from *separate* streams of randomness, "
           "so by construction neither knows the other exists. Now regress one on the "
           "other. A fair statistical test should call them 'related' only ~5% of the "
           "time (the false-positive rate you accept at the 5% level). Watch what OLS "
           "actually does."),
        code(
            "R = " + repr({k: R[k] for k in (
                "lvl_reject", "lvl_reject_x", "lvl_mean_abs_t", "lvl_mean_r2",
                "dif_reject", "dif_mean_r2")}) + "\n"
            "print('LEVELS  (regress y on x):')\n"
            "print(f\"  rejects 'no relation' at |t|>1.96 in {R['lvl_reject']:.0%} of pairs \"\n"
            "      f\"(a valid test would: ~5%) -> {R['lvl_reject_x']:.0f}x too often\")\n"
            "print(f\"  average |t| = {R['lvl_mean_abs_t']:.1f}   average R2 = {R['lvl_mean_r2']:.2f}\")\n"
            "print('FIRST DIFFERENCES (regress the day-to-day CHANGES) -- the fix:')\n"
            "print(f\"  rejects in {R['dif_reject']:.0%} of pairs (back to ~5%), R2 = {R['dif_mean_r2']:.3f}\")"
        ),
        md("## 2. See it live — a small simulation, no network\n\n"
           "Let's not take the frozen numbers on faith. Simulate a fresh batch of "
           "independent random walks right here and run both regressions."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from spurious_regression import data, strategy as st\n"
            "X, Y = data.independent_walks(1500, n_obs=250, seed=835)\n"
            "ex = st.regression_experiment(X, Y)\n"
            "print(f\"levels     : rejects in {ex['level']['reject_rate']:.0%} of pairs, \"\n"
            "      f\"mean|t| {ex['level']['mean_abs_t']:.1f}, mean R2 {ex['level']['mean_r2']:.2f}\")\n"
            "print(f\"differences: rejects in {ex['diff']['reject_rate']:.0%} of pairs, \"\n"
            "      f\"mean|t| {ex['diff']['mean_abs_t']:.1f}, mean R2 {ex['diff']['mean_r2']:.3f}\")\n"
            "print('\\n-> the level regression is a false-alarm machine; differencing fixes it.')"
        ),
        md("## 3. Trending makes it worse — and more data doesn't save you\n\n"
           f"If the two walks also **drift** (trend) in the same direction, the illusion "
           f"gets stronger: the level regression rejects **{R['drift_reject']:.0%}** of the "
           f"time with a mean R² of **{R['drift_mean_r2']:.2f}**. And — counter to every "
           f"instinct — *adding data makes it worse*: the spurious *t* grows with √T, so at "
           f"1,000 observations the level test rejects **{R['sweep'][-1][1]:.0%}** of the "
           f"time (vs {R['sweep'][0][1]:.0%} at 50). The differenced test stays at ~5% "
           f"throughout. A big-*n*, high-*t*, high-R² regression on **levels** is no "
           f"comfort at all."),
        code(
            "R_sweep = " + repr(R["sweep"]) + "\n"
            "print('n_obs | level rejects | diff rejects')\n"
            "for n, lr, mt, r2, dr in R_sweep:\n"
            "    print(f'{n:>5} | {lr:>11.0%} | {dr:>10.0%}')"
        ),
        md("## 4. The honest verdict\n\n"
           f"There is **nothing real here** — the series were built independent. The "
           f"level regression's 'significance' is manufactured by the trending "
           f"(non-stationary) structure, not by any relationship. The cures are old and "
           f"simple: **difference to stationarity**, or **test for cointegration** before "
           f"believing a levels regression. And you certainly can't *trade* the fake "
           f"relation — the spurious spread is itself a random walk, so a pairs trade on "
           f"it earns nothing you can distinguish from zero and bleeds costs. "
           f"**Signal: None** · **Tradability: Mirage** · **Do trending series "
           f"manufacture false significance? Confirmed.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 835 — Spurious Regression — the teardown\n\n"
           "The oversized level-OLS *t* and R², the differencing fix, the trending case, "
           "the √T sample-size divergence, the stationary-series size control, the "
           "Engle-Granger cointegration positive control, and the costed pairs timer. "
           "Frozen headline in `R`; the live cells re-run the fast controls."),
        code("R = %r" % (R,)),
        md("## 1. The pitfall vs the fix — level OLS on two independent random walks\n\n"
           "5,000 pairs × 250 obs, driftless. Nominal test size is 0.05."),
        code(
            "print(f\"levels      : reject {R['lvl_reject']:.3f} (Wilson 95% CI {R['lvl_wilson']}), \"\n"
            "      f\"{R['lvl_reject_x']:.1f}x oversized\")\n"
            "print(f\"              mean|t| {R['lvl_mean_abs_t']:.2f}  median|t| {R['lvl_median_abs_t']:.2f}  \"\n"
            "      f\"meanR2 {R['lvl_mean_r2']:.3f}  shareR2>0.25 {R['lvl_share_r2']:.3f}\")\n"
            "print(f\"differences : reject {R['dif_reject']:.3f} (~nominal), mean|t| {R['dif_mean_abs_t']:.2f}, \"\n"
            "      f\"meanR2 {R['dif_mean_r2']:.3f}  <- the fix\")"
        ),
        md("## 2. Trending series manufacture false significance (drift = 0.15/step)"),
        code(
            "print(f\"trending levels: reject {R['drift_reject']:.3f}, mean|t| {R['drift_mean_abs_t']:.2f}, \"\n"
            "      f\"meanR2 {R['drift_mean_r2']:.3f}, shareR2>0.25 {R['drift_share_r2']:.3f}\")"
        ),
        md("## 3. The √T divergence — more data makes the LEVEL test worse\n\n"
           "Phillips (1986): with `I(1)` regressors the *t*-stat diverges, so the "
           "rejection rate → 1 as `n` grows. The differenced test stays correctly sized."),
        code(
            "print('n_obs | level_reject  level_mean|t|  level_R2 | diff_reject')\n"
            "for n, lr, mt, r2, dr in R['sweep']:\n"
            "    print(f'{n:>5} | {lr:.3f}        {mt:>5.2f}        {r2:.3f} | {dr:.3f}')"
        ),
        md("## 4. Specificity control — the same OLS on STATIONARY series is correctly sized\n\n"
           "Live: the over-rejection is a property of the unit root, not of OLS."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from spurious_regression import data, strategy as st\n"
            "sc = st.size_control(data, n_pairs=3000, n_obs=250, phi=0.0, seed=835)\n"
            "print(f\"stationary levels: reject {sc['reject_rate']:.3f} (~0.05, correctly sized), \"\n"
            "      f\"mean|t| {sc['mean_abs_t']:.2f}, meanR2 {sc['mean_r2']:.3f}\")\n"
            "print(f\"(frozen headline at 5,000 pairs: {R['stat_reject']:.3f})\")"
        ),
        md("## 5. The other fix — Engle-Granger cointegration (positive control)\n\n"
           "Live, on a small sample: the test must NOT reject on independent walks and MUST "
           "reject on a genuinely cointegrated pair."),
        code(
            "Xi, Yi = data.independent_walks(120, n_obs=250, seed=835)\n"
            "Xc, Yc = data.cointegrated_pairs(120, n_obs=250, beta=1.0, noise_sd=1.0, seed=835)\n"
            "ci = st.cointegration_reject_rate(Xi, Yi)\n"
            "cc = st.cointegration_reject_rate(Xc, Yc)\n"
            "print(f\"independent walks : reject no-coint {ci['reject_rate']:.3f} (median p {ci['median_pvalue']:.3f}) -> nothing\")\n"
            "print(f\"cointegrated pair : reject no-coint {cc['reject_rate']:.3f} (median p {cc['median_pvalue']:.3f}) -> real relation\")\n"
            "print(f\"(frozen headline, 300 pairs: indep {R['coint_indep_reject']:.3f} / true {R['coint_true_reject']:.3f})\")"
        ),
        md("## 6. Tradability — a costed pairs trade on the spurious spread (no look-ahead)\n\n"
           "Trailing hedge ratio & z-score known at `t−1`; contrarian on the residual; "
           "one-way cost × NAV on turnover + short borrow."),
        code(
            "print('cost  | gross  ->  net  | t_net | Sharpe |  ~ann')\n"
            "for c, g, n, t, sh, ann in R['timer']:\n"
            "    print(f'{c:>4.1f} | {g:+.2f} -> {n:+.2f} | {t:+.2f} | {sh:+.2f} | {ann:+.1f}%')\n"
            "print('\\n-> gross |t|<2: the spread is a random walk, no reversion to harvest; costs only hurt. MIRAGE.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The two series are drawn independent; the level "
           f"regression's significance is a manufactured artefact of the unit root "
           f"(reject **{R['lvl_reject']:.3f}** vs nominal 0.05, a **{R['lvl_reject_x']:.0f}×** "
           f"oversized test; mean R² **{R['lvl_mean_r2']:.2f}**). First-differencing "
           f"restores the correct size (**{R['dif_reject']:.3f}**) and a stationary-series "
           f"control is correctly sized (**{R['stat_reject']:.3f}**), so the inflation is "
           f"nonstationarity, not OLS. No real tape a method demo could stamp → capped at None.\n"
           f"- **Tradability — Mirage.** The spurious spread is a random walk; the costed "
           f"pairs trade earns no edge distinguishable from zero (gross *t* = "
           f"**{R['timer'][0][3]:+.2f}**) and loses net of any friction.\n"
           f"- **Do trending series manufacture false significance? — Confirmed.** 85% false "
           f"rejection on driftless walks, **{R['drift_reject']:.0%}** with a shared trend, "
           f"and the inflation *grows with the sample* (mean |t| {R['sweep'][0][2]:.1f} → "
           f"{R['sweep'][-1][2]:.1f} as n: 50 → 1000). The cointegration test tells the "
           f"spurious from the genuine (reject {R['coint_indep_reject']:.2f} vs "
           f"{R['coint_true_reject']:.2f}).")
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
