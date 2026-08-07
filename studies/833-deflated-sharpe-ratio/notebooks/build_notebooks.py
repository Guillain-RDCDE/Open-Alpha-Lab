"""Generate the two narrative notebooks for Study 833 (Deflated Sharpe Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The heavy headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only fast synthetic checks
(a small null pool + the formula), so execution is quick and network-free.
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


# Frozen headline numbers — mirror of docs/results.md (null sim: N=1000, T=1260, ann-vol=15%,
# seed 833; expected-maximum-Sharpe + Deflated Sharpe Ratio, Bailey & López de Prado 2014).
R = dict(
    n_trials=1000, n_days=1260, ann_vol=0.15, seed=833, fingerprint="f7e4b81df8a2",
    mean_col_sharpe=0.026, obs_max_sharpe=1.251, exp_max_sharpe=1.456,
    sr_std=0.02725, sr_std_theory=0.02818,
    winner_sharpe=1.251, winner_sr0_ann=1.456, winner_excess=-0.205,
    winner_dsr=0.324, winner_naive_t=2.80,
    is_sharpe=1.769, oos_sharpe=-0.280, oos_t_nw=-0.47,
    timer1_gross=-1.649, timer1_net=-3.649, timer1_t=-0.98,
    timer5_net=-11.649, timer5_t=-3.13,
    curve_N=[2, 5, 10, 25, 50, 100, 250, 500, 1000],
    curve_obs=[0.237, 0.487, 0.705, 0.892, 1.007, 1.151, 1.277, 1.371, 1.467],
    curve_pred=[0.233, 0.534, 0.704, 0.894, 1.018, 1.132, 1.270, 1.366, 1.456],
    n_seeds=40, cal_naive_fire=40, cal_naive_rate=1.00,
    cal_dsr_fire=0, cal_dsr_rate=0.00, cal_mean_dsr=0.509, cal_mean_excess=0.011,
    honest_true=1.0, honest_realised=1.014, honest_mean_dsr=0.965, honest_fire=33, honest_rate=0.82,
    planted_true=2.0, planted_dsr=0.648,
)

BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
)

HEADER = f"""# Study 833 — Deflated Sharpe Ratio 🎏

**Try enough strategies and the luckiest one *always* dazzles.**

Bailey & López de Prado (2014) proved the arithmetic: run `N` **independent** strategies on a
tape with **zero** true edge, and the *best* sample Sharpe is not zero — it grows with `N`. Here
the best of **{R['n_trials']:,}** empty strategies posts an annualised Sharpe of
**{R['obs_max_sharpe']:+.2f}** with a *t* of **{R['winner_naive_t']:+.2f}**… and is, with
certainty, nothing. The **Deflated Sharpe Ratio** shrinks it back to a coin flip.

*Numbers below are the frozen headline (`docs/results.md`, sim fingerprint `{R['fingerprint']}`,
as-of 2026-06-30); the live cells run the fast synthetic checks. Signal is NONE **by
construction** — the tape is a certified null.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. A gorgeous backtest, built from nothing\n\n"
           "We generate 1,000 strategies whose *true* Sharpe is **exactly zero** — pure noise, no "
           "edge, by construction. Then we keep the one with the best backtest. Watch what the "
           "'winner' looks like."),
        code(
            "R = dict(mean_col_sharpe=%r, obs_max_sharpe=%r, winner_naive_t=%r,\n"
            "         exp_max_sharpe=%r, winner_dsr=%r)\n"
            "print(f\"mean strategy Sharpe (the truth): {R['mean_col_sharpe']:+.2f}  -> nothing real\")\n"
            "print(f\"BEST of 1,000 Sharpe (annualised): {R['obs_max_sharpe']:+.2f}  -> looks amazing\")\n"
            "print(f\"  its naive t-stat               : {R['winner_naive_t']:+.2f}  -> looks significant\")\n"
            "print(f\"  expected MAX under pure luck   : {R['exp_max_sharpe']:+.2f}  -> the bar luck alone clears\")\n"
            "print(f\"  Deflated Sharpe Ratio          : {R['winner_dsr']:.2f}   -> a coin flip (needs >=0.95 to matter)\")"
            % (R["mean_col_sharpe"], R["obs_max_sharpe"], R["winner_naive_t"],
               R["exp_max_sharpe"], R["winner_dsr"])
        ),
        md("## 2. Why the luck bar rises with every rule\n\n"
           "The more strategies you try, the luckier the luckiest one is — that is not intuition, "
           "it is a formula (the *expected maximum Sharpe*). Let's watch it climb, live, on tiny "
           "null pools, and check the formula nails it."),
        code(
            BOOT +
            "from deflated_sharpe import data, strategy as st\n"
            "import numpy as np\n"
            "for N in (10, 100, 1000):\n"
            "    panel = data.null_panel(N, n_days=1260, ann_vol=0.15, seed=1)\n"
            "    be = st.best_sharpe_experiment(panel)\n"
            "    print(f'N={N:>5}: best empty-strategy Sharpe {be[\"obs_max_sharpe_ann\"]:+.2f}  '\n"
            "          f'(formula says ~{be[\"exp_max_sharpe_ann\"]:+.2f})')\n"
            "print('\\n-> more trials, a luckier winner -- with ZERO real edge behind any of them')"
        ),
        md("## 3. The catch that spares an honest idea\n\n"
           "If deflation just punished big Sharpes, it would be useless. It punishes *searching*. "
           "An honestly-good **single** strategy (a true Sharpe of 1.0, no search) keeps a high "
           "Deflated Sharpe Ratio — live:"),
        code(
            BOOT +
            "from deflated_sharpe import data, strategy as st\n"
            "honest = data.honest_strategy(n_days=1260, true_ann_sharpe=1.0, seed=833)\n"
            "d = st.deflated_sharpe_ratio(honest, n_trials=1)   # a single hypothesis, no search\n"
            "print(f'honest single strategy: Sharpe {d[\"sharpe_ann\"]:+.2f}, DSR {d[\"dsr\"]:.3f}  '\n"
            "      f'-> survives (the correction spares genuine skill)')\n"
            "null_pool = data.null_panel(1000, 1260, 0.15, 833)\n"
            "win = null_pool[:, int(np.nanargmax(st.panel_sr_per_period(null_pool)))]\n"
            "dn = st.deflated_sharpe_ratio(win, n_trials=1000)\n"
            "print(f'best of 1,000 empties : Sharpe {dn[\"sharpe_ann\"]:+.2f}, DSR {dn[\"dsr\"]:.3f}  '\n"
            "      f'-> fails (consistent with luck)')"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"On a tape with **zero** real edge, the best of {R['n_trials']:,} strategies looks like "
           f"a Sharpe-{R['obs_max_sharpe']:.2f} winner (naive *t* = {R['winner_naive_t']:+.2f}) — and "
           f"is provably nothing. Out of sample it collapses (**{R['is_sharpe']:+.2f} → "
           f"{R['oos_sharpe']:+.2f}**) and bleeds on costs. The Deflated Sharpe Ratio "
           f"(**{R['winner_dsr']:.2f}**, a coin flip) sees straight through it, while sparing the "
           f"honest single strategy (DSR **{R['honest_mean_dsr']:.2f}**). **Signal: None** "
           f"(nothing real), **Tradability: Mirage**, and *does the trial count inflate the best "
           f"Sharpe?* — **Confirmed**. A Sharpe without its trial count is not evidence.")
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 833 — Deflated Sharpe Ratio — the teardown\n\n"
           "The expected-maximum-Sharpe asymptotics, the observed-vs-formula inflation curve, the "
           "moment-aware Deflated Sharpe Ratio, the in-sample→out-of-sample collapse, the costed "
           "timer, and the 40-seed null calibration + honest positive control. All offline, seed 833."),
        code("R = %r" % (R,)),
        md("## The headline — best of 1,000 EMPTY strategies\n\n"
           "Every column has a *true* Sharpe of exactly 0; the winner is pure selection luck."),
        code(
            "print(f\"mean column Sharpe (truth) : {R['mean_col_sharpe']:+.3f}  (~0)\")\n"
            "print(f\"observed MAX Sharpe        : {R['obs_max_sharpe']:+.3f}\")\n"
            "print(f\"expected MAX under null SR0: {R['exp_max_sharpe']:+.3f}  (N=1000)\")\n"
            "print(f\"cross-trial SR std sqrt(V) : {R['sr_std']:.5f}  (theory 1/sqrt(T-1) = {R['sr_std_theory']:.5f})\")"
        ),
        md("## The deflation — DSR of the winner\n\n"
           "DSR = Φ((SR − SR0)·√(T−1) / √(1 − g₃·SR + (g₄−1)/4·SR²)). Below 0.95 ⇒ consistent with luck."),
        code(
            "print(f\"winner Sharpe {R['winner_sharpe']:+.2f}  vs expected-max bar SR0 {R['winner_sr0_ann']:+.2f}\")\n"
            "print(f\"deflated EXCESS Sharpe (SR-SR0): {R['winner_excess']:+.2f}  (~0)\")\n"
            "print(f\"DSR = {R['winner_dsr']:.3f}   but naive one-sample t = {R['winner_naive_t']:+.2f} (fools you)\")"
        ),
        md("## The inflation curve — E[max] vs N (observed, 40 seeds/point, vs formula)"),
        code(
            "for N, o, p in zip(R['curve_N'], R['curve_obs'], R['curve_pred']):\n"
            "    print(f\"N={N:>5}: observed best {o:+.3f}   formula E[max] {p:+.3f}\")"
        ),
        md("## Out-of-sample collapse + the costed timer (the Mirage)"),
        code(
            "print(f\"in-sample Sharpe   {R['is_sharpe']:+.2f}  ->  out-of-sample {R['oos_sharpe']:+.2f} \"\n"
            "      f\"(NW t = {R['oos_t_nw']:+.2f})\")\n"
            "print(f\"timer @1 bp: gross {R['timer1_gross']:+.2f} -> net {R['timer1_net']:+.2f} bps/day (t={R['timer1_t']:+.2f})\")\n"
            "print(f\"timer @5 bp:                 net {R['timer5_net']:+.2f} bps/day (t={R['timer5_t']:+.2f})\")"
        ),
        md("## Live control — the machinery is calibrated\n\n"
           "A small live run: the naive screen fires on the null winner; the DSR does not; and an "
           "honest single strategy keeps a high DSR. (Fast: 12 null pools + 12 honest streams.)"),
        code(
            BOOT +
            "from deflated_sharpe import strategy as st\n"
            "cal = st.null_dsr_calibration(n_trials=1000, n_days=1260, ann_vol=0.15, n_seeds=12, base_seed=833)\n"
            "hc  = st.honest_control(true_ann_sharpe=1.0, n_days=1260, ann_vol=0.15, n_seeds=12, base_seed=833)\n"
            "print(f\"NULL pools: naive |t|>=2 fires {cal['naive_fire']}/{cal['n_seeds']}  \"\n"
            "      f\"vs DSR>=0.95 fires {cal['dsr_fire']}/{cal['n_seeds']}; mean DSR {cal['mean_dsr']:.3f} (~0.5)\")\n"
            "print(f\"HONEST single strategy: mean DSR {hc['mean_dsr']:.3f}, DSR>=0.95 in \"\n"
            "      f\"{hc['dsr_fire']}/{hc['n_seeds']}  -> the correction spares real skill\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** By construction: the tape is a certified null. The best of "
           f"{R['n_trials']:,} empty strategies looks like a Sharpe-{R['obs_max_sharpe']:.2f} winner "
           f"(naive *t* = {R['winner_naive_t']:+.2f}) and is provably nothing.\n"
           f"- **Tradability — Mirage.** In-sample {R['is_sharpe']:+.2f} → out-of-sample "
           f"{R['oos_sharpe']:+.2f} (NW *t* = {R['oos_t_nw']:+.2f}); net {R['timer1_net']:+.2f} bps/day "
           f"at 1 bp. Nothing to harvest.\n"
           f"- **Does the trial count inflate the best Sharpe? — Confirmed.** Observed max tracks "
           f"E[max] from N=2→1,000; the DSR shrinks the winner to a coin flip (mean "
           f"{R['cal_mean_dsr']:.3f}, deflated excess ≈ {R['cal_mean_excess']:+.2f}), firing on "
           f"{R['cal_dsr_fire']}/{R['n_seeds']} nulls vs {R['cal_naive_fire']}/{R['n_seeds']} for the "
           f"naive screen — while sparing the honest strategy (mean DSR {R['honest_mean_dsr']:.2f}). "
           f"*(The synthetic controls prove the machinery is calibrated — never cited to support a "
           f"real-tape stamp; there is no real tape.)*")
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
