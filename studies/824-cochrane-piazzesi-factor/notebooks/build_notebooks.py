"""Generate the two narrative notebooks for Study 824 (Cochrane-Piazzesi Factor).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily closes,
# ^IRX/^FVX/^TNX/^TYX yields + SHY/IEF/TLT ETFs, 2002-01-02 -> 2026-06-30; CP predictive
# regression of the avg 252-day excess return on the 4-forward vector).
R = dict(
    start="2002-01-02", end="2026-06-30", n_rows=6162, fingerprint="03a5e9844a31",
    n=5760, nw_lags=378, r2=0.2261, cp_slope=1.000, cp_slope_t=2.618, avg_rx_bps=166.7,
    load_const=-0.0696, load_yshort=-0.0587, load_f1=-2.1058, load_f2=4.9245, load_f3=-0.9744,
    load_t_const=-0.91, load_t_yshort=-0.06, load_t_f1=-1.12, load_t_f2=0.95, load_t_f3=-0.16,
    oos_r2=-0.2722, oos_npreds=227,
    placebo_obs=0.2261, placebo_mean=0.1795, placebo_sd=0.0541, placebo_p=0.2080,
    era1_n=2620, era1_r2=0.4118, era1_t=3.902,
    era2_n=2888, era2_r2=0.1139, era2_t=1.717,
    timer2_sharpe=0.016, timer5_sharpe=0.005, bh_sharpe=0.216, switches=3.7, invested=0.47,
    null_r2_mean=0.0007, null_r2_max=0.0019, null_t_mean=2.08, null_fire=11,
    null_oos=0.0186, plant_r2=0.6613, plant_t=100.09, plant_oos=0.6309,
)


HEADER = f"""# Study 824 — Cochrane-Piazzesi Factor 🧮📉

**Does a single tent of forward rates forecast bond excess returns?**

Cochrane & Piazzesi (2005) regress each Treasury bond's one-year-ahead **excess return** on
the whole vector of **forward rates** and find the fitted values collapse onto *one*
tent-shaped factor `CP = γ'f` that forecasts excess returns of every maturity — an R² a plain
curve slope can't touch. We rebuild it from the coarse constant-maturity yields yfinance
exposes (`^IRX` 0.25y, `^FVX` 5y, `^TNX` 10y, `^TYX` 30y, {R['start']} → {R['end']}) and
forecast the average 252-day excess return of the SHY/IEF/TLT bond ETFs.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. Signal caveat: a coarse 4-forward proxy of CP's
Fama-Bliss 1..5y zeros — named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "The **expectations hypothesis** says a long yield is just the average expected "
           "future short rate — so *no* combination of today's forwards should predict a bond's "
           "future *excess* return. Cochrane & Piazzesi found otherwise: regress next year's "
           "excess return on all the forwards and one **tent-shaped** loading vector "
           "(down-short, up-middle, down-long) pops out that forecasts returns across every "
           "maturity. Fat premium ⇒ own duration; thin ⇒ step aside."),
        code(
            "R = dict(r2=%r, cp_slope_t=%r, load_yshort=%r, load_f1=%r, load_f2=%r, load_f3=%r)\n"
            "print('in-sample predictive R2 : %%.3f' %% R['r2'])\n"
            "print('single-factor NW t      : %%+.2f' %% R['cp_slope_t'])\n"
            "print('tent loadings  y_short=%%+.2f  f_1=%%+.2f  f_2=%%+.2f  f_3=%%+.2f  '\n"
            "      '(peak on the 5->10y forward)'\n"
            "      %% (R['load_yshort'], R['load_f1'], R['load_f2'], R['load_f3']))"
            % (R["r2"], R["cp_slope_t"], R["load_yshort"], R["load_f1"], R["load_f2"], R["load_f3"])
        ),
        md("## 2. The catch — a fat R² on persistent yields is almost free\n\n"
           "Yields are *near-unit-root persistent* and annual returns *overlap* 252-fold. Regress a "
           "persistent series on persistent regressors and you get a big R² **even with no true "
           "link** (Bauer-Hamilton 2018). Our live synthetic control makes this concrete: a **null** "
           "world (no forward→return link) vs a **planted** one. Watch the *R²* (the honest detector) "
           "and note the raw HAC *t* fires even on the null."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from cp_factor import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_daily(edge=0.0, seed=824, n_days=3000))\n"
            "plant = st.synthetic_detect(data.synthetic_daily(edge=0.05, seed=824, n_days=3000))\n"
            "print('null world   : in-sample R2 = %.4f   (honest detector ~ 0)' % null['r2'])\n"
            "print('planted world: in-sample R2 = %.4f   (lights up)' % plant['r2'])\n"
            "print('null raw HAC t = %+.2f  <- size-distorted, fires even with no signal'\n"
            "      % null['cp_slope_t'])"
        ),
        md(f"## 3. The honest verdict — right shape, no robust edge\n\n"
           f"On the real tape the CP regression gives an in-sample **R² = {R['r2']:.3f}** with the "
           f"correct tent — the claim's fingerprint is visibly there. But it does **not** survive the "
           f"honesty rails: a block placebo puts that R² at **p = {R['placebo_p']:.2f}** (pure "
           f"persistence already delivers R² ≈ {R['placebo_mean']:.2f}), the **out-of-sample R² is "
           f"{R['oos_r2']:+.2f}** (worse than a constant), the second era is insignificant "
           f"(*t* = {R['era2_t']:+.2f}), and the headline HAC *t* = {R['cp_slope_t']:+.2f} sits inside "
           f"the synthetic null's own band for that statistic (mean {R['null_t_mean']:+.2f}). A timed "
           f"duration book earns a net Sharpe of ~{R['timer2_sharpe']:.02f} vs {R['bh_sharpe']:.2f} for "
           f"just holding TLT. **Signal: Weak** (right tent, spurious-looking fit), "
           f"**Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 824 — Cochrane-Piazzesi Factor — the teardown\n\n"
           "The HAC predictive regression, the tent loadings, the size-distorted *t*, the "
           "Campbell-Thompson out-of-sample R², the 1,000-draw block placebo, the two-era cut, the "
           "costed duration timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — regress avg 1y excess return on the forward vector\n\n"
           "`avg_rx_{t+252} = γ'·[1, y_short, f_1, f_2, f_3] + e`; the fitted value is the CP factor. "
           f"HAC lags = {R['nw_lags']} (≈ 1.5× the 252-day overlap)."),
        code(
            "print(f\"n = {R['n']}   in-sample R2 = {R['r2']:.4f}   avg excess = {R['avg_rx_bps']:+.1f} bps\")\n"
            "print(f\"single-factor predictive slope = {R['cp_slope']:.3f}  NW t = {R['cp_slope_t']:+.3f}\")\n"
            "print('tent loadings (NW t):')\n"
            "for nm,b,t in [('y_short',R['load_yshort'],R['load_t_yshort']),\n"
            "               ('f_1',R['load_f1'],R['load_t_f1']),\n"
            "               ('f_2',R['load_f2'],R['load_t_f2']),\n"
            "               ('f_3',R['load_f3'],R['load_t_f3'])]:\n"
            "    print(f'   {nm:>8}: {b:+.3f}  (t={t:+.2f})')\n"
            "print('  -> peak on f_2 (5->10y forward): the tent. But every single t is insignificant.')"
        ),
        md("## The naive HAC *t* is size-distorted — the placebo proves it"),
        code(
            "print(f\"block placebo: observed R2 {R['placebo_obs']:.4f} vs null mean {R['placebo_mean']:.4f} \"\n"
            "      f\"(sd {R['placebo_sd']:.4f}) -> p = {R['placebo_p']:.4f}\")\n"
            "print('  persistent regressors ALONE manufacture R2 ~ 0.18 -> observed is ~0.9 sigma up, p=0.21')"
        ),
        md("## Out-of-sample — does it beat the prevailing mean?"),
        code(
            "print(f\"Campbell-Thompson OOS R2 = {R['oos_r2']:+.4f}  ({R['oos_npreds']} forecasts)\")\n"
            "print('  negative -> the CP forecast is WORSE than a constant out of sample')"
        ),
        md("## Robustness — two eras (split 2014-01-01)"),
        code(
            "print(f\"2002-2013 (n={R['era1_n']}): R2={R['era1_r2']:.4f}  NW t={R['era1_t']:+.3f}\")\n"
            "print(f\"2014-2026 (n={R['era2_n']}): R2={R['era2_r2']:.4f}  NW t={R['era2_t']:+.3f}\")\n"
            "print('  whatever fit exists is concentrated in the first (QE / falling-rate) half')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Own TLT when the *out-of-sample* CP forecast is above its rolling median, else cash; "
           "one-way cost × traded NAV per switch."),
        code(
            "for tag,s in [('2 bps',R['timer2_sharpe']),('5 bps',R['timer5_sharpe'])]:\n"
            "    print(f\"{tag:>5} one-way: net Sharpe {s:+.3f}  vs buy-and-hold TLT {R['bh_sharpe']:+.3f}\")\n"
            "print(f\"  ({R['switches']:.1f} switches/yr, invested {R['invested']:.0%}) -> the signal subtracts value\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: R² and OOS R² must be silent on the null and light up on a planted edge. (The raw "
           "single-factor HAC *t* is NOT a valid detector here — it fires on the null.)"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from cp_factor import data, strategy as st\n"
            "null_r2 = np.array([st.synthetic_detect(data.synthetic_daily(edge=0.0, seed=824+s, n_days=3000))['r2'] for s in range(8)])\n"
            "null_t  = np.array([st.synthetic_detect(data.synthetic_daily(edge=0.0, seed=824+s, n_days=3000))['cp_slope_t'] for s in range(8)])\n"
            "plant = st.synthetic_detect(data.synthetic_daily(edge=0.05, seed=824, n_days=3000))\n"
            "print(f\"null (edge=0), 8 seeds: R2 mean {null_r2.mean():.4f} (max {null_r2.max():.4f}) -> SILENT\")\n"
            "print(f\"  but raw HAC t mean {null_t.mean():+.2f}, fires |t|>=2 on {(abs(null_t)>=2).sum()}/8 -> size-distorted\")\n"
            "print(f\"planted (edge=0.05): R2 = {plant['r2']:.4f} -> the machinery recovers a real edge\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The CP regression shows the right **tent** (peak on the 5→10y "
           f"forward) and a fat in-sample **R² = {R['r2']:.3f}**, but it is **≈0.9σ from a spurious "
           f"persistent-regressor fit** (placebo p = {R['placebo_p']:.2f}; null R² ≈ "
           f"{R['placebo_mean']:.2f}), goes **negative out of sample** (OOS R² = {R['oos_r2']:+.2f}), "
           f"is confined to the first era (*t* = {R['era1_t']:+.2f} vs {R['era2_t']:+.2f}), and its "
           f"headline HAC *t* = {R['cp_slope_t']:+.2f} lives inside the size-distorted null band "
           f"(mean {R['null_t_mean']:+.2f}, fires {R['null_fire']}/20). The synthetic control's R² is "
           f"clean (null {R['null_r2_mean']:.4f}, planted {R['plant_r2']:.3f}), so this is the honest "
           f"Bauer-Hamilton reading, not a bug. Coarse-grid proxy caveat on the Signal axis.\n"
           f"- **Tradability — Mirage.** The CP-timed duration book earns a net Sharpe of "
           f"~{R['timer2_sharpe']:.02f} (2 bps) / {R['timer5_sharpe']:.03f} (5 bps) versus "
           f"{R['bh_sharpe']:.2f} for simply holding TLT — the signal subtracts value. No paycheck."),
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
