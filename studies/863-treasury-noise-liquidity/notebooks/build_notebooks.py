"""Generate the two narrative notebooks for Study 863 (Treasury Noise Liquidity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily CMT yields
# ^IRX/^FVX/^TNX/^TYX + SPY/HYG/IEF, 2007-04-12 -> 2026-06-30, 4,833 signal days; noise =
# RMS residual of a quadratic-in-maturity fit; forward SPY / HYG-IEF returns regressed on
# noise_z; yields fingerprint ecbb818e0905).
R = dict(
    fp="ecbb818e0905", n_days=4834, n_sig=4833, start="2007-04-12", end="2026-06-30",
    noise_mean=0.0944, noise_sd=0.0631, noise_max=0.3936,
    # SPY headline by horizon: (label, slope %/1sd, NW t, OLS t, R2 %, hi-lo %)
    spy=[("5d", -0.171, -2.20, -4.75, 0.47, -0.29),
         ("21d", -0.536, -1.97, -7.61, 1.19, -1.04),
         ("63d", -0.755, -1.03, -6.52, 0.88, -1.62)],
    # HYG-IEF (credit) headline by horizon
    cred=[("5d", -0.077, -1.55, -2.99, 0.18, -0.17),
          ("21d", -0.206, -1.29, -3.94, 0.32, -0.51),
          ("63d", 0.117, 0.26, 1.31, 0.04, -0.06)],
    # era cut (split 2016-01-01)
    spy5_early_s=-0.351, spy5_early_t=-2.42, spy5_early_n=2192,
    spy5_late_s=-0.107, spy5_late_t=-1.14, spy5_late_n=2631,
    spy21_early_s=-0.722, spy21_early_t=-1.30, spy21_late_s=-0.476, spy21_late_t=-1.48,
    cred5_early_s=-0.314, cred5_early_t=-2.54, cred5_late_s=0.007, cred5_late_t=0.14,
    cred21_early_t=-1.86, cred21_late_t=0.07,
    # placebo (left-tail, 3000 draws): (label, sigma, p)
    plc_spy5_sig=-2.39, plc_spy5_p=0.0083, plc_spy21_sig=-1.88, plc_spy21_p=0.0303,
    plc_cred5_sig=-1.46, plc_cred5_p=0.0393, plc_cred21_sig=-0.96, plc_cred21_p=0.1603,
    # timer (own SPY when noise < expanding median)
    timer1_sharpe=0.311, bh_sharpe=0.528, timer1_spread=-2.53, timer1_t=-2.14,
    timer3_sharpe=0.292, timer3_spread=-2.63, timer3_t=-2.23,
    switches_yr=12.7, inv_frac=43,
    # synthetic control (20 seeds, 21d)
    null_spy_t=0.17, null_spy_fire=0, planted_spy_edge=0.03, planted_spy_s=-1.768,
    planted_spy_t=-7.33, planted_spy_r2=10.2, planted_spy_fire=20,
    null_cred_t=0.04, planted_cred_edge=0.05, planted_cred_t=-10.41, planted_cred_fire=20,
)


BOOT = (
    "import sys, os\n"
    'sys.path.insert(0, os.path.abspath(".."))          # the study package\n'
    'sys.path.insert(0, os.path.abspath("../../.."))    # repo root\n'
    "%matplotlib inline\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    'plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,\n'
    '                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})\n'
    'RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"\n'
    "\n"
    "from treasury_noise import data, strategy as st\n"
    "\n"
    "# Frozen real-tape headline (mirror of docs/results.md) — the notebook runs fully offline.\n"
    "R = " + repr(R) + "\n"
    "print('Study 863 — frozen real-tape headline loaded:', R['n_sig'], 'signal days,',\n"
    "      R['start'], '->', R['end'], '| yields fingerprint', R['fp'])\n"
)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Treasury 'noise' — does a *rough* yield curve warn of trouble? 🌊\n"
            "### When arbitrage money runs dry, the bond curve gets bumpy — and bumpy curves are "
            "supposed to precede weak stocks and wider credit\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![In--sample_vs_out--of--sample: Decayed](https://img.shields.io/badge/In--sample_vs_out--of--sample-Decayed-8b949e?style=flat-square)\n\n"
            "On a normal day, the yields of Treasury bonds line up along a smooth curve — 3-month, "
            "5-year, 10-year and 30-year rates sit almost exactly where a gentle bend through them "
            "says they should. That smoothness isn't luck: it's the work of well-capitalised "
            "relative-value desks who pounce on any bond that drifts off the curve and trade it back "
            "into line.\n\n"
            "The famous idea we test (Hu, Pan & Wang, 2013, *Noise as Information for Illiquidity*): "
            "when those desks run **short of capital** — a funding squeeze, a crisis — nobody irons "
            "the curve flat any more, and individual maturities wander away from the smooth shape. "
            "The curve gets **rough**. Measure that roughness (how far the four yields sit from a "
            "smooth fit) and you have a real-time gauge of market-wide illiquidity — which is said "
            "to **precede lower stock returns and wider credit spreads**.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a rougher curve precede weaker stocks? | **Yes — with the sign the theory "
            f"predicts.** Over the next week, high-noise days are followed by *lower* SPY returns "
            f"(a *t*-stat of **{R['spy'][0][2]}**, the bar is 2), and it's the real thing, not a "
            f"fluke (a placebo puts it **{R['plc_spy5_sig']}σ** into the left tail). |\n"
            f"| Does it hold up everywhere? | **No.** The punch is almost entirely from the "
            f"crisis-heavy **2007–2015** stretch (*t* = **{R['spy5_early_t']}**); after 2016 it "
            f"fades to an insignificant **{R['spy5_late_t']}**. And the credit-spread version only "
            f"fires during the 2008 crisis. |\n"
            f"| Could you trade it? | **No.** A timer that owns stocks only when the curve is smooth "
            f"actually **loses** to just buying and holding (Sharpe **{R['timer1_sharpe']}** vs "
            f"**{R['bh_sharpe']}**) — the noise spikes *with* the crash, too late to dodge it. |\n\n"
            "> Treasury noise is a genuine stress thermometer — it points the right way and it was "
            "sharp in the crisis years. But its edge is concentrated in exactly those crises, fades "
            "in calm modern markets, and you can't turn it into a paycheck."
        ),

        md("## 1. What 'roughness' means, in one picture\n\n"
           "We take four Treasury yields — 3-month, 5-year, 10-year, 30-year — and draw the "
           "smoothest gentle curve (a quadratic) through them. On a calm day the four points sit "
           "almost *on* that curve; on a stressed day they scatter. The **noise** is just the "
           "typical distance from the curve (root-mean-square of the gaps). Here's the idea on a "
           "toy stressed vs calm day:"),
        code(
            "m = data.MATURITIES\n"
            "P = st._perp_matrix()\n"
            "smooth = 3.0 + 0.25*m - 0.004*m**2\n"
            "rng = np.random.default_rng(0)\n"
            "calm  = smooth + 0.02*rng.standard_normal(4)\n"
            "rough = smooth + 0.30*rng.standard_normal(4)\n"
            "grid = np.linspace(0.25, 30, 100)\n"
            "def fit(y):\n"
            "    A = np.column_stack([np.ones_like(m), m, m**2]); b,_,_,_ = np.linalg.lstsq(A,y,rcond=None)\n"
            "    return b[0]+b[1]*grid+b[2]*grid**2\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(grid, fit(calm), color=GREEN, lw=1.5, label='smooth fit (calm)')\n"
            "ax.scatter(m, calm, color=GREEN, s=60, zorder=5, label=f'calm  (noise={np.sqrt(np.mean((P@calm)**2)):.3f})')\n"
            "ax.plot(grid, fit(rough), color=RED, lw=1.5, ls='--', label='smooth fit (rough)')\n"
            "ax.scatter(m, rough, color=RED, s=60, marker='s', zorder=5, label=f'rough (noise={np.sqrt(np.mean((P@rough)**2)):.3f})')\n"
            "ax.set_xlabel('maturity (years)'); ax.set_ylabel('yield (%)')\n"
            "ax.set_title('Treasury noise = RMS gap of the four yields from a smooth curve')\n"
            "ax.legend(); plt.show()"
        ),

        md("## 2. Is the detector honest? A live synthetic control\n\n"
           "Before trusting any real-tape number we plant the effect in a seeded toy world "
           "(`edge>0`: high noise really does drag next month's return down) and check the detector "
           "recovers it — and that it stays **silent** on the null (`edge=0`: the curve roughens but "
           "that roughness means nothing). No network."),
        code(
            "null    = st.synthetic_detect(data.synthetic_daily(edge=0.0,  seed=863), target='ret_spy', horizon=21)\n"
            "planted = st.synthetic_detect(data.synthetic_daily(edge=0.03, seed=863), target='ret_spy', horizon=21)\n"
            "print('null world   : noise->SPY slope %+.3f%%/1σ  NW t = %+.2f   (should be ~0)' % (null['slope_pct'], null['t_nw']))\n"
            "print('planted world: noise->SPY slope %+.3f%%/1σ  NW t = %+.2f   (should light up, negative)' % (planted['slope_pct'], planted['t_nw']))"
        ),

        md("## 3. The honest verdict\n\n"
           f"On the real tape ({R['start']} → {R['end']}) the noise measure points **exactly the "
           f"way Hu-Pan-Wang say** — a rougher curve today, weaker stocks tomorrow — and at the "
           f"one-week horizon it clears the bar (*t* = **{R['spy'][0][2]}**, {R['plc_spy5_sig']}σ "
           f"into a placebo's left tail). But the edge is a **crisis-era phenomenon**: strong in "
           f"2007–2015 (*t* = {R['spy5_early_t']}), gone after 2016 (*t* = {R['spy5_late_t']}), and "
           f"the credit-spread leg only fires in the 2008 meltdown. A noise-timer **loses** to "
           f"buy-and-hold. So: **Signal Weak** (right sign, real in stress, decays out of sample), "
           f"**Tradability Mirage**."),
    ]
    nb = new_notebook(); nb["cells"] = cells
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Study 863 — Treasury Noise Liquidity — the teardown 🌊\n\n"
            "The roughness construction (RMS residual of a quadratic-in-maturity fit), the forward "
            "SPY / HYG−IEF predictive regressions with a Newey-West slope *t*, the block-rotation "
            "placebo, the two-era cut, the costed regime timer, and the 20-seed synthetic control.\n\n"
            "*Real-tape numbers are the frozen headline (`docs/results.md`); the live cells run the "
            "fast synthetic control. The four CMT indices and the ETFs are continuously listed — no "
            "survivorship bias. The risk-free leg is proxied at 0 (named on the Signal axis: it "
            "moves the intercept, not the slope).*"
        ),
        code(BOOT),

        md("## The headline — forward return regressed on noise (per 1σ)\n\n"
           "Noise is z-scored; the slope is the forward return per 1σ of curve roughness. The claim "
           "(Hu-Pan-Wang) predicts a **negative** slope for both SPY and HYG−IEF."),
        code(
            "print('SPY (equity):')\n"
            "for lab,s,tnw,tols,r2,hilo in R['spy']:\n"
            "    print(f'  {lab:>4}: slope {s:+.3f}%/1σ  NW t {tnw:+.2f}  OLS t {tols:+.2f}  R² {r2:.2f}%  hi-lo {hilo:+.2f}%')\n"
            "print('HYG-IEF (credit-excess):')\n"
            "for lab,s,tnw,tols,r2,hilo in R['cred']:\n"
            "    print(f'  {lab:>4}: slope {s:+.3f}%/1σ  NW t {tnw:+.2f}  OLS t {tols:+.2f}  R² {r2:.2f}%  hi-lo {hilo:+.2f}%')"
        ),

        md("## The tell — the *t*-stats plotted against horizon\n\n"
           "The equity leg is negative (correct sign) and strongest at the shortest horizon; the "
           "credit leg is weaker and reverts by 63 days."),
        code(
            "hz = [5,21,63]\n"
            "spy_t = [t for _,_,t,_,_,_ in R['spy']]\n"
            "cred_t = [t for _,_,t,_,_,_ in R['cred']]\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(hz, spy_t, 'o-', color=RED, label='SPY')\n"
            "ax.plot(hz, cred_t, 's--', color=GREY, label='HYG-IEF')\n"
            "ax.axhline(-2, color=AMBER, lw=1, ls=':'); ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_xlabel('forward horizon (trading days)'); ax.set_ylabel('Newey-West slope t')\n"
            "ax.set_title('Full-tape predictive t by horizon (bar at −2)'); ax.legend(); plt.show()"
        ),

        md("## Robustness — two eras (split 2016-01-01)\n\n"
           "The whole edge lives in the crisis-heavy first half; the modern half is insignificant "
           "(equity) or dead (credit)."),
        code(
            "print('SPY  5d : early slope %+.3f%% (t %+.2f, n=%d) | late %+.3f%% (t %+.2f, n=%d)'\n"
            "      % (R['spy5_early_s'],R['spy5_early_t'],R['spy5_early_n'],R['spy5_late_s'],R['spy5_late_t'],R['spy5_late_n']))\n"
            "print('SPY  21d: early slope %+.3f%% (t %+.2f)         | late %+.3f%% (t %+.2f)'\n"
            "      % (R['spy21_early_s'],R['spy21_early_t'],R['spy21_late_s'],R['spy21_late_t']))\n"
            "print('CRED 5d : early slope %+.3f%% (t %+.2f)         | late %+.3f%% (t %+.2f)'\n"
            "      % (R['cred5_early_s'],R['cred5_early_t'],R['cred5_late_s'],R['cred5_late_t']))\n"
            "print('CRED 21d: early             (t %+.2f)          | late            (t %+.2f)'\n"
            "      % (R['cred21_early_t'],R['cred21_late_t']))"
        ),

        md("## Placebo — block-rotate forward returns vs noise (3,000 draws)\n\n"
           "Circular block rotation preserves the overlap-induced autocorrelation of the forward "
           "return; left-tail p (the claim is a *negative* slope)."),
        code(
            "print(f\"SPY   5d: {R['plc_spy5_sig']:+.2f}σ into the left tail, p = {R['plc_spy5_p']:.4f}\")\n"
            "print(f\"SPY  21d: {R['plc_spy21_sig']:+.2f}σ, p = {R['plc_spy21_p']:.4f}\")\n"
            "print(f\"CRED  5d: {R['plc_cred5_sig']:+.2f}σ, p = {R['plc_cred5_p']:.4f}\")\n"
            "print(f\"CRED 21d: {R['plc_cred21_sig']:+.2f}σ, p = {R['plc_cred21_p']:.4f}  (inside the null cloud)\")"
        ),

        md("## The timer — can you get paid for it?\n\n"
           "Own SPY when the noise known at `t−1` is below its expanding median (calm curve ⇒ own "
           "the market), else cash; one-way cost per switch, no borrow leg."),
        code(
            "print(f\"1 bp/switch: timer Sharpe {R['timer1_sharpe']:.3f} vs buy-and-hold {R['bh_sharpe']:.3f}; \"\n"
            "      f\"timer−BH {R['timer1_spread']:+.2f} bps/day (NW t {R['timer1_t']:+.2f})\")\n"
            "print(f\"3 bp/switch: timer Sharpe {R['timer3_sharpe']:.3f} vs {R['bh_sharpe']:.3f}; \"\n"
            "      f\"timer−BH {R['timer3_spread']:+.2f} bps/day (NW t {R['timer3_t']:+.2f})\")\n"
            "print(f\"switches/yr {R['switches_yr']}, invested {R['inv_frac']}% of the time — it sits out too much\")"
        ),

        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted (negative) "
           "relation, for both SPY and credit, across seeds."),
        code(
            "import numpy as np\n"
            "null_spy    = st.synthetic_mean_t(data, edge=0.0,  n_seeds=8, target='ret_spy',    horizon=21)\n"
            "plant_spy   = st.synthetic_mean_t(data, edge=0.03, n_seeds=8, target='ret_spy',    horizon=21)\n"
            "null_cred   = st.synthetic_mean_t(data, edge=0.0,  n_seeds=8, target='ret_credit', horizon=21)\n"
            "plant_cred  = st.synthetic_mean_t(data, edge=0.05, n_seeds=8, target='ret_credit', horizon=21)\n"
            "print(f\"SPY  null   : mean NW t {null_spy['mean_t']:+.2f}  fires {int(null_spy['fire_frac']*8)}/8\")\n"
            "print(f\"SPY  planted: mean NW t {plant_spy['mean_t']:+.2f}  fires {int(plant_spy['fire_frac']*8)}/8\")\n"
            "print(f\"CRED null   : mean NW t {null_cred['mean_t']:+.2f}  fires {int(null_cred['fire_frac']*8)}/8\")\n"
            "print(f\"CRED planted: mean NW t {plant_cred['mean_t']:+.2f}  fires {int(plant_cred['fire_frac']*8)}/8\")"
        ),

        md(
            "## Verdict\n\n"
            f"- **Signal — Weak.** The Hu-Pan-Wang noise measure has the **right sign** (a rough "
            f"curve precedes weaker equity and wider credit) and is **full-tape significant and "
            f"placebo-real at the short horizon** (SPY 5-day NW *t* = **{R['spy'][0][2]}**, "
            f"{R['plc_spy5_sig']}σ into a 3,000-draw placebo's left tail; 21-day *t* = "
            f"{R['spy'][1][2]}, a near-miss). But the edge is **concentrated in the crisis-heavy "
            f"2007–2015 era** (SPY 5d *t* = {R['spy5_early_t']}) and **decays to insignificance "
            f"after 2016** (*t* = {R['spy5_late_t']}); no sub-era clears |*t*|≥2 at the monthly "
            f"horizon, and the credit leg fires **only in the GFC** ({R['cred5_early_t']} early → "
            f"{R['cred5_late_t']} late). The 20-seed synthetic control recovers a planted relation "
            f"cleanly (SPY *t* = {R['planted_spy_t']}, fires 0/20 on the null), so the decay is real, "
            f"not machinery. Right sign, real in stress, **not certifiable out of sample.**\n"
            f"- **Tradability — Mirage.** A noise-conditioned long/flat SPY timer **loses to "
            f"buy-and-hold** (Sharpe **{R['timer1_sharpe']}** vs **{R['bh_sharpe']}**; "
            f"**{R['timer1_spread']:+.2f} bps/day**, NW *t* = {R['timer1_t']}) while churning "
            f"{R['switches_yr']}×/yr — the noise spikes *coincide* with the drawdown rather than "
            f"leading it, so stepping out on rough days forfeits the rebound. No paycheck.\n"
            f"- **In-sample vs out-of-sample — Decayed.** A ~2.4-*t* crisis-era edge collapses to an "
            f"insignificant −1.1 after 2016 — a post-publication fade."
        ),
    ]
    nb = new_notebook(); nb["cells"] = cells
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
