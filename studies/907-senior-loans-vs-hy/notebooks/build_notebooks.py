"""Generate the two narrative notebooks for Study 907 (Senior Loans vs High-Yield).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the only live-computed cell runs the fast
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (yfinance total-return closes; common window 2011-03-03 -> 2026-06-30; Fingerprint e09ddb919d86).
R = dict(
    start="2011-03-03", end="2026-06-30", n_days=3854, fp="e09ddb919d86",
    # arms: (cagr%, vol%, exSharpe, maxDD%)
    bkln=(3.71, 5.8, 0.414, -24.2), srln=(3.79, 5.4, 0.411, -22.3),
    hyg=(4.72, 8.2, 0.431, -22.0), jnk=(4.61, 8.1, 0.421, -22.9),
    loans=(3.82, 5.4, 0.457, -23.2), hy=(4.67, 8.1, 0.428, -22.5),
    ief=(2.38, 6.5, 0.177, -23.9),
    # flagship BKLN vs HYG
    flag_adv=-0.017, flag_spread_pct=-1.14, flag_spread_t=-0.95,
    # composite
    comp_adv=0.029, comp_spread_pct=-1.00, comp_spread_t=-0.83,
    boot_adv=0.029, boot_lo=-0.258, boot_hi=0.470, boot_win=62,
    # eras: (label, ShL, ShHY, adv, spread_t)
    eras=[("2011-15 energy build-up", 0.54, 0.43, 0.10, -0.49),
          ("2016-19", 1.64, 1.16, 0.48, -1.38),
          ("2020-22 COVID + hike", 0.05, -0.09, 0.13, 0.49),
          ("2023-26", 0.96, 0.69, 0.27, -0.44)],
    # stress: (episode, BKLN, SRLN, HYG, JNK) total return %
    stress=[("Energy wave 2015-16", -6.8, -5.3, -12.1, -15.0),
            ("COVID crash 2020", -23.8, -22.3, -21.9, -22.8),
            ("2022 rate shock", -4.5, -7.4, -14.6, -15.8)],
    # costed long-short net %/yr
    cost5_net=-2.82, cost5_t=-2.34, cost3_net=-2.14, cost3_t=-1.77, gross_ls=-1.02,
    # synthetic control
    null_adv=-0.061, null_sd=0.253, planted_adv=0.334, planted_win=93,
)


HEADER = f"""# Study 907 — Senior Loans vs High-Yield 🏦

**Senior secured loans sit *above* high-yield bonds in the capital stack — do you get paid a
"seniority premium" for it?**

Senior loans (BKLN, SRLN) are first-lien, better-recovery, floating-rate — and yield about
the same as high-yield bonds (HYG, JNK). The pitch: *same carry, less risk, a free seniority
premium.* We race the two sleeves, every Sharpe **excess of cash** (BIL), on the common
window {R['start']} → {R['end']} (bounded by BKLN's 2011 inception, so HY doesn't get the 2008
GFC the loans never saw).

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the live
cell runs the fast synthetic control. Short-history caveat: SRLN only lists 2013 — named on
the Signal axis.*
"""

PY_BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath('../../..'))\n"
    "import numpy as np\n"
    "from loans_vs_hy import data, strategy as st\n"
)

R_LITERAL = "R = " + repr(R) + "\n"


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What 'senior secured' actually buys\n\n"
           "In a default, first-lien **loans** get paid before **bonds** — historical "
           "recoveries ~60–80% for senior secured loans vs ~35–45% for senior unsecured "
           "bonds. And loans **float** (coupon resets with short rates), so they carry almost "
           "no interest-rate duration. Two real, valuable features. The question is whether "
           "they add up to a *higher risk-adjusted return* — or just a *calmer* one."),
        code(PY_BOOT + R_LITERAL +
             "print('LOANS sleeve : CAGR %.2f%%  vol %.1f%%  excess-Sharpe %+.3f  maxDD %.1f%%'\n"
             "      % R['loans'])\n"
             "print('HY    sleeve : CAGR %.2f%%  vol %.1f%%  excess-Sharpe %+.3f  maxDD %.1f%%'\n"
             "      % R['hy'])\n"
             "print()\n"
             "print('Loans are a THIRD less volatile (%.1f%% vs %.1f%%) ...' % (R['loans'][1], R['hy'][1]))\n"
             "print('... but earn a full point LESS per year (%.2f%% vs %.2f%% CAGR).' % (R['loans'][0], R['hy'][0]))\n"
             "print('The two forces nearly cancel: Sharpe %+.3f vs %+.3f.' % (R['loans'][2], R['hy'][2]))"),
        md("## 2. Lower vol is not a free premium\n\n"
           "The seniority discount buys **calm, not extra carry**. On a risk-adjusted basis "
           "the loan sleeve's advantage is a rounding error — and its *sign flips* depending "
           "on whether you use the single flagship pair or the two-ETF composite."),
        code(R_LITERAL +
             "print('Flagship BKLN vs HYG : Sharpe advantage %+.3f  (return spread %.2f%%/yr, t=%.2f)'\n"
             "      % (R['flag_adv'], R['flag_spread_pct'], R['flag_spread_t']))\n"
             "print('Composite  L vs HY   : Sharpe advantage %+.3f  (return spread %.2f%%/yr, t=%.2f)'\n"
             "      % (R['comp_adv'], R['comp_spread_pct'], R['comp_spread_t']))\n"
             "print()\n"
             "print('Bootstrap on the advantage: %+.3f, 95%% CI [%+.3f, %+.3f], loans win %d%% of draws'\n"
             "      % (R['boot_adv'], R['boot_lo'], R['boot_hi'], R['boot_win']))\n"
             "print('-> the interval straddles zero: risk-adjusted, loans and HY are a WASH.')"),
        md("## 3. Where seniority helps — and where it bites\n\n"
           "The honest split. When the pain is **spreads or rates**, seniority + the floating "
           "coupon deliver: loans lose about **half** what HY loses. But in a **pure liquidity "
           "crisis** the loan sleeve — the *less liquid* leg, sold at forced-seller discounts "
           "— gaps as hard or **harder** than HY, exactly when you wanted protection."),
        code(R_LITERAL +
             "print('%-22s %7s %7s %7s %7s' % ('episode','BKLN','SRLN','HYG','JNK'))\n"
             "for e in R['stress']:\n"
             "    print('%-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%%' % e)\n"
             "print()\n"
             "print('Energy & 2022: loans lose ~half of HY (seniority + floating rate work).')\n"
             "print('COVID: BKLN -23.8%% vs HYG -21.9%% -- loans gap WORSE (liquidity run).')"),
        md("## 4. The trade loses money\n\n"
           "To actually *harvest* seniority you'd go **long loans / short HY**. But loans earn "
           "**less**, so the spread is negative before you pay a cent — and after borrow on the "
           "short HY leg it bleeds 2–3%/yr."),
        code(R_LITERAL +
             "print('long loans / short HY, dollar-neutral:')\n"
             "print('  gross spread            : %+.2f%%/yr' % R['gross_ls'])\n"
             "print('  net (5bps + 60bps borrow): %+.2f%%/yr  (t=%.2f)' % (R['cost5_net'], R['cost5_t']))\n"
             "print('  net (3bps + 40bps borrow): %+.2f%%/yr  (t=%.2f)' % (R['cost3_net'], R['cost3_t']))"),
        md("## 5. The takeaway\n\n"
           "- **Signal — WEAK.** Lower vol is real; a higher *risk-adjusted return* is not — "
           "the advantage's sign flips with construction and its bootstrap CI straddles zero.\n"
           "- **Tradability — MIRAGE.** The natural trade is negative gross and −2 to −3%/yr "
           "after costs.\n"
           "- **Free premium? — BUSTED.** Seniority halves your loss in rate/spread selloffs "
           "but costs total return and gaps worse in a liquidity run.\n\n"
           "Senior loans are the **calmer** cousin of high-yield, not the **richer** one."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md(HEADER.replace("🏦", "🏦📊")),
        md("## 1. The race — excess-vs-excess Sharpe, both legs minus cash\n\n"
           "Every arm below is measured **excess of BIL** (the 1-3m T-bill ETF), on the common "
           "2011-inception window. The loan sleeve's lower vol is unmistakable; the "
           "risk-adjusted verdict is not."),
        code(PY_BOOT + R_LITERAL +
             "cols = ['cagr%','vol%','exSharpe','maxDD%']\n"
             "for name in ['bkln','srln','loans','hyg','jnk','hy','ief']:\n"
             "    c,v,s,d = R[name]\n"
             "    print('%-6s CAGR %5.2f%%  vol %4.1f%%  exSharpe %+.3f  maxDD %6.1f%%'\n"
             "          % (name.upper(), c, v, s, d))"),
        md("## 2. The construction sign-flip and the bootstrap\n\n"
           "A real premium survives *how you build the sleeve*. This one does not: HY noses "
           "ahead on the flagship single pair, loans nose ahead on the composite, and a "
           "21-day circular block bootstrap (5,000 draws) on the composite advantage can't "
           "push the CI off zero."),
        code(R_LITERAL +
             "print('flagship  advantage %+.3f' % R['flag_adv'])\n"
             "print('composite advantage %+.3f' % R['comp_adv'])\n"
             "print('bootstrap : %+.3f  95%% CI [%+.3f, %+.3f]  P(loans>HY)=%d%%'\n"
             "      % (R['boot_adv'], R['boot_lo'], R['boot_hi'], R['boot_win']))\n"
             "print('return spread (loans-HY) composite: %.2f%%/yr, NW t = %.2f  (NEGATIVE premium)'\n"
             "      % (R['comp_spread_pct'], R['comp_spread_t']))"),
        md("## 3. Era robustness — consistent sign, never significant\n\n"
           "The loan sleeve's Sharpe edges HY's in every era — a genuinely era-consistent "
           "*sign* — but the margins are small and no era's return spread clears |t| = 2. "
           "Era-consistent noise is still noise."),
        code(R_LITERAL +
             "print('%-24s %6s %6s %6s %8s' % ('era','ShL','ShHY','adv','spread_t'))\n"
             "for lbl,sl,sh,adv,t in R['eras']:\n"
             "    print('%-24s %+5.2f %+5.2f %+5.2f %+7.2f' % (lbl, sl, sh, adv, t))"),
        md("## 4. Tradability — costed long-short\n\n"
           "Gross spread + a monthly-rebalanced round-trip (2 × one-way × NAV) + borrow on the "
           "short HY leg. Negative gross → no cost schedule saves it."),
        code(R_LITERAL +
             "print('gross                    %+.2f%%/yr' % R['gross_ls'])\n"
             "print('net 5bps/side + 60bps     %+.2f%%/yr  (NW t %.2f)' % (R['cost5_net'], R['cost5_t']))\n"
             "print('net 3bps/side + 40bps     %+.2f%%/yr  (NW t %.2f)' % (R['cost3_net'], R['cost3_t']))"),
        md("## 5. The synthetic control (live — proves the machinery)\n\n"
           "A deterministic loans/HY/cash world driven by a shared credit factor, the loan leg "
           "engineered to **lower vol** with a tunable `sharpe_edge`. The **null** sets lower "
           "vol *exactly offset by lower carry* (same Sharpe); the **planted** world gives "
           "loans a genuine risk-adjusted edge. The detector must find nothing in the null and "
           "the edge in the planted world — the reason we trust its verdict of *nothing* on the "
           "real tape. This cell runs live (fast, offline)."),
        code(PY_BOOT +
             "null = []\n"
             "for s in range(12):\n"
             "    f0, _ = data.synthetic_pair(sharpe_edge=0.0, seed=907+s, n_days=4000)\n"
             "    r0 = st.to_returns(f0)\n"
             "    null.append(st.sharpe_advantage(st.excess(r0['LOANS'], r0['CASH']),\n"
             "                                    st.excess(r0['HY'], r0['CASH']))['advantage'])\n"
             "null = np.asarray(null)\n"
             "fp, _ = data.synthetic_pair(sharpe_edge=0.6, seed=907, n_days=4000)\n"
             "det = st.synthetic_detect(fp, n_boot=1500, seed=907)\n"
             "print('null  (edge=0), 12 seeds : mean advantage %+.3f (sd %.3f) -> no systematic edge'\n"
             "      % (null.mean(), null.std(ddof=1)))\n"
             "print('planted (edge=0.6)       : advantage %+.3f, loans win %d%% of bootstrap draws'\n"
             "      % (det['advantage'], round(det['frac_loans_wins']*100)))\n"
             "assert abs(null.mean()) < 0.15 and det['advantage'] > 0.15\n"
             "print('machinery OK: unbiased on the null, recovers a planted edge.')"),
        md("## Verdict\n\n"
           "**Signal WEAK · Tradability MIRAGE · Free-premium BUSTED.** Senior loans are the "
           "lower-vol, rate-proof, spread-cushioning cousin of high-yield — a real *defensive* "
           "tilt — but they earn less, tie on risk-adjusted return (a bootstrap that can't "
           "distinguish them from HY), gap worse in a liquidity run, and can't be harvested "
           "dollar-neutral without losing money. A volatility discount, not a premium."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())]:
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
