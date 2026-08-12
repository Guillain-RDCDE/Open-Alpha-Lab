"""Generate the two narrative notebooks for Study 899 (Cash + Call "90/10").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the frozen
``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic control, so
execution is quick and network-free. Every code cell that references ``R`` is a plain runtime
f-string; ``R`` itself is injected once, in its own cell.
"""

from __future__ import annotations

import os
import pprint

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (SPY total return, BIL cash, ^IRX
# rate; 2007-05-31 -> 2026-06-30; 90/10 = rolling 1-yr ATM SPY call marked daily with Black-Scholes,
# <=10% premium budget, notional capped at 100% of NAV; all excess-of-cash, BS-fair premium).
R = dict(
    start="2007-05-31", end="2026-06-30", n_days=4799, fp="b954e845292f",
    tt_sharpe=0.492, tt_sortino=0.497, tt_cagr=6.13, tt_vol=10.42, tt_dd=-19.8,
    bh_sharpe=0.543, bh_sortino=0.513, bh_cagr=10.70, bh_vol=19.78, bh_dd=-55.2,
    static_sharpe=0.543, static_cagr=8.09, static_vol=13.44, static_dd=-40.2,
    sharpe_vs_bh=-0.051, alpha_ann=0.46, t_alpha=0.33, beta=0.638, diff_t_nw=-1.37,
    up_cap=0.564, dn_cap=0.568, asym=-0.003,
    avg_w=0.68, turnover=0.18, n_rolls=19,
    boot_point=-0.051, boot_lo=-0.334, boot_hi=0.210, boot_win=33.8,
    crash08_bh=-36.8, crash08_bh_dd=-47.1, crash08_c=-8.1, crash08_c_dd=-10.2,
    crash20_bh=18.3, crash20_bh_dd=-33.7, crash20_c=-1.1, crash20_c_dd=-15.7,
    crash22_bh=-18.2, crash22_bh_dd=-24.5, crash22_c=-17.1, crash22_c_dd=-19.6,
    era_e_tt=0.453, era_e_bh=0.335, era_e_vs=0.119, era_e_t=0.94, era_e_n=2163,
    era_l_tt=0.999, era_l_bh=0.767, era_l_vs=0.232, era_l_t=1.68, era_l_n=2635,
    pm080_sh=0.629, pm080_vs=0.087, pm100_sh=0.492, pm100_vs=-0.051,
    pm125_sh=0.335, pm125_vs=-0.208, pm150_sh=0.193, pm150_vs=-0.350,
    pm200_sh=-0.039, pm200_vs=-0.582,
    bear_prot=0.369, bear_prot_min=0.043, bear_w=0.41, calm_w=0.79, n_seeds=30,
)

R_CELL = "R = " + pprint.pformat(R, width=100)

BOOT = ("import os, sys\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n")


HEADER = f"""# Study 899 — Cash + Call "90/10" 📈

**Bill Gross's "90/10": park ~90% in T-bills so your capital comes back to par, and spend ~10%
"renting" convex upside with call options. If the market crashes the calls expire worthless and the
bills carry you; if it rockets you get leveraged, capped-loss upside. Does "protect capital, rent
upside" actually beat just holding stocks on a risk-adjusted basis?**

Free listed-option history doesn't exist, so the ~10% convex sleeve is a **documented proxy**: a
rolling **1-year at-the-money SPY call, marked daily with Black–Scholes** (strike = spot at each
annual roll, priced off SPY's trailing realized vol and the ^IRX bill rate; the 10% budget buys as
much notional as the fair price affords, capped near 100% of capital). Real tape SPY/BIL,
{R['start']} → {R['end']}.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`); the live cells
run the fast synthetic control. Short history: BIL lists 2007 — a single, GFC-anchored cycle (which
is exactly where a capital floor should shine). The Black–Scholes mark **flatters** the strategy
(realized-vol pricing, no dividend drag) — the premium sweep restores reality.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        code(R_CELL),
        md("## 1. Capital protection really works — that part is true\n\n"
           "The profile does exactly what it says on the tin: it floors the drawdown. Over 2007–2026 "
           "the 90/10 book's worst peak-to-trough was **a third** of buy-and-hold's, at **half the "
           "volatility** — the calls simply expired worthless in the crashes while the bills carried "
           "the book."),
        code(
            "print(f\"90/10  : maxDD {R['tt_dd']:.1f}%   vol {R['tt_vol']:.1f}%\")\n"
            "print(f\"S&P 500: maxDD {R['bh_dd']:.1f}%   vol {R['bh_vol']:.1f}%\")\n"
            "print(f\"2008 crash: S&P {R['crash08_bh']:+.1f}% (DD {R['crash08_bh_dd']:.1f}%)\"\n"
            "      f\"  ->  90/10 {R['crash08_c']:+.1f}% (DD {R['crash08_c_dd']:.1f}%)\")"
        ),
        md("## 2. …but it doesn't actually *beat* stocks risk-adjusted — it ties\n\n"
           "The pitch is a better *risk-adjusted* outcome. It isn't there. Once you measure both books "
           f"in **excess of cash**, 90/10's **Sharpe {R['tt_sharpe']:.2f}** essentially **ties** "
           f"buy-and-hold's **{R['bh_sharpe']:.2f}** — the gap is a rounding error, and a bootstrap "
           "can't tell them apart (P(90/10 wins) ~34%). You gave up **4.6%/yr of compounding** "
           f"({R['tt_cagr']:.1f}% vs {R['bh_cagr']:.1f}%) to reshape the *tail*, not to earn more per "
           "unit of risk."),
        code(
            "print(f\"excess Sharpe : 90/10 {R['tt_sharpe']:.3f}   vs   S&P 500 {R['bh_sharpe']:.3f}   (gap {R['sharpe_vs_bh']:+.3f})\")\n"
            "print(f\"CAGR         : 90/10 {R['tt_cagr']:.2f}%   vs   S&P 500 {R['bh_cagr']:.2f}%\")\n"
            "print(f\"bootstrap 95% CI on the Sharpe gap: [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  \"\n"
            "      f\"P(90/10 wins) {R['boot_win']:.1f}%  -> a statistical tie\")"
        ),
        md("## 3. And a *real* option costs more than our model assumed\n\n"
           "Our call is priced off **realized** volatility. A real listed call trades at **implied** "
           "vol, which is systematically *higher* (the variance risk premium — that's how option "
           "sellers get paid). Charge the option what it actually costs — a **1.25–1.5×** markup — and "
           "the tie flips to a clear **loss** vs buy-and-hold. Renting upside is a negative-carry trade "
           "priced to leave the buyer behind."),
        code(
            "for lbl,sh,vs in [('BS-fair (realized vol)',R['pm100_sh'],R['pm100_vs']),\n"
            "                  ('1.25x (mild VRP)',R['pm125_sh'],R['pm125_vs']),\n"
            "                  ('1.50x (typical)',R['pm150_sh'],R['pm150_vs'])]:\n"
            "    print(f\"  option cost {lbl:24s}: 90/10 Sharpe {sh:+.3f}   vs S&P {vs:+.3f}\")"
        ),
        md("## 4. A live synthetic control — the engine is honest\n\n"
           "We plant a **bear** world (capital must be protected) and a **calm** steady bull (nothing "
           "to protect, so the premium just bleeds), and check the machinery reads each correctly. "
           "No network."),
        code(
            BOOT +
            "from cash_call import data, strategy as st\n"
            "bear = st.synthetic_detect(data.synthetic_prices(seed=899, n_days=2500, drift=-0.0004, sigma=0.016)[0])\n"
            "calm = st.synthetic_detect(data.synthetic_prices(seed=899, n_days=2500, drift=0.0005, sigma=0.006)[0])\n"
            "print(f\"bear world : drawdown protection {bear['dd_protection']:+.3f}, equity de-risked to {bear['avg_weight']:.2f}  (capital floored)\")\n"
            "print(f\"calm bull  : 90/10 Sharpe {calm['sharpe_tt']:+.3f} vs S&P {calm['sharpe_bh']:+.3f}, equity weight {calm['avg_weight']:.2f}  (premium bleeds, upside given up)\")"
        ),
        md("## 5. The honest verdict\n\n"
           f"- **Signal: Weak.** The capital protection is **real and mechanical** (maxDD "
           f"{R['tt_dd']:.1f}% vs {R['bh_dd']:.1f}%, 2008 cut from −47% to −10%). But the "
           f"*risk-adjusted-return* claim isn't there: excess Sharpe **{R['tt_sharpe']:.2f} vs "
           f"{R['bh_sharpe']:.2f}** is a statistical tie, the option's convexity adds no alpha "
           f"(*t* = {R['t_alpha']:+.2f}), and that's at a premium that *flatters* the strategy.\n"
           f"- **Tradability: Mirage.** The parity survives only at the fictional fair price — a "
           f"realistic option markup (1.25–1.5×) sinks it clearly below buy-and-hold. It's not a "
           f"trading-cost story (one roll a year); it's that renting upside is negative-carry. You run "
           f"90/10 to **sleep at night**, not to get paid."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 899 — Cash + Call \"90/10\" — the teardown\n\n"
           "The three-book race (90/10 / buy-and-hold / matched static), the excess-of-cash identity, "
           "the leverage-clean **convexity** spanning alpha, the block-bootstrap Sharpe-difference CI, "
           "the two-era cut, the **premium (variance-risk-premium) sweep** that is the tradability "
           "crux, and the synthetic control. The 10% sleeve is a Black–Scholes-marked 1-yr ATM SPY "
           "call — a documented proxy for a listed call (realized-vol priced, no dividend: both tilts "
           "*flatter* the strategy, named on the Signal axis)."),
        code(R_CELL),
        md("## The race — 90/10 vs buy-and-hold vs matched static mix (excess-of-cash, gross, BS-fair)\n\n"
           "A *constant* fraction of SPY funded from cash has the **same** excess-of-cash Sharpe as SPY "
           "— so the matched-static and buy-and-hold Sharpes coincide, and 90/10's only distinguishing "
           "act is the option's **convexity** (the spanning alpha)."),
        code(
            "print(f\"90/10       : exSharpe {R['tt_sharpe']:.3f}  Sortino {R['tt_sortino']:.3f}  CAGR {R['tt_cagr']:5.2f}%  vol {R['tt_vol']:5.2f}%  maxDD {R['tt_dd']:.1f}%\")\n"
            "print(f\"buy-and-hold: exSharpe {R['bh_sharpe']:.3f}  Sortino {R['bh_sortino']:.3f}  CAGR {R['bh_cagr']:5.2f}%  vol {R['bh_vol']:5.2f}%  maxDD {R['bh_dd']:.1f}%\")\n"
            "print(f\"static@avgw : exSharpe {R['static_sharpe']:.3f}                CAGR {R['static_cagr']:5.2f}%  vol {R['static_vol']:5.2f}%  maxDD {R['static_dd']:.1f}%\")\n"
            "print(f\"vs buy-and-hold {R['sharpe_vs_bh']:+.3f}   convexity alpha {R['alpha_ann']:+.2f}%/yr (HAC t {R['t_alpha']:+.2f}, beta {R['beta']:.3f})\")\n"
            "print(f\"up-capture {R['up_cap']:.3f} / down-capture {R['dn_cap']:.3f} (asym {R['asym']:+.3f})  avg Δ-w {R['avg_w']:.2f}  roll turnover {R['turnover']:.2f}x/yr\")"
        ),
        md("## Bootstrap — circular block CI on the excess-Sharpe difference (90/10 − buy-and-hold)"),
        code(
            "print(f\"gain {R['boot_point']:+.3f}  95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  \"\n"
            "      f\"P(90/10 wins) {R['boot_win']:.1f}%   -> the CI straddles zero: a statistical tie\")"
        ),
        md("## Crash years — capital protection bites, the recovery is rented not owned"),
        code(
            "for yr,bh,bd,c,cd in [(2008,R['crash08_bh'],R['crash08_bh_dd'],R['crash08_c'],R['crash08_c_dd']),\n"
            "                      (2020,R['crash20_bh'],R['crash20_bh_dd'],R['crash20_c'],R['crash20_c_dd']),\n"
            "                      (2022,R['crash22_bh'],R['crash22_bh_dd'],R['crash22_c'],R['crash22_c_dd'])]:\n"
            "    print(f\"{yr}: BH {bh:+6.1f}% (DD {bd:6.1f}%)  ->  90/10 {c:+6.1f}% (DD {cd:6.1f}%)\")"
        ),
        md("## Robustness — two eras (split 2016-01-01)\n\n"
           "Each half is *individually* a shade favourable to 90/10 (vs-BH +0.12, +0.23) yet the pooled "
           "gap is slightly negative — the usual Sharpe-not-additive artefact of mixing a high-vol GFC "
           "regime with a calmer one. Neither era's convexity alpha clears significance (*t* < 2)."),
        code(
            "print(f\"2007-2015 (n={R['era_e_n']}): 90/10-Sh {R['era_e_tt']:+.3f}  BH-Sh {R['era_e_bh']:+.3f}  vs-BH {R['era_e_vs']:+.3f}  alpha-t {R['era_e_t']:+.2f}\")\n"
            "print(f\"2016-2026 (n={R['era_l_n']}): 90/10-Sh {R['era_l_tt']:+.3f}  BH-Sh {R['era_l_bh']:+.3f}  vs-BH {R['era_l_vs']:+.3f}  alpha-t {R['era_l_t']:+.2f}\")"
        ),
        md("## The tradability crux — the premium (variance risk premium) sweep\n\n"
           "The Black–Scholes price uses **realized** vol; a real listed call trades at **implied** vol "
           "(IV/RV ≈ 1.1–1.4 on the S&P — the variance risk premium). `prem_mult` scales the option "
           "cost to that reality; the near-tie evaporates the moment you pay what the option costs."),
        code(
            "for lbl,sh,vs in [('0.80x (too cheap)',R['pm080_sh'],R['pm080_vs']),('1.00x BS-fair',R['pm100_sh'],R['pm100_vs']),\n"
            "                  ('1.25x mild VRP',R['pm125_sh'],R['pm125_vs']),('1.50x typical',R['pm150_sh'],R['pm150_vs']),\n"
            "                  ('2.00x stressed',R['pm200_sh'],R['pm200_vs'])]:\n"
            "    print(f\"  {lbl:18s}: 90/10 Sharpe {sh:+.3f}   vs S&P {vs:+.3f}\")"
        ),
        md("## Synthetic control — the machinery is unbiased (live, offline)"),
        code(
            BOOT +
            "import numpy as np\n"
            "from cash_call import data, strategy as st\n"
            "bear = np.array([st.synthetic_detect(data.synthetic_prices(seed=899+s, n_days=2500, drift=-0.0004, sigma=0.016)[0])['dd_protection'] for s in range(8)])\n"
            "bw   = np.array([st.synthetic_detect(data.synthetic_prices(seed=899+s, n_days=2500, drift=-0.0004, sigma=0.016)[0])['avg_weight'] for s in range(8)])\n"
            "cw   = np.array([st.synthetic_detect(data.synthetic_prices(seed=899+s, n_days=2500, drift=0.0005, sigma=0.006)[0])['avg_weight'] for s in range(8)])\n"
            "print(f\"bear (8 seeds): drawdown protection mean {bear.mean():+.3f}  (capital floored)\")\n"
            "print(f\"equity weight: bear {bw.mean():.2f}  <  calm {cw.mean():.2f}  -> the rule de-risks as vol rises\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — WEAK.** Capital protection is **real and mechanical**: maxDD **{R['tt_dd']:.1f}%** "
           f"vs **{R['bh_dd']:.1f}%** at half the vol, 2008 cut −47%→−10%, confirmed by a 30-seed synthetic "
           f"control. But the risk-adjusted-return claim fails: even at the BS-fair premium the excess "
           f"Sharpe **{R['tt_sharpe']:.3f} vs {R['bh_sharpe']:.3f}** is a tie (bootstrap CI "
           f"[{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]), the convexity adds no alpha "
           f"(*t* = {R['t_alpha']:+.2f}), and both eras are insignificant. Single GFC-anchored ~19-yr "
           f"window; the BS mark *flatters* (realized-vol, no dividend).\n"
           f"- **Tradability — MIRAGE.** The parity is an artefact of the fair-price assumption. A real "
           f"call carries the variance risk premium (IV>RV): at a **1.25–1.5×** markup the excess Sharpe "
           f"drops to **{R['pm125_vs']:+.3f}…{R['pm150_vs']:+.3f}**, clearly below buy-and-hold — and the "
           f"call-holder forgoes the ~1.8%/yr dividend. Not a friction story (0.18×/yr turnover). At best "
           f"it *matches* buy-and-hold's Sharpe while giving up ~4.6%/yr CAGR — a payoff reshaping, not a "
           f"paycheck."),
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
