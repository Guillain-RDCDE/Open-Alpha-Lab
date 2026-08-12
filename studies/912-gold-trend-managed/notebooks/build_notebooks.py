"""Generate the two narrative notebooks for Study 912 (Gold + Trend).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the only live cell runs the fast
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


# Frozen real-tape headline — mirror of docs/results.md. GLD (gold) vs BIL (cash),
# total-return, 200-day trend overlay, excess-of-cash, 2007-05-30 -> 2026-06-30.
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4802, fp="3ea3fb6649e8",
    in_frac=0.630, n_switches=150,
    bh_sharpe=0.520, bh_cagr=8.08, bh_vol=18.11, bh_dd=-45.56, bh_t=2.37,
    tr_sharpe=0.332, tr_cagr=3.78, tr_vol=14.23, tr_dd=-35.30, tr_t=1.52,
    rnd_sharpe=0.142, rnd_cagr=1.02, rnd_dd=-55.33,
    adv=-0.188, t_diff=-1.83, adv_vs_rnd=0.190, t_vs_rnd=0.96, dd_impr=10.25,
    ci_bh_lo=0.093, ci_bh_hi=0.943, ci_bh_neg=0.9,
    ci_tr_lo=-0.105, ci_tr_hi=0.753, ci_tr_neg=5.9,
    era_e_n=2164, era_e_bh=0.34, era_e_tr=0.06, era_e_adv=-0.28, era_e_t=-1.17,
    era_e_dd_bh=-45.6, era_e_dd_tr=-35.3,
    era_l_n=2636, era_l_bh=0.70, era_l_tr=0.50, era_l_adv=-0.20, era_l_t=-1.70,
    era_l_dd_bh=-26.2, era_l_dd_tr=-30.2,
    cost0_adv=-0.160, cost0_t=-1.68, cost25_adv=-0.299, cost25_t=-2.39,
    iau_adv=-0.183, iau_t=-1.82,
    syn_pl_ddimpr=23.4, syn_pl_adv=0.14, syn_nl_mean=-0.05, syn_nl_sd=0.13, syn_nl_fire=0,
)


HEADER = f"""# Study 912 — Gold + Trend 🥇

**Does a 200-day trend filter turn gold into a better drawdown-managed diversifier?**

Gold is a volatile diversifier (~18% vol) with famously long *dead decades* — 1980-2001,
2012-2018. The folk fix, borrowed from Faber's tactical playbook: hold gold only when its
price is **above** its 200-day moving average, else sit in T-bills. The promise: duck the
dead decades, earn a **better excess-of-cash Sharpe** and **much shallower drawdowns** than
buy-and-hold gold.

We test it on **GLD vs BIL** (cash) daily total-return closes, {R['start']} → {R['end']}
({R['n_days']:,} days), every leg **excess-of-cash**, 5 bps one-way cost.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the one
live cell runs the fast synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Gold spends years going nowhere, then years soaring. If a slow 200-day filter "
           "could keep you *in* for the soars and *out* for the dead decades, gold would "
           "become a smooth diversifier. Let's see whether it actually does."),
        code(
            "R = dict(bh_sharpe=%r, tr_sharpe=%r, adv=%r, t_diff=%r,\n"
            "         bh_dd=%r, tr_dd=%r, bh_cagr=%r, tr_cagr=%r)\n"
            "print('Buy-and-hold gold : excess Sharpe %%+.3f, CAGR %%+.2f%%%%, worst DD %%.1f%%%%'\n"
            "      %% (R['bh_sharpe'], R['bh_cagr'], R['bh_dd']))\n"
            "print('200-day trend     : excess Sharpe %%+.3f, CAGR %%+.2f%%%%, worst DD %%.1f%%%%'\n"
            "      %% (R['tr_sharpe'], R['tr_cagr'], R['tr_dd']))\n"
            "print('advantage (trend - hold): Sharpe %%+.3f  (HAC t = %%+.2f)'\n"
            "      %% (R['adv'], R['t_diff']))"
            % (R["bh_sharpe"], R["tr_sharpe"], R["adv"], R["t_diff"],
               R["bh_dd"], R["tr_dd"], R["bh_cagr"], R["tr_cagr"])
        ),
        md("## 2. The surprise — the overlay makes gold *worse* risk-adjusted\n\n"
           f"The trend overlay cuts the worst drawdown from **{R['bh_dd']:.1f}%** to "
           f"**{R['tr_dd']:.1f}%** (a {R['dd_impr']:.0f} pp cushion) — but it does it by "
           f"forfeiting so much return (CAGR **{R['bh_cagr']:.1f}% → {R['tr_cagr']:.1f}%**) "
           f"that its **excess-of-cash Sharpe falls** ({R['bh_sharpe']:.3f} → "
           f"{R['tr_sharpe']:.3f}). The advantage is **{R['adv']:+.3f}** with the *wrong sign* "
           f"(HAC *t* = {R['t_diff']:+.2f}). Why? Gold's biggest rallies are sharp V-shaped "
           "recoveries off the lows — and a 200-day filter always re-enters them late."),
        md("## 3. But it's not just random luck — it *is* genuine timing\n\n"
           f"A random control that goes to cash on *random* days at the same 63% in-market "
           f"rate does far worse (Sharpe {R['rnd_sharpe']:.3f}, DD {R['rnd_dd']:.1f}%). So the "
           f"trend rule really is picking *better* days than chance (+{R['adv_vs_rnd']:.3f} vs "
           f"random) — it's just that even good timing on gold under-performs simply holding "
           "gold. Genuine skill, wrong asset."),
        md("## 4. And the drawdown cushion doesn't last\n\n"
           f"Split the sample in 2016: the whole drawdown benefit comes from the *early* era "
           f"(ducking the 2013 gold crash, DD {R['era_e_dd_bh']:.0f}% → {R['era_e_dd_tr']:.0f}%). "
           f"In the *recent* era the overlay's drawdown is actually **deeper** than just "
           f"holding gold ({R['era_l_dd_bh']:.0f}% → {R['era_l_dd_tr']:.0f}%). The advantage is "
           f"negative in **both** halves ({R['era_e_adv']:+.2f} / {R['era_l_adv']:+.2f})."),
        md("## 5. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "On a *planted* dead-decade world the overlay works as advertised (cuts drawdown, "
           "lifts Sharpe); on a no-regime null it does nothing. So the real-tape miss is a "
           "fact about *gold*, not a broken backtest."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from gold_trend import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=912)[0])\n"
            "null    = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=912)[0])\n"
            "print('planted dead decade: DD improvement %+.1f pp, excess-Sharpe adv %+.2f (should help)'\n"
            "      % ((planted['dd_improvement'])*100, planted['excess_sharpe_adv']))\n"
            "print('flat-vol null      : excess-Sharpe adv %+.2f (should be ~0)'\n"
            "      % null['excess_sharpe_adv'])"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed *better-Sharpe* drawdown-managed gold does not "
           f"replicate: excess-Sharpe advantage **{R['adv']:+.3f}** (wrong sign, HAC *t* = "
           f"{R['t_diff']:+.2f}), CI includes zero, negative in both eras, and the drawdown "
           f"cushion reverses after 2016.\n"
           f"- **Tradability — Mirage.** Negative gross, more negative with cost. The overlay "
           f"buys a smaller (and recently absent) worst-loss with ~4.3 pp/yr of CAGR — a bad "
           f"trade. Hold gold and size it; don't trend-time it."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 912 — Gold + Trend — the teardown\n\n"
           "The excess-of-cash Sharpe race, the Newey-West return-difference *t*, the "
           "bootstrap Sharpe CIs, the random control, the two-era cut, the cost sweep, the "
           "IAU cross-check, and the live synthetic control. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`)." % R["fp"]),
        code("R = %r" % (R,)),
        md("## The headline — excess-of-cash Sharpe race\n\n"
           "GLD vs BIL, 200-day trend overlay vs buy-and-hold gold vs a 63%-matched random "
           "control. All excess-of-cash; drawdowns are absolute (lived)."),
        code(
            "print(f\"buy-and-hold gold : exSharpe {R['bh_sharpe']:+.3f}  CAGR {R['bh_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['bh_vol']:.1f}%  DD {R['bh_dd']:.1f}%  HAC t {R['bh_t']:+.2f}\")\n"
            "print(f\"200-day trend     : exSharpe {R['tr_sharpe']:+.3f}  CAGR {R['tr_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['tr_vol']:.1f}%  DD {R['tr_dd']:.1f}%  HAC t {R['tr_t']:+.2f}\")\n"
            "print(f\"random control    : exSharpe {R['rnd_sharpe']:+.3f}  CAGR {R['rnd_cagr']:+.2f}%  \"\n"
            "      f\"DD {R['rnd_dd']:.1f}%\")\n"
            "print(f\"\\nadvantage (trend - hold): {R['adv']:+.3f}  HAC t on return diff = {R['t_diff']:+.2f}\")\n"
            "print(f\"advantage vs random     : {R['adv_vs_rnd']:+.3f}  (t = {R['t_vs_rnd']:+.2f})  \"\n"
            "      f\"-> genuine timing, but it still loses to holding gold\")\n"
            "print(f\"in-gold fraction {R['in_frac']:.1%}  |  {R['n_switches']} switches (~8/yr)\")"
        ),
        md("## Bootstrap Sharpe CIs (excess-of-cash, 2,000 draws, 21-day blocks)\n\n"
           "The overlay's Sharpe CI includes zero and sits *below* buy-and-hold's."),
        code(
            "print(f\"buy-and-hold: Sharpe {R['bh_sharpe']:+.3f}  95% CI [{R['ci_bh_lo']:+.3f}, {R['ci_bh_hi']:+.3f}]  share<0 {R['ci_bh_neg']:.1f}%\")\n"
            "print(f\"trend       : Sharpe {R['tr_sharpe']:+.3f}  95% CI [{R['ci_tr_lo']:+.3f}, {R['ci_tr_hi']:+.3f}]  share<0 {R['ci_tr_neg']:.1f}%\")"
        ),
        md("## Robustness — two eras (split 2016-01-01)\n\n"
           "Advantage negative in **both** halves; the drawdown cushion is early-era only."),
        code(
            "print(f\"2007-2015 (n={R['era_e_n']}): BH {R['era_e_bh']:+.2f} / trend {R['era_e_tr']:+.2f}  \"\n"
            "      f\"adv {R['era_e_adv']:+.2f} (t={R['era_e_t']:+.2f})  DD {R['era_e_dd_bh']:.1f}%->{R['era_e_dd_tr']:.1f}%\")\n"
            "print(f\"2016-2026 (n={R['era_l_n']}): BH {R['era_l_bh']:+.2f} / trend {R['era_l_tr']:+.2f}  \"\n"
            "      f\"adv {R['era_l_adv']:+.2f} (t={R['era_l_t']:+.2f})  DD {R['era_l_dd_bh']:.1f}%->{R['era_l_dd_tr']:.1f}%  <- deeper!\")"
        ),
        md("## Cost sweep + IAU cross-check\n\n"
           "The edge is negative *before* any cost, and identical on a second gold ETF."),
        code(
            "print(f\"gross (0 bps): adv {R['cost0_adv']:+.3f} (t={R['cost0_t']:+.2f})\")\n"
            "print(f\"25 bps      : adv {R['cost25_adv']:+.3f} (t={R['cost25_t']:+.2f})  -> only more negative\")\n"
            "print(f\"IAU cross-check: adv {R['iau_adv']:+.3f} (t={R['iau_t']:+.2f})  -> not a GLD idiosyncrasy\")"
        ),
        md("## Live synthetic control — the machinery is unbiased\n\n"
           "Planted dead-decade: the overlay MUST help. Flat-vol null: it must do nothing. "
           "This proves the real-tape miss is a property of gold, not the harness."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from gold_trend import data, strategy as st\n"
            "pl = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=912)[0])\n"
            "print(f\"planted: DD improvement {pl['dd_improvement']*100:+.1f} pp, excess-Sharpe adv {pl['excess_sharpe_adv']:+.2f}\")\n"
            "nl = np.array([st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=912+s)[0])['excess_sharpe_adv'] for s in range(8)])\n"
            "print(f\"null x8: excess-Sharpe adv mean {nl.mean():+.2f} (sd {nl.std(ddof=1):.2f}), |adv|>=0.4 in {(abs(nl)>=0.4).sum()}/8\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The excess-of-cash Sharpe advantage is **{R['adv']:+.3f}** "
           f"with the wrong sign (HAC *t* = {R['t_diff']:+.2f}); the overlay's bootstrap Sharpe "
           f"CI [{R['ci_tr_lo']:+.3f}, {R['ci_tr_hi']:+.3f}] includes zero and sits below "
           f"buy-and-hold's [{R['ci_bh_lo']:+.3f}, {R['ci_bh_hi']:+.3f}]; the advantage is "
           f"negative in both eras ({R['era_e_adv']:+.2f} / {R['era_l_adv']:+.2f}); the drawdown "
           f"cushion (+{R['dd_impr']:.0f} pp overall) is early-era only and *reverses* post-2016. "
           f"The rule beats a random control (real timing) but still loses to holding gold — the "
           f"claimed better-Sharpe diversifier is absent. The synthetic control fires cleanly on "
           f"a planted regime (adv {R['syn_pl_adv']:+.2f}) and is silent on the null "
           f"(mean {R['syn_nl_mean']:+.2f}, {R['syn_nl_fire']}/8), so the miss is real.\n"
           f"- **Tradability — Mirage.** Negative gross, more negative net "
           f"({R['cost25_adv']:+.3f} at 25 bps). The overlay pays ~4.3 pp/yr of CAGR for a "
           f"worst-loss cut that has vanished in the recent decade. Not bankable."),
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
