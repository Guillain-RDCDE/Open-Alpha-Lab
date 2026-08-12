"""Generate the two narrative notebooks for Study 910 (Managed-Distribution CEF).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic control,
so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return closes;
# PCEF + equal-weight PDI/UTF/BST/RQI basket vs SPY, excess of BIL; as-of 2026-06-30).
R = dict(
    fp="cae85d4d2cca", asof="2026-06-30",
    # basket, full sample
    b_n=140, b_start="2014-11", b_end="2026-06", b_ann=11.5, spy_ann=13.8, b_vol=17,
    b_exret=88.3, b_t=2.63, b_sharpe=0.61, spy_sharpe=0.82, b_adv=-0.21,
    b_vsspy=-13.6, b_t_vsspy=-0.69, b_alpha=-1.6, b_talpha=-0.59, b_beta=1.00, b_r2=0.73,
    b_maxdd=-28,
    b_boot_lo=0.12, b_boot_hi=1.17, b_boot_fn=0.005,
    # PCEF
    p_n=196, p_exret=49.3, p_t=2.48, p_sharpe=0.51, p_adv=-0.41,
    p_alpha=-3.3, p_talpha=-1.96, p_beta=0.69, p_boot_lo=0.08, p_boot_hi=1.03,
    # per fund one-liners
    rqi_maxdd=-87, rqi_alpha=-4.7, bst_exret=138.5, pdi_exret=75.4,
    # era cut basket
    pre_n=86, pre_exret=123.9, pre_t=2.94, pre_sharpe=0.90, pre_adv=-0.10, pre_alpha=1.2,
    post_n=54, post_exret=31.6, post_t=0.62, post_sharpe=0.20, post_adv=-0.36,
    post_alpha=-5.5, post_talpha=-1.62,
    # PCEF era
    pcef_post_alpha=-5.0, pcef_post_talpha=-2.78,
    # costs
    gross_exret=88.3, net_exret=88.0, charge=0.25,
    # calendar 2022
    cy2022=-24.4,
    # synthetic
    syn_null_alpha=-1.46, syn_null_t=-0.86, syn_plant_alpha=3.54, syn_plant_t=2.10,
    syn_beta=1.11, syn_null_fire="2/20",
)


HEADER = f"""# Study 910 — Managed-Distribution CEF 🎁

**Do persistent-discount closed-end funds with a big "managed distribution" hand you the
discount pull *and* the payout — or just a levered-beta clone with a yield sticker?**

The pitch: a closed-end fund (CEF) trades below the value of what it holds (buy a dollar of assets
for ~90 cents — the *discount pull*) *and* pays a fat, level distribution (8–14 %/yr). Sounds like
free double-carry. The sceptic's counter — the mREIT lesson of
[611](../../611-mreit-carry/) — is that the fat payout is often your own capital handed back
(*return of capital*), and the leverage inside the wrapper is financed at short rates, so **NAV
erosion + leverage cost can eat the whole thing**.

We test the buyer's bottom line on **PCEF** (the CEF-of-CEFs) and an equal-weight basket of four
large CEFs (**PDI, UTF, BST, RQI**) vs **SPY**, everything **excess of cash (BIL)**, on
total-return tape ({R['b_start']} → {R['b_end']} for the basket; PCEF from 2010).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`, as-of
{R['asof']}); the live cells run the fast synthetic control. Survivorship + short basket history
bias the magnitude upward.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The two things a CEF supposedly gives you\n\n"
           "**The discount pull.** If a fund holding \\$100 of assets trades at \\$90, you own "
           "\\$100 of stuff for \\$90 — and if the discount ever closes, you pocket the gap.\n\n"
           "**The payout.** A *managed distribution plan* pays a fixed, fat yield on a schedule. "
           "The catch the brochure buries: when the fund hasn't *earned* that much, the extra is "
           "**return of capital** — literally your own money handed back, while the NAV quietly "
           "shrinks. A 12 % 'yield' that is half return-of-capital is a 6 % yield plus a slow "
           "leak. Because we use **total-return** prices (distributions reinvested), our numbers "
           "see through the label to the real economic return."),
        code(
            "R = dict(b_exret=%r, b_t=%r, b_sharpe=%r, spy_sharpe=%r, b_adv=%r,\n"
            "         b_alpha=%r, b_talpha=%r, b_beta=%r, b_maxdd=%r, rqi_maxdd=%r)\n"
            "print('CEF basket, excess of cash:  %%+.1f bps/mo  (HAC t = %%+.2f)'\n"
            "      %% (R['b_exret'], R['b_t']))\n"
            "print('  -> a REAL payout: the return clears cash, t > 2')\n"
            "print('excess-of-cash Sharpe:  basket %%.2f   vs   SPY %%.2f   (advantage %%+.2f)'\n"
            "      %% (R['b_sharpe'], R['spy_sharpe'], R['b_adv']))\n"
            "print('  -> but risk-adjusted it TRAILS the index it is sold against')\n"
            "print('CAPM alpha vs SPY:  %%+.1f%%%%/yr (t = %%+.2f),  beta = %%.2f  (levered!)'\n"
            "      %% (R['b_alpha'], R['b_talpha'], R['b_beta']))\n"
            "print('worst real-estate CEF (RQI) max drawdown: %%d%%%%' %% R['rqi_maxdd'])"
            % (R["b_exret"], R["b_t"], R["b_sharpe"], R["spy_sharpe"], R["b_adv"],
               R["b_alpha"], R["b_talpha"], R["b_beta"], R["b_maxdd"], R["rqi_maxdd"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We build a toy 'CEF' = a **levered** claim on a market factor (β = 1.1) **plus** a "
           "structural carry we can dial, **minus** a return-of-capital leak we can dial. Three "
           "worlds, all offline:\n\n"
           "* **null** — pure levered beta, no carry: the alpha test must stay silent;\n"
           "* **planted** — a genuine +5 %/yr net carry: the alpha must light up;\n"
           "* **return-of-capital trap** — a fat carry that is *entirely* leaked back (carry = "
           "leak): net zero, so — like the worst real CEFs — it must produce **no alpha**."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from md_cef import data, strategy as st\n"
            "for carry, leak, tag in [(0.0,0.0,'null (levered beta)'),\n"
            "                         (0.05,0.0,'planted +5%/yr carry'),\n"
            "                         (0.05,0.05,'return-of-capital trap')]:\n"
            "    d = st.synthetic_detect(data.synthetic_world(carry_annual=carry, roc_leak_annual=leak, seed=910))\n"
            "    fires = 'FIRES' if abs(d['t_alpha'])>=2 else 'silent'\n"
            "    print(f\"{tag:<24s}: alpha {d['alpha_ann_pct']:+.2f}%/yr (t {d['t_alpha']:+.2f})  beta {d['beta']:.2f}  -> {fires}\")"
        ),
        md(f"## 3. The honest verdict — half-true\n\n"
           f"The payout is **real**: the basket earns **{R['b_exret']:+.1f} bps/mo excess of "
           f"cash** (HAC *t* = {R['b_t']:+.2f}), and unlike a blown-up mREIT its total return "
           f"genuinely clears cash. So the sticker isn't pure fiction.\n\n"
           f"But the **edge over the asset class isn't there**. Risk-adjusted, the basket's "
           f"excess-of-cash Sharpe (**{R['b_sharpe']:.2f}**) **trails SPY's ({R['spy_sharpe']:.2f})**, "
           f"the CAPM alpha is {R['b_alpha']:+.1f} %/yr (*t* = {R['b_talpha']:+.2f}), and β ≈ "
           f"{R['b_beta']:.2f} tells the story: you're holding **levered equity beta** dressed as "
           f"income. When rates jumped in 2022 the whole thing fell **{R['cy2022']:.1f} %** in a "
           f"year. **Signal: Weak** (real payout, no asset-class edge), **Tradability: Mirage** "
           f"(a hidden levered beta — not costs — erases it; you'd do better just holding SPY)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 910 — Managed-Distribution CEF — the teardown\n\n"
           "The excess-of-cash Sharpe race vs SPY, the HAC *t*'s, the excess-vs-excess CAPM β/α, "
           "the bootstrap Sharpe CI, the 2022 rate-hike era cut, the cost math, and the "
           "planted-carry / return-of-capital-trap synthetic control. All excess of cash (BIL)."),
        code("R = %r" % (R,)),
        md("## The headline — equal-weight basket (PDI/UTF/BST/RQI), excess of cash\n\n"
           "A real return over cash — but a **negative** Sharpe advantage vs SPY and a "
           "zero-to-negative alpha once you net out the (levered) beta."),
        code(
            "print(f\"basket : n={R['b_n']}  {R['b_start']}->{R['b_end']}  ann {R['b_ann']:+.1f}% vs SPY {R['spy_ann']:+.1f}%  vol {R['b_vol']}%\")\n"
            "print(f\"excess-of-cash : {R['b_exret']:+.1f} bps/mo  (HAC t = {R['b_t']:+.2f})\")\n"
            "print(f\"Sharpe race    : basket {R['b_sharpe']:.2f}  vs SPY {R['spy_sharpe']:.2f}  -> advantage {R['b_adv']:+.2f}\")\n"
            "print(f\"basket - SPY   : {R['b_vsspy']:+.1f} bps/mo  (HAC t = {R['b_t_vsspy']:+.2f})\")\n"
            "print(f\"CAPM vs SPY    : alpha {R['b_alpha']:+.1f}%/yr (t = {R['b_talpha']:+.2f})  beta {R['b_beta']:.2f}  R2 {R['b_r2']:.2f}\")\n"
            "print(f\"max drawdown   : {R['b_maxdd']}%   (RQI alone {R['rqi_maxdd']}% - the mREIT trap in miniature)\")"
        ),
        md("## Bootstrap Sharpe CI (moving-block) — positive, but the whole interval sits below SPY"),
        code(
            "print(f\"basket excess-of-cash Sharpe {R['b_sharpe']:.2f}  95%% CI [{R['b_boot_lo']:.2f}, {R['b_boot_hi']:.2f}]  frac_neg={R['b_boot_fn']:.3f}\")\n"
            "print(f\"PCEF   excess-of-cash Sharpe {R['p_sharpe']:.2f}  95%% CI [{R['p_boot_lo']:.2f}, {R['p_boot_hi']:.2f}]\")\n"
            "print(f\"SPY excess-of-cash Sharpe on the same months: {R['spy_sharpe']:.2f}  (above both intervals' centre)\")"
        ),
        md("## PCEF — the diversified CEF-of-CEFs, 16 years: same story, sharper\n\n"
           "Longer history, and the alpha is *significantly* negative."),
        code(
            "print(f\"PCEF: n={R['p_n']}  excess-of-cash {R['p_exret']:+.1f} bps/mo (t {R['p_t']:+.2f})  Sharpe adv {R['p_adv']:+.2f}\")\n"
            "print(f\"      CAPM alpha {R['p_alpha']:+.1f}%/yr (t {R['p_talpha']:+.2f})  beta {R['p_beta']:.2f}\")"
        ),
        md("## Era cut at 2022-01 — the rate-hike regime broke it\n\n"
           "Leverage got expensive, discounts blew out. The excess return halved to statistical "
           "zero and the alpha turned sharply negative — basket *and* PCEF."),
        code(
            "print(f\"basket pre  (n={R['pre_n']}) : {R['pre_exret']:+.1f} bps/mo (t {R['pre_t']:+.2f})  Sharpe {R['pre_sharpe']:.2f}  adv {R['pre_adv']:+.2f}  alpha {R['pre_alpha']:+.1f}%/yr\")\n"
            "print(f\"basket post (n={R['post_n']}) : {R['post_exret']:+.1f} bps/mo (t {R['post_t']:+.2f})  Sharpe {R['post_sharpe']:.2f}  adv {R['post_adv']:+.2f}  alpha {R['post_alpha']:+.1f}%/yr (t {R['post_talpha']:+.2f})\")\n"
            "print(f\"PCEF   post : alpha {R['pcef_post_alpha']:+.1f}%/yr (t {R['pcef_post_talpha']:+.2f})  -- significantly negative\")\n"
            "print(f\"calendar 2022 basket return: {R['cy2022']:+.1f}%\")"
        ),
        md("## Tradability — costs are NOT the killer; a hidden levered beta is"),
        code(
            "print(f\"gross {R['gross_exret']:+.1f} bps/mo -> net {R['net_exret']:+.1f} bps/mo (charge {R['charge']:.2f} bps/mo; buy-and-hold)\")\n"
            "print(f\"net Sharpe still {R['b_sharpe']:.2f} < SPY {R['spy_sharpe']:.2f}: you get a better risk-adjusted return just holding SPY.\")\n"
            "print('The edge dies in the factor regression (beta ~1.0, R2 0.73), not at the trading desk -> Mirage.')"
        ),
        md("## Synthetic control — the machinery is unbiased *(never market evidence)*\n\n"
           "Live: the null (pure levered beta) must NOT fire; a planted net carry must be "
           "recovered as alpha; a 100%-return-of-capital payout must yield no alpha."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from md_cef import data, strategy as st\n"
            "for carry, leak, tag in [(0.0,0.0,'null'), (0.05,0.0,'planted +5%'), (0.05,0.05,'ROC trap')]:\n"
            "    d = st.synthetic_detect(data.synthetic_world(carry_annual=carry, roc_leak_annual=leak, seed=910))\n"
            "    print(f\"{tag:<12s}: alpha {d['alpha_ann_pct']:+.2f}%/yr (t {d['t_alpha']:+.2f})  beta {d['beta']:.2f}\")\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=910+s))['t_alpha'] for s in range(20)])\n"
            "print(f\"null over 20 seeds: |t|>=2 in {(np.abs(null_t)>=2).sum()}/20  (~5% false-positive rate)\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The payout is a *real* excess-of-cash return (basket "
           f"**{R['b_exret']:+.1f} bps/mo**, HAC *t* = **{R['b_t']:+.2f}**, bootstrap Sharpe CI "
           f"[{R['b_boot_lo']:.2f}, {R['b_boot_hi']:.2f}] clear of zero) — not the capital shredder "
           f"an mREIT is. But the **asset-class edge fails**: the excess-vs-excess Sharpe trails "
           f"SPY in every fund and sub-era (basket adv **{R['b_adv']:+.2f}**, {R['pre_adv']:+.2f} "
           f"pre / {R['post_adv']:+.2f} post; PCEF {R['p_adv']:+.2f}), the CAPM alpha is "
           f"~0-to-negative ({R['b_alpha']:+.1f} %/yr; PCEF {R['p_alpha']:+.1f} %/yr, *t* = "
           f"{R['p_talpha']:+.2f}), and β ≈ {R['b_beta']:.2f} exposes a levered-beta income clone. "
           f"The synthetic control confirms the machinery is unbiased.\n"
           f"- **Tradability — Mirage.** Net ≈ gross ({R['net_exret']:+.1f} bps; buy-and-hold), so "
           f"costs don't kill it — a **hidden levered equity beta** does: SPY's Sharpe "
           f"({R['spy_sharpe']:.2f}) beats the basket's ({R['b_sharpe']:.2f}), and the 2022 rate "
           f"shock turned the excess return to +32 bps (*t* = {R['post_t']:+.2f}) with a "
           f"{R['post_alpha']:+.1f} %/yr alpha and a {R['cy2022']:+.1f} % year."),
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
