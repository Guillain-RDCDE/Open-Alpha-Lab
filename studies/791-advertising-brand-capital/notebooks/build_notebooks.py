"""Generate the two narrative notebooks for Study 791 (Advertising-Brand-Capital).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR +
yfinance panels under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md. EDGAR advertising + revenue
# (46-name consumer basket that files the advertising line) x yfinance monthly total returns,
# 2010-02 -> 2026-06. Long heavy-advertisers / short light-advertisers, monthly, 1/3 tertiles.
R = dict(
    start="2010-02-28", end="2026-06-30", n_months=196,
    n_basket=46, n_adv_names=46, adv_fill_pct=88, adv_fy_min=2008, adv_fy_max=2026,
    n_marketing=9,
    # book summaries (annualised %)
    long_cagr=12.81, long_sh=0.92, long_dd=-28.2, long_mean=13.12,
    short_cagr=16.17, short_sh=1.11, short_dd=-22.3, short_mean=16.13,
    ls_cagr=-3.32, ls_sh=-0.35, ls_dd=-45.9,
    spy_cagr=14.56, spy_sh=1.02, spy_dd=-23.9, spy_mean=14.71,
    # HAC tests (%/yr, t)
    ls_mean=-3.00, ls_t=-1.46,
    ls_net_mean=-4.00, ls_net_t=-1.94,
    long_spy_mean=-1.47, long_spy_t=-0.62,
    short_spy_mean=1.54, short_spy_t=0.65,
    welch_legs=-0.60,
    hit=91, n_ls=196, hit_pct=46.4, wilson_lo=39.6, wilson_hi=53.4,
    turnover_pct=1.04,
    placebo_pctile=13.3, placebo_p=0.20, placebo_null_mean=-0.18, placebo_null_sd=2.34,
    # robustness: split -> (mean %/yr, t)
    rob={"half": (-0.68, -0.41), "tertile": (-3.00, -1.46), "quartile": (-3.47, -1.33),
         "quintile": (-2.24, -0.81)},
    # synthetic control
    syn_edge0=(0.47, 0.18), syn_edge6=(6.47, 2.47), syn_edge8=(8.47, 3.23),
    syn_null20_mean=-0.22, syn_null20_sd=0.43, syn_null20_fire=0,
    top_intens=[("ETSY", 27.4), ("EXPE", 26.5), ("CL", 13.3), ("CHD", 11.4),
                ("DPZ", 11.3), ("KO", 11.3)],
    bot_intens=[("TGT", 1.41), ("LOW", 1.1), ("TJX", 1.09), ("HD", 0.75),
                ("WMT", 0.75), ("ROST", 0.33)],
    long_final=["CHD", "CL", "CLX", "CMCSA", "DECK", "DIS", "DPZ", "EBAY", "ETSY", "EXPE",
                "KO", "NKE", "PG", "STZ", "TAP"],
    short_final=["BBY", "CAG", "CMG", "F", "HD", "LOW", "ROST", "SBUX", "SJM", "T",
                 "TGT", "TJX", "VZ", "WMT", "YUM"],
    cost_bps=10, borrow_bps=100,
    fp="979148cad57d",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from advertising_brand import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    REAL = data.load_real(allow_survivorship_bias=True)   # survivorship named on the Signal axis
    SIG = data.build_signal(REAL)
else:
    REAL = SIG = None
print("real cache present:", HAVE_REAL,
      "| basket names:", (0 if REAL is None else REAL["returns"].shape[1]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do big advertisers make better stocks? 📺\n"
            "### \"Brand capital\" — the intangible the market supposedly under-prices — and why "
            "the tape says the opposite\n\n"
            + BADGES +
            "Here's a genuinely appealing idea. When Coca-Cola or Procter & Gamble spends billions "
            "on advertising, that money doesn't vanish — it buys something durable: brand "
            "awareness, loyalty, pricing power. Accountants can't put \"the Coca-Cola brand\" on "
            "the balance sheet, so (the theory goes) the market *under-values* it — and firms that "
            "advertise heavily should therefore earn a quiet **return premium** as that hidden "
            "asset gets recognised.\n\n"
            "It's a real academic thesis (Belo-Lin-Vitorino; Chan-Lakonishok-Sougiannis). So we "
            "sorted a basket of brand-name companies by how much of their sales they plough into "
            "advertising, and bought the heavy spenders. The result is a clean, honest **no**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SEC-filed advertising ÷ sales, "
            f"{R['n_basket']} consumer names that actually disclose the line, monthly total "
            "returns 2010→2026. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do heavy advertisers out-earn light ones? | **No — the reverse, if anything.** "
            f"Buying the heaviest advertisers and shorting the lightest *lost* about "
            f"**{abs(R['ls_mean']):.1f}%/yr** over 2010-2026. |\n"
            f"| Is that loss statistically real? | **No.** It's inside the noise (*t* = "
            f"{R['ls_t']:.2f}, and you need \\|t\\| ≥ 2 to say anything). So the honest reading "
            "is: **no brand premium here at all**, in either direction. |\n"
            "| So who won — the heavy or the light advertisers? | The **light** ones — big-box and "
            "off-price retailers (Walmart, Ross, TJX) that advertise almost nothing beat the "
            "brand-heavy staples (Coke, P&G, Clorox). But that's the **sector story of the "
            "decade**, not an advertising signal. |\n"
            "| Could you trade it? | **No.** The spread is the wrong sign for the claim, "
            "insignificant, and a short book pays borrow on top. |\n\n"
            "> The brand-capital *idea* is lovely. On a clean tape of firms that actually report "
            "their ad spend, it simply isn't paid — and over this window it pointed the *other* way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Advertising builds brand capital — a durable intangible asset. GAAP expenses it "
            "immediately instead of capitalising it, so the market systematically under-values "
            "heavy advertisers. Buy them and collect the premium as the mispricing corrects.\"*\n\n"
            "This is the advertising-specific version of a serious research literature on "
            "intangibles (we test its cousins elsewhere: [R&D](../../525-r-and-d-intensity/), "
            "[intangible-adjusted value](../../526-intangible-value/), "
            "[patents](../../400-patent-intensity/)). The signal is simple and mechanical: "
            "**advertising expense ÷ sales**, straight off the income statement."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be a beautifully simple factor: read one line off a 10-K, rank, "
            "buy the top. No forecasting, no models. And it would say something deep about markets — "
            "that they can't see through accounting conventions to value the brands we all "
            "obviously recognise. That's exactly the kind of clean, intuitive claim worth testing "
            "hard, precisely because it's so easy to *want* to be true."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** {R['n_basket']} long-listed US consumer companies that *actually "
            "file* an advertising line — from ad-drenched marketplaces (Etsy, Expedia ~27% of "
            "sales) and classic brand builders (Coke, P&G, Clorox ~11%) down to retailers that "
            "barely advertise (Ross 0.3%, Walmart 0.8%).\n"
            "- **The sort.** Each month, rank by advertising/sales; go long the heavy-third, short "
            "the light-third; hold a month; repeat.\n"
            "- **The honest scope.** *Most* public firms don't report advertising at all — so this "
            "is a slice of disclosers, not the market. We say so loudly.\n"
            "- **What would make us say \"mirage.\"** If the long-short doesn't clear *t* = 2 in "
            "the **positive** direction, there is no brand premium to trade."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: who advertises, and who doesn't?** The spread across the basket is huge — "
            "two orders of magnitude."
        ),
        code(
            "if HAVE_REAL:\n"
            "    last = SIG.dropna(how='all').index.max()\n"
            "    row = SIG.loc[last].dropna().sort_values(ascending=False)\n"
            "    top = row.head(8) * 100; bot = row.tail(8) * 100\n"
            "else:\n"
            "    top = pd.Series(dict(R['top_intens'])); bot = pd.Series(dict(R['bot_intens']))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "vals = pd.concat([top, bot]);\n"
            "cols = [GREEN]*len(top) + [RED]*len(bot)\n"
            "ax.barh(range(len(vals)), vals.values, color=cols)\n"
            "ax.set_yticks(range(len(vals))); ax.set_yticklabels(vals.index)\n"
            "ax.invert_yaxis(); ax.set_xlabel('advertising as % of sales (latest filed)')\n"
            "ax.set_title('Heavy advertisers (green) vs light advertisers (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('heaviest:', [f'{k} {v:.0f}%' for k,v in top.head(4).items()])\n"
            "print('lightest:', [f'{k} {v:.1f}%' for k,v in bot.tail(4).items()])"
        ),
        md(
            "The ranking is economically sensible: digital marketplaces and classic consumer "
            "brands at the top, big-box and off-price retail at the bottom. So the sort is "
            "measuring something real. **Now — did the heavy advertisers earn more?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.race(REAL, SIG, frac=1/3, cost_bps=R['cost_bps'], borrow_bps=R['borrow_bps'], n_shuffles=200)\n"
            "    ln, sh_ = st.summarize(r['long'])['mean_ann']*100, st.summarize(r['short'])['mean_ann']*100\n"
            "    spy = st.summarize(r['spy'])['mean_ann']*100\n"
            "else:\n"
            "    ln, sh_, spy = R['long_mean'], R['short_mean'], R['spy_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['HEAVY\\nadvertisers','LIGHT\\nadvertisers','S&P 500'], [ln, sh_, spy],\n"
            "       color=[GREEN, RED, GREY], width=.6)\n"
            "for i,v in enumerate([ln, sh_, spy]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('mean annual return'); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_title('The light advertisers won — the opposite of the claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'heavy {ln:+.1f}%/yr | light {sh_:+.1f}%/yr | SPY {spy:+.1f}%/yr')"
        ),
        md(
            f"There it is. The **light** advertisers returned about **+{R['short_mean']:.0f}%/yr**, "
            f"the **heavy** advertisers about **+{R['long_mean']:.0f}%/yr** — so the brand-capital "
            f"long-short (heavy minus light) *lost* roughly **{abs(R['ls_mean']):.1f}%/yr**, the "
            "wrong sign for the claim. But before reading anything into that, the crucial "
            "question: **is the gap even real, or just noise?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t = r['test_ls']['tstat']; m = r['test_ls']['mean_ann']*100\n"
            "else:\n"
            "    t = R['ls_t']; m = R['ls_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 3.2))\n"
            "ax.barh([0], [t], color=(RED if abs(t)<2 else GREEN), height=.5)\n"
            "ax.axvline(2, ls='--', c=GREEN, lw=1.2); ax.axvline(-2, ls='--', c=GREEN, lw=1.2)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_yticks([]); ax.set_xlim(-3, 3)\n"
            "ax.set_xlabel('HAC t-stat of the heavy-minus-light spread')\n"
            "ax.set_title(f'Inside the noise: t = {t:+.2f} (need |t| >= 2 for a real signal)')\n"
            "ax.annotate('|t|=2 bar', (2, .35), color=GREEN, ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'spread {m:+.2f}%/yr at HAC t = {t:+.2f} -> indistinguishable from zero')"
        ),
        md(
            f"The spread's *t*-stat is **{R['ls_t']:.2f}** — nowhere near the ±2 bar. So the "
            "honest statement isn't \"heavy advertisers lose\" — it's **there is no advertising "
            "premium here, in either direction.** The negative point estimate is just this "
            "decade's sector weather (unglamorous retailers beat defensive staples) blowing "
            "through a signal that carries no independent information."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The claimed premium is a *positive* one; we find a **negative** "
            f"point estimate ({R['ls_mean']:+.1f}%/yr) that is statistically indistinguishable "
            f"from zero (*t* = {R['ls_t']:.2f}). No brand premium on this tape.\n"
            "- **Tradability — Mirage.** You can't trade a wrong-signed, insignificant spread — "
            "and a long-short pays short-borrow on top, making the net even worse.\n\n"
            "The idea isn't stupid — it's a real research thesis. It just doesn't survive contact "
            "with a clean, point-in-time tape of the firms that actually disclose their ad spend."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why might the premium be missing here when papers find it?** Two honest reasons: "
            "(1) *coverage* — since 2015 most firms stopped disclosing advertising, so any tradable "
            "basket is a thin, self-selected slice; (2) *style* — over 2010-2026 the heavy "
            "advertisers are mostly defensive staples, which simply had a poor decade versus "
            "everything else, swamping any faint intangible effect.\n"
            "- **The better denominator?** The R&D literature finds the mispricing lives when you "
            "scale by *market value*, not sales — see [study 525](../../525-r-and-d-intensity/). An "
            "advertising/market-cap version is the natural next test.\n"
            "- **Sibling studies (the dedup):** [525-r-and-d-intensity](../../525-r-and-d-intensity/), "
            "[526-intangible-value](../../526-intangible-value/), "
            "[400-patent-intensity](../../400-patent-intensity/) — same intangible-mispricing "
            "family, different intangibles. See [docs/references.md](docs/references.md).\n\n"
            "*Think advertising intensity pays once you scale it right, or clean up the survivorship? "
            "Show a net, certifiable spread on a defensible universe — then we'll talk.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Advertising brand capital — a quantitative teardown 🔬\n"
            "### The advertising-intensity long-short · HAC inference · a label-shuffle placebo · "
            "a fraction-robustness sweep · costs + short-borrow · a synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **heavy advertisers earn a forward premium because the market under-prices "
            "brand capital** (Belo-Lin-Vitorino 2014; Chan-Lakonishok-Sougiannis 2001) — is an "
            "intangible-mispricing claim, distinct from the R&D "
            "([525](../../525-r-and-d-intensity/)), intangible-value "
            "([526](../../526-intangible-value/)) and patent "
            "([400](../../400-patent-intensity/)) siblings. The job: measure the advertising/sales "
            "long-short honestly, on a clean point-in-time tape, and see whether it clears the "
            "desk's `t >= 2` bar in the predicted direction.\n\n"
            "> ⚠️ **Data note.** SEC EDGAR companyconcept (`AdvertisingExpense` / "
            "`MarketingAndAdvertisingExpense`, revenue) + yfinance monthly total returns, "
            f"{R['start']} → {R['end']}. Survivorship named on the Signal axis (current-membership "
            "consumer basket). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | heavy-minus-light advertising long-short "
            f"**{R['ls_mean']:+.2f}%/yr**, HAC **t = {R['ls_t']:+.2f}** (need +2; the estimate is "
            f"*negative*), placebo p = {R['placebo_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | wrong-signed & insignificant gross; net of "
            f"{R['cost_bps']}bps turnover + {R['borrow_bps']}bps/yr borrow = "
            f"{R['ls_net_mean']:+.2f}%/yr (t = {R['ls_net_t']:+.2f}) |\n\n"
            "> 💡 In plain words: the brand-capital premium is *positive* in theory. On this tape "
            "the advertising-intensity spread is **negative and inside its own error bar** — no "
            "signal, and nothing to trade. The negative sign is the decade's staples-vs-retail "
            "style rotation, not an advertising effect (the placebo confirms the ranking loads on "
            "no reliable return axis)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $A_{i}$ be firm $i$'s advertising expense and $S_i$ its sales in the most-recent "
            "reported fiscal year; the intensity signal is $x_{i} = A_i / S_i$. Form the monthly "
            "cross-sectional sort, long the top tertile (heavy advertisers), short the bottom. The "
            "claims:\n\n"
            "- **H₁ (premium).** $E[r^{\\text{heavy}} - r^{\\text{light}}] > 0$ — advertising "
            "intensity earns a positive forward spread (brand capital is under-priced).\n"
            "- **H₂ (real signal).** The spread clears an autocorrelation-robust `t >= 2` **and** "
            "sits in the tail of a label-shuffle placebo.\n"
            "- **H₃ (tradable).** It survives one-way turnover cost + short-borrow.\n\n"
            f"We find **H₁ rejected** (spread is *negative*, {R['ls_mean']:+.2f}%/yr), **H₂ "
            f"rejected** (HAC t = {R['ls_t']:+.2f}, placebo p = {R['placebo_p']:.2f}), **H₃ moot** "
            "(nothing to trade). The literature's premium does not appear on this tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The long-short is a monthly return series, so the primary Signal-axis test is a "
            "**Newey-West HAC** *t* of its mean (H₀: mean = 0), with the desk bar at 2 — and, "
            "because the claim is *directional*, **positive** 2. A **label-shuffle placebo** "
            "permutes the signal across names (same values, wrong names) and rebuilds the "
            "long-short 400× — if the real spread isn't in the tail, the ranking carries no "
            "return information. Costs are **one-way × NAV** per rebalance; the short leg pays an "
            "annual **borrow**. One execution lag (signal at month *t* → book held *t+1*), one "
            "reporting lag (a year on the fundamentals). Survivorship is on the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_basket']} consumer names filing the advertising line; advertising "
            f"fill ≈ {R['adv_fill_pct']}% of name-years (FY {R['adv_fy_min']}-{R['adv_fy_max']}); "
            f"{R['n_marketing']} names file the broader Marketing&Advertising tag (flagged). "
            f"{R['n_months']} holding months, {R['start']} → {R['end']}.\n"
            "- **Headline.** HAC t of the heavy-minus-light long-short + long-vs-SPY and "
            "short-vs-SPY decomposition + a positive-month hit rate with a Wilson interval.\n"
            "- **Placebo.** 400 label shuffles; percentile + two-sided p of the real spread.\n"
            "- **Robustness.** Tighten the sort halves → quintiles.\n"
            "- **Costs.** One-way turnover × NAV + annual short-borrow.\n"
            "- **Control.** Synthetic panel with a planted advertising premium; the null (edge=0) "
            "must not fire across 20 seeds, a planted edge must be recovered.\n\n"
            "> **What would flip the verdict to `REAL`:** a *positive* HAC t ≥ 2 that survives the "
            "placebo. We announce it before looking."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race and the HAC test\n\n"
            "Long heavy-advertisers, short light-advertisers, equal-weight, monthly. The long and "
            "short legs versus SPY, and the HAC *t* of the spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.race(REAL, SIG, frac=1/3, cost_bps=R['cost_bps'], borrow_bps=R['borrow_bps'], n_shuffles=200)\n"
            "    rows = {k: st.summarize(r[k]) for k in ['long','short','long_short','spy']}\n"
            "    tls, tl, ts = r['test_ls'], r['test_long_vs_spy'], r['test_short_vs_spy']\n"
            "    print('leg              CAGR    Sharpe   maxDD    mean/yr')\n"
            "    for k in ['long','short','long_short','spy']:\n"
            "        s=rows[k]; print(f\"{k:14s} {s['cagr']*100:7.2f}% {s['sharpe']:7.2f} {s['max_dd']*100:7.1f}% {s['mean_ann']*100:+7.2f}%\")\n"
            "    print(f\"\\nHAC  long-short {tls['mean_ann']*100:+.2f}%/yr  t={tls['tstat']:+.2f}\")\n"
            "    print(f\"HAC  long - SPY  {tl['mean_ann']*100:+.2f}%/yr  t={tl['tstat']:+.2f}\")\n"
            "    print(f\"HAC  short- SPY  {ts['mean_ann']*100:+.2f}%/yr  t={ts['tstat']:+.2f}\")\n"
            "    ln, sh_, ls_ = rows['long']['mean_ann']*100, rows['short']['mean_ann']*100, tls['mean_ann']*100\n"
            "    tval = tls['tstat']\n"
            "else:\n"
            "    ln, sh_, ls_, tval = R['long_mean'], R['short_mean'], R['ls_mean'], R['ls_t']\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(11,4.3))\n"
            "a1.bar(['heavy','light','L-S'], [ln, sh_, ls_], color=[GREEN,RED,GREY], width=.6)\n"
            "for i,v in enumerate([ln,sh_,ls_]): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('mean %/yr'); a1.set_title('Light beats heavy')\n"
            "a2.barh([0],[tval],color=RED,height=.5); a2.axvline(2,ls='--',c=GREEN); a2.axvline(-2,ls='--',c=GREEN)\n"
            "a2.axvline(0,c='k',lw=.8); a2.set_yticks([]); a2.set_xlim(-3,3)\n"
            "a2.set_xlabel('HAC t of long-short'); a2.set_title(f'Inside the noise (t={tval:+.2f})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the long-short is **{R['ls_mean']:+.2f}%/yr at HAC "
            f"t = {R['ls_t']:+.2f}** — negative and insignificant. Decomposed: the heavy leg vs "
            f"SPY is {R['long_spy_mean']:+.2f}%/yr (t = {R['long_spy_t']:+.2f}) and the light leg "
            f"vs SPY is {R['short_spy_mean']:+.2f}%/yr (t = {R['short_spy_t']:+.2f}) — neither leg "
            "does anything distinguishable from the market. There is simply no advertising premium "
            "here."
        ),
        md(
            "### 4b · The placebo — does the *signal* carry any return axis?\n\n"
            "Permute the advertising-intensity labels across names (same distribution of "
            "intensities, attached to the wrong companies) and rebuild the long-short. If the real "
            "spread is buried in the middle of that null, the ranking loads on no reliable return "
            "axis at all."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = r['placebo_null']*100; obs = r['test_ls']['mean_ann']*100\n"
            "    pctile, pval = r['placebo_pctile'], r['placebo_p']\n"
            "else:\n"
            "    rng = np.random.default_rng(791)\n"
            "    null = rng.normal(R['placebo_null_mean'], R['placebo_null_sd'], 400)\n"
            "    obs = R['ls_mean']; pctile, pval = R['placebo_pctile'], R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9,4.3))\n"
            "ax.hist(null, bins=40, color=GREY, alpha=.85, label='shuffled-label null')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed long-short {obs:+.2f}%/yr')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('annualised long-short of a shuffled signal (%/yr)'); ax.set_ylabel('freq')\n"
            "ax.set_title(f'Real spread sits in the crowd: {pctile:.0f}th pctile, p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%/yr | null mean {np.mean(null):+.2f} sd {np.std(null):.2f} | pctile {pctile:.0f} p {pval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: a *shuffled* advertising signal pays about the same as the real "
            f"one — the observed spread sits near the **{R['placebo_pctile']:.0f}th percentile** of "
            f"the null (two-sided p = {R['placebo_p']:.2f}). Unlike study 525's R&D/ME (which at "
            "least survived its placebo as a real-but-value axis), advertising intensity here "
            "doesn't even clear the placebo: the ranking carries **no** reliable cross-sectional "
            "return axis on this basket."
        ),
        md(
            "### 4c · Robustness — it isn't hiding at a tighter sort\n\n"
            "Tighten the long/short from halves to quintiles; a real signal often sharpens at the "
            "extremes. This one just stays negative and insignificant."
        ),
        code(
            "splits = ['half','tertile','quartile','quintile']; fracs=[1/2,1/3,1/4,1/5]\n"
            "if HAVE_REAL:\n"
            "    ms, tsv = [], []\n"
            "    for fr in fracs:\n"
            "        b = st.signal_books(SIG, REAL['returns'], frac=fr); d = st.hac_tstat(b['long_short'])\n"
            "        ms.append(d['mean_ann']*100); tsv.append(d['tstat'])\n"
            "else:\n"
            "    ms = [R['rob'][s][0] for s in splits]; tsv = [R['rob'][s][1] for s in splits]\n"
            "fig, ax = plt.subplots(figsize=(9,4.2))\n"
            "ax.bar(splits, tsv, color=[RED if abs(t)<2 else GREEN for t in tsv], width=.6)\n"
            "ax.axhline(2,ls='--',c=GREEN); ax.axhline(-2,ls='--',c=GREEN); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('HAC t of long-short'); ax.set_title('Negative and sub-2 at every sort width')\n"
            "for i,(m,t) in enumerate(zip(ms,tsv)): ax.annotate(f'{m:+.1f}%\\nt={t:+.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({s: (round(m,1), round(t,2)) for s,m,t in zip(splits,ms,tsv)})"
        ),
        md(
            "> 💡 In plain words: no sort width rescues it. The spread is negative and inside the "
            "noise from halves to quintiles — there is no concentrated tail where a hidden premium "
            "lives."
        ),
        md(
            "### 4d · Costs + short-borrow — the moot cost check\n\n"
            "Advertising intensity is a *slow* annual characteristic, so turnover is tiny; the only "
            "material friction is short-borrow. It makes a negative spread more negative — but the "
            "point is academic, since there is no positive edge to erode in the first place."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = r['test_ls']; nt = r['test_ls_net']; turn = r['avg_turnover']*100\n"
            "    gm, gt, nm, ntt = g['mean_ann']*100, g['tstat'], nt['mean_ann']*100, nt['tstat']\n"
            "else:\n"
            "    gm, gt, nm, ntt, turn = R['ls_mean'], R['ls_t'], R['ls_net_mean'], R['ls_net_t'], R['turnover_pct']\n"
            "fig, ax = plt.subplots(figsize=(7.6,4.2))\n"
            "ax.bar(['gross','net (cost+borrow)'], [gm, nm], color=[GREY, RED], width=.5)\n"
            "for i,(v,t) in enumerate([(gm,gt),(nm,ntt)]): ax.annotate(f'{v:+.1f}%/yr\\nt={t:+.2f}',(i,v),ha='center',va='top')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('long-short %/yr')\n"
            "ax.set_title(f'Turnover only {turn:.1f}%/mo; borrow makes a negative spread worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gm:+.2f}%/yr (t={gt:+.2f}) -> net {nm:+.2f}%/yr (t={ntt:+.2f}); turnover {turn:.2f}%/mo')"
        ),
        md(
            f"> 💡 In plain words: one-way turnover is only ~{R['turnover_pct']:.1f}%/month (the "
            f"signal moves once a year), so cost barely matters; short-borrow at "
            f"{R['borrow_bps']}bps/yr drags the gross {R['ls_mean']:+.2f}%/yr to "
            f"{R['ls_net_mean']:+.2f}%/yr — but you would never put on a wrong-signed, "
            "insignificant book to begin with. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic monthly panel (fixed seed, 15 years × 40 names) where the first half "
            "are persistently heavy advertisers, market beta is common to both legs, and a knob "
            "`edge` plants the *true* annual heavy-minus-light premium. The null (edge=0) is "
            "checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    sg, rr, _, _ = data.synthetic_panel(edge=0.0, seed=791 + s_)\n"
            "    b = st.signal_books(sg, rr, frac=1/3); null_ts.append(st.hac_tstat(b['long_short'])['tstat'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "sg, rr, _, _ = data.synthetic_panel(edge=0.06, seed=791)\n"
            "planted_t = st.hac_tstat(st.signal_books(sg, rr, frac=1/3)['long_short'])['tstat']\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1],[planted_t], color=GREEN, s=90, zorder=5, label='planted edge = +6%/yr')\n"
            "ax.axhline(2,ls='--',c=RED); ax.axhline(-2,ls='--',c=RED)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x20','planted']); ax.set_ylabel('HAC t')\n"
            "ax.set_title('Control: null never fires; a planted premium lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts)>=2).sum()}/20 | planted t={planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null20_mean']:+.2f} and **never** fires "
            f"({R['syn_null20_fire']}/20); a planted +6%/yr premium reads "
            f"t = {R['syn_edge6'][1]:+.2f}. The machinery finds a premium exactly when one exists — "
            f"so the real-tape null result ({R['ls_t']:+.2f}) is a true negative, not a broken "
            "pipeline. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — heavy-minus-light advertising long-short "
            f"**{R['ls_mean']:+.2f}%/yr**, HAC **t = {R['ls_t']:+.2f}** (the claim needs *positive* "
            f"t ≥ 2; the estimate is negative). The placebo doesn't clear it either "
            f"({R['placebo_pctile']:.0f}th pctile, p = {R['placebo_p']:.2f}), and no sort width "
            "(halves → quintiles) rescues it. Neither leg beats SPY. No brand premium on this tape.\n"
            f"- **Tradability `MIRAGE`** — a wrong-signed, insignificant spread; net of "
            f"{R['cost_bps']}bps turnover + {R['borrow_bps']}bps/yr borrow it is "
            f"{R['ls_net_mean']:+.2f}%/yr (t = {R['ls_net_t']:+.2f}). Nothing to allocate to.\n\n"
            "The literature (Belo-Lin-Vitorino; Chan-Lakonishok-Sougiannis) reports a brand/"
            "intangible premium; **this clean, point-in-time, disclosure-limited tape cannot "
            "certify it** — and over 2010-2026 the raw sign is backwards, driven by the "
            "staples-vs-retail style rotation rather than anything about advertising."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Denominator.** Chan-Lakonishok-Sougiannis find intangible mispricing concentrates "
            "when scaling by **market value**, not sales; an advertising/market-cap signal (the "
            "brand-capital analogue of study 525's R&D/ME) is the obvious next test.\n"
            "- **Brand-capital stock.** Belo-Lin-Vitorino accumulate advertising into a "
            "perpetual-inventory *stock* rather than using the raw annual flow; that smoother "
            "measure could carry signal the flow ratio doesn't.\n"
            "- **Coverage.** Post-2015 disclosure is thin and self-selected; a serious test needs a "
            "wider historical universe (pre-2015, when the line was more commonly filed) or a "
            "vendor brand-value dataset — both beyond a public-EDGAR basket.\n"
            "- **Dedup map:** [525-r-and-d-intensity](../../525-r-and-d-intensity/) (R&D, and the "
            "ME-vs-sales denominator contrast), [526-intangible-value](../../526-intangible-value/) "
            "(intangible-adjusted book-to-market), [400-patent-intensity](../../400-patent-intensity/) "
            "(patents). Same family; this is the **advertising** axis specifically.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
