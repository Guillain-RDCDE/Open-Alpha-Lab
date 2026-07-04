"""Generate the two narrative notebooks for Study 620 (A-H Share Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached pair closes
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 14 A/H pairs +
# CNY=X/HKD=X, 2010-01-01 -> 2026-06-30, 198 complete months, fingerprint f54aaa8986a9).
R = dict(
    start="2010-01-01", end="2026-06-30", years=16.5, n_pairs=14, n_months=198,
    # level
    mean_pct=29.69, hac12_t=5.12, hac36_t=3.29, share_pos=84.8,
    last_neg="2014-10", asof_pct=33.1, min_pct=-11.7, max_pct=77.1,
    post_mean=41.90, post_t=9.19, post_n=139,
    n_pairs_pos=13, widest=("Great Wall Motor", 83.6), second=("China Life", 74.5),
    narrowest=("Ping An Insurance", -0.5),
    # MC null placebo: (label, p_mean, p_share)
    mc=[("calibrated AR(1) (phi 0.977, innov sd 0.0501)", 0.0177, 0.0890),
        ("stress phi = 0.995", 0.2372, 0.2712),
        ("random walk from the 2010 start", 0.4405, 0.5118)],
    # non-convergence
    phi=0.977, se_phi=0.015, half_life=30.4, df_t=-1.49, df_crit=-2.86,
    hl_median=11.3, hl_min=5.5, hl_max=45.1,
    # co-movement
    pair_corr=0.322, pc1=39.2, indep=7.1, cm_months=176,
    # fade (paper): (label, bps/mo, t)
    fade=[("gross, headline", 61.9, 5.09), ("skip-a-month", 45.5, 4.84),
          ("2010-01 - 2018-06", 84.9, 5.17), ("2018-07 - 2026-06", 36.4, 2.36),
          ("ex 2 extremes (GWM, China Life)", 60.5, 6.32),
          ("price-only (no dividends)", 64.1, 5.38),
          ("random halves (20-seed baseline)", -2.6, -0.26),
          ("z-portfolio", 63.5, 5.04), ("own-history time-series fade", 43.0, 1.39),
          ("unconditional carry (long H / short A)", 6.8, 0.26)],
    fade_ann=7.68, fade_churn=11.4,
    # executable half: (label, bps/mo, t)
    feas=[("gross", 54.7, 2.09), ("net 25 bps + 1.5%/yr borrow", 30.9, 1.19),
          ("net 50 bps + 3.0%/yr borrow", 7.1, 0.27),
          ("gross 2010-01 - 2018-06", 82.7, 1.91), ("gross 2018-07 - 2026-06", 24.9, 0.95)],
    feas_net_ann=3.77,
    # third axis
    h_ann=9.36, a_ann=8.13, h_wealth=4.35, a_wealth=3.61, race_bps=11.6, race_t=0.42,
    race_years=16.4,
    # synthetic control
    syn_level=[(30.0, 29.79, 0, 40), (0.0, -0.21, 0, 26)],   # plant, recovered, n>=real, n fake t
    syn_fade=[(0.97, 5.01), (1.00, 0.44)],                   # phi, seed-avg t
    fingerprint="f54aaa8986a9",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Cheap H the better buy?: Busted](https://img.shields.io/badge/Cheap_H_the_better_buy%3F-Busted-8b949e?style=flat-square)\n\n"
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

from a_h_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    CLOSE, ADJ = data.load_real()
    PREM_M = st.monthly_premium(st.premium_panel(CLOSE))
    REL_M = st.monthly_rel_returns(ADJ)
    EW = PREM_M.mean(axis=1).dropna()
else:
    CLOSE = ADJ = PREM_M = REL_M = EW = None
print("real A-H cache present:", HAVE_REAL,
      "| months:", (0 if EW is None else len(EW)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The same company, two prices — and nobody can fix it 🇨🇳\n"
            "### The A-H share premium, in plain English\n\n"
            + BADGES +
            "Ping An, PetroChina, ICBC — many of China's biggest companies are listed **twice**: "
            "once in **Shanghai** (an *A-share*, priced in yuan) and once in **Hong Kong** (an "
            "*H-share*, priced in HK dollars). Same company, same profits, same dividend, share "
            "for share. The folklore says the Shanghai ticket costs about **30% more** — and that "
            "nobody can arbitrage it away, because you can't convert one share into the other and "
            "you can't short-sell the expensive one.\n\n"
            "Most market legends shrink when you measure them. This one **doesn't**: our 14-pair "
            "average over 16.5 years is **+29.7%** — the folklore number *is* the tape's number.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the unit-root tests and the "
            "Monte-Carlo placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Our panel is 14 pairs *still dual-listed today* — "
            "mega-cap state firms, the corner where the premium is best documented. That carries "
            "**survivorship**, and we say so. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does Shanghai really pay ~30% more? | **Yes.** 14-pair average **+29.7%** over "
            "2010–2026; positive **139 months in a row** since late 2014; as of June 2026 it sits "
            "at **+33%**. |\n"
            "| Does the gap ever close? | **No.** Statistically you cannot tell the premium from "
            "a random walk — there is no force pulling the two prices together. |\n"
            "| Can you trade it? | **Not really.** The one trade that works on paper needs to "
            "short-sell Shanghai shares — which is exactly what nobody is allowed to do. |\n"
            "| Is the cheap Hong Kong line at least the better buy? | **Couldn't prove it.** "
            "16 years of a 30% discount earned about +1.2 pp/yr — statistically indistinguishable "
            "from zero. |"
        ),

        md(
            "## 1 · One company, two tickets\n\n"
            "An A-share and an H-share of Ping An are the **same claim on the same company** — "
            "identical par value, identical dividend per share, identical vote. The only "
            "difference is the exchange, the currency, and **who is allowed to hold it**. "
            "Mainland savers mostly can't take money abroad; foreigners were long locked out of "
            "Shanghai. Two pools of money, two moods, one company — two prices.\n\n"
            "We convert every Hong Kong price into yuan at the day's exchange rate and ask: "
            "**how much more does Shanghai pay for the identical share?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(EW.index, EW.values * 100, color=GREEN, lw=2)\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.axhline(R['mean_pct'], color=AMBER, lw=1.2, ls='--')\n"
            "    ax.annotate(f\"16.5-yr mean {R['mean_pct']:+.1f}%\", xy=(EW.index[8], R['mean_pct'] + 2),\n"
            "                color=AMBER, fontsize=10)\n"
            "    connect = pd.Timestamp('2014-11-17')\n"
            "    ax.axvline(connect, color=GREY, lw=1, ls=':')\n"
            "    ax.annotate('Stock Connect opens', xy=(connect, EW.values.max() * 100 - 4),\n"
            "                color=GREY, fontsize=9, rotation=90, va='top')\n"
            "    ax.set_title('How much more Shanghai pays for the SAME share (14-pair average)')\n"
            "    ax.set_ylabel('A-share premium over H-share, %')\n"
            "    plt.show()\n"
            "    print(f\"mean {R['mean_pct']:+.2f}%  |  positive in {R['share_pos']:.1f}% of months\"\n"
            "          f\"  |  last negative month {R['last_neg']}\")\n"
            "else:\n"
            "    print('cache missing — headline:', R['mean_pct'], '% mean premium')"
        ),
        md(
            "The one moment the premium *almost* died — 2014 — is the moment **Stock Connect** "
            "opened, the pipe that lets mainlanders buy Hong Kong stocks and foreigners buy "
            "Shanghai. The textbook said the gap should close. Instead, mainland money poured "
            "into *both* sides, and since December 2014 the premium has been positive **every "
            "single month** (139 in a row), averaging **+41.9%**. The pipe lets you *buy* both "
            "tickets — it still doesn't let anyone *swap* them."
        ),

        md(
            "## 2 · It's not one weird stock — it's all of them\n\n"
            "Thirteen of our fourteen pairs cost more in Shanghai on average. And their premiums "
            "breathe **together** — one market-wide gap, not fourteen coincidences."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pp = PREM_M.mean().sort_values() * 100\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 5.5))\n"
            "    colors = [GREEN if v > 0 else GREY for v in pp.values]\n"
            "    ax.barh(pp.index, pp.values, color=colors, height=0.62)\n"
            "    ax.axvline(0, color=GREY, lw=1)\n"
            "    ax.set_title('Average A-over-H premium by company, 2010-2026')\n"
            "    ax.set_xlabel('mean premium, %')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"{R['n_pairs_pos']}/{R['n_pairs']} pairs positive; widest \"\n"
            "          f\"{R['widest'][0]} {R['widest'][1]:+.1f}%, narrowest {R['narrowest'][0]} \"\n"
            "          f\"{R['narrowest'][1]:+.1f}%\")"
        ),

        md(
            "## 3 · Why can't anyone fix it?\n\n"
            "Every arbitrage needs a **channel**. Here, every door is locked:\n\n"
            "1. **You can't convert.** An H-share never becomes an A-share. There is no "
            "redemption machine like an ETF's.\n"
            "2. **You can't short the expensive side.** Selling borrowed A-shares is the trade — "
            "but Stock Connect has no stock-lending, and China suspended the domestic relending "
            "channel in 2024. The people who can see the spread cannot short Shanghai.\n"
            "3. **Money can't move freely.** Capital controls keep mainland savings mostly at "
            "home, chasing a limited menu of domestic assets.\n\n"
            "So the gap isn't an oversight — it's the *price of segmentation*, quoted daily. "
            "The statisticians' version: the premium is indistinguishable from a **random walk** "
            "(no pull toward zero at all).\n\n"
            "> 🔬 **For the quants.** The level has AR(1) φ = 0.977 and a Dickey-Fuller *t* of "
            "−1.49 vs −2.86 — a unit root you cannot reject. The full inference (and why we "
            "refuse to certify a level with an HAC *t* alone) is in notebook 02."
        ),

        md(
            "## 4 · \"So buy the cheap one\" — we checked\n\n"
            "If the same dividends cost 30% less in Hong Kong, the H-share pays a ~30% higher "
            "dividend yield forever. Surely the cheap line wins in the long run? Here is the "
            "race: one yuan in the H basket vs one yuan in the A basket, dividends reinvested, "
            "both counted in yuan."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fx = data.hkd_cny(ADJ)\n"
            "    h_rets, a_rets = {}, {}\n"
            "    for name, a, h in data.PAIRS:\n"
            "        h_rets[name] = (ADJ[h] * fx).resample('ME').last().pct_change()\n"
            "        a_rets[name] = ADJ[a].resample('ME').last().pct_change()\n"
            "    h_m = pd.DataFrame(h_rets).mean(axis=1).dropna()\n"
            "    a_m = pd.DataFrame(a_rets).mean(axis=1).dropna()\n"
            "    common = h_m.index.intersection(a_m.index)\n"
            "    wh, wa = (1 + h_m[common]).cumprod(), (1 + a_m[common]).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(wh.index, wh.values, color=GREEN, lw=2, label='H basket (the cheap line)')\n"
            "    ax.plot(wa.index, wa.values, color=GREY, lw=2, label='A basket (the dear line)')\n"
            "    ax.annotate(f\"x{R['h_wealth']:.2f}\", xy=(wh.index[-1], wh.iloc[-1]), color=GREEN,\n"
            "                fontsize=11, ha='left')\n"
            "    ax.annotate(f\"x{R['a_wealth']:.2f}\", xy=(wa.index[-1], wa.iloc[-1]), color=GREY,\n"
            "                fontsize=11, ha='left')\n"
            "    ax.legend(frameon=False)\n"
            "    ax.set_title('One yuan in each basket, dividends reinvested, 2010-2026')\n"
            "    ax.set_ylabel('wealth (CNY)')\n"
            "    plt.show()\n"
            "    print(f\"H {R['h_ann']:+.2f}%/yr vs A {R['a_ann']:+.2f}%/yr — difference \"\n"
            "          f\"{R['race_bps']:+.1f} bps/mo at t = {R['race_t']:+.2f} (statistically nothing)\")"
        ),
        md(
            "The cheap line finished ahead — ×4.35 vs ×3.61 — but over sixteen years that's about "
            "**+1.2 percentage points a year with a *t*-stat of 0.4**: a coin flip. Why doesn't "
            "the discount pay? Because it **never closes** — you collect the higher dividend, but "
            "the price gap itself *widened* over the sample, handing the A side capital gains "
            "that ate most of your edge.\n\n"
            "> 💡 The discount is only free money if it converges. A discount that random-walks "
            "is just a different price."
        ),

        md(
            "## 5 · The cruel joke of the tradable version\n\n"
            "Here's the twist the quants found: *within* the family, the premium **does** "
            "mean-revert. Pairs whose premium is wide versus the pack tend to see it narrow; "
            "narrow pairs tend to widen. A paper strategy that fades this cross-section earns "
            "**~62 bps a month** with a *t*-stat of 5 — a genuinely strong signal.\n\n"
            "And you cannot have it. The profitable half of that trade is *short the Shanghai "
            "line* of wide-premium pairs — the exact transaction the system forbids. The half "
            "you *can* do (buy A / short H in narrow pairs) survives at *t* = 2.1 gross… and "
            "dies at *t* = 1.2 after real-world costs, fading to nothing since 2018.\n\n"
            "**The alpha lives exactly where you cannot go.** That is not a coincidence — it's "
            "*why* the alpha is still there."
        ),

        md(
            "## The moral\n\n"
            "1. **The folklore is true.** Shanghai really pays ~30% more for identical cash "
            "flows, and has for a decade and a half.\n"
            "2. **No convergence, no arbitrage.** The premium random-walks; every closing "
            "channel is locked by design.\n"
            "3. **A locked door is not a trade.** The paper fade (*t* ≈ 5) needs the one "
            "transaction nobody may do; the executable half nets to noise.\n"
            "4. **Cheap that never converges isn't cheap.** Sixteen years of a 30% discount "
            "certified... nothing.\n\n"
            "*Numbers frozen from [docs/results.md](../docs/results.md) (as-of 2026-06-30, "
            "fingerprint `" + R["fingerprint"] + "`). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "01_for_the_curious.ipynb"))


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# A-H Share Premium — the quant teardown 🇨🇳\n\n"
            + BADGES +
            "**Claim.** The same company costs ~30% more in Shanghai (A, CNY) than in Hong Kong "
            "(H, HKD), and no one can arbitrage it away.\n\n"
            "**Design.** 14 dual-listed pairs (1:1-par ordinary shares, identical per-share "
            "dividends), 2010-01 → 2026-06. Premium = raw `A / (H × HKD→CNY) − 1` (the Hang Seng "
            "AHP convention), equal-weight, monthly. Return races use dividend-adjusted closes, "
            "H leg converted to CNY (total-return, one currency). ONE documented lag everywhere: "
            "month-end signal, next-month holding. Survivorship (still-dual-listed mega-cap "
            "panel) named on the Signal axis.\n\n"
            "> 💡 **In plain words.** We measure the gap, test whether it ever closes, test "
            "whether it predicts which line does better, and then walk the arbitrage's locked "
            "doors one by one."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · The level — and why an HAC *t* is not enough here\n\n"
            "The equal-weight premium averaged **+29.69%** (HAC(12) *t* = 5.12, HAC(36) "
            "*t* = 3.29). But the level has AR(1) φ = 0.977 — near-unit-root — and our own "
            "synthetic control shows **26/40 zero-mean worlds fake |HAC *t*| ≥ 2** on such a "
            "level. So the level claim gets a **Monte-Carlo placebo**: zero-mean paths with the "
            "tape's own persistence and innovation vol.\n\n"
            "> 💡 **In plain words.** A wandering gap that mean-nothing could *look* "
            "significant by the usual test. We simulate thousands of such wanderers and ask how "
            "often they fake 16 years averaging +30%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lv = st.level_stats(PREM_M, lags=12)\n"
            "    _m36, _s36, t36 = st.nw_tstat(lv['ew'], lags=36)\n"
            "    print(f\"EW mean {lv['mean_pct']:+.2f}%  HAC(12) t {lv['hac_t']:+.2f}  \"\n"
            "          f\"HAC(36) t {t36:+.2f}  share>0 {lv['share_pos']*100:.1f}%  \"\n"
            "          f\"({lv['n_pairs_pos']}/{lv['n_pairs']} pairs positive)\")\n"
            "    post = lv['ew'].loc['2014-12':]\n"
            "    pmu, _s, pt = st.nw_tstat(post, lags=12)\n"
            "    print(f\"post-Connect: mean {pmu*100:+.2f}%  HAC t {pt:+.2f}  \"\n"
            "          f\"{int((post > 0).mean()*100)}% of {len(post)} months positive\")\n"
            "    fig, ax = plt.subplots()\n"
            "    for c in PREM_M.columns:\n"
            "        ax.plot(PREM_M.index, PREM_M[c] * 100, color=GREY, lw=0.6, alpha=0.35)\n"
            "    ax.plot(lv['ew'].index, lv['ew'] * 100, color=GREEN, lw=2.2, label='equal-weight')\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.legend(frameon=False)\n"
            "    ax.set_title('Per-pair premiums (grey) and the equal-weight level (green)')\n"
            "    ax.set_ylabel('A-over-H premium, %')\n"
            "    plt.show()"
        ),
        code(
            "if HAVE_REAL:\n"
            "    # Monte-Carlo placebo: zero-mean nulls with the tape's own dynamics (4,000 paths)\n"
            "    rows = []\n"
            "    for lbl, kw in [('calibrated AR(1)', {}), ('stress phi = 0.995', {'phi_override': 0.995}),\n"
            "                    ('random walk from 2010 start', {'rw': True})]:\n"
            "        mc = st.null_mc_level(EW, n_sims=4000, **kw)\n"
            "        rows.append((lbl, mc['p_mean'], mc['p_share']))\n"
            "        print(f\"{lbl:30s}: p(mean >= {mc['obs_mean_pct']:+.1f}%) = {mc['p_mean']:.4f}   \"\n"
            "              f\"p(share_pos >= {mc['obs_share']*100:.0f}%) = {mc['p_share']:.4f}\")\n"
            "    print()\n"
            "    print('calibrated null rejects at 5%; the pure-RW stress null does NOT —')\n"
            "    print('non-convergence is precisely what makes the level hard to certify,')\n"
            "    print('and precisely the claim. Structure is certified by the RETURN-based')\n"
            "    print('cross-section below (HAC t = +5.09 on returns).')"
        ),

        md(
            "## 2 · Non-convergence — the premium vs a random walk\n\n"
            "> 💡 **In plain words.** If arbitrage worked even slowly, the gap would drift back "
            "toward zero and a unit-root test would see it. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ar = st.ar1_stats(EW)\n"
            "    print(f\"AR(1) phi = {ar['phi']:.3f} (se {ar['se_phi']:.3f})  ->  \"\n"
            "          f\"half-life {ar['half_life_m']:.1f} months\")\n"
            "    print(f\"Dickey-Fuller t = {ar['df_t']:+.2f} vs 5% critical {ar['df_crit']:.2f}\"\n"
            "          f\"  ->  cannot reject a unit root\")\n"
            "    hl = st.per_pair_half_lives(PREM_M)\n"
            "    print(f\"per-pair half-lives: median {hl.median():.1f} m, \"\n"
            "          f\"range {hl.min():.1f} - {hl.max():.1f}\")\n"
            "    cm = st.comovement(PREM_M)\n"
            "    print(f\"co-movement: mean pairwise corr of changes {cm['mean_pair_corr']:.3f}; \"\n"
            "          f\"PC1 {cm['pc1_share']*100:.1f}% of variance \"\n"
            "          f\"(vs {100/cm['n_pairs']:.1f}% if independent)\")"
        ),

        md(
            "## 3 · The fade — the cross-section mean-reverts even though the level never does\n\n"
            "Long H / short A in the widest-premium half, the reverse in the narrow half; "
            "month-end signal, next-month holding, 0.5× per spread side.\n\n"
            "> 💡 **In plain words.** No pair's premium reverts to *its own* past — but pairs "
            "revert to the *pack*. The pack itself just sits 30–40% above parity."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f0 = st.fade_cross_section(PREM_M, REL_M)\n"
            "    fs = st.fade_cross_section(PREM_M.shift(1), REL_M)\n"
            "    rel_po = st.monthly_rel_returns_price_only(CLOSE)\n"
            "    fpo = st.fade_cross_section(PREM_M, rel_po)\n"
            "    rb = st.fade_random_baseline(PREM_M, REL_M, n_seeds=20)\n"
            "    ts = st.fade_time_series(PREM_M, REL_M)\n"
            "    cy = st.carry_stats(REL_M)\n"
            "    print(f\"gross         : {f0['mean_bps']:+.1f} bps/mo  HAC t {f0['hac_t']:+.2f}  \"\n"
            "          f\"(ann {f0['ann_pct']:+.2f}%, churn {f0['avg_churn']*100:.1f}%/mo)\")\n"
            "    print(f\"skip-a-month  : {fs['mean_bps']:+.1f} bps/mo  HAC t {fs['hac_t']:+.2f}\")\n"
            "    for lo, hi in [('2010-01', '2018-06'), ('2018-07', '2026-06')]:\n"
            "        f2 = st.fade_cross_section(PREM_M.loc[lo:hi], REL_M.loc[lo:hi])\n"
            "        print(f\"{lo} - {hi}: {f2['mean_bps']:+.1f} bps/mo  HAC t {f2['hac_t']:+.2f}\")\n"
            "    print(f\"price-only    : {fpo['mean_bps']:+.1f} bps/mo  HAC t {fpo['hac_t']:+.2f}  \"\n"
            "          f\"(not a dividend-carry artefact)\")\n"
            "    print(f\"random halves : {rb['mean_bps']:+.1f} bps/mo  mean t {rb['mean_t']:+.2f}  \"\n"
            "          f\"({rb['n_seeds']} seeds)\")\n"
            "    print(f\"own-history TS: {ts['mean_bps']:+.1f} bps/mo  HAC t {ts['hac_t']:+.2f}\")\n"
            "    print(f\"uncond. carry : {cy['mean_bps']:+.1f} bps/mo  HAC t {cy['hac_t']:+.2f}\")\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(f0['ret'].index, (1 + f0['ret']).cumprod(), color=GREEN, lw=2)\n"
            "    ax.set_title('Cross-sectional fade, gross paper wealth (log scale)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('wealth')\n"
            "    plt.show()"
        ),

        md(
            "## 4 · Tradability — walking the locked doors\n\n"
            "The wide half of the fade is long H / **short A** — and there is **no A-share "
            "borrow**: the lines are non-fungible, Stock Connect carries no securities lending, "
            "the domestic relending channel was suspended in 2024. The only executable half is "
            "narrow-pair **long A (northbound Connect) / short H (HK borrow)**:\n\n"
            "> 💡 **In plain words.** The half of the trade you're allowed to do is the weak "
            "half — and after costs it's noise. The strong half is illegal-by-plumbing."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for lbl, kw in [('gross', {}),\n"
            "                    ('net 25bp + 1.5%/yr borrow', {'cost_bps_oneway': 25.0, 'borrow_bps_yr': 150.0}),\n"
            "                    ('net 50bp + 3.0%/yr borrow', {'cost_bps_oneway': 50.0, 'borrow_bps_yr': 300.0})]:\n"
            "        fl = st.feasible_leg(PREM_M, REL_M, **kw)\n"
            "        print(f\"{lbl:26s}: {fl['mean_bps']:+.1f} bps/mo  HAC t {fl['hac_t']:+.2f}\")\n"
            "    for lo, hi in [('2010-01', '2018-06'), ('2018-07', '2026-06')]:\n"
            "        fl = st.feasible_leg(PREM_M.loc[lo:hi], REL_M.loc[lo:hi])\n"
            "        print(f\"gross {lo} - {hi}   : {fl['mean_bps']:+.1f} bps/mo  HAC t {fl['hac_t']:+.2f}\")\n"
            "    print()\n"
            "    print('borderline gross, sub-2 net, fading since 2018 — and it bets the premium')\n"
            "    print('WIDENS. Nothing here is an arbitrage of the premium. MIRAGE.')"
        ),

        md(
            "## 5 · Third axis — is the cheap H line at least the better buy?\n\n"
            "Equal-weight total-return baskets, both in CNY (gross of dividend withholding, "
            "≈10% either channel for an outside investor — near-symmetric)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    race = st.leg_race(ADJ)\n"
            "    print(f\"H basket {race['h_ann_pct']:+.2f}%/yr (x{race['h_wealth']:.2f})   \"\n"
            "          f\"A basket {race['a_ann_pct']:+.2f}%/yr (x{race['a_wealth']:.2f})\")\n"
            "    print(f\"H - A: {race['diff_bps_mo']:+.1f} bps/mo  HAC t {race['hac_t']:+.2f}  \"\n"
            "          f\"over {race['years']:.1f} years -> statistically NOTHING. BUSTED.\")"
        ),

        md(
            "## 6 · Synthetic control — machinery proof (never market evidence)\n\n"
            "Deterministic A/H worlds with a **planted** premium level and a knob on the "
            "premium's persistence. The level detector must recover the plant and stay quiet at "
            "zero; the fade detector must fire only when the premium truly mean-reverts.\n\n"
            "> 💡 **In plain words.** We test the measuring stick on worlds where we buried the "
            "answer ourselves — including the trap world where a zero gap merely *wanders*."
        ),
        code(
            "# always offline — 40-seed level suite + 10-seed fade suite (light, deterministic)\n"
            "for plant in (0.30, 0.0):\n"
            "    c = st.control_level_suite(plant, n_seeds=40)\n"
            "    print(f\"level  plant {plant*100:5.1f}%: seed-avg {c['avg_mean_pct']:+6.2f}% \"\n"
            "          f\"(sd {c['sd_pct']:.2f})  seeds |mean|>=29.7%: {c['n_ge_real']}/40  \"\n"
            "          f\"seeds |HAC t|>=2: {c['n_hac_t_ge2']}/40\")\n"
            "for phi in (0.97, 1.0):\n"
            "    c = st.control_fade_suite(phi, n_seeds=10)\n"
            "    print(f\"fade   phi {phi:.2f}    : seed-avg {c['avg_bps']:+5.1f} bps/mo  \"\n"
            "          f\"seed-avg t {c['avg_t']:+.2f}  (min {c['min_t']:+.2f}, max {c['max_t']:+.2f})\")\n"
            "print()\n"
            "print('zero-mean worlds NEVER reach the real +29.7% (0/40) — but 26/40 fake an')\n"
            "print('|HAC t| >= 2 on the level: that is why the real level carries an MC placebo.')\n"
            "print('The fade detector fires only under true mean reversion (t +5.01 vs +0.44).')"
        ),

        md(
            "## Verdict\n\n"
            "| Axis | Stamp | Decisive numbers |\n|---|---|---|\n"
            "| Signal | **REAL** | mean **+29.69%** (HAC(12) *t* 5.12, calibrated MC "
            "p 0.018), 139 straight positive months post-Connect, 13/14 pairs, PC1 39.2%, "
            "fade **HAC *t* +5.09 on returns** (skip-month 4.84, both halves ≥ 2.36, "
            "price-only 5.38, 20-seed baseline −0.26). RW-stress caveat named. |\n"
            "| Tradability | **MIRAGE** | no A borrow → the profitable half is unexecutable; "
            "executable half nets *t* 1.19 (and 0.95 gross since 2018) |\n"
            "| Cheap H the better buy? | **BUSTED** | ×4.35 vs ×3.61 = +11.6 bps/mo at "
            "*t* 0.42 over 16.4 years |\n\n"
            "*Frozen numbers mirror [docs/results.md](../docs/results.md) (as-of 2026-06-30, "
            "fingerprint `" + R["fingerprint"] + "`). Survivorship named. Not investment "
            "advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "02_for_the_quants.ipynb"))


if __name__ == "__main__":
    build_curious()
    build_quants()
    print("wrote 01_for_the_curious.ipynb, 02_for_the_quants.ipynb")
