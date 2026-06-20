"""Generate the two narrative notebooks for Study 318 (Election-Volatility).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the shared
cached ^GSPC / ^VIX parquets if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), bannering which tape each cell shows.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    gspc_start="1927-12-30", gspc_end="2026-06-12", gspc_n=24729, gspc_fp="530904cd97f3",
    vix_start="1990-01-02", vix_end="2026-06-11", vix_n=9179, vix_fp="c391a7a81017",
    # 1) realized vol ratio
    ratio_all_n=25, ratio_all=1.05, ratio_all_t=0.54, ratio_all_frac=44, ratio_all_pctile=66,
    ratio_vix_n=9, ratio_vix=1.21, ratio_vix_t=1.05, ratio_vix_frac=56,
    # 2) VRP
    vrp_n=9, vrp_mean=5.21, vrp_t=1.28, vrp_win=78, vrp_ci_lo=-2.8, vrp_ci_hi=11.8,
    vrp_2008_impl=53.7, vrp_2008_real=76.8, vrp_2008=-23.1,
    vrp_ex08_mean=8.76, vrp_ex08_t=3.83,
    # 3) carry + excess
    carry_gross=6.15, carry_gross_t=4.51, carry_win=89,
    carry_net=4.15, carry_net_t=3.04,
    baseline=3.80,
    excess=2.35, excess_t=1.72, excess_ci_lo=-0.2, excess_ci_hi=4.8, excess_pctile=88,
    carry_ex08_mean=7.11, carry_ex08_t=6.42,
    # synthetic control
    syn_null_ratio_t=0.93, syn_ctrl_ratio_t=5.84,
    syn_null_excess_t=1.93, syn_ctrl_excess_t=9.20,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from election_volatility import data, strategy as st

def load_real_or_none():
    \"\"\"Return (gspc, vix) from the shared cache, or (None, None) offline.\"\"\"
    try:
        return data.load_real("^GSPC"), data.load_vix()
    except Exception as exc:
        print("real tapes unavailable (offline) -> synthetic tape:", exc)
        return None, None

GSPC, VIX = load_real_or_none()
HAVE_REAL = GSPC is not None
print("real cached tapes present:", HAVE_REAL)

# The offline synthetic control tape (always available) — a planted spike + excess VRP.
SB, SV, SEL, STRUTH = data.synthetic_daily(vol_bump=0.8, vrp=8.0, seed=318)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Election-Volatility — does the vote really shake the market? 🗳️\n"
            "### The 'sell vol into the election and collect the crush' trade, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![An_election_premium%3F: Misattributed](https://img.shields.io/badge/An_election_premium%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "Every four years the same story makes the rounds: US elections inject huge "
            "*uncertainty*, volatility spikes into the vote, and the option market pays a fat "
            "premium for it — so the clever move is to **sell volatility into the election and "
            "collect the 'vol crush'** when the result lands. This notebook asks the only "
            "questions that matter in plain English: *does volatility actually spike, does the "
            "option market over-charge for it, and is selling vol into elections any better than "
            "selling vol on any random Tuesday?*\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the placebo "
            "distribution and the excess-vs-baseline race? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first 🎯\n\n"
            "| Question | Plain answer |\n"
            "|---|---|\n"
            "| Does *realized* vol spike into election day? | **Barely.** ~+5% on average across "
            f"25 elections — indistinguishable from a random week. |\n"
            "| Does *implied* vol (VIX) over-price it? | **A little, usually** — but one bad "
            "election (2008) wipes out the average, so we can't be sure. |\n"
            "| Is selling vol *into elections* a real edge? | **No.** The money is the everyday "
            "'sell vol, get paid' premium — elections add almost nothing, and you eat the rare "
            "disaster. |\n\n"
            f"> **In one line:** the eye-catching version of this trade wins ~{R['carry_win']}% of the time "
            "with a great-looking t-stat — but that's the *ordinary* variance-risk premium you'd "
            "earn any time, not an election edge, and it's paid for with a single "
            f"**{R['vrp_2008']:+.0f} vol-point** loss in 2008."
        ),
        md(
            "## 1 · The claim\n\n"
            "Stated at full strength (and it has real academic backing — "
            "[Kelly, Pástor & Veronesi 2016](https://doi.org/10.1111/jofi.12406)): political "
            "events resolve uncertainty, so the option market charges a **variance-risk premium** "
            "for the dates that span them. Around a US presidential election you should therefore "
            "see (a) volatility rising into the vote, (b) implied vol (VIX) priced *above* the "
            "volatility that actually shows up, and (c) a profitable trade in **selling that "
            "implied vol** and buying it back cheaper after the result."
        ),
        md(
            "## 2 · So what?\n\n"
            "If true, it's a calendar you can mark years in advance — election dates are fixed by "
            "law, no forecasting needed. A reliable 'sell vol into November every fourth year' "
            "overlay would be a rare thing: a *scheduled* risk premium. The catch every options "
            "trader knows: selling volatility is **picking up pennies in front of a steamroller** "
            "— it wins most of the time and then, occasionally, hands back years of pennies at "
            "once. So the real question isn't 'does it usually win' (it does); it's *'does the "
            "election add anything over selling vol all the time, after the steamroller?'*"
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest tests, announced before we run them:\n\n"
            "1. **Realized-vol spike** — for each election, compare actual volatility in the run-up "
            "to a normal baseline. If elections are special, the ratio is clearly above 1; if it "
            "hovers around 1, the spike is a myth.\n"
            "2. **The premium** — compare the VIX going into the vote to the volatility that "
            "*actually* shows up after. Implied-minus-realized is what a vol seller pockets.\n"
            "3. **The fair race** — run the *same* short-vol trade on hundreds of random "
            "non-election dates. If the election trade isn't clearly better than that baseline, "
            "what looks like an 'election premium' is just the everyday premium.\n\n"
            "**What would make us say 'mirage':** a vol ratio near 1, a premium whose error bars "
            "cross zero, and an election trade that lands in the *middle* of the random-date pile."
        ),
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, the realized-vol spike. Does volatility actually climb into the vote? We draw "
            "it on whichever tape is available — the real S&P 500 if cached, otherwise the "
            "offline synthetic tape (clearly bannered)."
        ),
        code(
            "fig, ax = plt.subplots()\n"
            "if HAVE_REAL:\n"
            "    el = data.election_table()\n"
            "    ratios = st.vol_ratio_around(GSPC, el['election_date'])\n"
            "    tape = 'REAL ^GSPC (1928-2024 elections)'\n"
            "else:\n"
            "    ratios = st.vol_ratio_around(SB, SEL['election_date'])\n"
            "    tape = 'SYNTHETIC control tape (planted spike)'\n"
            "ax.bar(range(len(ratios)), ratios, color=[GREEN if r>1 else AMBER for r in ratios])\n"
            "ax.axhline(1.0, color=RED, lw=2, label='no spike (ratio = 1)')\n"
            "ax.set_title(f'Pre-election realized-vol ratio  ·  {tape}')\n"
            "ax.set_xlabel('election #'); ax.set_ylabel('pre-election vol / baseline vol')\n"
            "ax.legend()\n"
            "print(f'mean ratio = {ratios.mean():.2f}   (1.0 = no spike)')\n"
            "if HAVE_REAL:\n"
            "    print(f'on the real tape: t vs 1.0 = {st.tstat(ratios,1.0):+.2f} -> '\n"
            "          f'{\"a real spike\" if abs(st.tstat(ratios,1.0))>2 else \"basically no spike\"}')\n"
        ),
        md(
            f"On the **real** S&P 500 the average pre-election vol ratio is **{R['ratio_all']:.2f}** "
            f"(t vs 1.0 = **{R['ratio_all_t']:+.2f}**) — only {R['ratio_all_frac']}% of elections show "
            "*any* elevation, and the whole thing sits at the 66th percentile of random weeks. The "
            "'markets convulse into election day' image is, on realized volatility, **not there.**"
        ),
        md(
            "Now the premium and the trade. Does the VIX over-price the vol that follows — and is "
            "the election trade better than the same trade on random dates?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    elv = data.election_table(min_year=1992)\n"
            "    led = st.short_vol_carry(GSPC, VIX, elv['election_date'])\n"
            "    rd = st.random_carry_dates(GSPC.loc['1992':], 6000, seed=99)\n"
            "    base = st.short_vol_carry(GSPC, VIX, rd)['pnl_gross'].mean()\n"
            "    tape = 'REAL (VIX era 1992-2024)'\n"
            "else:\n"
            "    led = st.short_vol_carry(SB, SV, SEL['election_date'])\n"
            "    rd = st.random_carry_dates(SB, 6000, seed=99)\n"
            "    base = st.short_vol_carry(SB, SV, rd)['pnl_gross'].mean()\n"
            "    tape = 'SYNTHETIC control tape'\n"
            "elec = led['pnl_gross'].to_numpy()\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(range(len(elec)), elec, color=[GREEN if x>0 else RED for x in elec], label='election trades')\n"
            "ax.axhline(base, color=GREY, lw=2, ls='--', label=f'always-on carry = {base:.1f}')\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_title(f'Short-vol carry per election (vol points)  ·  {tape}')\n"
            "ax.set_xlabel('election #'); ax.set_ylabel('implied - realized (vol pts)'); ax.legend()\n"
            "print(f'election carry mean = {elec.mean():.2f} vol pts ; always-on baseline = {base:.2f}')\n"
            "print(f'election EXCESS over always-on = {elec.mean()-base:+.2f} vol pts')\n"
        ),
        md(
            f"There's the trap, in one chart. The election trade earns **+{R['carry_gross']:.1f} vol points** "
            f"on average and wins {R['carry_win']}% of the time — *but the grey dashed line is the "
            f"same trade on random dates, and it already earns **+{R['baseline']:.1f}**.* The election "
            f"**excess** over that baseline is only **+{R['excess']:.1f} vol points**, and notice the one "
            f"deep-red bar: **2008**, election week inside the financial crisis, where realized vol "
            f"({R['vrp_2008_real']:.0f}) blew past the {R['vrp_2008_impl']:.0f} implied for a "
            f"**{R['vrp_2008']:+.0f}** disaster. That single bar is the steamroller."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Realized vol doesn't spike (ratio {R['ratio_all']:.2f}). Implied vol "
            f"over-prices it *most* of the time (premium positive {R['vrp_win']}% of elections, "
            f"averaging +{R['vrp_mean']:.1f} vol points) — but the average is dragged to a t of just "
            f"+{R['vrp_t']:.2f} by 2008, below the bar we need to call it real.\n"
            f"- **Tradability — Mirage.** The pretty win-rate and t-stat are the *everyday* "
            "variance-risk premium, not an election edge. Strip out the always-on carry and the "
            f"election adds an insignificant +{R['excess']:.1f} vol points — while you carry the full "
            "2008-style tail.\n"
            "- **An election premium? — Misattributed.** What looks like 'election risk' is mostly "
            "the premium you'd collect selling vol any time."
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Sure — and most months you'd be happy: short vol, collect the crush, ~89% winners. "
            "But you'd collect almost exactly what a 'short vol all the time' fund collects, you'd "
            "have your capital tied up only every fourth November (a *calendar*-capacity problem), "
            "and you'd be on the hook for the 2008-shaped loss with no extra premium to compensate "
            "for choosing the election as your moment. The desk's rule decides it: race a part-time "
            "carry against the always-on carry on **excess**, and the excess here can't be told "
            "from zero. There's no free election lunch — only the usual vol-selling lunch, eaten "
            "less often."
        ),
        md(
            "## 7 · Going further 🚪\n\n"
            "Threads worth a fork: (a) **politically-sensitive sectors** (Boutchkova et al. 2012) "
            "may carry a real election vol premium even when the index doesn't; (b) **the VIX term "
            "structure** — does the election 'kink' in the futures curve pay better than the "
            "spot trade; (c) **non-US elections** (UK, France, EM) where the sample is bigger and "
            "the surprises larger; (d) **referendums and contested results** (Brexit, 2000) as a "
            "separate, higher-variance event class. The synthetic tape here lets you plant any of "
            "these and check the harness can find them before you trust it on the real thing."
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
            "# Election-Volatility — a quantitative teardown 🔬\n"
            "### Vol event study · variance-risk premium · excess-vs-always-on race · placebo · capacity\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![An_election_premium%3F: Misattributed](https://img.shields.io/badge/An_election_premium%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We separate the "
            "**election-specific** volatility-risk premium from the **always-on** VRP that selling "
            "vol pays on any date, and show the election component cannot clear the inference bar "
            "on the 9 US elections of the VIX era.\n\n"
            "> ⚠️ **Not investment advice.** Real tapes: ^GSPC (price-only, realized vol) and ^VIX "
            "(implied vol) from the shared `_cache`; offline cells fall back to the deterministic "
            "synthetic tape. Methods + citations in [`docs/references.md`](../docs/references.md); "
            "frozen headline numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\nfrom election_volatility import data, strategy as st\n"),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number (real tape, as-of 2026-06-12) |\n"
            "|---|---|---|\n"
            f"| **Signal** | `WEAK` | realized vol ratio {R['ratio_all']:.2f} (t={R['ratio_all_t']:+.2f}); "
            f"VRP +{R['vrp_mean']:.1f} (t={R['vrp_t']:+.2f}, CI [{R['vrp_ci_lo']:.1f},{R['vrp_ci_hi']:.1f}]); "
            f"election carry **excess** +{R['excess']:.1f} (t={R['excess_t']:+.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | gross t={R['carry_gross_t']:+.2f} is the always-on carry "
            f"(+{R['baseline']:.1f} of +{R['carry_gross']:.1f}); excess CI [{R['excess_ci_lo']:.1f},"
            f"{R['excess_ci_hi']:.1f}] crosses 0; 2008 = {R['vrp_2008']:+.0f} vol pts |\n"
            f"| **Election premium?** | `MISATTRIBUTED` | excess at {R['excess_pctile']}th pctile of random "
            "9-date baskets — inside the bulk |\n\n"
            "> 💡 **In plain words:** vol selling is paid every day; the election barely adds to it, "
            "and what it adds is within noise."
        ),
        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (realized spike):** mean pre-election realized-vol ratio > 1 at t ≥ 2.\n"
            "- **H₂ (variance-risk premium):** mean (implied − subsequent realized) > 0 at t ≥ 2 "
            "across elections.\n"
            "- **H₃ (tradable election edge):** election short-vol carry **excess over the "
            "always-on carry** > 0 at t ≥ 2 — the only H that, if true, would make the *election* "
            "(not vol-selling in general) the source of the edge."
        ),
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "H₂ alone is the Kelly-Pástor-Veronesi prediction and would be academically "
            "interesting; H₃ is the one a desk would *trade*. The desk's racing rule is the hinge: "
            "a part-time premium must be measured **excess of the full-time premium it's a subset "
            "of**, or you simply rediscover the variance-risk premium and mislabel it."
        ),
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Measure** the realized-vol ratio per election (pre-window std / baseline std) and "
            "the VRP per election (VIX in − realized out), exact identities, no fitting.\n"
            "2. **Robust inference** — cross-event t (events ~independent in time), percentile "
            "bootstrap CIs on the mean, and a **random-date placebo distribution** as the "
            "falsification test.\n"
            "3. **Alpha vs the carry** — subtract the always-on short-vol carry; race the election "
            "trade on *excess*.\n"
            "4. **Capacity / costs** — costs in vol points charged per round trip; the binding "
            "constraint is *calendar* capacity (one event every four years) and the short-vol tail.\n"
            "5. **Positive control** — the deterministic synthetic tape must recover a planted "
            "spike / planted election-excess, and find nothing on the null."
        ),
        md(
            "## 4 · The teardown\n\n"
            "### 4.1 Realized-vol spike + placebo\n"
            "💡 *In plain words: is the run-up genuinely more volatile than a random stretch?*"
        ),
        code(
            "if HAVE_REAL:\n"
            "    el = data.election_table()\n"
            "    ratios = st.vol_ratio_around(GSPC, el['election_date'])\n"
            "    s = st.summarize(ratios); t1 = st.tstat(ratios, 1.0)\n"
            "    plac = st.placebo_vol_ratio(GSPC, s['n'], n_draws=3000)\n"
            "    pct = st.percentile_in(ratios.mean(), plac); tape='REAL ^GSPC'\n"
            "else:\n"
            "    ratios = st.vol_ratio_around(SB, SEL['election_date'])\n"
            "    s = st.summarize(ratios); t1 = st.tstat(ratios, 1.0)\n"
            "    plac = st.placebo_vol_ratio(SB, s['n'], n_draws=3000)\n"
            "    pct = st.percentile_in(ratios.mean(), plac); tape='SYNTHETIC (planted spike)'\n"
            "fig, ax = plt.subplots()\n"
            "ax.hist(plac, bins=40, color=GREY, alpha=.6, label='placebo (random dates)')\n"
            "ax.axvline(ratios.mean(), color=RED, lw=2, label=f'elections mean = {ratios.mean():.2f}')\n"
            "ax.axvline(1.0, color='k', lw=1, ls='--', label='no spike')\n"
            "ax.set_title(f'Pre-election vol ratio vs placebo  ·  {tape}'); ax.legend()\n"
            "print(f'n={s[\"n\"]}  mean ratio={s[\"mean\"]:.3f}  t(vs1)={t1:+.2f}  placebo pctile={pct:.0f}')\n"
        ),
        md(
            f"**Real tape:** ratio {R['ratio_all']:.2f}, t(vs 1) = {R['ratio_all_t']:+.2f}, "
            f"{R['ratio_all_pctile']}th placebo percentile. **H₁ rejected** — no realized spike."
        ),
        md(
            "### 4.2 The variance-risk premium, per election\n"
            "💡 *In plain words: did the VIX charge more than the vol that actually arrived?*"
        ),
        code(
            "if HAVE_REAL:\n"
            "    elv = data.election_table(min_year=1992)\n"
            "    vrp = st.vrp_around(GSPC, VIX, elv['election_date'], pre=1, post=21)\n"
            "    tape='REAL (VIX era)'\n"
            "else:\n"
            "    vrp = st.vrp_around(SB, SV, SEL['election_date'], pre=1, post=21); tape='SYNTHETIC'\n"
            "v = vrp['vrp'].to_numpy(); s = st.summarize(v); ci = st.bootstrap_ci(v)\n"
            "fig, ax = plt.subplots()\n"
            "yr = pd.to_datetime(vrp['election_date']).dt.year.astype(str)\n"
            "ax.bar(yr, v, color=[GREEN if x>0 else RED for x in v])\n"
            "ax.axhline(s['mean'], color=GREY, lw=2, ls='--', label=f\"mean = {s['mean']:.1f}\")\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_title(f'Variance-risk premium per election (implied - realized)  ·  {tape}')\n"
            "ax.set_ylabel('vol points'); ax.legend(); plt.xticks(rotation=45)\n"
            "print(f'n={s[\"n\"]}  mean VRP={s[\"mean\"]:.2f}  t={s[\"tstat\"]:+.2f}  '\n"
            "      f'win={s[\"win_rate\"]*100:.0f}%  CI=[{ci[0]:.1f},{ci[1]:.1f}]')\n"
        ),
        md(
            f"**Real tape:** mean VRP +{R['vrp_mean']:.2f}, **t = {R['vrp_t']:+.2f}**, "
            f"positive {R['vrp_win']}% of elections, 95% CI [{R['vrp_ci_lo']:.1f}, {R['vrp_ci_hi']:.1f}] — "
            f"**straddles zero**. Drop the 2008 GFC election ({R['vrp_2008']:+.0f}) and it jumps to "
            f"+{R['vrp_ex08_mean']:.1f} at t = {R['vrp_ex08_t']:+.2f} — but that is a *snooped* removal of "
            "the exact tail a vol seller exists to survive. **H₂ not cleared.**"
        ),
        md(
            "### 4.3 The race that decides it — election excess over the always-on carry\n"
            "💡 *In plain words: subtract the money you'd make selling vol on any random day. "
            "What's left that's specifically about the election?*"
        ),
        code(
            "if HAVE_REAL:\n"
            "    elv = data.election_table(min_year=1992)\n"
            "    led = st.short_vol_carry(GSPC, VIX, elv['election_date'])\n"
            "    allrand = st.random_carry_dates(GSPC.loc['1992':], 8000, seed=99)\n"
            "    base = st.short_vol_carry(GSPC, VIX, allrand)['pnl_gross'].mean(); tape='REAL'\n"
            "else:\n"
            "    led = st.short_vol_carry(SB, SV, SEL['election_date'])\n"
            "    allrand = st.random_carry_dates(SB, 8000, seed=99)\n"
            "    base = st.short_vol_carry(SB, SV, allrand)['pnl_gross'].mean(); tape='SYNTHETIC'\n"
            "elec = led['pnl_gross'].to_numpy()\n"
            "g = st.summarize(elec); excess = elec - base\n"
            "ge = st.summarize(excess); cie = st.bootstrap_ci(excess)\n"
            "print(f'[{tape}] gross carry mean={g[\"mean\"]:.2f}  t={g[\"tstat\"]:+.2f}  win={g[\"win_rate\"]*100:.0f}%')\n"
            "print(f'[{tape}] always-on baseline = {base:.2f}')\n"
            "print(f'[{tape}] EXCESS mean={ge[\"mean\"]:.2f}  t={ge[\"tstat\"]:+.2f}  CI=[{cie[0]:.1f},{cie[1]:.1f}]')\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(['gross carry','always-on\\n(random)','election\\nEXCESS'],\n"
            "       [g['mean'], base, ge['mean']], color=[AMBER, GREY, RED])\n"
            "ax.set_title(f'Where the carry comes from (vol points)  ·  {tape}'); ax.set_ylabel('vol points')\n"
        ),
        md(
            f"**Real tape:** gross carry +{R['carry_gross']:.1f} at a gorgeous t = {R['carry_gross_t']:+.2f} — "
            f"but **+{R['baseline']:.1f} of it is the always-on VRP**. The election **excess** is "
            f"+{R['excess']:.1f} vol points at **t = {R['excess_t']:+.2f}**, CI [{R['excess_ci_lo']:.1f}, "
            f"{R['excess_ci_hi']:.1f}] (crosses zero), and a 9-election basket sits at the "
            f"{R['excess_pctile']}th percentile of random 9-date baskets. **H₃ rejected.** The headline "
            "t is the variance-risk premium wearing an election costume."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"| Test | real-tape stat | clears t ≥ 2? |\n"
            "|---|--:|:--:|\n"
            f"| H₁ realized-vol spike | ratio {R['ratio_all']:.2f}, t={R['ratio_all_t']:+.2f} | ❌ |\n"
            f"| H₂ variance-risk premium | +{R['vrp_mean']:.1f}, t={R['vrp_t']:+.2f} | ❌ |\n"
            f"| H₃ election carry excess | +{R['excess']:.1f}, t={R['excess_t']:+.2f} | ❌ |\n\n"
            "Signal **WEAK** (a real-looking premium the tape can't certify), Tradability "
            "**MIRAGE** (no excess over the carry you'd run anyway, plus the tail), 3rd axis "
            "**MISATTRIBUTED**."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            f"Break-even is generous on *cost* — a 2-vol-point round trip only trims gross "
            f"+{R['carry_gross']:.1f} to net +{R['carry_net']:.1f} (t {R['carry_net_t']:+.2f}). The trade doesn't "
            "die on costs; it dies on **attribution and tail**: the marginal election edge over an "
            "always-on short-vol book is statistically zero, capacity is one trade every four "
            f"years, and a single GFC-style election ({R['vrp_2008']:+.0f} vol points) erases many "
            "cycles of pennies. A short-vol fund should keep selling vol; it gains nothing by "
            "*timing* the sale to elections."
        ),
        md(
            "## 7 · Going further\n\n"
            "- **Sector dispersion** (Boutchkova et al. 2012): politically-exposed industries may "
            "carry a genuine election vol premium the index washes out.\n"
            "- **VIX term-structure kink**: trade the election bump in VIX *futures* rather than "
            "spot — does the roll pay better?\n"
            "- **Bigger samples**: non-US elections and referendums multiply the events and the "
            "surprises; the same engine and the excess-vs-carry rule port directly.\n"
            "- **Fork the synthetic tape** to plant a sector-specific or term-structure effect and "
            "confirm the harness recovers it before trusting any real-tape claim."
        ),
        md(
            "## Appendix · the synthetic positive control\n\n"
            "💡 *A harness that can't bank a planted signal proves nothing by finding nothing.*"
        ),
        code(
            "rows = []\n"
            "for name, vb, vr in [('null (vb=0, vrp=0)', 0.0, 0.0), ('planted (vb=.8, vrp=8)', 0.8, 8.0)]:\n"
            "    b, v, e, _ = data.synthetic_daily(vol_bump=vb, vrp=vr, seed=318)\n"
            "    t_ratio = st.tstat(st.vol_ratio_around(b, e['election_date']), 1.0)\n"
            "    led = st.short_vol_carry(b, v, e['election_date'])\n"
            "    rd = st.random_carry_dates(b, 6000, seed=5)\n"
            "    base = st.short_vol_carry(b, v, rd)['pnl_gross'].mean()\n"
            "    t_ex = st.tstat(led['pnl_gross'].to_numpy() - base)\n"
            "    rows.append({'tape': name, 'vol-ratio t(vs1)': round(t_ratio,2), 'excess-carry t': round(t_ex,2)})\n"
            "ctrl = pd.DataFrame(rows).set_index('tape'); print(ctrl)\n"
        ),
        md(
            f"The null clears nothing (vol-ratio t≈{R['syn_null_ratio_t']:.2f}, excess t≈{R['syn_null_excess_t']:.2f}); "
            f"the planted tape recovers both (t≈{R['syn_ctrl_ratio_t']:.2f} and t≈{R['syn_ctrl_excess_t']:.2f}). "
            "The engine is a faithful detector — which is exactly why its **null finding on the "
            "real election tape is informative**, not a failure to look."
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
