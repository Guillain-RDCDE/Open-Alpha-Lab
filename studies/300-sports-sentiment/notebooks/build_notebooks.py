"""Generate the two narrative notebooks for Study 300 (Sports-Sentiment).

    python notebooks/build_notebooks.py

This emits AND executes both notebooks (offline). The hardcoded elimination table
and the synthetic generator run anywhere, deterministically. The REAL cells use the
per-ticker parquet cache under ``_cache/`` if present; otherwise they fall back to
the frozen headline numbers in ``R`` (mirroring docs/results.md) and synthetic data,
so the notebook re-runs for any reader offline.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it. Execution uses nbconvert (no --allow-errors).
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-18).
R = dict(
    n=59,
    mean_bps=33.46,
    hac_t=1.606,
    hac_se_bps=20.84,
    plain_t=1.633,
    boot_p_ge0=0.955,
    frac_neg=0.441,
    n_down=26,
    n_up=33,
    shoot_n=18, shoot_mean_bps=-14.48, shoot_t=-0.484,
    nonshoot_n=41, nonshoot_mean_bps=54.51, nonshoot_t=2.147,
    wc_n=33, wc_mean_bps=12.13, wc_t=0.532,
    euro_n=26, euro_mean_bps=60.54, euro_t=1.676,
    net_mean_bps=-43.46, net_hit_rate=0.390, gross_mean_bps=-33.46,
    edmans_bps=-49.0,
    fp="95d210fa8556",
    date_start="1998-07-01", date_end="2024-07-15",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from sports_sentiment import data, strategy as st

def _have_cache():
    try:
        data.fetch_event_returns(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("ETF return cache present:", HAVE_REAL)
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-18)
R = dict(
    n=59, mean_bps=33.46, hac_t=1.606, hac_se_bps=20.84, plain_t=1.633,
    boot_p_ge0=0.955, frac_neg=0.441, n_down=26, n_up=33,
    shoot_n=18, shoot_mean_bps=-14.48, shoot_t=-0.484,
    nonshoot_n=41, nonshoot_mean_bps=54.51, nonshoot_t=2.147,
    wc_n=33, wc_mean_bps=12.13, wc_t=0.532,
    euro_n=26, euro_mean_bps=60.54, euro_t=1.676,
    net_mean_bps=-43.46, net_hit_rate=0.390, gross_mean_bps=-33.46,
    edmans_bps=-49.0,
)
"""

SIG = "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)"
TRAD = "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sports-Sentiment -- does a national-team loss sink the market next day?\n"
            "### The Edmans 'loss effect', tested honestly, in plain English\n\n"
            f"{SIG}\n{TRAD}\n\n"
            "There is a beautiful, much-cited finding in behavioural finance: when a country's "
            "national soccer team is **knocked out** of a major tournament, the home stock market "
            "drops the **next trading day**. Edmans, Garcia & Norli (2007) measured roughly a "
            "**-49 bps** abnormal next-day return after World Cup elimination losses. The story is "
            "irresistible: a whole nation wakes up heartbroken, risk appetite sags, and stocks dip.\n\n"
            "This notebook asks the only question that matters for a trader: **does it show up on a "
            "tape you could actually have traded?** We pair 60 marquee elimination losses (1998-2024) "
            "with the *investable single-country ETF* for each nation and look at the next session.\n\n"
            "> **This is the plain-language layer.** Want the Newey-West HAC t-stat, the bootstrap, "
            "and the sub-group cuts? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the famous -49 bps loss effect real in the academic data? | **Yes** -- Edmans et al. "
            "(2007) document it across dozens of countries with broad indices. |\n"
            "| Does it replicate on the *investable* country-ETF tape (1998-2024)? | **No.** The mean "
            "next-day return is **+33 bps** -- the *wrong sign* -- with HAC t = **+1.6** "
            "(not significant). |\n"
            "| Could you trade it? | **No.** Shorting the ETF the day after a loss loses **-43 bps "
            "net** per event and sits idle 99% of the time. |\n\n"
            "> The literature is real and important. But the headline effect does **not** survive the "
            "move from broad national indices to tradable single-country ETFs at this sample size -- "
            "so the *signal* is weak and the *trade* is a mirage."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"When a country's national team is eliminated from the World Cup, that country's "
            "stock market earns a significantly negative return the next day -- about half a percent "
            "below normal. The effect is asymmetric: wins do not produce an equal-and-opposite gain.\"*\n\n"
            "This is Edmans, Garcia & Norli (2007), *Journal of Finance* -- one of the cleanest "
            "demonstrations that **investor mood** moves prices. The mechanism is psychological, not "
            "fundamental: a painful, nationally-shared loss depresses risk appetite for a day."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a soccer result reliably moved a national market by half a percent, it would be a "
            "rare *dated, exogenous* sentiment shock -- the kind academics dream of, because the "
            "fixture list is set years in advance and a loss is unrelated to earnings. For a trader "
            "it would be a calendar of pre-scheduled short opportunities.\n\n"
            "The honest question: the original study used **broad national stock indices** measured in "
            "local currency. A retail or institutional trader outside that country reaches the market "
            "through a **single-country ETF** (EWU, EWG, EWQ, EWZ...). Does the effect survive that "
            "translation -- different basket, USD pricing, US trading hours, a much smaller event set?"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps:\n\n"
            "1. **Sign and significance.** The hypothesis is *directional* (returns should be "
            "**negative**). We must check both the sign of the mean and whether a robust t-stat clears "
            "|t| >= 2. A positive or insignificant mean kills it.\n"
            "2. **Tiny, clustered n.** We have ~60 marquee losses across 25 years, and they cluster in "
            "six-week tournament windows. Same-day eliminations of two countries can co-move, so we use "
            "a Newey-West (HAC) standard error, not a naive one.\n"
            "3. **Proxy mismatch.** A single-country ETF priced in USD during US hours is not the local "
            "index Edmans measured. We label this honestly: this is a **proxy** test of the *tradable* "
            "version, not a re-run of the original.\n\n"
            "We test on real ETF prices (cached) when present, and fall back to the frozen headline "
            "numbers + synthetic data when offline."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** every hardcoded elimination loss with its country ETF."
        ),
        code(
            "ev = data.elimination_table()\n"
            "print('Hardcoded elimination losses:', len(ev))\n"
            "print('Countries:', ', '.join(sorted(ev['country'].unique())))\n"
            "print('Tournaments:', dict(ev['tournament'].value_counts()))\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_event_returns(fetch=False)\n"
            "    n = len(df); mean_bps = float(df['next_day_ret'].mean()*1e4)\n"
            "    frac_neg = float((df['next_day_ret']<0).mean())\n"
            "else:\n"
            "    n, mean_bps, frac_neg = R['n'], R['mean_bps'], R['frac_neg']\n"
            "print(f'Events with ETF coverage: {n}')\n"
            "print(f'Mean NEXT-DAY return: {mean_bps:+.1f} bps  (Edmans hypothesis: ~{R[\"edmans_bps\"]:.0f} bps)')\n"
            "print(f'Fraction of losses followed by a DOWN day: {frac_neg:.1%}  (coin = 50%)')"
        ),
        md(
            "The very first number is decisive: the mean next-day return on the real ETF tape is "
            "**+33 bps** -- *positive*. The Edmans hypothesis predicts a ~**-49 bps** drop. On this "
            "investable proxy the sign is simply wrong, and only **44%** of losses are followed by a "
            "down day (a coin would give 50%)."
        ),
        md("**The distribution of next-day returns -- centred just above zero, not below.**"),
        code(
            "if HAVE_REAL:\n"
            "    rets = df['next_day_ret'].values*1e4\n"
            "else:\n"
            "    rng = np.random.default_rng(300)\n"
            "    rets = rng.normal(R['mean_bps'], 120, R['n'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(rets, bins=18, color=GREY, alpha=0.75, edgecolor='white')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(rets.mean(), c=GREEN, lw=2.5, label=f'Observed mean: {rets.mean():+.0f} bps')\n"
            "ax.axvline(R['edmans_bps'], c=RED, lw=2.5, ls='--', label=f'Edmans hypothesis: {R[\"edmans_bps\"]:.0f} bps')\n"
            "ax.set_xlabel('Next-day return after elimination (bps)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('Next-day returns after a national-team loss: mean is on the WRONG side of zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Observed mean {rets.mean():+.1f} bps vs hypothesised {R[\"edmans_bps\"]:.0f} bps')"
        ),
        md(
            "The mass sits slightly to the **right** of zero. Whatever mood effect exists in local "
            "indices, it does not print on the USD single-country ETF the next session in this sample."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Weak.** The academic effect (Edmans et al. 2007) is genuine and famous, "
            "so this is *not* pure folklore. But on the tradable ETF tape the mean is **+33 bps** with "
            "HAC t = **+1.6** -- wrong sign, not significant. Literature support without a confirming "
            "real-tape t-stat earns **Weak**, not Real.\n"
            "- **Tradability -- Mirage.** Shorting the ETF after each loss loses **-43 bps net** per "
            "event; the strategy is flat 99% of the time and single-country ETFs carry wide spreads "
            "and borrow.\n"
            "- **Why the gap?** Proxy mismatch (USD ETF vs local index), a much smaller event set, and "
            "US trading hours all dilute a fragile one-day mood effect."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The natural trade: short the country ETF for the single session after each loss."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tradable_pnl(df, cost_bps_oneway=5.0)\n"
            "    gross, net, hit = tr['gross_mean_bps'], tr['net_mean_bps'], tr['net_hit_rate']\n"
            "else:\n"
            "    gross, net, hit = R['gross_mean_bps'], R['net_mean_bps'], R['net_hit_rate']\n"
            "\n"
            "print('Strategy: short the country ETF, next session only, per elimination loss')\n"
            "print(f'  Gross mean PnL: {gross:+.1f} bps/event')\n"
            "print(f'  Net mean PnL (5 bps one-way x2): {net:+.1f} bps/event')\n"
            "print(f'  Net hit-rate: {hit:.1%}')\n"
            "print()\n"
            "print('Because the underlying market actually rose (+33 bps) after these losses,')\n"
            "print('SHORTING loses money even before costs -- and costs make it worse.')"
        ),
        md(
            "> The trade loses on average, before and after costs, because the market drifted **up** "
            "after these losses in our sample. There is no edge to harvest on this tape."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control** (companion notebook): plant a -49 bps loss effect in a "
            "synthetic panel and the same machinery recovers it cleanly -- so a null result here is "
            "about the *tape*, not a broken test.\n"
            "- **The cleaner test** would use broad local indices in local currency, as Edmans did. "
            "Our ETF proxy is the *tradable* version, and it is what fails.\n"
            "- **Cousins:** other mood/calendar anomalies on this desk -- the weather/SAD effect, the "
            "weekend effect, and the Super Bowl indicator ([Study 158](../../158-super-bowl/)).\n\n"
            "*Think the loss effect is tradable? Fork this, swap in local-index total-return series, "
            "and show a HAC t <= -2 with the right (negative) sign. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Sports-Sentiment -- a quantitative teardown\n"
            "### 60 elimination losses * country-ETF next-day returns * Newey-West HAC t * "
            "bootstrap * sub-group cuts * synthetic positive control\n\n"
            f"{SIG}\n{TRAD}\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same beats, every claim carrying its standard error.* We test the Edmans, Garcia & "
            "Norli (2007) loss effect on the **investable single-country ETF** tape: mean next-day "
            "return, a Newey-West (HAC) t-stat for H0: mean = 0, an i.i.d. bootstrap for the one-sided "
            "p-value, sub-group cuts (penalty shootouts; World Cup vs Euro/Copa), and a synthetic "
            "positive control that plants the -49 bps effect and recovers it.\n\n"
            "> **Not investment advice.** Real data: iShares single-country ETFs + ^GSPC daily closes "
            "(price-only, USD), per-ticker parquet cache. Elimination dates hardcoded in `data.py` "
            "(1998-2024). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `WEAK` | Real-tape mean **+33 bps**, HAC t = **+1.61** (wrong sign, "
            "not significant). Famous in the literature, absent on the tradable proxy. |\n"
            "| **Tradability** | `MIRAGE` | Short-on-loss nets **-43 bps/event**; idle 99% of the "
            "time; wide single-country ETF costs. |\n\n"
            "> A genuine REAL candidate that **fails its own real-tape t-test**: literature support "
            "without |HAC t| >= 2 (with the right sign) is Weak, by the desk rubric."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_i$ be the next-trading-day simple return of country $c(i)$'s ETF after "
            "elimination event $i$. The loss-effect hypothesis is:\n\n"
            "- **H1 (sign & size).** $E[r_i] < 0$, of order $-40$ to $-50$ bps.\n"
            "- **H2 (significance).** A robust (HAC) t-stat for $E[r_i]=0$ clears $t \\le -2$.\n"
            "- **H3 (intensity).** The effect is *stronger* for the most painful losses -- penalty "
            "shootouts -- per Edmans et al.\n\n"
            "We will see H1 fails (sign is positive), so H2 cannot hold; H3's shootout cut is the only "
            "negative-signed group, but with n=18 and t = -0.48 it is noise."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The original paper is a landmark precisely because the shock is **dated and exogenous**: "
            "the fixture is scheduled years ahead and a loss is orthogonal to fundamentals, so reverse "
            "causality and omitted-variable stories are weak. If the effect were tradable it would be a "
            "pre-printed calendar of short setups. Our test isolates whether the *tradable* "
            "implementation (USD single-country ETF, next US session) carries the signal -- a strictly "
            "harder, more honest bar than the local-index event study."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Events.** 60 marquee national-team elimination losses, 1998-2024, hardcoded in "
            "`data.py` with each country's ETF (^GSPC for the US proxy).\n"
            "- **Response.** Close-to-close simple return of the first trading session strictly after "
            "the match date -- an explicit **one-day execution lag** (matches are evening events).\n"
            "- **Returns.** PRICE-only, USD, from cached daily ETF closes. Survivorship: ETFs are "
            "live single-country funds; no dead-fund bias, but the basket differs from the local index.\n"
            "- **Inference.** Newey-West HAC t (Bartlett kernel, 1 lag) for H0: mean = 0; i.i.d. "
            "bootstrap (10,000) for the one-sided p; sub-group HAC t's.\n"
            "- **Positive control.** Synthetic panel with a planted -49 bps loss effect.\n"
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The headline test: mean next-day return and HAC t\n\n"
            "The single most important number, with a heteroskedasticity- and autocorrelation-"
            "consistent standard error (guards against same-day co-movement of two eliminated teams)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_event_returns(fetch=False)\n"
            "    s = st.loss_effect_stats(df, hac_lags=1, n_boot=10_000, seed=300)\n"
            "    mean_bps, hac_t, plain_t = s['mean_bps'], s['hac_t'], s['plain_t']\n"
            "    boot_p, frac_neg = s['boot_p_ge0'], s['frac_neg']\n"
            "    n = s['n']\n"
            "else:\n"
            "    mean_bps, hac_t, plain_t = R['mean_bps'], R['hac_t'], R['plain_t']\n"
            "    boot_p, frac_neg, n = R['boot_p_ge0'], R['frac_neg'], R['n']\n"
            "\n"
            "print(f'n events: {n}')\n"
            "print(f'Mean next-day return: {mean_bps:+.2f} bps   (Edmans hypothesis ~ {R[\"edmans_bps\"]:.0f} bps)')\n"
            "print(f'Newey-West HAC t (H0: mean=0): {hac_t:+.3f}')\n"
            "print(f'Plain i.i.d. t:                {plain_t:+.3f}')\n"
            "print(f'Bootstrap P(mean >= 0):        {boot_p:.3f}  (one-sided; loss effect wants this LOW)')\n"
            "print(f'Fraction of down days:         {frac_neg:.1%}')\n"
            "print()\n"
            "print('Bar for a REAL stamp: HAC t <= -2 with a NEGATIVE mean. We have +1.6 -- it fails twice.')"
        ),
        md(
            "> The HAC t is **+1.6** -- not only short of significance but on the *wrong side of zero*. "
            "The bootstrap puts ~95% of resampled means at or above zero. The loss effect does not exist "
            "on this tradable tape; if anything the market drifted mildly up after these losses."
        ),
        md(
            "### 4b - Sub-group cuts: does the painful-loss (shootout) story survive?\n\n"
            "Edmans et al. find the effect strongest for the most agonising defeats. Penalty shootouts "
            "are the cleanest 'painful loss' proxy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sg = st.subgroup_summary(df, hac_lags=1, seed=300)\n"
            "else:\n"
            "    sg = pd.DataFrame({\n"
            "        'group': ['All eliminations','Penalty-shootout losses','Non-shootout losses',\n"
            "                  'World Cup only','Euro / Copa'],\n"
            "        'n': [R['n'], R['shoot_n'], R['nonshoot_n'], R['wc_n'], R['euro_n']],\n"
            "        'mean_bps': [R['mean_bps'], R['shoot_mean_bps'], R['nonshoot_mean_bps'],\n"
            "                     R['wc_mean_bps'], R['euro_mean_bps']],\n"
            "        'hac_t': [R['hac_t'], R['shoot_t'], R['nonshoot_t'], R['wc_t'], R['euro_t']],\n"
            "        'frac_neg': [R['frac_neg'], 0.444, 0.439, 0.424, 0.462],\n"
            "    })\n"
            "print(sg.to_string(index=False))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "colors = [GREEN if m < 0 else RED for m in sg['mean_bps']]\n"
            "bars = ax.barh(sg['group'], sg['mean_bps'], color=colors)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(R['edmans_bps'], c=AMBER, ls='--', lw=2, label=f'Edmans ~{R[\"edmans_bps\"]:.0f} bps')\n"
            "ax.set_xlabel('Mean next-day return (bps)  [green = negative = supports hypothesis]')\n"
            "ax.set_title('Only the small shootout sub-group is negative -- and its HAC t is -0.5 (noise)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "Only the **penalty-shootout** sub-group (n=18) has a negative mean (-14 bps), consistent in "
            "*sign* with the painful-loss story -- but HAC t = **-0.48**, indistinguishable from zero. "
            "Every other cut is positive. This is the texture of noise, not a hidden real effect."
        ),
        md(
            "### 4c - The bootstrap distribution of the mean\n\n"
            "Resample events i.i.d. and view the sampling distribution of the mean next-day return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = df['next_day_ret'].values\n"
            "else:\n"
            "    rng = np.random.default_rng(300)\n"
            "    r = rng.normal(R['mean_bps']*1e-4, 120e-4, R['n'])\n"
            "rng_b = np.random.default_rng(300)\n"
            "boot = rng_b.choice(r, size=(10_000, len(r)), replace=True).mean(axis=1)*1e4\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(boot, bins=40, color=GREY, alpha=0.75)\n"
            "ax.axvline(0, c='k', lw=1, label='zero')\n"
            "ax.axvline(boot.mean(), c=GREEN, lw=2.5, label=f'mean {boot.mean():+.0f} bps')\n"
            "ax.axvline(R['edmans_bps'], c=RED, ls='--', lw=2.5, label=f'Edmans {R[\"edmans_bps\"]:.0f} bps')\n"
            "ax.set_xlabel('Bootstrapped mean next-day return (bps)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('The hypothesised -49 bps sits far in the LEFT tail of the resampled mean')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'P(mean >= 0) = {(boot >= 0).mean():.3f}   |   P(mean <= -49 bps) = {(boot <= R[\"edmans_bps\"]).mean():.4f}')"
        ),
        md(
            "The hypothesised **-49 bps** is essentially never reproduced by resampling our data -- the "
            "observed mean is in the wrong place entirely. The loss effect is not hiding in sampling error."
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `WEAK`** -- real-tape mean +33 bps, HAC t +1.61, bootstrap P(mean>=0)=0.95. "
            "Wrong sign, not significant. The desk rubric: a famous, literature-backed effect that does "
            "not clear |HAC t| >= 2 on the real tape is **Weak**, not Real.\n"
            "- **Tradability `MIRAGE`** -- short-on-loss nets -43 bps/event and is idle 99% of the time; "
            "single-country ETFs carry wide spreads and borrow.\n"
            "- **Honest caveat** -- this is the *tradable proxy* (USD ETF, next US session), not the "
            "local-index event study Edmans ran. The signal may live in the local tape; it does not "
            "survive the translation to something you could trade."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- gross vs net\n\n"
            "Short the country ETF for the single next session after each loss; round-trip cost = "
            "2 x one-way (shorts also pay borrow)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    out = []\n"
            "    for c in [0.0, 5.0, 10.0]:\n"
            "        tr = st.tradable_pnl(df, cost_bps_oneway=c)\n"
            "        out.append({'one_way_bps': c, 'gross_bps': tr['gross_mean_bps'],\n"
            "                    'net_bps': tr['net_mean_bps'], 'net_hit': tr['net_hit_rate']})\n"
            "    tbl = pd.DataFrame(out)\n"
            "else:\n"
            "    tbl = pd.DataFrame({'one_way_bps':[0.0,5.0,10.0],\n"
            "        'gross_bps':[R['gross_mean_bps']]*3,\n"
            "        'net_bps':[R['gross_mean_bps'], R['gross_mean_bps']-10, R['gross_mean_bps']-20],\n"
            "        'net_hit':[R['net_hit_rate']]*3})\n"
            "print(tbl.to_string(index=False))\n"
            "print()\n"
            "print('Short PnL = -(next-day return). Since the market ROSE +33 bps after losses,')\n"
            "print('the short bleeds -33 bps gross, worse after costs. No viable trade.')"
        ),
        md(
            "> Even at zero cost the short loses money, because the underlying drifted up. Costs only "
            "deepen the hole. The strategy also concentrates all its (negative) action into a handful "
            "of six-week tournament windows -- terrible capital utilisation on top of a negative edge."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Is the machinery capable of *finding* the loss effect when it exists? Plant a -49 bps "
            "effect in a synthetic panel and sweep the sample size."
        ),
        code(
            "rows = []\n"
            "for n_ev in [60, 150, 300, 600]:\n"
            "    dfp, _ = data.synthetic_panel(n_events=n_ev, loss_bps=49.0, seed=300)\n"
            "    sp = st.loss_effect_stats(dfp, hac_lags=1, n_boot=2000, seed=300)\n"
            "    df0, _ = data.synthetic_panel(n_events=n_ev, loss_bps=0.0, seed=300)\n"
            "    s0 = st.loss_effect_stats(df0, hac_lags=1, n_boot=2000, seed=300)\n"
            "    rows.append({'n': n_ev, 'planted_t': sp['hac_t'], 'planted_mean_bps': sp['mean_bps'],\n"
            "                 'null_t': s0['hac_t']})\n"
            "res = pd.DataFrame(rows)\n"
            "print(res.to_string(index=False))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(res['n'], res['planted_t'], 'o-', c=GREEN, lw=2, label='planted -49 bps effect')\n"
            "ax.plot(res['n'], res['null_t'], 's--', c=GREY, lw=2, label='null (no effect)')\n"
            "ax.axhline(-2, c=RED, ls=':', lw=1.5, label='t = -2 (significance)')\n"
            "ax.set_xlabel('Number of events'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine detects a real -49 bps effect -- the null tape does not')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('When the effect is real, HAC t goes strongly negative. On the real ETF tape it is +1.6.')"
        ),
        md(
            "The control confirms the test is calibrated: a planted -49 bps effect drives the HAC t "
            "well below -2, while the null stays near zero. The real-tape **+1.6** is therefore a "
            "statement about the **tradable proxy and sample**, not a broken method.\n\n"
            "> **Bottom line:** the Edmans loss effect is a real and important behavioural finding, but "
            "the version you could actually trade -- USD single-country ETFs, next US session -- shows "
            "no usable signal here. **Signal: Weak. Tradability: Mirage.**"
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


def _execute(name):
    """Execute a notebook in-place via nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    with open(path, encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
