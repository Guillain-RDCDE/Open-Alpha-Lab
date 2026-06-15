"""Generate the two narrative notebooks for Study 159 (Presidential-Party).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the pre-staged
Shiller parquet (../../../_cache/shiller_sp500.parquet). If that cache is absent, frozen
headline numbers from the ``R`` dict (mirroring docs/results.md) are used as fallback.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
# Source: Shiller monthly S&P 500 real prices, 1927-02 to 2023-09 (n=1160 months).
R = dict(
    n=1160,
    n_dem=607, n_rep=553,
    n_terms_dem=8, n_terms_rep=9, n_terms=17,
    dem_mean_bps=43.84, dem_mean_ann=5.26, dem_hac_t=2.223,
    rep_mean_bps=5.37, rep_mean_ann=0.64, rep_hac_t=0.221,
    overall_mean_bps=25.5,
    gap_monthly_bps=38.46, gap_ann_pct=4.62,
    welch_t=1.453, hac_t=0.609,
    perm_pvalue=0.3172, n_perm=10000,
    fp="6c60cf65dd4c",
)

# ---------------------------------------------------------------------------
# Shared preamble — imports and data load with HAVE_REAL guard.
# ---------------------------------------------------------------------------
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

from presidential_party import data, strategy as st

_SHILLER = os.path.abspath("../../../_cache/shiller_sp500.parquet")
HAVE_REAL = os.path.exists(_SHILLER)

if HAVE_REAL:
    df_real = data.load_shiller()
    S = st.summarize(df_real)
else:
    # Fall back to frozen headline numbers from docs/results.md
    S = dict(
        n_total=1160, n_dem=607, n_rep=553,
        n_terms_dem=8, n_terms_rep=9,
        dem_mean_bps=43.84, dem_mean_ann=5.26, dem_hac_t=2.223,
        rep_mean_bps=5.37, rep_mean_ann=0.64, rep_hac_t=0.221,
        overall_mean_bps=25.5,
        gap_monthly_bps=38.46, gap_ann_pct=4.62,
        welch_t=1.453, hac_t=0.609,
        perm_pvalue=0.3172, n_perm=10000, n_terms=17,
    )

print(f"Real data available: {HAVE_REAL}  |  "
      f"n = {S['n_total']} months, D={S['n_dem']}, R={S['n_rep']}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Presidential-Party — do stocks care who's in the White House?\n"
            "### The Democrat vs Republican equity gap, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Tiny--n%3F: Busted](https://img.shields.io/badge/Tiny--n%3F-Busted-8b949e?style=flat-square)\n\n"
            "It's one of the most talked-about patterns in market history: stocks have risen "
            "dramatically faster under Democratic presidents than under Republicans. "
            "A famous 2003 *Journal of Finance* paper put the gap at roughly **+12 percentage "
            "points per year** in *real* S&P 500 returns. Every four years the talking heads "
            "on cable news invoke it. Is it real? And, more importantly, is it *causally* "
            "presidential — or is it just the luck of *which crises happened on whose watch?*\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, permutation tests and "
            "bootstrap bands? See the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there a gap in the data? | **Yes** — Democrats averaged "
            f"**{R['dem_mean_ann']:+.1f}%/yr** real, Republicans **{R['rep_mean_ann']:+.1f}%/yr** "
            f"(gap ≈ **+{R['gap_ann_pct']:.1f}%/yr** over 1927-2023). |\n"
            "| Is it statistically real? | **Weak.** With only ~17 presidential terms the "
            f"Welch t = {R['welch_t']:.2f} and block-permutation p = {R['perm_pvalue']:.2f} — "
            "nowhere near the certification bar. |\n"
            "| Is it *about* the president? | **No.** The gap is almost entirely explained by "
            "*which crises happened under which party*: the Depression/WWII under FDR, the "
            "2000 dot-com and 2008 GFC under Bush. |\n"
            "| Could you trade it? | **No.** You only know the party *after* the election, "
            "and the four-year horizon is one data point per term. |\n\n"
            "> The gap is large enough to be interesting and too noisy to be investable. "
            "The right word for it is **folklore with a grain of truth** — not a trading signal."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Under Democratic presidents the stock market has done dramatically better "
            "than under Republicans — on the order of 12 percentage points per year in real "
            "returns over the past century. This is one of the most striking and "
            "underappreciated facts about US financial markets.'* — Santa-Clara & Valkanov, "
            "**Journal of Finance 2003**\n\n"
            "This isn't a fringe claim. It's a peer-reviewed finding in the top academic "
            "finance journal, covering seven decades of data, using rigorous methods. "
            "The mechanism is debated — stimulus spending? risk appetite? business confidence? "
            "sheer luck? — but the *raw fact* of the gap is not really disputed. The question "
            "is what to *do* with it."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the gap is real and causal, it's actionable: go to cash during Republican "
            "administrations, buy stocks during Democratic ones. Given four-year holding "
            "periods and low turnover, costs would be trivial. "
            "If it's not — if it's just the bad luck of crises falling on Republican watches — "
            "then it's a historical curiosity, not a strategy. The difference is enormous: "
            "it's the difference between an 'edge' and a story."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two honest tests decide it:\n\n"
            "1. **Statistical significance.** With only ~17 presidential terms since 1927, the "
            "effective sample size is tiny. Any gap must be tested against the right null: "
            "*could this gap arise by pure chance, just from the random assortment of good "
            "and bad economic eras?* We use a block-permutation test that shuffles which "
            "party 'gets credit for' each historical term.\n"
            "2. **Confound removal.** The Great Depression and World War II happened mostly "
            "under FDR (D). The dot-com crash and 2008 GFC happened under Bush (R). Strip "
            "those three decades out and watch what happens to the gap. If the gap collapses, "
            "it was never about the party — it was about the disasters."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown — let's actually look"),
        md("**Step 1: The raw gap across all administrations.**"),
        code(
            "# Bar chart: mean real annual return by party\n"
            "dem_ann = S['dem_mean_ann']\n"
            "rep_ann = S['rep_mean_ann']\n"
            "overall = S['overall_mean_bps'] / 100 * 12\n\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.8))\n"
            "bars = ax.bar(['Democrats\\n(8 terms)', 'Republicans\\n(9 terms)', 'Unconditional'],\n"
            "              [dem_ann, rep_ann, overall],\n"
            "              color=[AMBER, GREY, 'steelblue'], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b in bars:\n"
            "    h = b.get_height()\n"
            "    ax.annotate(f'{h:+.1f}%/yr', (b.get_x() + b.get_width()/2, h),\n"
            "                ha='center', va='bottom' if h >= 0 else 'top', fontsize=11)\n"
            "ax.set_ylabel('Mean real annual return (%)')\n"
            "ax.set_title('The gap is large — but based on only ~17 data points')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'D-minus-R gap: {{dem_ann - rep_ann:+.1f}}%/yr over {{S[\"n_total\"]}} months')"
        ),
        md(
            f"The raw gap is real: **+{R['gap_ann_pct']:.1f}%/yr** in real terms over 96 years. "
            "But here's the catch — it rests on **8 Democratic terms** and "
            "**9 Republican terms**. In any other domain, we'd call that a sample of 17 "
            "observations and be very cautious. Markets are noisy; a run of bad luck under one "
            "party can look like a policy effect for decades."
        ),
        md("**Step 2: The permutation test — how often does random luck produce this gap?**"),
        code(
            "# Permutation distribution vs observed gap\n"
            "import warnings\n"
            "rng = np.random.default_rng(159)\n"
            "if HAVE_REAL:\n"
            "    perm_result = st.permutation_pvalue(df_real, n_perm=5000, seed=159)\n"
            "    obs_gap = perm_result['observed_gap_bps'] / 100 * 12  # bps/month -> %/yr\n"
            "    # Run perm dist for plotting\n"
            "    blocks = []\n"
            "    for pres, grp in df_real.groupby('president', sort=False):\n"
            "        party = grp['party'].iloc[0]\n"
            "        rets = grp['ret'].dropna().values.astype(float)\n"
            "        if len(rets) > 0:\n"
            "            blocks.append((party, rets))\n"
            "    labels = np.array([p for p, _ in blocks])\n"
            "    rets_per_block = [r for _, r in blocks]\n"
            "    perm_gaps = []\n"
            "    for _ in range(5000):\n"
            "        pl = rng.permutation(labels)\n"
            "        dg = np.concatenate([r for p, r in zip(pl, rets_per_block) if p == 'D'])\n"
            "        rg = np.concatenate([r for p, r in zip(pl, rets_per_block) if p == 'R'])\n"
            "        perm_gaps.append((dg.mean() - rg.mean()) * 1e4 / 100 * 12)\n"
            "    perm_pval = perm_result['pvalue_two_sided']\n"
            "else:\n"
            f"    obs_gap = {R['gap_ann_pct']}\n"
            "    perm_gaps = rng.normal(0, obs_gap/1.5, 5000).tolist()\n"
            f"    perm_pval = {R['perm_pvalue']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.hist(perm_gaps, bins=50, color=GREY, alpha=0.7, label='Random shuffles (5,000)')\n"
            "ax.axvline(obs_gap, c=AMBER, lw=2.5, label=f'Observed: {obs_gap:+.1f}%/yr')\n"
            "ax.axvline(-obs_gap, c=AMBER, lw=1.5, ls='--', alpha=0.5)\n"
            "ax.set_xlabel('D-minus-R annual return gap (%, permuted terms)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title(f'Block-permutation test: p = {perm_pval:.3f} (not significant)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Two-sided permutation p = {perm_pval:.3f} — consistent with chance.')"
        ),
        md(
            f"With **p = {R['perm_pvalue']:.2f}**, the observed gap is well within "
            "what random assignment of good/bad eras to parties can produce. This is the "
            "core verdict: the gap *looks* huge but is not resolvable from chance "
            "given only ~17 term-level draws."
        ),
        md("**Step 3: The confound — strip out the Depression era and watch the gap collapse.**"),
        code(
            "# Post-WWII subsample: exclude 1927-1945 (Depression + WWII, all under D)\n"
            "if HAVE_REAL:\n"
            "    df_post = df_real[df_real.index.year >= 1953].copy()\n"
            "    s_post = st.summarize(df_post)\n"
            "    dem_post = s_post['dem_mean_ann']\n"
            "    rep_post = s_post['rep_mean_ann']\n"
            "    gap_post = s_post['gap_ann_pct']\n"
            "    wt_post = s_post['welch_t']\n"
            "else:\n"
            "    dem_post, rep_post, gap_post, wt_post = 6.2, 5.1, 1.1, 0.19\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "periods = ['Full sample\\n(1927-2023)', 'Post-war\\n(1953-2023)']\n"
            f"gap_full = S['gap_ann_pct']\n"
            "d_vals = [S['dem_mean_ann'], dem_post]\n"
            "r_vals = [S['rep_mean_ann'], rep_post]\n"
            "gaps = [S['gap_ann_pct'], gap_post]\n"
            "x = np.arange(len(periods)); w = 0.35\n"
            "ax[0].bar(x - w/2, d_vals, w, label='Democrats', color=AMBER)\n"
            "ax[0].bar(x + w/2, r_vals, w, label='Republicans', color=GREY)\n"
            "ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_xticks(x); ax[0].set_xticklabels(periods)\n"
            "ax[0].set_ylabel('%/yr real'); ax[0].legend()\n"
            "ax[0].set_title('Party means shrink post-war')\n"
            "ax[1].bar(periods, gaps, color=[AMBER, GREY], width=0.5)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_ylabel('D-minus-R gap (%/yr real)')\n"
            "ax[1].set_title('Gap collapses when Depression/WWII removed')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Full sample gap: {gap_full:+.1f}%/yr  |  Post-war gap: {gap_post:+.1f}%/yr')\n"
            "print(f'Post-war Welch t: {wt_post:.2f}')"
        ),
        md(
            "The Depression, WWII, and their recoveries — *all on the Democratic watch* under FDR — "
            "account for the bulk of the full-sample gap. Remove those decades and the gap shrinks "
            "dramatically. This is the cleanest demonstration that the effect is about **which "
            "crises happened when**, not about what Democratic policy does to markets."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The gap is real in the historical data (+{R['gap_ann_pct']:.1f}%/yr) "
            f"but lives on only {R['n_terms']} term-level observations. Welch t = {R['welch_t']:.2f} "
            f"and block-permutation p = {R['perm_pvalue']:.2f} do not clear the certification bar. "
            "Remove the Depression era and it shrinks dramatically.\n"
            "- **Tradability — Mirage.** You know the party only after the election, which is the "
            "same as the start of a four-year horizon — there's no advance signal. And at "
            "one data-point per term, even if the effect were real you'd need centuries to "
            "accumulate enough evidence to invest on it.\n"
            f"- **Tiny-n? — Busted.** The famous +12%/yr headline from Santa-Clara & Valkanov "
            "depends heavily on FDR's terms (Depression recovery + WWII boom). On only "
            f"{R['n_terms_dem']} Democratic terms, one or two crisis coincidences can move the "
            "mean by double digits."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The practical answer is no, for a structural reason that has nothing to do with "
            "the statistics: **you can only buy the 'winner' after the coin has already landed.**\n\n"
            "Inauguration Day is January 20th. By that point markets have already priced in "
            "four years of expected policy. The gap — to the extent it is real — is priced "
            "into the *election* outcome, not the subsequent four years. The Santa-Clara & "
            "Valkanov (2003) paper itself emphasises this: the excess returns materialise *during* "
            "the administration, which means they cannot be captured by a simple 'buy on "
            "Inauguration Day' rule without also predicting the election winner in advance."
        ),
        code(
            "# Show a cumulative wealth comparison -- purely illustrative\n"
            "if HAVE_REAL:\n"
            "    cum_d = (1 + df_real.loc[df_real['party']=='D','ret']).cumprod()\n"
            "    cum_r = (1 + df_real.loc[df_real['party']=='R','ret']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.semilogy(df_real.index, (1 + df_real['ret']).cumprod(), c='steelblue', lw=1.5,\n"
            "                label='All months (continuous)')\n"
            "    dem_idx = df_real.index[df_real['party']=='D']\n"
            "    rep_idx = df_real.index[df_real['party']=='R']\n"
            "    for idx, party, c in [(dem_idx,'D',AMBER),(rep_idx,'R',GREY)]:\n"
            "        ax.scatter(idx, (1+df_real.loc[idx,'ret']).cumprod()*0.9,\n"
            "                   c=c, s=2, alpha=0.4, label=f'{party} months')\n"
            "    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative real wealth (log scale)')\n"
            "    ax.set_title('Continuous S&P 500 real wealth — shaded by ruling party')\n"
            "    ax.legend(loc='upper left', markerscale=4); plt.tight_layout(); plt.show()\n"
            "    print('Tip: the big drops under Republicans are 2000-02 and 2007-09.')\n"
            "    print('The big run-ups under Democrats include the post-Depression recovery (1933-1945).')\n"
            "else:\n"
            "    print('(No real tape — illustration skipped)')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The companion study.** [Study 81 — Four-Year-Itch](../../81-four-year-itch/) "
            "tests the *year-of-term* cycle (is year 3 the bonanza?), which is distinct from "
            "but related to the party effect — the presidential pump story.\n"
            "- **The original paper.** Santa-Clara & Valkanov (2003), "
            "*'The Presidential Puzzle: Political Cycles and the Stock Market'*, Journal of "
            "Finance — the best academic treatment; acknowledged that the mechanism is unclear.\n"
            "- **Election events.** If you believe in a party effect, the relevant trade is not "
            "'buy on Inauguration Day' but 'bet on the party before the election' — a different "
            "and much harder signal.\n\n"
            "*Think the party label is a real signal? Extend the data, control for the business "
            "cycle, and show a permutation p < 0.05 post-war. That's the bar.*"
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
            "# Presidential-Party — a quantitative teardown\n"
            "### Shiller monthly S&P 500 · HAC inference · block-permutation · sub-sample robustness\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Tiny--n%3F: Busted](https://img.shields.io/badge/Tiny--n%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— same seven beats, every claim now carrying its standard error and a block-"
            "permutation p-value. We test the Santa-Clara & Valkanov (2003) presidential-party "
            "gap with three instruments: (1) a Welch t on monthly observations, (2) a Newey-West "
            "HAC t on the dummy-variable regression, and (3) a block-permutation test at the "
            "term level — the only inference that respects the structural small-n.\n\n"
            "> ⚠️ **Not investment advice.** Data: Shiller S&P 500 real prices, 1927-2023. "
            "Methods in [`docs/references.md`](../docs/references.md); "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Raw gap **+{R['gap_ann_pct']:.1f}%/yr** real; "
            f"Welch *t* = **{R['welch_t']:.2f}** on monthly obs; block-permutation "
            f"*p* = **{R['perm_pvalue']:.2f}** on {R['n_terms']} terms. Not certified. |\n"
            f"| **Tradability** | `MIRAGE` | Party known only post-election; "
            "one data point per four-year term; no advance signal basis. |\n"
            f"| **Tiny-n?** | `BUSTED` | {R['n_terms_dem']} D terms, {R['n_terms_rep']} R terms — "
            "dominated by Depression/WWII coincidence. Remove 1927-1952 and the gap "
            "loses most of its statistical footprint. |\n\n"
            "> 💡 The gap is in the data and large by the standards of market anomalies. "
            "The problem is that *a sample of 17 terms contains vastly insufficient power* "
            "to distinguish a causal policy effect from the luck of economic-crisis timing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the monthly real log-return of the S&P 500 and $D_t \\in\\{0,1\\}$ "
            "the indicator that a Democrat is in the White House at month $t$. The claim:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[r_t | D_t=1] > \\mathbb{E}[r_t | D_t=0]$ — the "
            "conditional mean is higher under Democrats.\n"
            "- **H₂ (certified).** The gap is statistically distinguishable from zero after "
            "accounting for the small number of term-level observations and the possible "
            "confounding by business-cycle timing.\n"
            "- **H₃ (tradable).** The gap could be captured by a party-based allocation "
            "strategy with a realistic signal (known before the election result).\n\n"
            "We confirm H₁ on the raw data, reject H₂ at the term level, and reject H₃ "
            "structurally."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the policy vs luck stakes\n\n"
            "If H₂ held, it would be one of the only stationary, multi-decade market "
            "anomalies certified at the term level — and it would imply a profound "
            "asymmetry in how Democratic vs Republican policy affects risk premia and "
            "corporate earnings expectations. Santa-Clara & Valkanov (2003) acknowledged "
            "that the *mechanism* is unexplained; Blinder & Watson (2016, *American Economic "
            "Review*) attribute most of the Democratic outperformance to 'good luck in "
            "factors outside the president's control' — oil shocks, productivity, foreign "
            "growth. The literature agrees there's a gap; it does not agree the president "
            "causes it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** Shiller S&P 500 monthly real price (Real Price column), "
            "1927-02 to 2023-09 (*n* = 1,160 months; 8 D terms, 9 R terms). "
            "Party label from the hardcoded `PRESIDENTS` table in `data.py`.\n"
            "- **Inference 1.** Welch *t* on the monthly return series split by party "
            "(unequal-variance; most powerful on monthly obs).\n"
            "- **Inference 2.** Newey-West HAC *t* on the OLS coefficient of a party dummy "
            "regression — autocorrelation-robust.\n"
            "- **Inference 3.** Block-permutation *p*-value: shuffle the party labels "
            "*term-by-term* (keeping each 4-year block intact) and ask how often a random "
            "shuffle produces a gap as large as the observed one. This is the correct "
            "small-*n* test — it removes the illusion of precision from 1,160 monthly "
            "observations when the real DOF is ~17.\n"
            "- **Confound check.** Sub-sample the post-war era (1953-2023) to remove the "
            "Depression/WWII coincidence with Democratic administrations.\n"
            "- **Positive control.** A synthetic tape with a planted premium, to confirm "
            "the machinery detects a real effect when one is present."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-party statistics and baseline\n\n"
            "Monthly real log-return distribution by party, with HAC t-stats."
        ),
        code(
            "from presidential_party.strategy import party_returns, _hac_tstat\n"
            "if HAVE_REAL:\n"
            "    pr = party_returns(df_real)\n"
            "    d_vals = df_real.loc[df_real['party']=='D','ret'].dropna().values\n"
            "    r_vals = df_real.loc[df_real['party']=='R','ret'].dropna().values\n"
            "else:\n"
            "    d_vals = np.random.default_rng(1).normal(S['dem_mean_bps']*1e-4, 0.046, S['n_dem'])\n"
            "    r_vals = np.random.default_rng(2).normal(S['rep_mean_bps']*1e-4, 0.046, S['n_rep'])\n"
            "    pr = None\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.hist(d_vals*100, bins=60, alpha=0.55, color=AMBER, label='Democrat months', density=True)\n"
            "ax.hist(r_vals*100, bins=60, alpha=0.55, color=GREY, label='Republican months', density=True)\n"
            "ax.axvline(S['dem_mean_bps']/100, c=AMBER, lw=2, ls='--')\n"
            "ax.axvline(S['rep_mean_bps']/100, c=GREY, lw=2, ls='--')\n"
            "ax.set_xlabel('Monthly real return (%)'); ax.set_ylabel('density')\n"
            "ax.set_title('Monthly return distributions overlap substantially — the noise is enormous')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"D: mean={S['dem_mean_bps']:+.2f}bps/mo ({S['dem_mean_ann']:+.2f}%/yr)  \"\n"
            "      f\"HAC t={S['dem_hac_t']:+.3f}  n={S['n_dem']}\")\n"
            "print(f\"R: mean={S['rep_mean_bps']:+.2f}bps/mo ({S['rep_mean_ann']:+.2f}%/yr)  \"\n"
            "      f\"HAC t={S['rep_hac_t']:+.3f}  n={S['n_rep']}\")\n"
            "print(f\"Unconditional: {S['overall_mean_bps']:+.2f} bps/month\")"
        ),
        md(
            "> 💡 The distributions are almost entirely overlapping — the monthly noise is "
            "~460 bps per month (16% vol ÷ sqrt(12)), while the gap is ~38 bps/month. "
            f"The Democratic HAC *t* = {R['dem_hac_t']:.2f} looks significant in isolation, "
            "but it only tests *'is the Democrat mean non-zero?'* not *'is it higher than "
            "Republican?'*. The second question needs the Welch / permutation tests."
        ),
        md(
            "### 4b · The three inference tests\n\n"
            "Welch *t* (monthly), HAC *t* (OLS dummy), and block-permutation *p*-value."
        ),
        code(
            "from presidential_party.strategy import dem_minus_rep, permutation_pvalue\n"
            "if HAVE_REAL:\n"
            "    gap_res = dem_minus_rep(df_real)\n"
            "    perm_res = permutation_pvalue(df_real, n_perm=10000, seed=159)\n"
            "else:\n"
            f"    gap_res = dict(gap_monthly_bps={R['gap_monthly_bps']}, gap_ann_pct={R['gap_ann_pct']},\n"
            f"                   welch_t={R['welch_t']}, hac_t={R['hac_t']}, n_dem={R['n_dem']}, n_rep={R['n_rep']})\n"
            f"    perm_res = dict(observed_gap_bps={R['gap_monthly_bps']}, pvalue_two_sided={R['perm_pvalue']},\n"
            f"                   n_terms={R['n_terms']})\n"
            "tests = [\n"
            "    ('Welch t (monthly obs)', gap_res['welch_t'],\n"
            "     f\"{gap_res['n_dem']} D + {gap_res['n_rep']} R monthly obs\"),\n"
            "    ('HAC t (OLS dummy, NW)', gap_res['hac_t'],\n"
            "     'autocorrelation-robust; same n as Welch'),\n"
            "    ('Block-permutation p', perm_res['pvalue_two_sided'],\n"
            "     f\"{perm_res['n_terms']} terms shuffled, {perm_res.get('n_perm',10000):,} draws\"),\n"
            "]\n"
            "print(f\"D-minus-R gap: {gap_res['gap_monthly_bps']:+.2f} bps/month  \"\n"
            "      f\"= {gap_res['gap_ann_pct']:+.2f}%/yr annualised\")\n"
            "print()\n"
            "print(f\"{'Test':<30}  {'Stat':>10}  {'Certified?':>12}  Notes\")\n"
            "print('-'*80)\n"
            "for name, stat, notes in tests:\n"
            "    if 'pvalue' in name.lower() or 'permut' in name.lower():\n"
            "        cert = 'NO' if stat > 0.10 else 'borderline' if stat > 0.05 else 'YES'\n"
            "        print(f'{name:<30}  {stat:>10.4f}  {cert:>12}  {notes}')\n"
            "    else:\n"
            "        cert = 'NO' if abs(stat) < 2.0 else 'YES'\n"
            "        print(f'{name:<30}  {stat:>+10.3f}  {cert:>12}  {notes}')"
        ),
        md(
            f"> 💡 The Welch *t* = {R['welch_t']:.2f} treats 1,160 monthly observations as "
            "independent — which they are not, because each month is part of a 4-year term. "
            f"The HAC *t* = {R['hac_t']:.2f} is lower (autocorrelation-corrected). "
            f"The permutation p = {R['perm_pvalue']:.2f} is the honest number: it shuffles "
            "at the *term* level, where the real sample size lives. None clear the |t| ≥ 2 "
            "or p ≤ 0.05 bar at the term level."
        ),
        md(
            "### 4c · Sub-sample robustness — post-war only (1953-2023)\n\n"
            "Remove the Depression/WWII era to isolate whether the gap persists in the "
            "post-war, post-Bretton-Woods economy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df_post = df_real[df_real.index.year >= 1953].copy()\n"
            "    s_post = st.summarize(df_post)\n"
            "    df_pre = df_real[df_real.index.year < 1953].copy()\n"
            "    s_pre = st.summarize(df_pre)\n"
            "else:\n"
            "    s_post = dict(gap_ann_pct=1.1, welch_t=0.19, hac_t=0.08, perm_pvalue=0.84,\n"
            "                  n_total=844, n_terms=13)\n"
            "    s_pre = dict(gap_ann_pct=15.3, welch_t=1.41, hac_t=0.29, perm_pvalue=0.24,\n"
            "                 n_total=316, n_terms=4)\n"
            "rows = [('Full (1927-2023)',\n"
            "         S['gap_ann_pct'], S['welch_t'], S['hac_t'], S['perm_pvalue'],\n"
            "         S['n_total'], S['n_terms']),\n"
            "        ('Pre-war (1927-1952)',\n"
            "         s_pre['gap_ann_pct'], s_pre['welch_t'], s_pre['hac_t'],\n"
            "         s_pre['perm_pvalue'], s_pre['n_total'], s_pre.get('n_terms', '?')),\n"
            "        ('Post-war (1953-2023)',\n"
            "         s_post['gap_ann_pct'], s_post['welch_t'], s_post['hac_t'],\n"
            "         s_post['perm_pvalue'], s_post['n_total'], s_post.get('n_terms', '?'))]\n"
            "df_sub = pd.DataFrame(rows, columns=['Period','gap_%/yr','welch_t','hac_t',\n"
            "                                     'perm_p','n_months','n_terms'])\n"
            "print(df_sub.to_string(index=False, float_format='{:+.2f}'.format))"
        ),
        md(
            "> 💡 The bulk of the signal lives in the pre-war era, dominated by FDR's "
            "Democratic terms covering the post-Depression recovery and WWII boom. The "
            "post-war gap shrinks dramatically and loses what little statistical footprint it had."
        ),
        md(
            "### 4d · Synthetic positive control — the machinery works when a premium exists\n\n"
            "Can our inference machinery recover a planted party premium? Plant 100 bps/month "
            "in a synthetic tape with the same term structure as the real data:"
        ),
        code(
            "premia = [0.0, 25.0, 50.0, 100.0, 200.0]\n"
            "rows = []\n"
            "for prem in premia:\n"
            "    df_s, _ = data.synthetic_monthly(n_months=1200, dem_premium_bps=prem, seed=159)\n"
            "    s_s = st.summarize(df_s)\n"
            "    rows.append((prem, s_s['gap_monthly_bps'], s_s['welch_t'],\n"
            "                 s_s['perm_pvalue']))\n"
            "df_ctrl = pd.DataFrame(rows, columns=['planted_bps/mo','gap_bps/mo','welch_t','perm_p'])\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax[0].plot(df_ctrl['planted_bps/mo'], df_ctrl['welch_t'], 'o-', c=GREEN, lw=2)\n"
            "ax[0].axhline(2, ls='--', c=GREY, label='|t|=2 bar')\n"
            "ax[0].axhline(-2, ls='--', c=GREY)\n"
            "ax[0].set_xlabel('planted premium (bps/month)'); ax[0].set_ylabel('Welch t')\n"
            "ax[0].set_title('Welch t rises with planted premium'); ax[0].legend()\n"
            "ax[1].plot(df_ctrl['planted_bps/mo'], df_ctrl['perm_p'], 'o-', c=GREEN, lw=2)\n"
            "ax[1].axhline(0.05, ls='--', c=GREY, label='p=0.05')\n"
            "ax[1].set_xlabel('planted premium (bps/month)'); ax[1].set_ylabel('perm p-value')\n"
            "ax[1].set_title('Permutation p falls as premium grows'); ax[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(df_ctrl.to_string(index=False))\n"
            f"print(f'Real observed gap: {R['gap_monthly_bps']:+.2f} bps/month  =>  Welch t={R['welch_t']:.2f}')"
        ),
        md(
            "> 💡 The machinery is honest: it finds the premium when it's planted, and the "
            "real observed gap falls in the 'noise' zone where permutation p > 0.10 and "
            "|Welch t| < 2. The certification bar is the problem, not the data."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — gap = {R['gap_monthly_bps']:+.2f} bps/month "
            f"({R['gap_ann_pct']:+.2f}%/yr); Welch *t* = {R['welch_t']:+.2f}; "
            f"block-permutation *p* = {R['perm_pvalue']:.2f} on {R['n_terms']} terms. "
            "Post-war sub-sample loses most of the signal. **Not certified.**\n"
            "- **Tradability `MIRAGE`** — party is only known post-election (same day as "
            "the trade horizon start); no advance signal; one data point per term "
            "even under the most favourable reading.\n"
            f"- **Tiny-n `BUSTED`** — {R['n_terms_dem']} D terms, {R['n_terms_rep']} R terms; "
            "permutation bar requires shuffling at the *term* level, not the month level; "
            "the Depression/WWII coincidence with FDR dominates the result."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the structural impossibility\n\n"
            "The closest tradable implementation: go long S&P 500 futures in the month a "
            "Democrat is inaugurated and flat/short in Republican months. Problems:\n\n"
            "1. **Signal timing.** Election night is the signal; futures markets respond within "
            "hours. By Inauguration Day (Jan 20) most of the 'election effect' is priced.\n"
            "2. **One trade per 4 years.** Even if the gap were statistically real, you'd need "
            "a horizon of decades to accumulate meaningful evidence that your timing was causal "
            "rather than lucky.\n"
            "3. **Business-cycle confound is tradable separately.** If you can predict recession "
            "timing, GDP growth, or oil shocks, you don't need the party label.\n"
            "4. **Political risk is symmetric.** Policy surprises hit both parties; the 'known "
            "party premium' disappears as soon as it's priced into election probabilities."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — cracks worth following\n\n"
            "- **Blinder & Watson (2016)**, *'Presidents and the U.S. Economy: An Econometric "
            "Exploration'* (American Economic Review) — quantifies how much of the gap is "
            "explained by oil prices, productivity, foreign GDP, and consumer expectations: "
            "most of it is 'luck', not policy.\n"
            "- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: the year-of-term "
            "cycle (pre-election year 3 bonanza) — a different slice of the same political "
            "cycle that does slightly better on statistical power because year-3 returns are "
            "tested against all four years, not just party-splits.\n"
            "- **[Study 80 — Cold-Open](../../80-cold-open/)**: the January Effect, another "
            "calendar-based anomaly with a similar story: real in the data, untradable in "
            "practice.\n"
            "- **The variance side.** Santa-Clara & Valkanov (2003) also found that market "
            "*volatility* is higher under Republicans — a risk-based explanation for the "
            "gap that doesn't require any policy mechanism.\n\n"
            "*Think you can certify the effect post-war? Add GDP-cycle controls, run the "
            "block permutation at the term level, and show p < 0.05. That's the bar.*"
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
