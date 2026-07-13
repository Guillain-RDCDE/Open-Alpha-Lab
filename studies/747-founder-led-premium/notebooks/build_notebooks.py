"""Generate the two narrative notebooks for Study 747 (Founder-Led-Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached monthly
closes under ../_cache/ (each basket ticker + SPY) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance monthly adjusted
# closes = total-return proxy; two hardcoded baskets frozen as of the 2016-01 formation;
# 108 months 2016-01..2024-12; founder basket 11 priced of 13 (SQ reticker, FIT delisted),
# professional 13; market model = CAPM vs SPY, Newey-West HAC t).
R = dict(
    asof="2026-07-13", data_start="2016-01", data_end="2024-12", n_months=108,
    fingerprint="c363d7e5e578",
    n_founder_priced=11, n_founder=13, n_pro=13, dropped=("SQ", "FIT"),
    # raw basket performance
    founder_ann=34.5, pro_ann=11.8, ls_ann=20.5, ls_bps=156.5, ls_sharpe=0.73,
    ls_rawmean_t=2.00,
    # the abnormal return (CAPM alpha, HAC t)
    ls_alpha_bps=52.1, ls_beta=0.85, ls_alpha_t=0.76, ls_r2=0.26,
    beta_part_bps=104.3, beta_share_pct=67,
    founder_alpha_bps=50.6, founder_beta=1.62, founder_alpha_t=0.84,
    pro_alpha_bps=-1.5, pro_beta=0.77, pro_alpha_t=-0.09,
    # jackknife: (dropped, alpha_bps, t) sorted by resulting alpha
    jk=[("NVDA", 21.9, 0.33), ("TSLA", 35.6, 0.52), ("SHOP", 37.6, 0.58),
        ("NFLX", 47.3, 0.72), ("META", 50.3, 0.71), ("AMZN", 51.4, 0.73),
        ("GOOGL", 52.3, 0.72), ("CRM", 54.8, 0.77), ("YELP", 62.6, 0.82),
        ("W", 72.8, 1.11), ("GPRO", 86.9, 1.17)],
    # placebo: obs alpha, null mean, two-sided p, frac beats
    placebo_obs=52.1, placebo_null_mean=0.4, placebo_p=0.331, placebo_beats_pct=16,
    # costs
    gross_bps=156.5, net_bps=128.5, drag_bps=28.0,
    # synthetic control: (planted_bps, alpha_bps, t)
    syn=[(0.0, 25.2, 0.82), (200.0, 225.2, 7.29)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Founder_premium%3F: Misattributed](https://img.shields.io/badge/Founder_premium%3F-Misattributed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from founder_led_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RETS, FIRMS = data.load_real()
    LS = st.long_short(RETS, data.FOUNDER_TICKERS, data.PRO_TICKERS)
    MKT = RETS["SPY"].reindex(LS.index)
else:
    RETS = FIRMS = LS = MKT = None
print("real price cache present:", HAVE_REAL,
      "| months:", (0 if LS is None else len(LS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do founder-run companies beat the market? 🧑‍💼\n"
            "### The most flattering chart in finance — and the trick hiding inside it\n\n"
            + BADGES +
            "It's one of the stickiest ideas in business: **the founder cares more.** Skin in the "
            "game, a long horizon, a reality-distortion field — so *founder-led firms outperform*. "
            "There's even a famous paper (Fahlenbrach, 2009) and a Bain best-seller ('The Founder's "
            "Mentality') behind it. And when you actually build the basket — Amazon, Nvidia, Meta, "
            "Tesla, Nvidia again — it **crushes** a basket of professionally-run blue chips.\n\n"
            "So we did the honest thing: put a founder-led basket **long**, a professional-CEO basket "
            "**short**, and asked not *'did it win?'* (it did, enormously) but *'is the win a founder "
            "effect — or something much more boring?'* The answer is a small masterclass in how a real "
            "number can tell a false story.\n\n"
            "> 📓 **Plain-language layer.** Want the CAPM alpha, the Newey-West *t*-stats, the "
            "jackknife and the placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** There's no free 'founder-CEO' database, so we **hardcode a "
            "transparent basket** of the founder firms one *remembers in 2024* — and that memory is "
            "exactly the problem, which we flag loudly. Prices are yfinance monthly *total returns*; "
            "every chart is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the founder basket beat professional-CEO peers? | **Yes — by a mile.** "
            f"**+{R['founder_ann']:.0f}%/yr** vs **+{R['pro_ann']:.0f}%/yr**, a "
            f"**+{R['ls_ann']:.0f}-point** annual spread. The chart is gorgeous. |\n"
            f"| Is that a real *founder* edge? | **No.** Once you subtract the plain stock-market "
            f"ride, **{R['beta_share_pct']}%** of the gap is just **beta** — a high-octane tech "
            f"basket vs sleepy staples. The leftover 'alpha' is **+{R['ls_alpha_bps']:.0f} bps/mo** "
            f"at *t* = **{R['ls_alpha_t']:.2f}** — statistically **nothing**. |\n"
            f"| Could you have bought it in 2016? | **No.** The basket is the founders who turned out "
            f"to be Nvidia — chosen *because* we know that now. Two of the original names "
            f"(**{', '.join(R['dropped'])}**) literally delisted and can't even be priced. |\n"
            "| So is 'founder magic' a strategy? | **It's survivorship in a nice suit.** Drop the "
            f"single name **NVDA** and the whole edge halves. It's one or two rockets and a lot of "
            "hindsight — not a repeatable characteristic. |\n\n"
            "> Founder firms really did win. But 'they won' and 'founders win' are different "
            "sentences — and the gap between them is the whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Founders have skin in the game and a long horizon, so founder-led companies "
            "outperform professionally-managed ones.\"*\n\n"
            "This isn't a strawman — it's in the literature. **Fahlenbrach (2009)** finds S&P 500 "
            "founder-CEO firms earned positive abnormal returns; **Bain & Company** built a whole "
            "'Founder's Mentality' franchise on it. The intuition is genuinely appealing, and the "
            "*raw* data seems to shout agreement. We'll take the strongest version — a long/short "
            "basket, founders vs professionals — and ask whether the outperformance is a **founder** "
            "effect or a mirage made of the two things founders and survivors share: **tech-sector "
            "beta** and **being remembered because they won**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were a real, repeatable characteristic, it would be one of the great free lunches: "
            "screen for founder-CEOs, tilt toward them, collect a premium — no forecasting required. "
            "That's why the claim is everywhere in venture decks and factor pitches. But 'founder-led' "
            "has to earn its keep against two impostors. **(1) Beta.** Founder firms skew young, "
            "techy, high-beta; in a decade-long bull market that alone beats staples — and beta is "
            "free, you don't pay a manager for it. **(2) Survivorship.** The founders we *name* are "
            "the ones who made it; the founder-run flame-outs (Theranos, WeWork, a graveyard of "
            "de-SPACs) never enter a 2024 basket. Strip both away and see if anything is left."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We freeze two baskets as of **{R['data_start']}** (as-of {R['asof']}; fingerprint "
            f"`{R['fingerprint']}`) and run a **long/short sort** over **{R['n_months']} months**:\n\n"
            "1. **Two baskets, equal weight.** *Founder-led* (Amazon, Nvidia, Meta, Tesla, Shopify, "
            "Netflix, Salesforce… plus the duds we *do* remember — GoPro, Wayfair, Yelp) minus "
            "*professional-CEO* blue chips (Walmart, Coke, J&J, Cisco, JPMorgan…). Long the first, "
            "short the second.\n"
            "2. **Subtract the market.** Fit `long/short = alpha + beta·market`. The **alpha** is the "
            "part the stock market *doesn't* explain — the only thing that could be a founder effect.\n"
            "3. **Stress it three ways.** A **Newey-West** *t* (honest with monthly data); a "
            "**jackknife** (drop one name — does the edge survive?); a **placebo** (shuffle the "
            "founder/professional labels on these same names — is the *real* labelling special?).\n\n"
            "**What would make us say 'mirage':** the alpha is insignificant, most of the raw gap is "
            "beta, and the edge collapses when you drop a name or two. (Spoiler: all three.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the flattering chart.** Cumulative growth of \\$1 in each basket. This is the "
            "picture the founder thesis is sold on — and it's not fake, it's just *misread*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.basket_returns(RETS, data.FOUNDER_TICKERS).reindex(LS.index)\n"
            "    pr = st.basket_returns(RETS, data.PRO_TICKERS).reindex(LS.index)\n"
            "    sp = MKT\n"
            "    cf, cp, cs = (1+fr).cumprod(), (1+pr).cumprod(), (1+sp).cumprod()\n"
            "    x = LS.index\n"
            "else:\n"
            "    x = np.arange(R['n_months']); g=lambda a:(1+np.full(R['n_months'],a))\n"
            "    cf=np.cumprod(1+np.full(R['n_months'],0.025)); cp=np.cumprod(1+np.full(R['n_months'],0.009)); cs=np.cumprod(1+np.full(R['n_months'],0.012))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.0))\n"
            "ax.plot(x, cf, c=GREEN, lw=2.2, label=f'founder basket (+{R[\"founder_ann\"]:.0f}%/yr)')\n"
            "ax.plot(x, cp, c=GREY, lw=2.0, label=f'professional basket (+{R[\"pro_ann\"]:.0f}%/yr)')\n"
            "ax.plot(x, cs, c='k', lw=1.2, ls='--', label='S&P 500 (SPY)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.set_title('The chart the founder thesis is sold on')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'founder ${float(cf[-1]) if not hasattr(cf,\"iloc\") else cf.iloc[-1]:.1f} vs professional ${float(cp[-1]) if not hasattr(cp,\"iloc\") else cp.iloc[-1]:.1f} — a blowout')"
        ),
        md(
            f"A blowout: the founder basket compounds to roughly **3–4×** the professional one. If the "
            "story ended here, 'founders outperform' would be settled. It doesn't end here — because "
            "*two* different things could draw this exact chart, and only one of them is a founder "
            "effect."
        ),
        md(
            "**So we split the spread into its two pieces.** The long/short return each month is "
            "`founder − professional`. Regress it on the market and you get **beta** (the free "
            "stock-market ride) and **alpha** (everything left over). Here's how the average monthly "
            "spread divides."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.capm_alpha(LS['ls'].to_numpy(), MKT.to_numpy())\n"
            "    raw = LS['ls'].mean()*1e4; alpha=c['alpha_bps']; beta_part = c['beta']*MKT.mean()*1e4\n"
            "else:\n"
            "    raw=R['ls_bps']; alpha=R['ls_alpha_bps']; beta_part=R['beta_part_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['raw spread'], [raw], color=GREEN, width=.5, label='what you see')\n"
            "ax.bar(['split'], [beta_part], color=GREY, width=.5, label='just market beta (free)')\n"
            "ax.bar(['split'], [alpha], bottom=[beta_part], color=AMBER, width=.5, label='founder alpha (the claim)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('avg long/short return (bps/mo)')\n"
            "ax.annotate(f'{beta_part:.0f} bps\\n= {R[\"beta_share_pct\"]}% is BETA', (1, beta_part/2), ha='center', va='center', fontsize=9)\n"
            "ax.annotate(f'{alpha:.0f} bps alpha', (1, beta_part+alpha+6), ha='center', fontsize=9, color='#8a6d00')\n"
            "ax.set_title('Two-thirds of the founder premium is just beta'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'raw {raw:.0f} bps/mo = beta {beta_part:.0f} + alpha {alpha:.0f}  ->  beta is {R[\"beta_share_pct\"]}%')"
        ),
        md(
            f"There's the first cut: **{R['beta_share_pct']}%** of the monthly spread is just the "
            "founder basket being a **higher-beta** collection of stocks in a bull market. You don't "
            "need a founder to get beta — you can buy it in an index fund for a few basis points. What "
            f"remains — the actual *founder* claim — is **+{R['ls_alpha_bps']:.0f} bps/month** of "
            "alpha. Is *that* real?"
        ),
        md(
            "**Is the leftover alpha more than luck?** Two honest tests. Left: the founder alpha "
            "against a **placebo** — shuffle the founder/professional labels across these very same "
            "names, thousands of times, and see where the real labelling lands. Right: the "
            "**jackknife** — drop each founder name in turn and watch the alpha move."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pool = data.FOUNDER_TICKERS + data.PRO_TICKERS\n"
            "    kf = sum(t in RETS.columns for t in data.FOUNDER_TICKERS); kp = sum(t in RETS.columns for t in data.PRO_TICKERS)\n"
            "    null = st.placebo_alpha_dist(RETS, pool, k_long=kf, k_short=kp, n_draws=2500)\n"
            "    obs = st.capm_alpha(LS['ls'].to_numpy(), MKT.to_numpy())['alpha_bps']\n"
            "    pval = st.placebo_pvalue(obs, null)\n"
            "    jk = st.jackknife_alpha(RETS, data.FOUNDER_TICKERS, data.PRO_TICKERS)\n"
            "    jn = jk['dropped'].tolist(); ja = jk['alpha_bps'].tolist()\n"
            "else:\n"
            "    rng=np.random.default_rng(747); null=rng.normal(0.4, 55, 2500); obs=R['placebo_obs']; pval=R['placebo_p']\n"
            "    jn=[j[0] for j in R['jk']]; ja=[j[1] for j in R['jk']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.hist(null, bins=45, color=GREY, alpha=.85, label='random label (luck)')\n"
            "a1.axvline(obs, c=GREEN, lw=2.5, label=f'real founder tag ({obs:+.0f} bps)')\n"
            "a1.set_xlabel('long/short alpha (bps/mo)'); a1.set_ylabel('freq')\n"
            "a1.set_title(f'Placebo: p = {pval:.2f} (not special)'); a1.legend(fontsize=8)\n"
            "cols=[RED if n in ('NVDA','TSLA','SHOP') else GREEN for n in jn]\n"
            "a2.barh(range(len(jn)), ja, color=cols)\n"
            "a2.axvline(R['ls_alpha_bps'], c='k', lw=1, ls='--', label='keep all names')\n"
            "a2.set_yticks(range(len(jn))); a2.set_yticklabels(jn, fontsize=7); a2.invert_yaxis()\n"
            "a2.set_xlabel('alpha after dropping that name (bps/mo)'); a2.set_title('Jackknife: drop NVDA, lose half'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p={pval:.2f}; dropping NVDA takes alpha {R[\"ls_alpha_bps\"]:.0f} -> {ja[0]:.0f} bps')"
        ),
        md(
            f"Both tests say the same thing. The placebo *p* is **{R['placebo_p']:.2f}** — a random "
            f"way of splitting these same names into 'long' and 'short' beats the real founder "
            f"labelling **{R['placebo_beats_pct']}%** of the time, so the *founder tag itself* adds "
            f"nothing. And the jackknife is brutal: drop **NVDA** alone and the alpha falls from "
            f"**{R['ls_alpha_bps']:.0f}** to **{R['jk'][0][1]:.0f} bps** — dropping the *losers* "
            "(GoPro, Wayfair) *raises* it. The 'premium' is one or two rockets, not a characteristic."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The market-model alpha is **+{R['ls_alpha_bps']:.0f} bps/mo** at "
            f"Newey-West *t* = **{R['ls_alpha_t']:.2f}** — indistinguishable from zero. "
            f"**{R['beta_share_pct']}%** of the raw spread is beta; the placebo says the founder tag "
            f"isn't special (*p* = {R['placebo_p']:.2f}); the jackknife says it's one name. No founder edge survives.\n"
            "- **Tradability — Mirage.** You couldn't have picked this basket in 2016 — it's the "
            f"founders who *turned into* Nvidia. Two originals ({', '.join(R['dropped'])}) delisted "
            "outright. The part you could actually buy is free beta plus a coin-flip.\n"
            "- **\"Founder premium?\" — Misattributed.** The outperformance is **real** and "
            "**mis-explained**: it's survivorship (we only name the winners) and tech-sector beta "
            "concentrated in a handful of names — not a leadership characteristic you can harvest."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the hindsight problem\n\n"
            "Forget statistics for a second. The fatal flaw is **timing**: the basket only looks "
            "brilliant because we drew it *after* the winners revealed themselves. Here's the "
            "jackknife alpha as a ladder — every bar below the dashed line is a name whose *removal* "
            "shrinks the edge, i.e. a name doing the heavy lifting. A real characteristic wouldn't "
            "care which name you drop."
        ),
        code(
            "if HAVE_REAL:\n"
            "    jk = st.jackknife_alpha(RETS, data.FOUNDER_TICKERS, data.PRO_TICKERS)\n"
            "    jn=jk['dropped'].tolist(); ja=jk['alpha_bps'].tolist()\n"
            "else:\n"
            "    jn=[j[0] for j in R['jk']]; ja=[j[1] for j in R['jk']]\n"
            "full=R['ls_alpha_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols=[GREEN if a<full else GREY for a in ja]\n"
            "ax.bar(jn, ja, color=cols)\n"
            "ax.axhline(full, c='k', ls='--', lw=1.2, label=f'keep all names ({full:.0f} bps)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('L/S alpha after dropping (bps/mo)'); ax.set_title('The edge is a few names, not a characteristic')\n"
            "ax.tick_params(axis='x', labelrotation=45); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('green bars = load-bearing winners; drop them and the edge falls. drop the DUDS and it rises.')"
        ),
        md(
            "> The green bars — the winners — are the whole story; grey bars (the founder *duds*) "
            "*add* to the edge when removed. To 'trade the founder premium' you'd have needed to know, "
            "in 2016, that Nvidia and Tesla would be Nvidia and Tesla. That's not a strategy — that's "
            "a memory. Costs barely matter here (net is a hair below gross); the binding constraint is "
            "that **the basket is built from the answer key.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🧑‍💼\n\n"
            "- **Do it survivorship-clean.** Freeze the basket on a *contemporaneous* founder list "
            "(e.g. every S&P 500 founder-CEO named in a 2016 filing, winners and losers alike) and "
            "rerun — the honest version of Fahlenbrach. The tech beta will still be there; the alpha "
            "probably won't.\n"
            "- **Neutralise the sector.** Match each founder name to a same-industry professional peer "
            "so the short leg cancels the beta. Bet: the residual is a wash.\n"
            "- **A cousin study.** [Study 391 — CEO-Turnover](../391-ceo-turnover/) runs the same "
            "abnormal-return machinery on *changing* the CEO; here we test *who founded* it. Both end "
            "up as small-sample, hindsight-shaped mirages.\n\n"
            "*Think there's a real founder premium? Show a founder-minus-professional alpha that "
            "clears *t* = 2 on a **contemporaneously-selected** basket and survives dropping its top "
            "name — then we'll talk.*"
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
            "# Founder-Led-Premium — a quantitative long/short teardown 🔬\n"
            "### Founder vs professional-CEO baskets · CAPM alpha with a Newey-West HAC *t* · a "
            "beta/alpha decomposition · a leave-one-out jackknife · a label-shuffle placebo · costs "
            "+ borrow · a synthetic plant-and-recover control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We take "
            "the strongest form of the Fahlenbrach (2009) / Bain 'founder's mentality' claim — a "
            "founder-minus-professional long/short — and separate the three things it fuses: a "
            "**real, large raw spread**, a **market-beta tilt**, and **survivorship**. The decisive "
            "object is not the spread's sign (a hindsight basket of today's founder-winners is "
            "*designed* to win) but the **market-model alpha's HAC *t***, its **concentration** (one "
            "name), and its **label-specificity** (a placebo).\n\n"
            "> ⚠️ **Data + selection note.** No free founder-CEO database, so we use a hardcoded, "
            "labelled basket of the founder firms *salient in 2024* — a deliberate, **hindsight / "
            "survivor-biased** sample (the founder flame-outs that delisted never enter; two of our "
            f"own originals, {', '.join(R['dropped'])}, already dropped out). That bias is named on "
            "the **Signal axis** and it points **for** the claim, so an insignificant alpha here is a "
            "*conservative* refutation. Real data: yfinance monthly adjusted closes (**total-return** "
            "proxy), each name + SPY. Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | L/S CAPM alpha **+{R['ls_alpha_bps']:.0f} bps/mo** at HAC "
            f"*t* = **{R['ls_alpha_t']:.2f}**; **{R['beta_share_pct']}%** of the raw "
            f"**+{R['ls_bps']:.0f} bps** spread is beta; placebo **p = {R['placebo_p']:.2f}**; drop "
            f"NVDA ⇒ alpha **{R['jk'][0][1]:.0f} bps**. |\n"
            f"| **Tradability** | `MIRAGE` | Hindsight basket (couldn't be formed ex-ante); "
            f"{', '.join(R['dropped'])} delisted; net **{R['net_bps']:.0f} bps** is mostly free beta "
            "+ an insignificant, one-name residual. |\n"
            f"| **Founder premium?** | `MISATTRIBUTED` | The raw outperformance is real but "
            "mis-explained: survivorship + tech-sector beta concentrated in ~2 names, not a "
            "leadership characteristic. |\n\n"
            "> 💡 In plain words: the founder basket genuinely trounced the professional one, but "
            "'they won' isn't 'founders win' — once you remove the market ride and remember that we "
            "picked the winners on purpose, there is no founder alpha left to certify."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Form the equal-weighted founder basket return $r^{F}_t$ and professional basket return "
            "$r^{P}_t$, and the dollar-neutral long/short $r^{LS}_t = r^{F}_t - r^{P}_t$. Fit the "
            "market model\n\n"
            "$$r^{LS}_t = \\alpha + \\beta\\, r^{m}_t + \\varepsilon_t,$$\n\n"
            "so $\\alpha$ is the **abnormal return** — the founder-vs-professional spread the market "
            "factor does not explain (Jensen's alpha) — judged by a **Newey-West HAC** *t*.\n\n"
            "- **H₁ (raw outperformance).** $\\mathbb{E}[r^{LS}] > 0$. *(Trivially true for a "
            "hindsight basket — and it is.)*\n"
            "- **H₂ (a real founder premium).** $\\alpha \\neq 0$ with $|t_{\\alpha}| \\ge 2$ — the "
            "spread is more than beta.\n"
            "- **H₃ (robust characteristic).** $\\alpha$ survives dropping the top name (jackknife) "
            "and the founder *label* beats a random relabelling (placebo).\n\n"
            f"We find **H₁ supported** (raw {R['ls_ann']:.0f}%/yr), **H₂ rejected** "
            f"($\\alpha$ *t* = {R['ls_alpha_t']:.2f}), **H₃ rejected** (drop NVDA ⇒ "
            f"{R['jk'][0][1]:.0f} bps; placebo *p* = {R['placebo_p']:.2f}). The claim is true exactly "
            "in the form that carries no information (a survivor basket rose) and false exactly in the "
            "form that would pay (a harvestable founder alpha)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the three impostors\n\n"
            "A positive $\\mathbb{E}[r^{LS}]$ can come from any of:\n\n"
            "$$\\underbrace{\\mathbb{E}[r^{LS}]}_{\\text{raw spread}} = "
            "\\underbrace{\\alpha}_{\\text{founder effect?}} + "
            "\\underbrace{\\beta\\,\\mathbb{E}[r^m]}_{\\text{sector beta}} + "
            "\\underbrace{\\text{selection}}_{\\text{survivorship}}.$$\n\n"
            "The market regression separates the first two; the placebo and jackknife attack the "
            "third. **Beta** is free (an index fund sells it for basis points), so it cannot be a "
            "*premium*. **Survivorship** is the subtle one: our basket is the founder firms that "
            "*won and are therefore remembered*, so even a genuinely zero founder effect would show a "
            "positive raw spread. Because that bias points **for** H₂, an insignificant $\\alpha$ is "
            "a *conservative* result — the true effect is, if anything, weaker than what a "
            "survivor-tilted sample reports."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Baskets.** Hardcoded founder ({R['n_founder']} names, {R['n_founder_priced']} "
            f"priced — {', '.join(R['dropped'])} delisted/reticker) vs professional ({R['n_pro']} "
            f"names), membership frozen at **{R['data_start']}**, fingerprint `{R['fingerprint']}`, "
            f"as-of {R['asof']}.\n"
            f"- **Returns.** yfinance monthly adjusted closes (total-return proxy), "
            f"**{R['n_months']} months** {R['data_start']}..{R['data_end']}; equal weight, "
            "monthly rebalance; a delisted name drops from that month's average.\n"
            "- **Market model.** $r^{LS} = \\alpha + \\beta\\,r^{SPY} + \\varepsilon$; **Newey-West** "
            "HAC SE on $\\alpha$ (Bartlett kernel, rule-of-thumb lags). Also the long-only founder and "
            "professional alphas.\n"
            "- **Decomposition.** Split the raw spread into $\\alpha$ vs $\\beta\\,\\mathbb{E}[r^m]$.\n"
            "- **Jackknife.** Drop each founder name; recompute $\\alpha$ — load-bearing-name test.\n"
            "- **Placebo.** Random $k_F$-long / $k_P$-short relabelling of the *pooled* names; "
            "$p = \\Pr[|\\text{random }\\alpha| \\ge |\\text{founder }\\alpha|]$ — isolates the "
            "founder *tag* from membership luck (it does **not** undo pool-level survivorship, which "
            "is named separately).\n"
            "- **Costs.** One-way turnover both legs + short borrow on the professional leg; gross vs "
            "net.\n"
            "- **Positive control.** Deterministic two-basket panel with a **plantable founder "
            "alpha**: recover a large plant, stay insignificant at zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The raw spread and its beta — a high-octane long vs a sleepy short\n\n"
            "Long-only CAPM fits for each leg. The founder basket carries a **much higher beta**; the "
            "professional basket a low one. The long/short therefore inherits a large *net* beta — "
            "before any alpha is discussed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.basket_returns(RETS, data.FOUNDER_TICKERS).reindex(LS.index).to_numpy()\n"
            "    pr = st.basket_returns(RETS, data.PRO_TICKERS).reindex(LS.index).to_numpy()\n"
            "    m = MKT.to_numpy()\n"
            "    fc = st.capm_alpha(fr, m); pc = st.capm_alpha(pr, m); lc = st.capm_alpha(LS['ls'].to_numpy(), m)\n"
            "else:\n"
            "    fc={'beta':R['founder_beta'],'alpha_bps':R['founder_alpha_bps'],'t_alpha':R['founder_alpha_t']}\n"
            "    pc={'beta':R['pro_beta'],'alpha_bps':R['pro_alpha_bps'],'t_alpha':R['pro_alpha_t']}\n"
            "    lc={'beta':R['ls_beta'],'alpha_bps':R['ls_alpha_bps'],'t_alpha':R['ls_alpha_t']}\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(11.0,4.3))\n"
            "a1.bar(['founder','professional','long/short'], [fc['beta'],pc['beta'],lc['beta']], color=[GREEN,GREY,AMBER])\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('market beta'); a1.set_title('Founder leg is high-beta; the L/S inherits it')\n"
            "for i,b in enumerate([fc['beta'],pc['beta'],lc['beta']]): a1.annotate(f'{b:.2f}',(i,b),ha='center',va='bottom')\n"
            "a2.bar(['founder','professional','long/short'], [fc['alpha_bps'],pc['alpha_bps'],lc['alpha_bps']], color=[GREEN,GREY,AMBER])\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('CAPM alpha (bps/mo)'); a2.set_title('...and every alpha is insignificant')\n"
            "for i,(al,t) in enumerate([(fc['alpha_bps'],fc['t_alpha']),(pc['alpha_bps'],pc['t_alpha']),(lc['alpha_bps'],lc['t_alpha'])]): a2.annotate(f'{al:.0f}\\n(t={t:+.2f})',(i,al),ha='center',va='bottom' if al>=0 else 'top',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'betas: founder {fc[\"beta\"]:.2f} / pro {pc[\"beta\"]:.2f} / L/S {lc[\"beta\"]:.2f} | L/S alpha {lc[\"alpha_bps\"]:.0f} bps t={lc[\"t_alpha\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: founder beta **{R['founder_beta']:.2f}** vs professional "
            f"**{R['pro_beta']:.2f}** ⇒ the long/short runs a **+{R['ls_beta']:.2f}** net beta. In a "
            "decade-long bull market that beta *is* the outperformance. Both long-only alphas "
            f"(founder *t* = {R['founder_alpha_t']:.2f}, professional *t* = "
            f"{R['pro_alpha_t']:.2f}) and the L/S alpha (*t* = {R['ls_alpha_t']:.2f}) are inside their "
            "error bars."
        ),
        md(
            "### 4b · The decomposition — where the raw spread actually comes from\n\n"
            "The average monthly $r^{LS}$ split into $\\alpha$ and $\\beta\\,\\mathbb{E}[r^m]$. The "
            "beta slice is money you were paid for taking market risk; only the alpha slice could be a "
            "founder premium."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.capm_alpha(LS['ls'].to_numpy(), MKT.to_numpy())\n"
            "    raw = LS['ls'].mean()*1e4; alpha=c['alpha_bps']; beta_part=c['beta']*MKT.mean()*1e4\n"
            "else:\n"
            "    raw=R['ls_bps']; alpha=R['ls_alpha_bps']; beta_part=R['beta_part_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.2))\n"
            "ax.barh(['raw L/S spread'], [raw], color=GREEN)\n"
            "ax.barh(['beta * E[mkt]'], [beta_part], color=GREY)\n"
            "ax.barh(['alpha (residual)'], [alpha], color=AMBER)\n"
            "ax.axvline(0,c='k',lw=.8); ax.set_xlabel('bps / month')\n"
            "ax.set_title(f'{R[\"beta_share_pct\"]}% of the founder premium is beta you can buy for pennies')\n"
            "for i,v in enumerate([raw,beta_part,alpha]): ax.annotate(f'{v:.0f}',(v,i),ha='left' if v>=0 else 'right',va='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'raw {raw:.0f} = beta-part {beta_part:.0f} ({R[\"beta_share_pct\"]}%) + alpha {alpha:.0f} (t={R[\"ls_alpha_t\"]:.2f}, n.s.)')"
        ),
        md(
            f"> 💡 In plain words: of the **{R['ls_bps']:.0f} bps/mo** headline spread, "
            f"**{R['beta_part_bps']:.0f}** is beta and only **{R['ls_alpha_bps']:.0f}** is residual "
            "'alpha' — and that residual can't clear *t* = 2. The famous founder premium is mostly a "
            "sector bet in disguise."
        ),
        md(
            "### 4c · Placebo + jackknife — the residual isn't even label-specific\n\n"
            "Left: shuffle the founder/professional labels across the pooled names $N$ times; the real "
            "founder alpha vs the null. Right: leave-one-out — the founder alpha after dropping each "
            "name. If the effect were a characteristic, neither would move it much."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pool = data.FOUNDER_TICKERS + data.PRO_TICKERS\n"
            "    kf = sum(t in RETS.columns for t in data.FOUNDER_TICKERS); kp = sum(t in RETS.columns for t in data.PRO_TICKERS)\n"
            "    null = st.placebo_alpha_dist(RETS, pool, k_long=kf, k_short=kp, n_draws=3000)\n"
            "    obs = st.capm_alpha(LS['ls'].to_numpy(), MKT.to_numpy())['alpha_bps']; pval = st.placebo_pvalue(obs, null)\n"
            "    jk = st.jackknife_alpha(RETS, data.FOUNDER_TICKERS, data.PRO_TICKERS); jn=jk['dropped'].tolist(); ja=jk['alpha_bps'].tolist()\n"
            "else:\n"
            "    rng=np.random.default_rng(747); null=rng.normal(R['placebo_null_mean'],55,3000); obs=R['placebo_obs']; pval=R['placebo_p']\n"
            "    jn=[j[0] for j in R['jk']]; ja=[j[1] for j in R['jk']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.2,4.3))\n"
            "a1.hist(null,bins=45,color=GREY,alpha=.85,label='random label')\n"
            "a1.axvline(obs,c=GREEN,lw=2.5,label=f'founder tag {obs:+.0f} bps')\n"
            "a1.axvline(0,c='k',lw=.8); a1.set_xlabel('L/S alpha (bps/mo)'); a1.set_ylabel('freq')\n"
            "a1.set_title(f'Placebo p = {pval:.2f}'); a1.legend(fontsize=8)\n"
            "cols=[RED if n in ('NVDA','TSLA','SHOP') else GREEN for n in jn]\n"
            "a2.barh(range(len(jn)), ja, color=cols); a2.axvline(R['ls_alpha_bps'],c='k',ls='--',lw=1,label='keep all')\n"
            "a2.set_yticks(range(len(jn))); a2.set_yticklabels(jn,fontsize=7); a2.invert_yaxis()\n"
            "a2.set_xlabel('alpha after dropping (bps/mo)'); a2.set_title('Drop NVDA -> half the alpha'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p={pval:.3f}; jackknife range {min(ja):.0f}..{max(ja):.0f} bps (full {R[\"ls_alpha_bps\"]:.0f})')"
        ),
        md(
            f"> 💡 In plain words: the placebo *p* is **{R['placebo_p']:.2f}** — a *random* long/short "
            f"split of these same names reaches the founder alpha **{R['placebo_beats_pct']}%** of the "
            f"time, so the founder *label* is not doing the work. And the jackknife swings from "
            f"**{R['jk'][0][1]:.0f}** (drop NVDA) to **{R['jk'][-1][1]:.0f} bps** (drop the dud GPRO): "
            "the residual is a concentration artefact, red bars (winners) load-bearing, green duds "
            "*subtracting*."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic two-basket panel (higher founder beta, matched idiosyncratic noise). With "
            "a **zero** planted founder alpha the L/S alpha must stay below *t* = 2; with a **+200 "
            "bps/mo** plant it must light up. Both hold — the engine separates alpha from beta and "
            "won't fabricate an edge."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0, 200.0):\n"
            "    syn = data.synthetic_baskets(alpha_bps=edge, seed=747)\n"
            "    c = st.capm_alpha(syn['ls'], syn['mkt']); res.append((edge, c['alpha_bps'], c['t_alpha'], c['beta']))\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "labs=['planted 0 bps\\n(null)','planted +200 bps\\n(large)']; ts=[r[2] for r in res]\n"
            "ax.bar(labs, ts, color=[GREY,GREEN], width=.5)\n"
            "ax.axhline(2,ls='--',c=RED,label='t = 2'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('L/S alpha Newey-West t'); ax.set_title('Control: no false positive at 0, clean recovery at +200 bps'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,a,t,b in res: print(f'planted {e:+.0f} bps: alpha {a:+.1f} bps  t={t:+.2f}  beta={b:.2f}')"
        ),
        md(
            f"> 💡 In plain words: at a **zero** plant the control alpha *t* is **{R['syn'][0][2]:.2f}** "
            f"(no false positive); a **+200 bps/mo** plant gives *t* = **{R['syn'][1][2]:.2f}** (clean "
            f"recovery). So the machinery is honest — and the real-tape alpha *t* of "
            f"**{R['ls_alpha_t']:.2f}** is exactly what *no founder premium* looks like through this "
            "lens, even with survivorship tilting the sample in the claim's favour."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — L/S CAPM alpha **+{R['ls_alpha_bps']:.0f} bps/mo**, Newey-West "
            f"*t* = **{R['ls_alpha_t']:.2f}**; **{R['beta_share_pct']}%** of the raw "
            f"**{R['ls_bps']:.0f} bps** spread is beta; placebo **p = {R['placebo_p']:.2f}**; drop-NVDA "
            f"alpha **{R['jk'][0][1]:.0f} bps**. Literature support + a survivor-tilted sample that "
            "*still* can't clear *t* = 2 ⇒ NONE, not even WEAK.\n"
            f"- **Tradability `MIRAGE`** — the basket is hindsight-selected (formed from the answer "
            f"key); {', '.join(R['dropped'])} delisted; net **{R['net_bps']:.0f} bps** is mostly free "
            "beta plus a one-name residual. Nothing harvestable ex-ante at any size.\n"
            "- **Founder premium? `MISATTRIBUTED`** — the raw outperformance is real and "
            "**mis-explained**: survivorship + tech-sector beta concentrated in ~2 names, not a "
            "leadership characteristic."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the survivorship arrow\n\n"
            "The operational truth: our sample is biased **toward** the claim, and it *still* fails. "
            "The chart makes the logic explicit — the raw spread is big, but each honest correction "
            "(remove beta, isolate the label, drop the top name) walks it toward zero, and every "
            "correction we *couldn't* apply (the founder firms that delisted before 2024) would push "
            "it **further** down."
        ),
        code(
            "steps=['raw\\nspread','minus\\nbeta','drop\\nNVDA','placebo\\nnull mean']\n"
            "vals=[R['ls_bps'], R['ls_alpha_bps'], R['jk'][0][1], R['placebo_null_mean']]\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "cols=[GREEN,AMBER,AMBER,GREY]\n"
            "ax.bar(steps, vals, color=cols)\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('L/S return / alpha (bps/mo)')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.0f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.annotate('and the founder firms that\\ndelisted would drag it LOWER', (2.5, R['ls_bps']*0.6), ha='center', color=RED, fontsize=8.5)\n"
            "ax.set_title('Every honest correction walks the premium toward zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('raw', R['ls_bps'], '-> minus beta', R['ls_alpha_bps'], '-> drop NVDA', R['jk'][0][1], '-> random label ~', R['placebo_null_mean'])"
        ),
        md(
            "> 💡 In plain words: a signal that only exists **before** you correct for the obvious "
            "confounds, on a sample **hand-picked from the winners**, is the textbook shape of a "
            "non-effect. The founder premium isn't destroyed by costs (net ≈ gross); it's destroyed by "
            "**asking where it came from.** You cannot form this basket without already knowing the "
            "answer — and the professional-CEO firms that quietly compounded (or the founder firms that "
            "died) never get a vote."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Survivorship-clean replication.** Build the founder set from a *contemporaneous* 2016 "
            "filing list (à la Fahlenbrach's hand-collection), winners and losers alike, and rerun. "
            "The tech beta persists; the alpha almost certainly doesn't. The gap between that number "
            "and ours *is* the survivorship bias.\n"
            "- **Sector-neutral pairs.** Match each founder name to a same-GICS professional peer so "
            "the short leg cancels the beta; test the residual with the same HAC *t*.\n"
            "- **Governance angle.** Split founders by voting control (dual-class) vs economic-only — "
            "is any residual a *control* premium/discount rather than 'founder magic'?\n"
            "- **Sibling study.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): the same "
            "abnormal-return + HAC + synthetic-control machinery on *replacing* a CEO — another "
            "small-sample corporate-leadership mirage.\n\n"
            "*The reproducible core is offline and deterministic; the basket is an explicit hardcoded, "
            "hindsight-labelled stand-in. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
