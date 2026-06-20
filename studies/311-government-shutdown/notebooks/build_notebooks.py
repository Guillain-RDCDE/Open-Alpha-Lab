"""Generate the two narrative notebooks for Study 311 (Government-Shutdown).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the shared
total-return SPY cache if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    fp="fbc1be821974", n_days=8390, n_events=5,
    # Per-horizon: mean %, HAC t, random-date baseline %, excess %, permutation p
    h5_mean=1.76, h5_t=1.65, h5_base=0.24, h5_excess=1.52, h5_p=0.138,
    h20_mean=3.83, h20_t=2.33, h20_base=0.93, h20_excess=2.90, h20_p=0.129,
    h20_ci_lo=-1.19, h20_ci_hi=8.69, h20_win=80,
    h40_mean=6.57, h40_t=3.48, h40_base=1.90, h40_excess=4.67, h40_p=0.081,
    # lag robustness (H=20)
    lag1_mean=2.20, lag1_t=1.90, lag1_win=60,
    # per-shutdown (H=20)
    ev_1995a=6.00, ev_1995b=0.05, ev_2013=4.62, ev_2018a=-3.99, ev_2018b=12.46,
    # synthetic controls (H=20)
    pos_n=40, pos_mean=6.31, pos_t=8.31, pos_excess=5.88, pos_p=0.0000,
    null_n=40, null_mean=0.99, null_t=1.08, null_excess=0.43, null_p=0.5899,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/, _cache/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from government_shutdown import data, strategy as st

def _try_real():
    try:
        bars = data.load_real("SPY", end="2026-05-31")
        return bars, True
    except Exception as exc:
        print("real tape unavailable, falling back to synthetic:", repr(exc))
        # offline fallback: a synthetic tape, clearly bannered where shown
        bars, _, _ = data.synthetic_daily(event_effect=0.0, seed=311)
        return bars, False

BARS, HAVE_REAL = _try_real()
EVENTS = data.shutdown_starts()

def baseline(H, n_draws=20000, seed=311):
    c = BARS["close"].to_numpy(float); N = len(c)
    rng = np.random.default_rng(seed)
    l = rng.integers(0, N - H, size=n_draws)
    return c[l + H] / c[l] - 1.0

print("real total-return SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Government-Shutdown — should you buy the dip every time Washington closes?\n"
            "### A favourite piece of market folklore, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Buy_the_dip_every_time%3F: Not_supported](https://img.shields.io/badge/Buy_the_dip_every_time%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Every time the US government runs out of funding and shuts down, the same advice "
            "makes the rounds: **stocks always bounce back, so buy the dip.** It *feels* true — "
            "2013 bounced, 2018-19 bounced hard. This notebook lines up every modern federal "
            "shutdown, buys SPY on the day each one starts, and asks the only two questions that "
            "matter: *did stocks really rise more than usual, and is five events enough to know?*\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the random-date control "
            "and the permutation test? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks rise after a shutdown starts? | **Usually, yes** — 4 of the last 5 "
            f"shutdowns saw SPY up ~+{R['h20_mean']:.0f}% in the next month. |\n"
            "| Is that *more* than a normal month? | **Not provably.** On a random date SPY is "
            f"already up ~+{R['h20_base']:.0f}% over the same span; the *extra* bit isn't "
            f"statistically significant (p = {R['h20_p']:.2f}). |\n"
            "| Is 5 events enough to trust it? | **No.** An 80% win rate on five trades is "
            "**4 out of 5** — and one of those (Christmas Eve 2018) was a famous one-off. |\n"
            "| So should you buy the dip *every time*? | **There's no edge to bank.** You're being "
            "paid the market's normal drift, not a shutdown bonus. The folklore is real-ish but "
            "unprovable, and rests on the rebounds people remember. |\n\n"
            "> Shutdowns make scary headlines and stocks usually recover — but \"usually recover\" "
            "is what stocks do *all the time*. The shutdown adds nothing you can measure."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Don't panic about the shutdown. Look at the history — the S&P 500 rose during "
            "most past government shutdowns and bounced back fast afterward. Shutdowns are a "
            "buying opportunity.\"*\n\n"
            "— the recurring financial-media take (CNBC, LPL Financial, Reuters wrap-ups) every "
            "time Washington fights over funding\n\n"
            "The idea has a kernel of plausibility: a shutdown is political theatre, not an "
            "economic shock; the government reopens; nothing fundamental changed; so the dip "
            "should reverse. The bet is that you can systematically *harvest* that reversal."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two reasons to care. First, it's a perfect specimen of **event folklore** — a story "
            "stitched together from a handful of memorable episodes, repeated until it sounds like "
            "a law. If we can show *how* five data points get spun into a 'rule', that lesson "
            "transfers to every crisis-buying pitch you'll ever hear. Second, it's a clean test of "
            "the single most common event-study mistake: **forgetting that stocks drift up anyway.** "
            "Over a month, 'buy and hold' looks great after *any* date, not just a shutdown."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Line up every shutdown.** Not just the famous bounces — *all* the modern "
            "funding-gap shutdowns, including the flat and negative ones.\n"
            "2. **Race it against a random date.** Buy SPY on the shutdown day and hold a month; "
            "then do the same thing starting on 20,000 *random* days. If shutdowns are special, "
            "the shutdown return beats the random-date return by a clear margin.\n"
            "3. **Count the sample.** Five events is five events. We let the (wide) error bars and a "
            "permutation test say out loud how little we can actually conclude.\n\n"
            "Tape: SPY total-return daily bars back to 1993."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the five shutdowns themselves.** Here is SPY's return in the 20 trading days "
            "after each modern shutdown started:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = st.event_forward_returns(BARS, EVENTS, horizon=20)\n"
            "    labels = list(data.SHUTDOWNS['name'])\n"
            "    rets = led['ret_gross'].to_numpy()*100\n"
            "else:\n"
            f"    labels = list(data.SHUTDOWNS['name'])\n"
            f"    rets = np.array([{R['ev_1995a']},{R['ev_1995b']},{R['ev_2013']},{R['ev_2018a']},{R['ev_2018b']}])\n"
            "short = ['1995 Nov','1995-96','2013','2018 Jan','2018-19']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "col = [GREEN if r > 0 else RED for r in rets]\n"
            "b = ax.bar(short, rets, color=col, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('SPY +20-session return (%)')\n"
            "ax.set_title('Four of five shutdowns bounced — one (Jan 2018) did not')\n"
            "for bar_, r in zip(b, rets):\n"
            "    ax.annotate(f'{r:+.1f}%', (bar_.get_x()+bar_.get_width()/2,\n"
            "        r + (0.3 if r>=0 else -0.6)), ha='center', va='bottom' if r>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean', f'{rets.mean():+.2f}%', '  win-rate', f'{(rets>0).mean()*100:.0f}%')"
        ),
        md(
            f"Mean **+{R['h20_mean']:.1f}%**, win-rate **{R['h20_win']}%** (4 of 5). Looks great — "
            "until you notice the whole story leans on **+12.5% from the December 2018 shutdown**, "
            "which happened to start two days before the exact bottom of the Christmas-Eve 2018 "
            "sell-off. Pure coincidence with the calendar, not the shutdown."
        ),
        md(
            "**Now the punchline: is +3.8% in a month actually special?** Here's the same "
            "'buy-and-hold-20-days' trade, but starting on 20,000 *random* days instead of "
            "shutdown days:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = st.event_forward_returns(BARS, EVENTS, horizon=20)\n"
            "    sh_mean = led['ret_gross'].mean()*100\n"
            "    base = baseline(20)*100\n"
            "else:\n"
            f"    sh_mean = {R['h20_mean']}\n"
            "    base = baseline(20)*100  # synthetic-tape baseline\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.hist(base, bins=60, color=GREY, alpha=.7, label='any random date (+20 sessions)')\n"
            "ax.axvline(base.mean(), c='k', lw=1.2, ls='--', label=f'random-date mean {base.mean():+.1f}%')\n"
            "ax.axvline(sh_mean, c=GREEN, lw=2.5, label=f'shutdown mean {sh_mean:+.1f}%')\n"
            "ax.set_xlabel('+20-session SPY return (%)'); ax.set_ylabel('count')\n"
            "ax.set_title('The shutdown mean sits well inside the everyday distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'shutdown {sh_mean:+.2f}%   vs random date {base.mean():+.2f}%   excess {sh_mean-base.mean():+.2f}%')"
        ),
        md(
            f"The shutdown mean (+{R['h20_mean']:.1f}%) is higher than the random-date mean "
            f"(+{R['h20_base']:.1f}%), but it sits **comfortably inside** the ordinary spread of "
            "monthly outcomes. The 'extra' return from the shutdown is "
            f"**+{R['h20_excess']:.1f}%** — and with only five events, a permutation test can't "
            f"distinguish that from luck (**p = {R['h20_p']:.2f}**, far from significant). "
            "Most of what you'd 'earn' is just the market drifting up like it always does."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Stocks usually do rise after a shutdown, but no more than after "
            f"any random date once you account for normal drift; the extra bit isn't significant "
            f"(p = {R['h20_p']:.2f}) and rests on **5 events**.\n"
            "- **Tradability — Mirage.** There's no shutdown 'edge' to bank — you'd be collecting "
            "the market's everyday drift and calling it a shutdown trade.\n"
            "- **Buy the dip *every time*? — Not supported.** One of the five was negative, and the "
            "headline is carried by a single Christmas-Eve coincidence. The rule survives on the "
            "rebounds people remember."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs aren't the issue — one round trip per shutdown is nothing. The issue is that "
            "delaying entry by even **one day** guts the result, which is the tell of a non-robust "
            "effect:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    m0 = st.event_forward_returns(BARS, EVENTS, horizon=20, lag=0)['ret_gross'].mean()*100\n"
            "    m1 = st.event_forward_returns(BARS, EVENTS, horizon=20, lag=1)['ret_gross'].mean()*100\n"
            "else:\n"
            f"    m0, m1 = {R['h20_mean']}, {R['lag1_mean']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "b = ax.bar(['Buy on day 0','Buy one day later'], [m0, m1], color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean +20-session return (%)')\n"
            "ax.set_title('One day of delay cuts the \"edge\" by a third')\n"
            "for bar_, m in zip(b, [m0, m1]):\n"
            "    ax.annotate(f'{m:+.1f}%', (bar_.get_x()+bar_.get_width()/2, m+0.15), ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'day 0: {{m0:+.2f}}%   one day later: {{m1:+.2f}}%')"
        ),
        md(
            f"Entering one session later drops the mean from +{R['h20_mean']:.1f}% to "
            f"+{R['lag1_mean']:.1f}% and the win-rate from 80% to {R['lag1_win']}%. A real, robust "
            "effect doesn't evaporate because you bought lunch first. This one does — it's a "
            "story, not a strategy."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The engine works — there just isn't a signal.** The companion notebook plants a "
            "fake post-event bounce in a synthetic tape and shows the same code finds it at "
            "p < 0.0001. So the null result on real data is about the *market*, not a broken test.\n"
            "- **Other event folklore, same trap.** [Study 287 — Easter-Effect](../../287-easter-effect/) "
            "and the pre-holiday family: a real-looking calendar drift that mostly survives only as "
            "the market's normal behaviour.\n"
            "- **Want to revive it?** Try debt-ceiling standoffs, or condition on the *size* of the "
            "pre-shutdown drop — but you'll keep running into the same wall: there just aren't "
            "enough shutdowns to learn anything from.\n\n"
            "*Think the shutdown trade is real? Fork this, add the events you think we missed, and "
            "show the excess over a random date clearing significance — with the sample size stated.*"
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
            "# Government-Shutdown — a quantitative teardown\n"
            "### Real SPY total-return tape · event study · synthetic event-null · "
            "permutation test · block-bootstrap CI · n = 5\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Buy_the_dip_every_time%3F: Not_supported](https://img.shields.io/badge/Buy_the_dip_every_time%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the forward "
            "return after a federal funding-gap shutdown exceeds the unconditional drift, with an "
            "honest random-date benchmark, a permutation test, and a frank accounting of n = 5.\n\n"
            "> **Not investment advice.** Real data: Yahoo total-return SPY back to 1993; as-of "
            "2026-05-31; the offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Naive HAC *t* clears 2 at long H (+{R['h20_t']:.2f} at H=20, "
            f"+{R['h40_t']:.2f} at H=40) **but against zero, not drift**: excess over random dates "
            f"is +{R['h20_excess']:.1f}% at permutation *p* = {R['h20_p']:.2f} (n = {R['n_events']}). |\n"
            f"| **Tradability** | `MIRAGE` | The 'edge' is unconditional drift "
            f"(+{R['h20_base']:.1f}% on a random date); costs irrelevant; no significant excess. |\n"
            f"| **Buy the dip every time?** | `NOT SUPPORTED` | 5 events, one negative "
            f"({R['ev_2018a']:+.1f}%), headline carried by a single +{R['ev_2018b']:.1f}% Christmas-Eve "
            "coincidence; fragile to a +1-session lag. |\n\n"
            "> In plain words: stocks rise after shutdowns about as much as they rise after any "
            "random date — and five events is far too few to claim otherwise. The folk rule is "
            "drift wearing a shutdown costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_i(H)$ be the forward total return of buying SPY at the close of the first "
            "session on/after shutdown $i$ and holding $H$ sessions, and $\\bar r_0(H)$ the "
            "unconditional $H$-session return on a random date.\n\n"
            "- **H₁ (positive).** $\\mathbb{E}[r_i(H)] > 0$.\n"
            "- **H₂ (abnormal).** $\\mathbb{E}[r_i(H)] > \\bar r_0(H)$ — it *beats* the drift "
            "benchmark, not just zero.\n"
            "- **H₃ (robust).** It survives a conservative +1-session entry lag.\n"
            "- **H₄ (large enough sample).** $n$ is sufficient for the *t*/permutation test to "
            "discriminate.\n\n"
            "We find **H₁ supported** (stocks usually rise), **H₂ NOT supported** (excess not "
            "significant), **H₃ rejected** (fragile to lag), **H₄ rejected** (n = 5)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The interesting content is not the binary 'does it work' — it's the **mechanism of the "
            "illusion**. A shutdown event study is the cleanest possible demonstration of the "
            "MacKinlay (1997) point: an 'abnormal return' measured against zero is meaningless when "
            "the asset has a strong unconditional drift. Over 20–40 sessions SPY's drift alone "
            "produces a positive, high-*t* number for *any* entry date. Naming that precisely — and "
            "showing the right null kills it — is worth more than the verdict."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Events.** The hardcoded funding-gap table in `data.SHUTDOWNS` (CRS-sourced), "
            "snapped to the first trading session on/after each start.\n"
            "- **Trade.** Enter at the event close, hold $H \\in \\{5,20,40\\}$, exit at the close; "
            "a +1-session lag variant for the look-ahead check.\n"
            "- **Benchmark (the synthetic event-null).** The same $H$-session return measured "
            "around 20,000 random dates on the *same* tape.\n"
            "- **Inference.** Newey-West HAC *t* on the per-event returns; circular block-bootstrap "
            "CI; a permutation test of the shutdown mean vs the random-date distribution.\n"
            "- **Positive control.** A deterministic synthetic tape with a *planted* post-event "
            "bounce — the engine must recover it, and must find nothing when nothing is planted.\n\n"
            "Tape: SPY total-return daily bars from 1993."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The average path around the event (the CAR curve)\n\n"
            "Cross-event mean cumulative return, rebased to the event bar (t = 0), with a ±1 SE "
            "band. With n = 5 the band is wide — which is the whole point."
        ),
        code(
            "if HAVE_REAL:\n"
            "    car = st.car_curve(BARS, EVENTS, pre=10, post=40)\n"
            "else:\n"
            "    # synthetic fallback (clearly bannered): plant a small bounce so the cell renders\n"
            "    sb, sev, _ = data.synthetic_daily(event_effect=0.001, n_events=5, seed=311)\n"
            "    car = st.car_curve(sb, sev, pre=10, post=40)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(car.index, car['mean']*100, c=GREEN, lw=2)\n"
            "ax.fill_between(car.index, (car['mean']-car['se'])*100, (car['mean']+car['se'])*100,\n"
            "                color=GREEN, alpha=.15)\n"
            "ax.axvline(0, c='k', ls='--', lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading sessions relative to shutdown start'); ax.set_ylabel('mean cum. return (%)')\n"
            "ttl = 'Average SPY path around shutdowns' + ('' if HAVE_REAL else ' [SYNTHETIC fallback]')\n"
            "ax.set_title(ttl + ' — drift up, but a wide n=5 band')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(car.round(4).iloc[[0,10,20,30,40,50]].to_string() if len(car)>=51 else car.round(4).to_string())"
        ),
        md(
            "> 💡 In plain words: the average post-shutdown path drifts up — but the shaded "
            "uncertainty band, built from just five events, is too wide to call the drift a "
            "*shutdown* effect rather than the market's everyday upward grind."
        ),
        md(
            "### 4b · The horizon table — naive *t* vs excess-over-random\n\n"
            "The decisive contrast: the naive HAC *t* (against zero) **clears the bar**, while the "
            "permutation *p* against the random-date null **does not**."
        ),
        code(
            "rows = []\n"
            "for H in [5, 20, 40]:\n"
            "    if HAVE_REAL:\n"
            "        led = st.event_forward_returns(BARS, EVENTS, horizon=H)\n"
            "        sh = led['ret_gross'].to_numpy()\n"
            "        base = baseline(H); s = st.summarize(sh)\n"
            "        obs, p = st.permutation_pvalue(sh, base, n_perm=20000)\n"
            "        rows.append((H, s['n'], s['win_rate']*100, s['mean']*100, s['tstat'], base.mean()*100, obs*100, p))\n"
            "    else:\n"
            "        d = {5:(R5:=None)}; pass\n"
            "if not HAVE_REAL:\n"
            "    rows = [\n"
            f"        (5, {R['n_events']}, 80, {R['h5_mean']}, {R['h5_t']}, {R['h5_base']}, {R['h5_excess']}, {R['h5_p']}),\n"
            f"        (20, {R['n_events']}, {R['h20_win']}, {R['h20_mean']}, {R['h20_t']}, {R['h20_base']}, {R['h20_excess']}, {R['h20_p']}),\n"
            f"        (40, {R['n_events']}, 80, {R['h40_mean']}, {R['h40_t']}, {R['h40_base']}, {R['h40_excess']}, {R['h40_p']}),\n"
            "    ]\n"
            "tab = pd.DataFrame(rows, columns=['H','n','win%','mean%','HAC_t','rand%','excess%','perm_p'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "x = np.arange(len(tab)); w=.38\n"
            "ax.bar(x-w/2, tab['HAC_t'], w, color=GREEN, label='naive HAC t (vs zero)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.bar(x+w/2, tab['perm_p'], w, color=RED, label='permutation p (vs random date)')\n"
            "ax.axhline(2, ls='--', c=GREY); ax2.axhline(0.05, ls=':', c=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'H={h}' for h in tab['H']])\n"
            "ax.set_ylabel('HAC t', color=GREEN); ax2.set_ylabel('permutation p', color=RED)\n"
            "ax.set_title('Naive t clears 2; the honest permutation p never clears 0.05')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: at H=20 the naive *t* is +{R['h20_t']:.2f} (above 2) but the "
            f"permutation *p* against random dates is {R['h20_p']:.2f} (nowhere near 0.05). The *t* "
            "is measuring drift, not a shutdown effect. **H₂ fails.**"
        ),
        md(
            "### 4c · Look-ahead robustness — the +1-session lag\n\n"
            "The shutdown date is calendar-known, so no lag is required; but a robust effect should "
            "not care about a one-day delay. This one does."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r0 = st.summarize(st.event_forward_returns(BARS, EVENTS, horizon=20, lag=0)['ret_gross'].to_numpy())\n"
            "    r1 = st.summarize(st.event_forward_returns(BARS, EVENTS, horizon=20, lag=1)['ret_gross'].to_numpy())\n"
            "    vals=[r0['mean']*100, r1['mean']*100]; ts=[r0['tstat'], r1['tstat']]; wr=[r0['win_rate']*100, r1['win_rate']*100]\n"
            "else:\n"
            f"    vals=[{R['h20_mean']}, {R['lag1_mean']}]; ts=[{R['h20_t']}, {R['lag1_t']}]; wr=[{R['h20_win']}, {R['lag1_win']}]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "b = ax.bar(['lag 0 (canonical)','lag +1 (conservative)'], vals, color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean +20-session return (%)')\n"
            "ax.set_title('Fragile to a one-day entry shift')\n"
            "for bar_, t_, w_ in zip(b, ts, wr):\n"
            "    ax.annotate(f't={t_:+.2f}\\nwin {w_:.0f}%', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()+0.1), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'lag0 mean {vals[0]:+.2f}% t={ts[0]:+.2f}   lag1 mean {vals[1]:+.2f}% t={ts[1]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: a one-session delay drops the mean to +{R['lag1_mean']:.1f}% and "
            f"the *t* to +{R['lag1_t']:.2f} (below 2), win-rate {R['h20_win']}%→{R['lag1_win']}%. "
            "**H₃ rejected** — robust effects don't behave like this."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — naive HAC *t* +{R['h20_t']:.2f} (H=20) / +{R['h40_t']:.2f} "
            f"(H=40) clears 2, but only against zero; excess over random dates +{R['h20_excess']:.1f}% "
            f"at permutation *p* = {R['h20_p']:.2f}, fragile to a +1-session lag (*t* → "
            f"+{R['lag1_t']:.2f}), n = {R['n_events']}. The tape cannot certify it REAL.\n"
            "- **Tradability `MIRAGE`** — the return is unconditional drift "
            f"(+{R['h20_base']:.1f}% on a random date); costs irrelevant; no significant abnormal "
            "return to bank.\n"
            f"- **Buy the dip every time? `NOT SUPPORTED`** — 5 events, one negative "
            f"({R['ev_2018a']:+.1f}%), headline carried by a single +{R['ev_2018b']:.1f}% calendar "
            "coincidence."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs are a red herring\n\n"
            "One round trip per event makes costs negligible. The strategy doesn't die on "
            "frictions — it dies on the absence of a measurable abnormal return. The 'capacity' "
            "question is moot: with ~one shutdown every few years, this could never be more than a "
            "rare tactical overlay, and even that overlay has no proven edge."
        ),
        code(
            "costs = [0, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    net = [st.event_forward_returns(BARS, EVENTS, horizon=20, cost_bps=c)['ret_net'].mean()*100 for c in costs]\n"
            "else:\n"
            f"    net = [{R['h20_mean']}, {R['h20_mean']-0.05:.2f}, {R['h20_mean']-0.10:.2f}]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.0))\n"
            "ax.plot(costs, net, 'o-', c=GREY, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (%)')\n"
            "ax.set_ylim(0, max(net)*1.2); ax.set_title('Costs barely move it — that is NOT good news here')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Cost is irrelevant; the problem is significance, not friction.')"
        ),
        md(
            "> 💡 In plain words: when a strategy is cost-insensitive *and* statistically empty, "
            "the lesson is that there was never anything to trade — costs only matter once you have "
            "an edge to erode."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the engine is a faithful detector\n\n"
            "Is the null result a broken test, or a true absence of signal? Plant a real post-event "
            "bounce in a synthetic tape and confirm the *same* engine recovers it — and finds "
            "nothing when nothing is planted."
        ),
        code(
            "# positive control vs null, on deterministic synthetic tapes\n"
            "out = []\n"
            "for eff, lab, sd in [(0.002,'planted bounce',311), (0.0,'null (no effect)',314)]:\n"
            "    b, ev, _ = data.synthetic_daily(event_effect=eff, n_events=40, seed=sd)\n"
            "    led = st.event_forward_returns(b, ev, horizon=20)\n"
            "    nul = data.synthetic_null_events(b, n_events=400, window=20, seed=sd+1)\n"
            "    nret = st.all_horizon_returns(b, st.snap_to_sessions(b, nul), 20)\n"
            "    s = st.summarize(led['ret_gross'].to_numpy())\n"
            "    obs, p = st.permutation_pvalue(led['ret_gross'].to_numpy(), nret, n_perm=20000)\n"
            "    out.append((lab, s['n'], s['mean']*100, s['tstat'], obs*100, p))\n"
            "dfc = pd.DataFrame(out, columns=['tape','n','mean%','HAC_t','excess%','perm_p'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(dfc['tape'], dfc['HAC_t'], color=[GREEN, GREY], width=.5); a1.axhline(2, ls='--', c=GREY)\n"
            "a1.set_ylabel('HAC t'); a1.set_title('Engine recovers a planted effect...')\n"
            "a2.bar(dfc['tape'], dfc['perm_p'], color=[GREEN, GREY], width=.5); a2.axhline(0.05, ls=':', c=RED)\n"
            "a2.set_ylabel('permutation p'); a2.set_title('...and finds nothing in the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfc.round(4).to_string(index=False))"
        ),
        md(
            f"The harness banks the planted bounce (HAC *t* +{R['pos_t']:.1f}, permutation "
            f"*p* < 0.0001) and correctly finds nothing in the null (*p* = {R['null_p']:.2f}). "
            "Across 20 seeds the null's excess-over-random averages ~0 and ~5% of seeds give "
            "p < 0.05 — the nominal false-positive rate. So the real-data null is a statement "
            "about **the market**: five shutdowns simply don't carry enough signal to clear the "
            "right benchmark.\n\n"
            "For the same illusion in other guises, see "
            "[Study 287 — Easter-Effect](../../287-easter-effect/) and "
            "[Study 301 — Triple-RSI](../../301-triple-rsi/)."
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
