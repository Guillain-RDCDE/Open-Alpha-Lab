"""Generate the two narrative notebooks for Study 545 (IPO-Birthday).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached prices under ../_cache/ if
present and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; event window
# [-5,+5]; 30-name basket, 29 active; 312 firm-year events; prices fp 19fee811b30d, cars fp
# 73d3051eb6f4).
R = dict(
    fp_px="19fee811b30d", fp_ipos="d7888035240a", fp_cars="73d3051eb6f4",
    n_events=312, n_names=29, pre=5, post=5,
    mean_car_bps=62.75, t=1.36, se_bps=45.99,
    median_bps=12.0, share_pos=50.6,
    placebo_p=0.648, placebo_null_bps=86.23, placebo_null_sd_bps=68.0,
    gross_bps=62.75, net_bps=52.75, cost_bps=5.0,
    # window sweep: (label, n, mean_car_bps, t)
    sweep=[("[-1,+1]", 313, 3.7, 0.15), ("[-3,+3]", 313, 66.0, 1.65),
           ("[-5,+5]", 312, 62.7, 1.36), ("[-10,+10]", 312, 129.0, 1.99)],
    # synthetic control: (bump_bps, mean_car_bps over 25 seeds, mean t over 25 seeds)
    ctrl=[(0.0, -2.4, -0.04), (40.0, 37.6, 0.57), (120.0, 117.6, 1.78), (200.0, 197.6, 2.99)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from ipo_birthday import data, strategy as st

PRE, POST = 5, 5

def load_real():
    \"\"\"Cache-first real abnormal-return frame + IPO dates (empty offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    if px.empty:
        return None, None
    ar = st.abnormal_returns(px)
    return ar, data.ipo_dates()

AR, IPOS = load_real()
HAVE_REAL = AR is not None
print("real IPO-birthday tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({AR.shape[1]} names, {AR.index[0].date()} -> {AR.index[-1].date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The IPO Birthday Effect — do stocks pop on their listing anniversary? 🎂\n"
            "### Testing a piece of market folklore, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Companies celebrate the anniversary of the day they went public — press writes "
            "'one year on the market' retrospectives, analysts run their calendar notes, and the "
            "folklore says all that **attention** should give the stock a little bump around its "
            "**IPO birthday** each year. If true, you could buy just before the anniversary and sell "
            "just after, every year, on any listed company.\n\n"
            "So we test it the honest way: take 30 well-known US IPOs, and for every year each one has "
            "been public, look at how it did around its birthday *versus the market* — the "
            "'abnormal return'. Then we check whether a **random** date does any worse.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "event-study maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do stocks pop on their IPO birthday? | **No.** Over **{R['n_events']} firm-years** the "
            f"average abnormal return in the 11-day window around the anniversary is **+{R['mean_car_bps']} "
            f"basis points** (that's {R['mean_car_bps']/100:.2f}%), with a *t* of just **{R['t']}** — well "
            "below the 'real' bar of 2. |\n"
            f"| Is that tiny bump even real? | **No** — half the birthdays ({R['share_pos']}%) were "
            f"actually *negative*, and the median was a rounding-error **+{R['median_bps']} bps**. |\n"
            f"| Could it be luck? | Worse than luck. A **random** date (not the birthday) did *better* "
            f"on average (+{R['placebo_null_bps']} bps) — so the birthday window is nothing special "
            f"(*p* = {R['placebo_p']}). |\n"
            "| Could you trade it? | **No.** The 'effect' is just that these famous survivors beat the "
            "market in general — you'd see it in *any* window, birthday or not. |\n\n"
            "> The bump you *do* see is a mirage: this basket is winners that are still around in 2026, "
            "so they beat the market on average — and that shows up on random dates just as much as on "
            "birthdays."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A stock earns a little extra around the anniversary of its IPO — a 'birthday "
            "effect' from renewed attention.\"*\n\n"
            "The idea rides on a real behavioural finding: **attention moves prices** (Barber & Odean "
            "2008; Da, Engelberg & Gao 2011). An IPO anniversary is a natural attention magnet — "
            "'one year since the float' headlines, calendar reminders, retrospective coverage. If that "
            "attention nudges buyers in, the stock should drift up a touch around the date."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it were real it would be a rare *free* calendar trade: the anniversary is perfectly "
            "predictable years in advance, and it applies to *every* listed company. But that is "
            "exactly why we should be suspicious — a date everyone can see coming is a date the market "
            "has already priced. The desk has looked at the IPO *launch* pop "
            "([219 IPO-Pop](../../219-ipo-pop/)); here we go after the *recurring birthday*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Pick a birthday.** Each company's IPO date is public. Its 1st, 2nd, 3rd… anniversary "
            "are just that date in later years.\n"
            "2. **Measure the window.** Around each anniversary, add up the stock's return **minus the "
            "market's** (SPY) over an 11-day window (5 days before to 5 after). That's its 'abnormal "
            "return' — how much it beat the market *that week*.\n"
            "3. **Average over everything.** 30 companies × up-to-~29 years of birthdays = hundreds of "
            "events. If birthdays pay, the average should be clearly positive.\n"
            "4. **The catch.** Compare against **random** dates. If random dates do just as well, the "
            "birthday isn't special.\n\n"
            "*Timing:* the window is fixed by the (known) calendar date — nothing about the future "
            "leaks in."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**How big is the average birthday bump, and how does it compare to a random date?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cars = st.event_cars(AR, IPOS, PRE, POST); t = st.car_tstat(cars)\n"
            "    pv = st.placebo_pvalue(AR, IPOS, PRE, POST, n_perm=2000, seed=545)\n"
            "    obs, tt = t['mean_car']*1e4, t['t']\n"
            "    null_m = pv['placebo_mean']*1e4; p = pv['p']; n = t['n']\n"
            "    banner = f'REAL tape ({n} firm-year events, {AR.shape[1]} names)'\n"
            "else:\n"
            f"    obs, tt, null_m, p, n = {R['mean_car_bps']}, {R['t']}, {R['placebo_null_bps']}, {R['placebo_p']}, {R['n_events']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['IPO birthday\\nwindow','RANDOM date\\n(placebo)'], [obs, null_m], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('abnormal return, bps')\n"
            "ax.set_title(f'The random date did BETTER (+{null_m:.0f} vs +{obs:.0f} bps) — birthday isn\\'t special')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'birthday CAR +{obs:.1f} bps (t {tt:+.2f}) vs random +{null_m:.1f} bps -> placebo p {p:.3f}')"
        ),
        md(
            f"There it is: the birthday window earns **+{R['mean_car_bps']} bps** on average — but a "
            f"**random** date earns *more* (**+{R['placebo_null_bps']} bps**). The birthday is not "
            f"special; the placebo *p* = {R['placebo_p']} says it is indistinguishable from noise. Why "
            "positive at all? These are famous survivors — they beat the market in general, so *any* "
            "window looks good."
        ),
        md(
            "**Is the bump on the *day*, or just slow drift?** Widen and narrow the window and watch "
            "the CAR track the *width*:"
        ),
        code(
            "sweep = " + repr([(w[0], w[2]) for w in R["sweep"]]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.window_sweep(AR, IPOS, [(1,1),(3,3),(5,5),(10,10)])\n"
            "    labs = list(tab['window']); vals = list(tab['mean_car_bps'])\n"
            "else:\n"
            "    labs = [s[0] for s in sweep]; vals = [s[1] for s in sweep]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(labs, vals, color=GREY, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean abnormal return, bps')\n"
            "ax.set_title('The bump grows with the WINDOW — a sign of generic drift, not a birthday spike')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean CAR by window (bps):', [round(v,1) for v in vals])"
        ),
        md(
            "The tell: the tight **3-day** window around the actual date is **+3.7 bps** — essentially "
            "zero. The number only grows as we widen the window to 21 days, which just captures more of "
            "the stock's general upward drift. A real *birthday* spike would sit tight on the date; "
            "this doesn't."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Mean birthday CAR **+{R['mean_car_bps']} bps** at *t* {R['t']} "
            f"(median +{R['median_bps']} bps, only {R['share_pos']}% positive), and a random date does "
            "*better* (*p* = "
            f"{R['placebo_p']}). No birthday effect.\n"
            "- **Tradability — Mirage.** The tiny positive average is just these survivors beating the "
            "market in any window; there is no date-specific edge to harvest."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even the un-adjusted average is a fraction of a percent, it isn't statistically real, "
            "and it evaporates the moment you compare it to a random date. Add a round-trip cost each "
            "year and you are paying to collect noise. The birthday is a story, not a signal."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The launch, not the birthday.** [219 IPO-Pop](../../219-ipo-pop/) tackles the "
            "*first-day* pop and the long-run hangover — a different (and better documented) object.\n"
            "- **Other calendar folklore.** [89 Turn-of-the-Month](../../89-turn-of-the-month/), "
            "[95 Holiday-Cheer](../../95-holiday-cheer/), [544 Oyster-R-Months](../../544-oyster-r-months/) "
            "test market-wide calendar dates with the same placebo discipline.\n\n"
            "*Think there's a real birthday effect on some other universe? Fork this, swap the basket, "
            "and show the tight `[-1,+1]` window clearing t = 2 against the random-anchor placebo.*"
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
            "# The IPO Birthday Effect — a quantitative teardown 🔬\n"
            "### Market-adjusted CAR event study · one-sample *t* · random-anchor placebo · window sweep · costs · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run a classic "
            "cumulative-abnormal-return (CAR) event study (Fama-Fisher-Jensen-Roll 1969; MacKinlay "
            "1997) around each firm-year IPO anniversary and test whether the mean CAR beats zero — "
            "and beats a random date.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily prices for a 30-name US IPO "
            f"basket + SPY ({R['n_events']} firm-year events, as-of 2026-06-30, prices fp "
            f"`{R['fp_px']}`, CARs fp `{R['fp_cars']}`); the offline core and tests run on a "
            "deterministic synthetic world. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Mean `[-5,+5]` anniversary CAR **+{R['mean_car_bps']} bps** "
            f"(one-sample *t* **{R['t']}**), median **+{R['median_bps']} bps**, {R['share_pos']}% "
            f"positive; random-anchor placebo null mean **+{R['placebo_null_bps']} bps** (*higher*), "
            f"*p* **{R['placebo_p']}**. |\n"
            f"| **Tradability** | `MIRAGE` | Survivor growth basket; CAR grows with window width "
            f"(generic drift). Gross +{R['gross_bps']} → net +{R['net_bps']} bps (round trip, "
            f"{R['cost_bps']:.0f} bps/side). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this "
            "basket the anniversary CAR is insignificant and *below* the random-date null — there is "
            "no birthday effect."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $AR_{i,\\tau} = r_{i,\\tau} - r_{SPY,\\tau}$ be name $i$'s market-adjusted abnormal "
            "return on trading day $\\tau$, and for firm-year event $e$ let "
            "$CAR_e = \\sum_{\\tau \\in [-5,+5]} AR_{i,\\tau}$ over the window around the IPO "
            "anniversary.\n\n"
            "- **H₁ (the birthday effect).** $\\mathbb{E}[CAR] > 0$ at one-sample *t* ≥ 2.\n"
            "- **H₂ (specificity).** The anniversary CAR exceeds the CAR of *random* anchor dates "
            "(placebo tail).\n"
            "- **H₃ (localisation).** The effect sits on the date — the tight `[-1,+1]` window shows "
            "it, not only wide windows.\n"
            "- **H₄ (net).** H₁ survives the round-trip cost.\n\n"
            "We find **H₁ rejected** (*t* 1.36), **H₂ rejected** (placebo *p* 0.65, null mean *higher*), "
            "**H₃ rejected** (`[-1,+1]` = +3.7 bps, *t* 0.15; CAR grows with width), **H₄ moot**."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why it's a mirage**. Two forces conspire: (a) a **predictable "
            "calendar date** carries no information an efficient market hasn't already priced; and (b) "
            "the basket is a **survivor set of growth IPOs** that beat SPY on average over 1997-2026, "
            "so *every* window — birthday or random — shows a positive market-adjusted return. The "
            "placebo isolates (b): its null mean (+86 bps) *exceeds* the observed anniversary CAR "
            "(+63 bps). That maps to the survivorship lesson in "
            "[219 IPO-Pop](../../219-ipo-pop/), where a few giant survivors inflate every mean."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Abnormal returns.** $AR = r_i - r_{SPY}$ (beta-1 market model — the standard "
            "short-window convention; an estimation-window beta adds noise on a 30-name basket).\n"
            "- **Events.** For each ticker and each anniversary year in-sample (skipping the immediate "
            "post-IPO period), cumulate $AR$ over `[-5,+5]` trading days → one CAR per firm-year.\n"
            "- **Inference.** One-sample *t* on the mean CAR (H0: 0).\n"
            "- **Placebo.** Re-run the identical window on **random non-anniversary dates**, matched to "
            "each name's event count (2000 perms); two-sided tail *p*.\n"
            "- **Robustness.** Window sweep `[-1,+1] … [-10,+10]`.\n"
            "- **Frictions.** Round-trip cost (enter before, exit after): 2 × one-way bps.\n"
            "- **Positive control.** A deterministic synthetic world with a planted anniversary bump "
            "(`bump_bps > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the window is fixed by the historical IPO calendar date — no future data enters. "
            "Day 0 is the first trading day on/after the anniversary; shifting it a day changes the CAR "
            "trivially and no stamp."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The mean CAR — H₁\n\n"
            "The distribution of firm-year CARs and the one-sample *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cars = st.event_cars(AR, IPOS, PRE, POST); t = st.car_tstat(cars)\n"
            "    x = cars['car'].to_numpy()*1e4\n"
            "    obs, tt, med, shpos, n = t['mean_car']*1e4, t['t'], float(np.median(x)), float((x>0).mean()*100), t['n']\n"
            "else:\n"
            f"    rng = np.random.default_rng(0); x = rng.normal({R['mean_car_bps']}, {R['se_bps']}*np.sqrt({R['n_events']}), {R['n_events']})\n"
            f"    obs, tt, med, shpos, n = {R['mean_car_bps']}, {R['t']}, {R['median_bps']}, {R['share_pos']}, {R['n_events']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.4))\n"
            "ax.hist(x, bins=40, color=GREY, alpha=.8)\n"
            "ax.axvline(0, c='k', lw=1); ax.axvline(obs, c=RED, lw=2, label=f'mean +{obs:.0f} bps')\n"
            "ax.set_xlabel('firm-year CAR, bps'); ax.set_ylabel('events')\n"
            "ax.set_title(f'Mean anniversary CAR +{obs:.1f} bps (t {tt:+.2f}) — median +{med:.0f}, {shpos:.0f}% positive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'n {n} | mean {obs:+.2f} bps | t {tt:+.2f} | median {med:+.1f} bps | share+ {shpos:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** Mean CAR **+{R['mean_car_bps']} bps** at *t* "
            f"**{R['t']}** — below 2 — with a near-zero median (**+{R['median_bps']} bps**) and a "
            f"coin-flip **{R['share_pos']}%** positive. No birthday spike."
        ),
        md(
            "### 4b · Specificity — H₂ (vs a random date)\n\n"
            "The random-anchor placebo: mean CAR of 2000 sets of random, non-anniversary anchors."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pv = st.placebo_pvalue(AR, IPOS, PRE, POST, n_perm=2000, seed=545)\n"
            "    obs = st.car_tstat(st.event_cars(AR, IPOS, PRE, POST))['mean_car']*1e4\n"
            "    null_m, null_sd, p = pv['placebo_mean']*1e4, pv['placebo_sd']*1e4, pv['p']\n"
            "else:\n"
            f"    obs, null_m, null_sd, p = {R['mean_car_bps']}, {R['placebo_null_bps']}, {R['placebo_null_sd_bps']}, {R['placebo_p']}\n"
            "xs = np.linspace(null_m-3*null_sd, null_m+3*null_sd, 200)\n"
            "pdf = np.exp(-0.5*((xs-null_m)/null_sd)**2)\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "ax.fill_between(xs, pdf, color=GREY, alpha=.35, label='random-anchor null')\n"
            "ax.axvline(null_m, c=GREY, lw=1.5, ls='--', label=f'null mean +{null_m:.0f} bps')\n"
            "ax.axvline(obs, c=RED, lw=2, label=f'observed +{obs:.0f} bps')\n"
            "ax.set_yticks([]); ax.set_xlabel('mean CAR, bps'); ax.legend()\n"
            "ax.set_title(f'Observed birthday CAR sits BELOW the random-date null (placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed +{obs:.1f} bps | random null +{null_m:.1f} +/- {null_sd:.1f} bps | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected — decisively.** The random-date null mean "
            f"(**+{R['placebo_null_bps']} bps**) is *higher* than the anniversary CAR "
            f"(**+{R['mean_car_bps']} bps**): *p* = {R['placebo_p']}. The birthday window is not "
            "special; the positive number is just these survivors beating the market in any window."
        ),
        md(
            "### 4c · Localisation — H₃ (is it on the date?)\n\n"
            "The window sweep. A real birthday spike is tight; generic drift scales with width."
        ),
        code(
            "sweep = " + repr([(w[0], w[1], w[2], w[3]) for w in R["sweep"]]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.window_sweep(AR, IPOS, [(1,1),(3,3),(5,5),(10,10)])\n"
            "else:\n"
            "    tab = pd.DataFrame(sweep, columns=['window','n','mean_car_bps','t'])\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "ax.bar(range(len(tab)), tab['mean_car_bps'], color=GREY, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab['window'])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean CAR, bps')\n"
            "for i,(v,tv) in enumerate(zip(tab['mean_car_bps'], tab['t'])):\n"
            "    ax.text(i, v+2, f't={tv:+.2f}', ha='center', fontsize=9, color=GREY)\n"
            "ax.set_title('CAR grows with window width — the signature of drift, not a date spike')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** The tight `[-1,+1]` window is **+3.7 bps (_t_ 0.15)** "
            "— nothing on the date itself. The CAR only climbs as the window widens (`[-10,+10]` +129 "
            "bps, *t* 1.99), which is generic drift, and its placebo is non-significant too."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (mean +{R['mean_car_bps']} bps, *t* {R['t']}); H₂ "
            f"rejected (placebo *p* {R['placebo_p']}, null mean higher); H₃ rejected (`[-1,+1]` +3.7 "
            "bps, CAR scales with width).\n"
            f"- **Tradability `MIRAGE`** — survivor growth basket, no date-specific edge; gross "
            f"+{R['gross_bps']} → net +{R['net_bps']} bps."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs\n\n"
            "Enter before the window, exit after: one round trip per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = st.car_tstat(st.event_cars(AR, IPOS, PRE, POST))['mean_car']\n"
            "    gross = m*1e4; net = st.net_mean_car(m, cost_bps=5.0)*1e4\n"
            "else:\n"
            f"    gross, net = {R['gross_bps']}, {R['net_bps']}\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.2))\n"
            "ax.bar(['gross','net\\n(round trip)'], [gross, net], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean CAR, bps')\n"
            "ax.set_title(f'Insignificant before costs (+{gross:.0f} bps), still noise after (+{net:.0f} bps)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{gross:.1f} bps -> net +{net:.1f} bps (5 bps/side round trip)')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the CAR is not "
            "distinguishable from zero (or from a random window) before frictions. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant an anniversary bump "
            "(`bump_bps > 0`) of increasing size and watch the mean CAR and *t* rise — averaged over "
            "25 seeds so no single lucky seed can fake it."
        ),
        code(
            "bumps = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]\n"
            "res = [st.synthetic_mean_t(data, bump_bps=b, n_seeds=25) for b in bumps]\n"
            "cbps = [r['mean_car_bps'] for r in res]; ts = [r['mean_t'] for r in res]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(bumps, cbps, 'o-', c=GREEN, lw=2); a1.plot(bumps, bumps, ls=':', c=GREY, label='planted')\n"
            "a1.set_xlabel('planted bump (bps)'); a1.set_ylabel('recovered mean CAR (bps)'); a1.legend()\n"
            "a1.set_title('Recovers the planted bump almost exactly')\n"
            "a2.plot(bumps, ts, 'o-', c=GREEN, lw=2); a2.axhline(2, ls='--', c=GREY, label='t = 2 bar')\n"
            "a2.axhline(0, c='k', lw=1); a2.set_xlabel('planted bump (bps)'); a2.set_ylabel('mean t (25 seeds)'); a2.legend()\n"
            "a2.set_title('Flat at the null, past t=2 as the bump grows')\n"
            "plt.tight_layout(); plt.show()\n"
            "for b, c, tv in zip(bumps, cbps, ts): print(f'bump {b:5.0f} bps -> CAR {c:+.1f} bps, t {tv:+.2f}')"
        ),
        md(
            "The mean CAR tracks the planted bump almost one-for-one, *t* is ≈ 0 at the null and clears "
            "2 by a ~200 bps bump — so the engine is a faithful detector. The flat real-tape result is "
            "therefore a genuine **absence** of an IPO-birthday effect on this basket, not a broken "
            "test. For the *launch* pop (a different, better-documented object) see "
            "[219 IPO-Pop](../../219-ipo-pop/)."
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
