"""Generate the two narrative notebooks for Study 929 (Rights Offering).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
**synthetic** control, and they are always labelled as synthetic — never under a
real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. 39 US rights offerings by 20
# issuers, market-model CARs vs SPY, total-return closes, as-of 2026-06-30.
R = dict(
    start="2005-01-03", end="2026-06-30", n_days=5406, fp="9acc9535dfdd",
    n_events=39, n_issuers=20, first_deal="2013-05", last_deal="2023-09",
    # event windows: mean %, cross-sectional t, hit rate, bootstrap CI
    ann_mean=0.27, ann_t=0.34, ann_hit=49, ann_lo=-1.20, ann_hi=1.95,
    run_mean=1.08, run_t=1.20,
    sub_mean=-2.06, sub_t=-2.03, sub_hit=41, sub_lo=-4.07, sub_hi=-0.05,
    post_mean=0.21, post_t=0.25, post_hit=59, post_lo=-1.51, post_hi=1.87,
    # discount band split
    deep_n=18, rest_n=21,
    sub_deep=-2.74, sub_rest=-1.47, sub_gap=-1.27, sub_welch=-0.60, sub_perm=0.530,
    post_deep=-0.09, post_rest=0.48, post_gap=-0.57, post_welch=-0.31, post_perm=0.751,
    # placebo A — 300 random anchors drawn from the WHOLE tape (2005-2026). Flattering:
    # it hands the null the 2008 and 2020 regimes no deal on this list ever saw.
    plc_sub_mean=-0.18, plc_sub_sd=2.03, plc_sub_z=-0.93, plc_sub_p=0.21,
    plc_ann_z=0.31, plc_ann_p=0.69, plc_post_z=0.37, plc_post_p=0.68,
    # placebo B — ERA-MATCHED anchors (2012-01..2024-06). The fair yardstick, and the
    # one the verdict quotes.
    era_plc_sub_sd=1.28, era_plc_sub_z=-1.60, era_plc_sub_p=0.090,
    era_plc_ann_z=0.46, era_plc_ann_p=0.563,
    era_plc_post_z=0.25, era_plc_post_p=0.803,
    # placebo C — CLUSTERED: slide the whole list by one random offset, keeping every
    # gap between deals (so the cross-event dependence survives).
    clu_plc_sub_sd=2.07, clu_plc_sub_z=-0.91, clu_plc_sub_p=0.281,
    clu_plc_ann_z=0.34, clu_plc_ann_p=0.747,
    clu_plc_post_z=0.19, clu_plc_post_p=0.768,
    # jackknives / eras / sweeps on the subscription window
    jk_issuer_best=-2.50, jk_issuer_best_drop="SPE",
    jk_issuer_worst=-1.14, jk_issuer_worst_drop="CRF",
    era_e_n=17, era_e_mean=-2.65, era_e_t=-2.07,
    era_l_n=22, era_l_mean=-1.59, era_l_t=-1.05,
    tt28_mean=-1.07, tt28_t=-1.43, tt40_mean=-2.06, tt40_t=-2.03,
    tt55_mean=-1.43, tt55_t=-1.29,
    jit_lo_mean=-1.18, jit_lo_t=-1.57, jit_hi_mean=-2.30, jit_hi_t=-1.88,
    # tradability — full (+1,+49) span: swallows the mechanical ex-rights drop
    long_sharpe=0.098, long_t=0.44, long_cagr=0.47, long_dd=-39.7,
    long_alpha=0.03, long_alpha_t=0.01, long_beta=0.10,
    long_adv=-0.477, long_adv_t=-2.53, invested=25.3, avg_names=1.4,
    short_sharpe=-0.322, short_t=-1.43, short_cagr=-4.28, short_dd=-73.6,
    short_alpha=-2.60, short_alpha_t=-1.02,
    # tradability — clean (+29,+49) span: ex-rights free, and the strongest positive
    # number in the study (which is still rented beta, not alpha)
    cln_long_sharpe=0.448, cln_long_t=2.03, cln_long_cagr=3.17, cln_long_dd=-22.1,
    cln_long_alpha=2.98, cln_long_alpha_t=1.79, cln_long_beta=0.04,
    cln_long_adv=-0.127, cln_long_adv_t=-2.05, cln_invested=13.5, cln_avg_names=1.1,
    cln_short_sharpe=-0.730, cln_short_t=-3.24, cln_short_dd=-71.4,
    cln_short_alpha=-5.12, cln_short_alpha_t=-3.02,
    spy_sharpe=0.575,
    cost0=0.168, cost10=0.140, cost25=0.098, cost50=0.028, cost100=-0.110,
    borrow0=-0.256, borrow100=-0.278, borrow300=-0.322, borrow800=-0.431,
    cln_borrow0=-0.677, cln_borrow800=-0.818,
    # synthetic control
    syn_ann=-6.62, syn_ann_t=-7.88, syn_sub=-9.42, syn_sub_t=-5.73,
    syn_post=5.81, syn_post_t=4.96,
    syn_null_post=-0.19, syn_null_post_sd=1.40, syn_null_ann=0.03, syn_null_fire=0,
    # sample hygiene
    dropped_no_tape="CUBA, NHF, ENZN, SPLP, TUEM",
)


BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "from rights_offering import data, strategy as st\n"
)


HEADER = f"""# Study 929 — Rights Offering 🎟️

**Is the deep subscription discount on a rights issue a gift, or a dilution warning?**

When a listed company or fund runs a **rights offering**, every existing holder is handed
the right to buy new shares at a price well below the market — 20% or more below, in the
small closed-end funds and BDCs that dominate US rights activity. That looks like free
money. It usually isn't: the rights go out *pro rata*, so a holder who subscribes in full
owns the same slice of a slightly larger pie, and the share price mechanically falls by
about the value of the right. The discount is a bookkeeping choice made to guarantee
take-up, not a transfer.

We test both readings on **{R['n_events']} US rights offerings** by **{R['n_issuers']}
issuers** ({R['first_deal']} → {R['last_deal']}), measuring abnormal returns against
**SPY** around the announcement, across the subscription period and after expiry.

*Real-tape numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`, as-of {R['end']}); the live cells run the fast **synthetic** control and are
labelled as such.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Why a discount you are *given* can't make you richer\n\n"
           "Imagine a fund worth £100 a share, and every holder is offered one new share "
           "at £80 for each two they own. Afterwards the fund holds more cash per share "
           "than before but has more shares outstanding; the price settles near £93. If "
           "you subscribed, you paid £80 for something now worth £93 — but your existing "
           "two shares each fell from £100 to £93. You are exactly where you started.\n\n"
           "The only person the discount actually *costs* is the holder who ignores the "
           "letter: they get the price fall without the cheap share. That is the whole "
           "economics of a rights issue. The interesting empirical question is what is "
           "left over — does the market read the *decision* to raise equity as bad news, "
           "and does a deeper discount signal a worse issuer?\n\n"
           "> 🔬 **For the quants** — the theoretical ex-rights price is the pre-deal "
           "market cap plus the cash raised, divided by the post-deal share count. Yahoo's "
           "`auto_adjust` back-adjusts for splits and dividends but **not** for a rights "
           "issue, so that mechanical fall stays in the tape as a raw price drop. Any "
           "window spanning the ex-rights date therefore mixes news with arithmetic — "
           "which is why our verdict is formed on the announcement and post-expiry "
           "windows, not the subscription window."),
        md("## 2. What the tape says\n\n"
           "Three windows around each deal, measured against SPY (so a general market "
           "move doesn't count as news):"),
        code(
            "R = " + repr(R) + "\n"
            "rows = [('announcement (-1..+5)', R['ann_mean'], R['ann_t'], R['ann_hit']),\n"
            "        ('subscription (+1..+28)', R['sub_mean'], R['sub_t'], R['sub_hit']),\n"
            "        ('post-expiry (+29..+49)', R['post_mean'], R['post_t'], R['post_hit'])]\n"
            "print(f\"{'window':24s}{'mean':>9s}{'t':>9s}{'hit':>8s}\")\n"
            "for name, m, t, h in rows:\n"
            "    print(f'{name:24s}{m:+8.2f}%{t:+9.2f}{h:7d}%')\n"
            "print()\n"
            "print(f\"SPY-relative, {R['n_events']} deals, {R['n_issuers']} issuers, \"\n"
            "      f\"as-of {R['end']}\")"
        ),
        md(f"**The announcement is a non-event.** +{R['ann_mean']:.2f}% with a *t* of "
           f"{R['ann_t']:+.2f} and a {R['ann_hit']}% hit rate — a coin toss. The textbook "
           f"−3% reaction to a seasoned equity issue (Asquith & Mullins, 1986) simply does "
           f"not show up here, which makes sense: these issuers are portfolios of "
           f"marked-to-market securities, so there is little private information for the "
           f"market to be alarmed by.\n\n"
           f"**And nobody is compensated afterwards.** The 20 sessions after the deal "
           f"closes return {R['post_mean']:+.2f}% (*t* = {R['post_t']:+.2f}). If the "
           f"discount were a gift, this is where you would collect it."),
        md("## 3. Do the *deepest* discounts behave differently?\n\n"
           "We hand-labelled each deal `deep`, `moderate` or `shallow`. If a deep discount "
           "were a warning, the deep bucket should drift worse; if it were a gift, better."),
        code(
            "print('post-expiry return, by discount depth')\n"
            "print(f\"  deep     (n={R['deep_n']}): {R['post_deep']:+.2f}%\")\n"
            "print(f\"  the rest (n={R['rest_n']}): {R['post_rest']:+.2f}%\")\n"
            "print(f\"  gap: {R['post_gap']:+.2f}%   -- shuffle the labels at random and you \"\n"
            "      f\"beat that gap {R['post_perm']*100:.0f}% of the time\")"
        ),
        md("Nothing. A random relabelling of the bands reproduces the gap three times out "
           "of four. The depth of the discount carries no information about what happens "
           "next."),
        md(f"## 4. The one number that looked real\n\n"
           f"Across the subscription period the deals lose **{R['sub_mean']:.2f}%** against "
           f"SPY with a *t* of **{R['sub_t']:+.2f}** — nominally significant. Two reasons "
           f"not to believe it.\n\n"
           f"**First**, that window is exactly where the mechanical ex-rights drop lives "
           f"(section 1), so a fall there is arithmetic, not news.\n\n"
           f"**Second**, it is not robust. Drop one issuer's five deals and the *t* falls to "
           f"{R['jk_issuer_worst']:+.2f}; split the sample at 2018 and the recent half gives "
           f"{R['era_l_t']:+.2f}; change the assumed expiry date from 40 to 28 days and it "
           f"is {R['tt28_t']:+.2f}; shift the (month-precision) anchors by a fortnight and "
           f"it never clears −2 with room to spare. A result that needs one issuer, one "
           f"decade and one guessed date to exist is not a finding.\n\n"
           f"> 🔬 **For the quants — and a caution about our own favourite chart.** Re-run "
           f"the study on 300 sets of *random* dates and the mean abnormal return over a "
           f"28-day window wanders with a standard deviation of "
           f"{R['plc_sub_sd']:.2f}%, which would make our result a *z* of "
           f"{R['plc_sub_z']:+.2f} — a nothing. But that placebo draws its dates from the "
           f"whole 2005-2026 tape, including 2008 and 2020, regimes in which none of these "
           f"deals was announced: it is a rigged-easy yardstick. Restrict the draws to the "
           f"years the deals actually happened and the dispersion falls to "
           f"{R['era_plc_sub_sd']:.2f}%, giving *z* = {R['era_plc_sub_z']:+.2f} "
           f"(*p* = {R['era_plc_sub_p']:.2f}) — still not significant, but not the rout the "
           f"first version implied. We quote the harder number."),
        md(f"## 5. Could you trade it either way?\n\n"
           f"Buy every announcing name the day *after* the news and hold 49 sessions, 25 bps "
           f"one-way, everything else in T-bills. Because that holding period straddles the "
           f"ex-rights date — where the price mechanically falls for reasons that are "
           f"arithmetic, not news — we also run the book over the clean stretch *after* the "
           f"deal closes, which is the version that gives the discount its best shot:"),
        code(
            "print(f\"long,  full  window : excess Sharpe {R['long_sharpe']:+.3f}, \"\n"
            "      f\"worst loss {R['long_dd']:.1f}%\")\n"
            "print(f\"long,  clean window : excess Sharpe {R['cln_long_sharpe']:+.3f}, \"\n"
            "      f\"worst loss {R['cln_long_dd']:.1f}%\")\n"
            "print(f\"short, full  window : excess Sharpe {R['short_sharpe']:+.3f}\")\n"
            "print(f\"just holding SPY    : excess Sharpe {R['spy_sharpe']:+.3f}\")\n"
            "print()\n"
            "print('...but strip out the market exposure the long book rents while it is in:')\n"
            "print(f\"   its edge over SPY is {R['cln_long_alpha']:+.2f}%/yr \"\n"
            "      f\"(t = {R['cln_long_alpha_t']:+.2f}) -- inside the noise\")\n"
            "print(f\"the book is live only {R['cln_invested']:.1f}% of days, \"\n"
            "      f\"averaging {R['cln_avg_names']:.1f} names\")"
        ),
        md(f"Neither direction works. On the full window the long book is indistinguishable "
           f"from cash while carrying a 40% drawdown in a handful of illiquid funds. On the "
           f"clean window it *looks* better — Sharpe {R['cln_long_sharpe']:+.3f} — but that "
           f"is the stock market, not the discount: it is long ordinary equity an eighth of "
           f"the time, and once you charge it for that borrowed exposure the edge is "
           f"{R['cln_long_alpha']:+.2f}% a year with a *t* of {R['cln_long_alpha_t']:+.2f}. "
           f"It still loses to simply owning SPY. The short book just loses money, at every "
           f"borrow rate we tried, on both windows."),
        md("## 6. Live check — the machinery does work (offline **synthetic** data)\n\n"
           "Before believing a null result, prove the detector can see something. We plant "
           "a rights effect into a synthetic panel — a drop on announcement, a slide through "
           "the subscription period, a bounce afterwards — and check the same code finds it. "
           "**These are synthetic numbers, not the real tape.**"),
        code(
            BOOT +
            "import numpy as np\n"
            "px, ev, truth = data.synthetic_panel(signal_strength=1.0, seed=929)\n"
            "d = st.synthetic_detect(px, ev)\n"
            "print(f\"SYNTHETIC planted world ({d['n']} deals):\")\n"
            "for w in ('announce', 'subscription', 'post_expiry'):\n"
            "    print(f\"   {w:13s}{d[w+'_mean_pct']:+7.2f}%  (t={d[w+'_t']:+6.2f})\")\n"
            "nulls = {w: [] for w in ('announce', 'subscription', 'post_expiry')}\n"
            "for s in range(8):\n"
            "    p0, e0, _ = data.synthetic_panel(signal_strength=0.0, seed=929 + s)\n"
            "    d0 = st.synthetic_detect(p0, e0)\n"
            "    for w in nulls:\n"
            "        nulls[w].append(d0[w + '_mean_pct'])\n"
            "print('SYNTHETIC null world (effect switched off), averaged over 8 seeds:')\n"
            "for w, vals in nulls.items():\n"
            "    print(f\"   {w:13s}{np.mean(vals):+7.2f}%  (spread {np.std(vals, ddof=1):.2f}%)\")"
        ),
        md("The detector fires hard on a planted effect and, averaged across seeds, sits on "
           "zero when there is none. (Averaging matters: any single null seed can throw a "
           "two-sigma window — which is precisely the trap the real tape sets in section 4.) "
           "So the flat real-tape result is a fact about rights offerings, not a broken "
           "harness."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** No announcement reaction "
           f"({R['ann_mean']:+.2f}%, *t* = {R['ann_t']:+.2f}), no post-expiry compensation "
           f"({R['post_mean']:+.2f}%, *t* = {R['post_t']:+.2f}), no gradient in the depth of "
           f"the discount (permutation *p* = {R['post_perm']:.2f}). The single significant "
           f"number lives in the window contaminated by mechanical dilution, misses the fair "
           f"placebo (*p* = {R['era_plc_sub_p']:.2f}) and does not survive a jackknife, an "
           f"era cut, a timetable sweep or a fortnight of anchor jitter.\n"
           f"- **Tradability — Mirage.** Long excess Sharpe {R['long_sharpe']:+.3f} on the "
           f"full window, {R['cln_long_sharpe']:+.3f} on the clean one — but the clean "
           f"version's edge over the market is {R['cln_long_alpha']:+.2f}%/yr "
           f"(*t* {R['cln_long_alpha_t']:+.2f}), and both lose to SPY's "
           f"{R['spy_sharpe']:+.3f}. The short is negative at every borrow rate.\n"
           f"- **Three honest caveats, and they all cut the same way.** The event list is "
           f"hand-compiled and **month-precision** — which means the book is sometimes long "
           f"*before* the news is public, a look-ahead that can only help it. Five further "
           f"deals ({R['dropped_no_tape']}) have no retrievable tape at all, so the sample "
           f"is **survivor-biased towards deals that ended well**. And the dilution artefact "
           f"sits inside the only negative window. Every bias we know of pushes towards "
           f"finding an effect, and there is still none."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 929 — Rights Offering — the teardown\n\n"
           "Market-model CARs against SPY over three event windows, three placebos "
           "(whole-tape, era-matched and clustered), a leave-one-issuer-out jackknife, a "
           "discount-band permutation test, an anchor jitter, a timetable sweep, the costed "
           "long/short calendar-time book over two holding spans with its beta-adjusted "
           "alpha and a borrow sweep, and the live synthetic control. Every real number is "
           "frozen from `docs/results.md` (Fingerprint `%s`, as-of %s)." % (R["fp"], R["end"])),
        code("R = %r" % (R,)),
        md("## Design\n\n"
           "- **Sample.** %d rights offerings, %d issuers, %s → %s. Hand-compiled anchors, "
           "**month precision** (PROXY). Five candidate deals (%s) have no retrievable "
           "Yahoo tape — the sample is **survivor-only**, named on the Signal axis.\n"
           "- **Abnormal return.** `AR_t = r_i,t − (α_i + β_i r_SPY,t)`, with α, β from OLS "
           "on the `(−250, −31)` trading-day window before the anchor. Nothing inside any "
           "event window enters the estimate.\n"
           "- **Windows.** announcement `(−1, +5)`, run-up `(−20, −2)`, subscription "
           "`(+1, +28)`, post-expiry `(+29, +49)`. The subscription window straddles the "
           "modelled ex-rights date and therefore contains **mechanical, un-adjusted "
           "dilution**; announcement and post-expiry are clean.\n"
           "- **One execution lag, and the hole in it.** Every tradable leg enters at the "
           "close of the session *after* the anchor; the announcement-day return is an "
           "event-study quantity only. But the anchors are month-precision, so for roughly "
           "half the deals the anchor precedes the true press release and the book is long "
           "before the news is public — a **genuine look-ahead channel**, named here. It can "
           "only manufacture a reaction a real-time trader would have missed, so it biases "
           "the study *towards* an effect; the ±10-day jitter sweeps it.\n"
           "- **Total return.** `auto_adjust=True` throughout — vital for CEFs, which "
           "distribute most of their return.\n"
           "- **Assumptions swept.** The rights timetable (ex-rights +10d, expiry +40d) and "
           "the `deep`/`moderate`/`shallow` band labels."
           % (R["n_events"], R["n_issuers"], R["first_deal"], R["last_deal"],
              R["dropped_no_tape"])),
        md("## The event-window table"),
        code(
            "print(f\"run-up       (-20,-2)  mean {R['run_mean']:+6.2f}%  t {R['run_t']:+6.2f}\")\n"
            "print(f\"announcement ( -1,+5)  mean {R['ann_mean']:+6.2f}%  t {R['ann_t']:+6.2f}  \"\n"
            "      f\"hit {R['ann_hit']}%  boot95 [{R['ann_lo']:+.2f}%, {R['ann_hi']:+.2f}%]\")\n"
            "print(f\"subscription ( +1,+28) mean {R['sub_mean']:+6.2f}%  t {R['sub_t']:+6.2f}  \"\n"
            "      f\"hit {R['sub_hit']}%  boot95 [{R['sub_lo']:+.2f}%, {R['sub_hi']:+.2f}%]  <- dilution mixed in\")\n"
            "print(f\"post-expiry  (+29,+49) mean {R['post_mean']:+6.2f}%  t {R['post_t']:+6.2f}  \"\n"
            "      f\"hit {R['post_hit']}%  boot95 [{R['post_lo']:+.2f}%, {R['post_hi']:+.2f}%]\")"
        ),
        md("## Three placebos — and why the flattering one is not the one to quote\n\n"
           "The naive cross-sectional *t* treats %d deals as %d independent draws. Brown & "
           "Warner (1985) and Kolari & Pynnönen (2010) say it is not, so we resample. But a "
           "placebo is itself a modelling choice, and the obvious version is rigged in our "
           "favour:\n\n"
           "- **A — whole-tape anchors.** 300 draws, same tickers, dates uniform over "
           "2005-2026. That range hands the null the GFC and the COVID crash, regimes in "
           "which **no deal on this list was announced**. Its dispersion is inflated and it "
           "makes our observed drift look tame.\n"
           "- **B — era-matched anchors.** Same, but drawn only from 2012-2024, the years "
           "the deals actually happened. **This is the fair yardstick and the one the "
           "verdict quotes.**\n"
           "- **C — clustered.** Slide the *whole list* by one random offset, so every gap "
           "between deals survives and the cross-event dependence (both Cornerstone funds "
           "every September, the Gabelli funds every April) is preserved. This is the one "
           "that actually answers the Brown & Warner objection; the wide range means it "
           "inherits A's regime problem too."
           % (R["n_events"], R["n_events"])),
        code(
            "print(f\"observed subscription CAR: {R['sub_mean']:+.2f}%  \"\n"
            "      f\"(naive parametric SE {abs(R['sub_mean']/R['sub_t']):.2f}%)\\n\")\n"
            "print(f\"{'placebo':28s}{'sd':>7s}{'z':>8s}{'p':>8s}\")\n"
            "print(f\"{'A whole tape 2005-2026':28s}{R['plc_sub_sd']:6.2f}%\"\n"
            "      f\"{R['plc_sub_z']:+8.2f}{R['plc_sub_p']:8.3f}   <- flattering\")\n"
            "print(f\"{'B era-matched 2012-2024':28s}{R['era_plc_sub_sd']:6.2f}%\"\n"
            "      f\"{R['era_plc_sub_z']:+8.2f}{R['era_plc_sub_p']:8.3f}   <- quote this one\")\n"
            "print(f\"{'C clustered common shift':28s}{R['clu_plc_sub_sd']:6.2f}%\"\n"
            "      f\"{R['clu_plc_sub_z']:+8.2f}{R['clu_plc_sub_p']:8.3f}\")\n"
            "print()\n"
            "print(f\"announcement, era-matched : z {R['era_plc_ann_z']:+.2f}, \"\n"
            "      f\"p {R['era_plc_ann_p']:.3f}\")\n"
            "print(f\"post-expiry,  era-matched : z {R['era_plc_post_z']:+.2f}, \"\n"
            "      f\"p {R['era_plc_post_p']:.3f}\")\n"
            "print()\n"
            "print(f\"-> the fair placebo still does not clear the drift \"\n"
            "      f\"(p = {R['era_plc_sub_p']:.2f}), but it is 0.09, not 0.21:\")\n"
            "print('   the placebo is a supporting witness here, not the case. The case is')\n"
            "print('   that the drift needs one issuer, one era and one guessed date to exist.')"
        ),
        md("## Robustness on the subscription window\n\n"
           "Leave-one-issuer-out (the list has %d deals from %d issuers), the era cut, the "
           "assumed timetable, and the ±10 calendar-day anchor jitter."
           % (R["n_events"], R["n_issuers"])),
        code(
            "print(f\"leave-one-issuer-out: t from {R['jk_issuer_best']:+.2f} (drop {R['jk_issuer_best_drop']}) \"\n"
            "      f\"to {R['jk_issuer_worst']:+.2f} (drop {R['jk_issuer_worst_drop']})\")\n"
            "print(f\"era 2013-2017 (n={R['era_e_n']}): {R['era_e_mean']:+.2f}% (t={R['era_e_t']:+.2f})   \"\n"
            "      f\"era 2018-2023 (n={R['era_l_n']}): {R['era_l_mean']:+.2f}% (t={R['era_l_t']:+.2f})\")\n"
            "print(f\"timetable expiry 28d {R['tt28_mean']:+.2f}% (t={R['tt28_t']:+.2f})  \"\n"
            "      f\"40d {R['tt40_mean']:+.2f}% (t={R['tt40_t']:+.2f})  \"\n"
            "      f\"55d {R['tt55_mean']:+.2f}% (t={R['tt55_t']:+.2f})\")\n"
            "print(f\"anchor jitter +/-10d: mean spans {R['jit_hi_mean']:+.2f}% to {R['jit_lo_mean']:+.2f}%, \"\n"
            "      f\"t spans {R['jit_hi_t']:+.2f} to {R['jit_lo_t']:+.2f} -> never robustly past |2|\")"
        ),
        md("## Is the discount compensated? Band split + label permutation"),
        code(
            "print(f\"subscription: deep (n={R['deep_n']}) {R['sub_deep']:+.2f}%  vs rest (n={R['rest_n']}) \"\n"
            "      f\"{R['sub_rest']:+.2f}%  gap {R['sub_gap']:+.2f}%  Welch t {R['sub_welch']:+.2f}  \"\n"
            "      f\"perm p {R['sub_perm']:.3f}\")\n"
            "print(f\"post-expiry : deep {R['post_deep']:+.2f}%  vs rest {R['post_rest']:+.2f}%  \"\n"
            "      f\"gap {R['post_gap']:+.2f}%  Welch t {R['post_welch']:+.2f}  perm p {R['post_perm']:.3f}\")\n"
            "print('\\n-> the band is an ASSUMPTION and it buys us nothing: a random relabelling')\n"
            "print('   reproduces both gaps more than half the time.')"
        ),
        md("## Tradability — calendar-time book, excess-of-cash on both legs, one lag\n\n"
           "Equal-weight every name inside the holding span, 25 bps one-way × NAV on entry "
           "and exit, flat days in BIL. Two spans, because the obvious one is contaminated:\n\n"
           "- **full `(+1, +49)`** — spans the modelled ex-rights date, so the long leg eats "
           "a price fall a real subscriber was compensated for and the short leg pockets it. "
           "It **understates** the long book.\n"
           "- **clean `(+29, +49)`** — after the modelled expiry, no dilution artefact. This "
           "is the discount's best case, and we report it as such.\n\n"
           "Both legs are excess-of-cash; the short leg is **credited its cash collateral** "
           "and pays borrow on top (charging borrow while withholding the rebate would be a "
           "one-sided cost). `alpha` is the beta-adjusted intercept vs SPY with a HAC *t* — "
           "the statistic to read, because the book is long-only equity %.1f–%.1f%% of the "
           "time and its **own** *t* mostly prices the beta it rents, while the vs-SPY "
           "Sharpe race is exposure-mismatched the other way."
           % (R["cln_invested"], R["invested"])),
        code(
            "print(f\"{'leg / span':26s}{'exSharpe':>10s}{'own t':>8s}{'alpha/yr':>10s}\"\n"
            "      f\"{'t(a)':>7s}{'beta':>7s}{'DD':>8s}\")\n"
            "for tag, s, t, a, at, b, dd in [\n"
            "    ('long  full  (+1,+49)', R['long_sharpe'], R['long_t'], R['long_alpha'],\n"
            "     R['long_alpha_t'], R['long_beta'], R['long_dd']),\n"
            "    ('long  clean (+29,+49)', R['cln_long_sharpe'], R['cln_long_t'],\n"
            "     R['cln_long_alpha'], R['cln_long_alpha_t'], R['cln_long_beta'], R['cln_long_dd']),\n"
            "    ('short full  (+1,+49)', R['short_sharpe'], R['short_t'], R['short_alpha'],\n"
            "     R['short_alpha_t'], -R['long_beta'], R['short_dd']),\n"
            "    ('short clean (+29,+49)', R['cln_short_sharpe'], R['cln_short_t'],\n"
            "     R['cln_short_alpha'], R['cln_short_alpha_t'], -R['cln_long_beta'],\n"
            "     R['cln_short_dd']),\n"
            "]:\n"
            "    print(f'{tag:26s}{s:+10.3f}{t:+8.2f}{a:+9.2f}%{at:+7.2f}{b:+7.2f}{dd:+8.1f}')\n"
            "print(f\"{'SPY (excess of cash)':26s}{R['spy_sharpe']:+10.3f}\")\n"
            "print(f\"\\ncost sweep (long full, one-way bps): 0 {R['cost0']:+.3f} | 10 {R['cost10']:+.3f} | \"\n"
            "      f\"25 {R['cost25']:+.3f} | 50 {R['cost50']:+.3f} | 100 {R['cost100']:+.3f}\")\n"
            "print(f\"borrow sweep (short full, bps/yr) : 0 {R['borrow0']:+.3f} | 100 {R['borrow100']:+.3f} | \"\n"
            "      f\"300 {R['borrow300']:+.3f} | 800 {R['borrow800']:+.3f}\")\n"
            "print(f\"borrow sweep (short clean, bps/yr): 0 {R['cln_borrow0']:+.3f} \"\n"
            "      f\"... 800 {R['cln_borrow800']:+.3f}\")\n"
            "print('\\n-> the clean long book is the strongest positive in this study and it is')\n"
            "print('   still not an edge: Sharpe +0.448 is rented beta, alpha +2.98%/yr t=+1.79,')\n"
            "print('   on a window chosen AFTER seeing the full span was contaminated, and it')\n"
            "print('   loses the race to simply owning SPY.')"
        ),
        md("## Live synthetic control — the harness is unbiased\n\n"
           "**Synthetic data, not the real tape.** Planted world: an announcement drop, a "
           "subscription slide and a post-expiry bounce (1.5× for the deep band) that the "
           "event study must recover. Null world: identical anchors, effect switched off — "
           "the study must stay quiet, and must stay quiet *across seeds*, since any single "
           "seed can throw a 2-sigma window."),
        code(
            BOOT +
            "import numpy as np\n"
            "px, ev, truth = data.synthetic_panel(signal_strength=1.0, seed=929)\n"
            "d = st.synthetic_detect(px, ev)\n"
            "print('planted (n=%d): announce %+.2f%% (t=%+.2f)  subscription %+.2f%% (t=%+.2f)  '\n"
            "      'post-expiry %+.2f%% (t=%+.2f)'\n"
            "      % (d['n'], d['announce_mean_pct'], d['announce_t'],\n"
            "         d['subscription_mean_pct'], d['subscription_t'],\n"
            "         d['post_expiry_mean_pct'], d['post_expiry_t']))\n"
            "post, ann, ts = [], [], []\n"
            "for s in range(12):\n"
            "    p0, e0, _ = data.synthetic_panel(signal_strength=0.0, seed=929 + s)\n"
            "    d0 = st.synthetic_detect(p0, e0)\n"
            "    post.append(d0['post_expiry_mean_pct']); ann.append(d0['announce_mean_pct'])\n"
            "    ts.append(d0['post_expiry_t'])\n"
            "post, ann, ts = np.array(post), np.array(ann), np.array(ts)\n"
            "print('null x12: post-expiry mean %+.2f%% (sd %.2f), announce mean %+.2f%%, '\n"
            "      '|t|>=2 in %d/12' % (post.mean(), post.std(ddof=1), ann.mean(), (abs(ts) >= 2).sum()))"
        ),
        md("## Discount-band recovery on the planted world\n\n"
           "**Synthetic.** Deep-band names carry 1.5× the planted effect, so the same split "
           "that finds nothing on the real tape must find something here."),
        code(
            BOOT +
            "px, ev, _ = data.synthetic_panel(signal_strength=1.0, seed=929)\n"
            "panel = st.event_panel(px, ev)\n"
            "sp = st.discount_split(panel, 'post_expiry')\n"
            "pm = st.permutation_discount_test(panel, 'post_expiry', n_perm=2000, seed=929)\n"
            "print('SYNTHETIC deep %+.2f%% vs rest %+.2f%%  gap %+.2f%% (Welch t %+.2f), perm p %.3f'\n"
            "      % (sp['mean_deep_pct'], sp['mean_rest_pct'], sp['diff_pct'],\n"
            "         sp['welch_t'], pm['p_two_sided']))\n"
            "print('-> the split works when there is something to find; on the real tape p = %.2f'\n"
            "      % 0.751)"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Announcement {R['ann_mean']:+.2f}% "
           f"(*t* = {R['ann_t']:+.2f}, bootstrap [{R['ann_lo']:+.2f}%, {R['ann_hi']:+.2f}%]); "
           f"post-expiry {R['post_mean']:+.2f}% (*t* = {R['post_t']:+.2f}, "
           f"[{R['post_lo']:+.2f}%, {R['post_hi']:+.2f}%]); discount-band gap "
           f"{R['post_gap']:+.2f}% with permutation *p* = {R['post_perm']:.2f}. The lone "
           f"|*t*| ≥ 2 — the subscription window's {R['sub_mean']:+.2f}% "
           f"(*t* = {R['sub_t']:+.2f}) — sits in the window carrying **mechanical, "
           f"un-adjusted ex-rights dilution** and fails every robustness cut: the fair "
           f"era-matched placebo *z* = {R['era_plc_sub_z']:+.2f} "
           f"(*p* = {R['era_plc_sub_p']:.2f}; the flattering whole-tape version says "
           f"{R['plc_sub_p']:.2f} and we do not lean on it), leave-one-issuer-out "
           f"*t* → {R['jk_issuer_worst']:+.2f}, recent era *t* = {R['era_l_t']:+.2f}, "
           f"timetable *t* ∈ [{R['tt28_t']:+.2f}, {R['tt40_t']:+.2f}], anchor jitter never "
           f"clearing |2| with room. The synthetic control recovers a planted effect "
           f"(announce {R['syn_ann']:+.2f}%, *t* = {R['syn_ann_t']:+.2f}) and is centred on "
           f"the null ({R['syn_null_post']:+.2f}%, sd {R['syn_null_post_sd']:.2f}, "
           f"{R['syn_null_fire']}/12 firings), so the silence is real.\n"
           f"- **Tradability — Mirage.** On the full span the long book earns excess Sharpe "
           f"{R['long_sharpe']:+.3f} with a beta-adjusted alpha of {R['long_alpha']:+.2f}%/yr "
           f"(HAC *t* {R['long_alpha_t']:+.2f}) — and that span is penalised by the dilution "
           f"artefact. Give the discount its best case on the clean "
           f"`(+29,+49)` span and the Sharpe is {R['cln_long_sharpe']:+.3f}, but the alpha "
           f"is {R['cln_long_alpha']:+.2f}%/yr (HAC *t* {R['cln_long_alpha_t']:+.2f}) on a "
           f"book live {R['cln_invested']:.1f}% of days averaging {R['cln_avg_names']:.1f} "
           f"names through a {R['cln_long_dd']:.1f}% drawdown: rented beta, not edge, and "
           f"still behind SPY's {R['spy_sharpe']:+.3f}. The short leg is negative on both "
           f"spans at every borrow rate ({R['borrow0']:+.3f} to {R['borrow800']:+.3f} full, "
           f"{R['cln_borrow0']:+.3f} to {R['cln_borrow800']:+.3f} clean).\n"
           f"- **Limits, stated plainly.** A month-precision, hand-compiled, "
           f"survivor-only list of {R['n_events']} deals from {R['n_issuers']} issuers can "
           f"rule out a *large* rights-offering effect; it cannot rule out a small one. "
           f"Month precision also means the book is sometimes long *before* the press "
           f"release — a look-ahead we name rather than hide. All three biases we know of "
           f"(that look-ahead, survivorship, and the dilution contaminating the only "
           f"negative window) push **towards** finding an effect, and we still find none."),
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
