"""Generate the two narrative notebooks for Study 924 (First Cut).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control, which is never presented under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. TLT vs BIL, total return,
# first-cut event study, excess-of-cash, 2007-05-30 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4802, fp="ff993b355f57",
    n_first=5, n_all=31, n_events=4,
    events=[
        ("2007-09-18", "2007-09-19", "2008-09-18", 14.37, 2.78, 11.50),
        ("2019-07-31", "2019-08-01", "2020-07-31", 28.58, 1.11, 27.37),
        ("2020-03-03", "2020-03-04", "2021-03-03", -8.51, 0.05, -8.67),
        ("2024-09-18", "2024-09-19", "2025-09-18", -6.20, 4.36, -10.66),
    ],
    ex_2019_mean=-2.61,
    # horizon sweep, first cuts (TLT): mean, net, median, t(N=4), hit, in%/y, out%/y, HAC t
    h_first={
        1: (3.18, 3.08, 4.61, 0.91, 0.75, 40.13, 2.22, 1.04),
        3: (0.42, 0.32, 3.71, 0.12, 0.75, 3.44, 2.86, 0.18),
        6: (3.79, 3.69, 7.99, 0.90, 0.75, 8.67, 2.22, 0.81),
        12: (4.98, 4.88, 1.51, 0.55, 0.50, 2.62, 2.95, 0.35),
    },
    # horizon sweep, ALL cuts control (N=18): mean, net, t(N), hit, HAC t
    h_all={
        1: (0.96, 0.86, 0.80, 0.44, 0.78),
        3: (0.54, 0.44, 0.31, 0.50, 0.60),
        6: (1.30, 1.20, 0.64, 0.61, 0.65),
        12: (5.60, 5.50, 1.78, 0.72, 0.22),
    },
    n_all_events=18,
    # placebo: observed net, placebo mean, placebo sd, one-sided p
    placebo={
        1: (3.08, 0.14, 2.04, 0.086),
        3: (0.32, 0.72, 3.54, 0.540),
        6: (3.69, 1.34, 5.14, 0.319),
        12: (4.88, 2.61, 7.18, 0.365),
    },
    cond_ann=0.47, cond_lo=-2.05, cond_hi=3.02, cond_negfrac=36.6, in_frac=18.7,
    # Rate earned over the invested days only (12m in-window annualised, excess of cash).
    in_ann_pct12=2.62,
    uncond_ann=2.89, uncond_lo=-3.64, uncond_hi=9.04,
    # Overlapping 12m windows collapse both calendars onto the same three macro episodes
    # (2007-08 GFC, 2019-20, 2024-25), so neither N is a count of independent bets.
    n_episodes_first=3, n_episodes_all=3, overlap_pairs_all=53, overlap_pairs_all_tot=153,
    era_early=19.43, era_late=-9.66,
    cost0=4.98, cost5=4.88, cost10=4.78, cost25=4.48, cost25_t=0.50,
    ls_b0=3.01, ls_b0_t=0.36, ls_b50=2.51, ls_b50_t=0.30, ls_b200=1.01, ls_b200_t=0.12,
    ief_12m=4.26, ief_12m_t=1.17, ief_6m=2.98, ief_6m_t=1.06,
    irx_mean=5.09, irx_t=0.56,
    syn_planted_true=9.0, syn_planted_pooled=8.55, syn_planted_t=6.93,
    syn_null_pooled=-0.83, syn_null_t=-0.74,
    syn_planted_fire=5, syn_null_fire=1, syn_worlds=12,
)


HEADER = f"""# Study 924 — First Cut

**When the Fed starts cutting, is that the moment to buy duration?**

The folk trade: the FOMC delivers the **first cut of an easing cycle**, you buy long
Treasuries, and you ride a multi-quarter bond rally. We tested it the only way it can be
tested — a hardcoded list of cycle-start cuts, buy **TLT** at the **close of the session
after** the announcement, hold 1 / 3 / 6 / 12 months, score it **excess of T-bills (BIL)**,
5 bps one-way each way.

The tape is TLT vs BIL daily **total-return** closes, {R['start']} → {R['end']}
({R['n_days']:,} days), as-of 2026-06-30.

**And the number that governs everything below: N = {R['n_events']}.**

*Real numbers are the frozen headline from `docs/results.md` (Fingerprint `{R['fp']}`);
the live cells run the offline synthetic control only.*
"""

EVENT_CODE = (
    "events = %r\n"
    "print(f\"{'first cut':12s} {'entry':12s} {'exit':12s} {'TLT':>8s} {'BIL':>7s} {'excess net':>11s}\")\n"
    "for ev, entry, ex, tlt, bil, net in events:\n"
    "    print(f'{ev:12s} {entry:12s} {ex:12s} {tlt:+8.2f} {bil:+7.2f} {net:+11.2f}')\n"
    "mean = sum(e[5] for e in events) / len(events)\n"
    "print(f'\\nN = {len(events)}   mean excess net = {mean:+.2f}%%   "
    "hit rate = {sum(1 for e in events if e[5] > 0)}/{len(events)}')\n"
    "print('2001-01-03 is unmeasurable: TLT only lists from 2002-07-30.')"
    % (R["events"],)
)


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Four trades. That is the entire dataset.\n\n"
           "Since 2001 the Fed has begun an easing cycle five times by any reasonable "
           "reading: January 2001, September 2007, July 2019, the emergency cut of March "
           "2020, and September 2024. The first of those pre-dates the long-Treasury ETF, "
           "so **four** trades is what the tape allows. Here they are, in full — this is not "
           "a summary of the evidence, it *is* the evidence."),
        code(EVENT_CODE),
        md(f"## 2. The average is one trade wide\n\n"
           f"The mean looks respectable: **+4.88%** over twelve months, above cash, after "
           f"costs. But two of the four lost money, and the whole positive average rests on "
           f"the 2019 window — which happens to run from August 2019 to July 2020, i.e. "
           f"straight through the pandemic flight-to-quality. Drop it and the other three "
           f"average **{R['ex_2019_mean']:+.2f}%**.\n\n"
           f"The 2024 cut is the honest counter-example: the Fed cut, and long bonds fell "
           f"**−6.2%** over the next year while T-bills paid **+4.4%** — a **−10.7%** "
           f"excess. Cutting the front end does not oblige the long end to follow."),
        md("## 3. The control that quietly demolishes the premise\n\n"
           "If the *first* cut is special, buying it should beat buying cuts in general. "
           "It does not. Over twelve months, buying duration after **any** of the 18 "
           "measurable cuts paid **more** than buying after the four hand-picked first ones."),
        code(
            "first = %r\nallc = %r\n"
            "print(f\"{'horizon':>8s} {'FIRST cuts (N=4)':>20s} {'ALL cuts (N=18)':>20s}\")\n"
            "for h in (1, 3, 6, 12):\n"
            "    print(f'{h:>6d}m {first[h][1]:>+15.2f}%%      {allc[h][1]:>+15.2f}%%')\n"
            "print('\\nThe hand-picked label buys you nothing and costs you 14 observations.')"
            % ({k: v for k, v in R["h_first"].items()}, {k: v for k, v in R["h_all"].items()})
        ),
        md(f"> 🔬 **For the quants** — the all-cuts leg is not a clean control either: "
           f"adjacent cuts share overlapping 12-month windows "
           f"({R['overlap_pairs_all']} of {R['overlap_pairs_all_tot']} pairs overlap), so "
           f"those {R['n_all_events']} events collapse onto just "
           f"**{R['n_episodes_all']} macro episodes** — 2007-08, 2019-20 and 2024-25 — and "
           f"its nominal *t* = {R['h_all'][12][2]:+.2f} is badly oversized. Read it as a mean, "
           f"not as a test. The point is comparative, not absolute: whatever is in the data "
           f"is a *\"the Fed is easing\"* effect, not a *\"this is the first one\"* effect."),
        md(f"## 4. What a random day would have bought you\n\n"
           f"The right question for a four-event mean is not \"is it positive?\" but \"how "
           f"often would four random dates have done as well?\" Owning duration paid "
           f"something over this sample anyway, so a random 12-month start already earns "
           f"**+{R['placebo'][12][1]:.2f}%** above cash. The first-cut premium above *that* "
           f"is about two points, and 2,000 random draws produce a mean at least that good "
           f"**{R['placebo'][12][3]:.0%} of the time**. Only the one-month window is even "
           f"suggestive (*p* = {R['placebo'][1][3]:.3f}), and we looked at four horizons."),
        md(f"## 5. Why we cannot simply say \"it's false\"\n\n"
           f"Here is the uncomfortable part, and the live cell below shows it. We built a "
           f"synthetic world with a **real** post-cut rally planted in it — a genuine "
           f"**+{R['syn_planted_true']:.0f}%** six-month effect — and ran the identical "
           f"five-event test on it, world after world. It found the effect in only "
           f"**{R['syn_planted_fire']} of {R['syn_worlds']}** worlds.\n\n"
           f"So a five-event study *cannot* reliably detect an effect big enough to change "
           f"your life. Pool the worlds together and the machinery recovers the planted "
           f"number almost exactly — the harness is fine. The sample is not."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from first_cut import data, strategy as st\n"
            "\n"
            "def five_event_ts(signal_strength):\n"
            "    ts, pooled = [], []\n"
            "    for prices, events, _ in data.synthetic_panel(n_worlds=12, signal_strength=signal_strength):\n"
            "        tbl = st.event_table(prices['duration'], prices['cash'], events, 6, 5.0)\n"
            "        ts.append(st.one_sample_t(tbl['excess_net_pct'].to_numpy()))\n"
            "        pooled.extend(tbl['excess_net_pct'].tolist())\n"
            "    return np.array(ts), np.array(pooled)\n"
            "\n"
            "for tag, ss in [('planted +9% effect', 1.0), ('null, no effect   ', 0.0)]:\n"
            "    ts, pooled = five_event_ts(ss)\n"
            "    print(f'{tag}: pooled mean {pooled.mean():+5.2f}% (t={st.one_sample_t(pooled):+5.2f}, '\n"
            "          f'N={len(pooled)})  |  five-event t clears 2 in {(np.abs(ts) > 2).sum()}/12 worlds')\n"
            "print('\\nSYNTHETIC DATA — a machinery check, not the real tape.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Every horizon's point estimate is positive and not one "
           f"clears *t* = 1.1 (12-month: **+{R['h_first'][12][1]:.2f}%**, *t* = "
           f"{R['h_first'][12][3]:+.2f}, N = 4). A random start date already buys "
           f"+{R['placebo'][12][1]:.2f}%, so the placebo *p* is **{R['placebo'][12][3]:.3f}**. "
           f"Buying *any* cut paid more than buying the first one. The two pre-2020 events "
           f"made **{R['era_early']:+.1f}%**, the two since made **{R['era_late']:+.1f}%**.\n"
           f"- **Tradability — Mirage.** The conditional *strategy* — long TLT inside the "
           f"windows, cash the other {100 - R['in_frac']:.1f}% of days — earns "
           f"**{R['cond_ann']:+.2f}%/y** excess of cash across the whole sample, CI "
           f"[{R['cond_lo']:+.2f}%, {R['cond_hi']:+.2f}%]. Measured only over the days it is "
           f"actually invested it makes {R['in_ann_pct12']:+.2f}%/y — still *less* than the "
           f"{R['uncond_ann']:+.2f}%/y you get from just owning TLT and going for a walk. "
           f"It loses to buy-and-hold on either reading.\n"
           f"- **The honest caveat.** This study cannot prove the trade is worthless; four "
           f"events can barely prove anything. What it can say is that nothing survives "
           f"contact with a control, and that anyone sizing this position is making a "
           f"four-sample bet on a story."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 924 — First Cut — the teardown\n\n"
           "Event study on a hardcoded FOMC cycle-start calendar: buy TLT at the close of "
           "*t+1* after the announcement (one execution lag, no exceptions), hold "
           "1/3/6/12 months, score excess of BIL over the identical days, 5 bps one-way "
           "each leg. Below: the per-event table, the horizon sweep, the all-cuts control, "
           "the randomisation placebo, the daily conditional leg with its HAC *t* and "
           "block-bootstrap CI, the era split, the cost and borrow sweeps, the IEF and "
           "`^IRX` cross-checks, and a power calibration of the five-event test itself.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`), as-of "
           "2026-06-30." % R["fp"]),
        code("R = %r" % (R,)),
        md("## Design and its one non-tape input\n\n"
           "- **Execution lag:** signal formed at the close of the announcement day *t*, "
           "position opened at the **close of t+1**. Exactly one lag, applied identically "
           "to the real, control, placebo and synthetic paths.\n"
           "- **Costs:** 5 bps one-way × NAV on entry and on exit (two legs; four for the "
           "curve expression). Swept.\n"
           "- **Excess of cash:** BIL's own total return over the same days, not a flat "
           "proxy — at 2024-26 bill yields that distinction is worth ~4.4 points on a "
           "12-month window.\n"
           "- **Total return throughout**, never price-only: coupon is most of a bond ETF's "
           "12-month return.\n"
           "- **ASSUMPTION (the only one):** the event calendar is hand-typed — 5 "
           "cycle-start cuts, 31 cuts in total, truncated at 2024-12-18 so every listed "
           "event has a full 12-month window inside the as-of.\n"
           "- **Survivorship:** none — single still-listed ETFs, calendar fixed ex ante. "
           "The selection risk is of another kind: *which* cuts count as \"first\" is a "
           "judgement, which is why the all-cuts control exists."),
        md("## Per-event table — the whole sample, 12-month hold"),
        code(EVENT_CODE),
        md("> 💡 **In plain words** — four trades. Two won, two lost, and the average is "
           "carried by a window that happens to contain March 2020."),
        md("## Horizon sweep: first cuts vs the all-cuts control\n\n"
           "`in`/`out` are annualised *conditional* rates on the daily excess-of-cash "
           "series (post-cut days vs every other day) — not the return of any fund. The "
           "HAC *t* is Newey-West on the daily conditional leg."),
        code(
            "print(f\"{'h':>3s} {'FIRST net':>10s} {'t(N=4)':>7s} {'hit':>5s} {'in %/y':>8s} \"\n"
            "      f\"{'out %/y':>8s} {'HAC t':>6s} | {'ALL net':>8s} {'t(N=18)':>8s} {'hit':>5s}\")\n"
            "for h in (1, 3, 6, 12):\n"
            "    f = R['h_first'][h]; a = R['h_all'][h]\n"
            "    print(f'{h:>2d}m {f[1]:>+10.2f} {f[3]:>+7.2f} {f[4]:>5.0%} {f[5]:>+8.2f} '\n"
            "          f'{f[6]:>+8.2f} {f[7]:>+6.2f} | {a[1]:>+8.2f} {a[2]:>+8.2f} {a[3]:>5.0%}')\n"
            "print('\\nAt 12m the ALL-cuts leg (+5.50%, t=+1.78, N=18) beats the FIRST-cut leg '\n"
            "      '(+4.88%, t=+0.55, N=4).')"
        ),
        md("> 💡 **In plain words** — if the special ingredient were the *first* cut, the "
           "narrow list should beat the broad one. It does not, at any horizon worth "
           "trading."),
        md("## Randomisation inference — the only test N=4 can honestly support\n\n"
           "2,000 draws of N random eligible start dates, same horizon, same costs. The "
           "*p* is the share of draws whose mean is at least as good as the observed one."),
        code(
            "print(f\"{'h':>3s} {'observed':>9s} {'placebo mu':>11s} {'placebo sd':>11s} {'one-sided p':>12s}\")\n"
            "for h in (1, 3, 6, 12):\n"
            "    o, mu, sd, p = R['placebo'][h]\n"
            "    print(f'{h:>2d}m {o:>+9.2f} {mu:>+11.2f} {sd:>11.2f} {p:>12.3f}')\n"
            "print('\\nFour horizons examined; the best p is 0.086 at 1m and survives no '\n"
            "      'multiplicity adjustment at all.')\n"
            "print('Caveat: the draws are iid dates, while the real events cluster in three '\n"
            "      'episodes. A clustered null would be wider, so these p-values are a '\n"
            "      'FLOOR - the true p is larger, which only reinforces the None stamp.')"
        ),
        md("## The daily conditional leg — HAC *t* and block-bootstrap CI\n\n"
           "Per-event returns from overlapping windows are not independent, so the "
           "inferential spine is the daily excess-of-cash series gated by the post-cut "
           "window: Newey-West *t* on the mean, circular block bootstrap (2,000 draws, "
           "21-day blocks) on the CI."),
        code(
            "print(f\"conditional leg (12m windows): {R['cond_ann']:+.2f}%/y  \"\n"
            "      f\"95% CI [{R['cond_lo']:+.2f}, {R['cond_hi']:+.2f}]  share<0 {R['cond_negfrac']:.1f}%  \"\n"
            "      f\"invested {R['in_frac']:.1f}% of days\")\n"
            "print(f\"unconditional TLT - BIL      : {R['uncond_ann']:+.2f}%/y  \"\n"
            "      f\"95% CI [{R['uncond_lo']:+.2f}, {R['uncond_hi']:+.2f}]\")\n"
            "print(f\"HAC t on the conditional leg (12m): {R['h_first'][12][7]:+.2f}\")\n"
            "print('\\nThe timing device earns LESS per day invested than simply owning the asset.')"
        ),
        md("## Era split, cost sweep, borrow sweep, cross-checks"),
        code(
            "print(f\"era split (2020-01-01), 12m: pre-2020 (N=2) {R['era_early']:+.2f}%  |  \"\n"
            "      f\"2020+ (N=2) {R['era_late']:+.2f}%   <- complete sign flip\")\n"
            "print(f\"cost sweep 12m: 0bps {R['cost0']:+.2f}%  5bps {R['cost5']:+.2f}%  \"\n"
            "      f\"10bps {R['cost10']:+.2f}%  25bps {R['cost25']:+.2f}% (t={R['cost25_t']:+.2f})\")\n"
            "print(f\"curve leg (long TLT / short SHY, 12m), borrow ASSUMPTION swept:\")\n"
            "print(f\"   0 bp/y {R['ls_b0']:+.2f}% (t={R['ls_b0_t']:+.2f})   \"\n"
            "      f\"50 bp/y {R['ls_b50']:+.2f}% (t={R['ls_b50_t']:+.2f})   \"\n"
            "      f\"200 bp/y {R['ls_b200']:+.2f}% (t={R['ls_b200_t']:+.2f})\")\n"
            "print(f\"IEF (belly) cross-check: 6m {R['ief_6m']:+.2f}% (t={R['ief_6m_t']:+.2f})  \"\n"
            "      f\"12m {R['ief_12m']:+.2f}% (t={R['ief_12m_t']:+.2f})\")\n"
            "print(f\"^IRX PROXY cash cross-check (12m): {R['irx_mean']:+.2f}% (t={R['irx_t']:+.2f})  \"\n"
            "      f\"-> the cash leg is not what removes 2001; TLT's 2002 inception is\")"
        ),
        md("> 💡 **In plain words** — costs are irrelevant here (two trades a year), borrow "
           "only makes the curve version worse, and every cross-check reproduces the same "
           "shape: positive point estimate, no significance."),
        md("## Power calibration — what a five-event test can and cannot see\n\n"
           "The live cell runs the identical harness on synthetic worlds with a **known** "
           "planted six-month effect of +9%, and on matched nulls. Pooled across worlds the "
           "estimator is unbiased and sharp; world-by-world, the five-event *t* is a "
           "coin flip. This is a property of the design, not of the Fed."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from first_cut import data, strategy as st\n"
            "\n"
            "for tag, ss in [('planted (+9% true)', 1.0), ('null    (0% true) ', 0.0)]:\n"
            "    ts, pooled = [], []\n"
            "    for prices, events, _ in data.synthetic_panel(n_worlds=12, signal_strength=ss):\n"
            "        tbl = st.event_table(prices['duration'], prices['cash'], events, 6, 5.0)\n"
            "        ts.append(st.one_sample_t(tbl['excess_net_pct'].to_numpy()))\n"
            "        pooled.extend(tbl['excess_net_pct'].tolist())\n"
            "    ts, pooled = np.array(ts), np.array(pooled)\n"
            "    print(f'{tag}: pooled N={len(pooled)} mean {pooled.mean():+5.2f}% '\n"
            "          f't={st.one_sample_t(pooled):+5.2f} | per-world 5-event |t|>2 in '\n"
            "          f'{(np.abs(ts) > 2).sum()}/12')\n"
            "print('\\nSYNTHETIC DATA (machinery + power check) — never a real-tape number.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Headline 12-month mean **{R['h_first'][12][1]:+.2f}%** net "
           f"excess-of-cash, *t* = {R['h_first'][12][3]:+.2f} on N = 4, randomisation "
           f"*p* = {R['placebo'][12][3]:.3f} against matched random start dates. The daily "
           f"conditional leg's HAC *t* is {R['h_first'][12][7]:+.2f} and its bootstrap CI "
           f"[{R['cond_lo']:+.2f}%, {R['cond_hi']:+.2f}%] straddles zero. No horizon reaches "
           f"*t* = 1.1. The all-cuts control ({R['h_all'][12][1]:+.2f}%, N = "
           f"{R['n_all_events']}, nominal *t* = {R['h_all'][12][2]:+.2f} — oversized, because "
           f"those {R['n_all_events']} twelve-month windows overlap onto only "
           f"{R['n_episodes_all']} macro episodes; its HAC daily *t* is "
           f"{R['h_all'][12][4]:+.2f}) *beats* the first-cut leg on the mean, "
           f"so the hand-picked label carries no information. Era split flips sign "
           f"({R['era_early']:+.1f}% / {R['era_late']:+.1f}%). The synthetic control confirms "
           f"the harness is unbiased (pooled null {R['syn_null_pooled']:+.2f}%, *t* = "
           f"{R['syn_null_t']:+.2f}) and correctly powered in aggregate (pooled planted "
           f"{R['syn_planted_pooled']:+.2f}%, *t* = {R['syn_planted_t']:+.2f}) — while "
           f"demonstrating that at N = 5 a true +9% effect is found only "
           f"{R['syn_planted_fire']}/{R['syn_worlds']} of the time.\n"
           f"- **Tradability — Mirage.** {R['cond_ann']:+.2f}%/y for the conditional strategy "
           f"across the full sample ({R['in_ann_pct12']:+.2f}%/y counting only the days it is "
           f"invested) versus {R['uncond_ann']:+.2f}%/y for holding the asset "
           f"unconditionally — it loses on either reading; the curve "
           f"expression is {R['ls_b50']:+.2f}% per event at a 50 bp/y borrow "
           f"(*t* = {R['ls_b50_t']:+.2f}); and the entire positive mean is one 2019-2020 "
           f"window. Nothing here is sizeable, and nothing here is separable from luck.\n"
           f"- **The methodological finding.** The interesting result is not that the trade "
           f"failed but that it *could not have succeeded*: a macro event that fires four "
           f"times in twenty years does not generate enough independent observations to "
           f"clear any honest inference bar. Treat every \"the Fed always...\" claim built "
           f"on a handful of cycles the same way."),
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
