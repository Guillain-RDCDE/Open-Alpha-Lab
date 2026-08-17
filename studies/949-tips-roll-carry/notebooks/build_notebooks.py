"""Generate the two narrative notebooks for Study 949 (Riding the TIPS Curve).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast offline synthetic control, and they are never placed under a real-tape banner.
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


def code_with(assignments, body):
    """A code cell that opens with literal ``name = <repr>`` lines, then runs ``body``.

    Keeps the frozen numbers visible at the top of the cell without forcing every
    per-cent sign in the body through ``%``-formatting.
    """
    head = "\n".join(f"{name} = {value!r}" for name, value in assignments)
    return new_code_cell(head + "\n" + body)


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — mirror of docs/results.md.
# VTIP / SCHP / TIP / LTPZ vs SHY / IEF / TLT vs BIL, daily total return,
# excess-of-cash, 2012-10-16 -> 2026-06-30, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2012-10-16", end="2026-06-30", n_days=3444, fp="9da861d9d566",
    cash_cagr=1.59,
    # ladder, excess-of-cash
    ladder={
        "VTIP": dict(dur="~2.5y", exret=0.65, vol=2.54, sharpe=0.256, t=0.99,
                     exdd=-7.4, cagr=2.23, absdd=-6.3, ci=(-0.193, 0.767), neg=13.6),
        "SCHP": dict(dur="~7y", exret=0.35, vol=5.48, sharpe=0.064, t=0.24,
                     exdd=-18.4, cagr=1.80, absdd=-14.3, ci=(-0.409, 0.575), neg=38.8),
        "TIP": dict(dur="~7y", exret=0.27, vol=5.61, sharpe=0.049, t=0.19,
                    exdd=-18.7, cagr=1.71, absdd=-14.5, ci=(-0.417, 0.545), neg=41.8),
        "LTPZ": dict(dur="~20y", exret=-0.02, vol=14.68, sharpe=-0.001, t=-0.00,
                     exdd=-44.8, cagr=0.49, absdd=-41.0, ci=(-0.462, 0.454), neg=49.8),
    },
    diffs={"SCHP": (-0.30, -0.32, -0.192), "TIP": (-0.38, -0.39, -0.207),
           "LTPZ": (-0.67, -0.21, -0.257)},
    # duration-hedged sleeves: beta, gross, t_gross, net, t_net, net Sharpe, vol
    sleeves={
        "VTIP-SHY": dict(beta=1.12, gross=0.98, tg=1.55, net=0.60, tn=0.94,
                         sharpe=0.277, vol=2.16, turn=2.3),
        "SCHP-IEF": dict(beta=0.68, gross=0.69, tg=0.68, net=0.47, tn=0.47,
                         sharpe=0.141, vol=3.35, turn=0.6),
        "TIP-IEF": dict(beta=0.69, gross=0.55, tg=0.54, net=0.33, tn=0.33,
                        sharpe=0.096, vol=3.49, turn=0.6),
        "LTPZ-TLT": dict(beta=0.86, gross=0.34, tg=0.15, net=0.07, tn=0.03,
                         sharpe=0.009, vol=8.22, turn=0.6),
    },
    static={"VTIP~SHY": (0.86, 1.016, 1.47, 0.317), "SCHP~IEF": (0.46, 0.682, 0.48, 0.637),
            "TIP~IEF": (0.38, 0.693, 0.39, 0.627), "LTPZ~TLT": (0.24, 0.850, 0.11, 0.699)},
    boot_gross=(0.98, -0.27, 2.11, 5.8),
    boot_net=(0.60, -0.66, 1.73, 15.9),
    # era cut on the NET sleeve: (pre-shock, t), (shock, t), (post, t)
    eras={
        "VTIP-SHY": ((-0.20, -0.22), (2.36, 1.72), (0.79, 1.05)),
        "SCHP-IEF": ((0.09, 0.06), (1.54, 0.65), (0.29, 0.26)),
        "TIP-IEF": ((-0.05, -0.04), (1.39, 0.59), (0.19, 0.17)),
        "LTPZ-TLT": ((0.15, 0.05), (0.18, 0.03), (-0.28, -0.12)),
    },
    # the SAME era cut on the GROSS sleeve -- this is where the study's only |t| >= 2 sits
    eras_gross={
        "VTIP-SHY": ((0.18, 0.19), (2.81, 2.04), (1.09, 1.45)),
        "SCHP-IEF": ((0.31, 0.22), (1.75, 0.74), (0.50, 0.44)),
        "TIP-IEF": ((0.17, 0.12), (1.60, 0.68), (0.39, 0.35)),
        "LTPZ-TLT": ((0.42, 0.13), (0.44, 0.08), (-0.02, -0.01)),
    },
    # the pre-shock era starts at the sleeve's first day, NOT 2013-01-01 (beta warmup)
    era_spans=(("pre-shock", "2013-10-21", "2020-12-31", 1813, 7.2),
               ("shock", "2021-01-04", "2023-12-29", 753, 3.0),
               ("post-shock", "2024-01-02", "2026-06-30", 625, 2.5)),
    # multiplicity audit: every real-tape HAC t, ranked (top 8 of 56)
    tmax=[(2.04, "VTIP-SHY gross, sub-window 'shock 2021-2023'"),
          (1.72, "VTIP-SHY net, sub-window 'shock 2021-2023'"),
          (1.55, "VTIP-SHY sleeve, gross, 252d beta  [best FULL-SAMPLE]"),
          (1.47, "VTIP~SHY full-sample OLS alpha (IN-SAMPLE hedge)"),
          (1.45, "VTIP-SHY gross, sub-window 'post-shock 2024+'"),
          (1.24, "VTIP-SHY sleeve, net, 504d beta"),
          (1.24, "VTIP-IEI sleeve, gross, 252d beta"),
          (1.09, "VTIP-IEF sleeve, gross, 252d beta")],
    n_tests=56, n_over_2=1, bonferroni=3.2, t_max_full_sample=1.55,
    ex2021={"VTIP-SHY": (0.53, 0.80, 0.17, 0.26), "SCHP-IEF": (0.02, 0.02, -0.19, -0.19),
            "TIP-IEF": (-0.10, -0.09, -0.32, -0.30),
            "LTPZ-TLT": (-0.59, -0.25, -0.86, -0.37)},
    years={2013: -0.47, 2014: -2.36, 2015: -0.54, 2016: 1.54, 2017: 0.55, 2018: -0.86,
           2019: 1.53, 2020: 1.79, 2021: 6.38, 2022: 1.63, 2023: 0.51, 2024: 0.63,
           2025: 1.19, 2026: 0.87},
    cost_sweep=[(0.0, 0.64, 1.02), (2.0, 0.60, 0.94), (5.0, 0.53, 0.84), (10.0, 0.42, 0.66)],
    borrow_sweep=[(0.0, 0.93, 1.47, 0.433), (25.0, 0.65, 1.03, 0.303),
                  (30.0, 0.60, 0.94, 0.277), (50.0, 0.37, 0.59, 0.173),
                  (100.0, -0.19, -0.29, -0.087)],
    pairings=[("VTIP", "SHY", 1.12, 0.98, 1.55, 0.60, 0.94),
              ("VTIP", "IEI", 0.37, 0.76, 1.24, 0.64, 1.04),
              ("VTIP", "IEF", 0.19, 0.68, 1.09, 0.61, 0.99),
              ("SCHP", "IEF", 0.68, 0.69, 0.68, 0.47, 0.47),
              ("SCHP", "IEI", 1.17, 0.86, 0.81, 0.49, 0.46),
              ("LTPZ", "TLT", 0.86, 0.34, 0.15, 0.07, 0.03),
              ("LTPZ", "IEF", 1.83, 0.64, 0.26, 0.06, 0.02)],
    windows=[(126, 0.28, 0.45), (252, 0.60, 0.94), (504, 0.84, 1.24)],
    syn_planted=(2.00, 2.19, 3.65), syn_null=(0.19, 0.32), syn_seeds=(0.50, 1.04, 0, 8),
    syn_panel=(2.07, 2.08, 1.55, 2.00),
)


HEADER = f"""# Study 949 — Riding the TIPS Curve 🪜

**Is there a roll-down carry in real yields, or only duration risk?**

The pitch is old and tidy. The *real* yield curve slopes upward, so an inflation-linked
bond held for a year ages down that curve and gets repriced at a lower real yield — a
capital gain on top of the running real coupon. Extend from short linkers into long ones,
the story goes, and you pick up the roll.

We test both halves of that claim on **VTIP / SCHP / TIP / LTPZ vs BIL** (cash), daily
**total-return** closes, {R['start']} → {R['end']} ({R['n_days']:,} days):

1. **Does extending pay?** Race the four maturity buckets **excess-of-cash** — minus
   BIL's actual total return, not a flat proxy.
2. **Is what is left a carry?** Strip the duration out of each linker by shorting a
   duration-matched **nominal** Treasury fund, with the hedge beta fitted through day
   *t*−1 and applied at *t*, and put a Newey-West *t* on the residual.

*Every real number below is the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`); the live cells run the offline synthetic control only. As-of 2026-06-30.*
"""


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
def build_curious():
    L = R["ladder"]
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. First, the simple question: did going longer pay?\n\n"
           "Four funds, same asset class, different maturities. VTIP holds linkers with "
           "under five years to run; LTPZ holds the fifteen-year-plus end. If riding the "
           "real curve pays, the long one should have earned more over cash than the "
           "short one. Here is what thirteen and a half years actually delivered."),
        code_with(
            [("L", L), ("CASH_CAGR", R["cash_cagr"])],
            "print(f\"{'bucket':6s} {'duration':9s} {'over cash':>10s} {'vol':>7s} \"\n"
            "      f\"{'exSharpe':>9s} {'worst DD':>9s}\")\n"
            "for k in ('VTIP','SCHP','TIP','LTPZ'):\n"
            "    d = L[k]\n"
            "    print(f\"{k:6s} {d['dur']:9s} {d['exret']:+9.2f}% {d['vol']:6.2f}% \"\n"
            "          f\"{d['sharpe']:+9.3f} {d['absdd']:8.1f}%\")\n"
            "print(f\"\\ncash (BIL) itself compounded at {CASH_CAGR:+.2f}%/yr over the same window\")"
        ),
        md(f"## 2. The uncomfortable answer\n\n"
           f"Going longer did not pay more. It paid **less**, all the way down the "
           f"ladder — and the longest bucket did not beat cash at all. Over thirteen and "
           f"a half years, long TIPS (LTPZ) compounded at **+{L['LTPZ']['cagr']:.2f}%/yr** "
           f"while the T-bill leg paid **+{R['cash_cagr']:.2f}%/yr** for doing nothing — "
           f"and it cost a **{L['LTPZ']['absdd']:.0f}%** drawdown to find that out. Short "
           f"linkers (VTIP) were the only bucket to clear cash, at a tenth of the "
           f"volatility.\n\n"
           f"To be scrupulous: none of these gaps is statistically significant. The honest "
           f"sentence is not *\"long linkers lose\"* — it is **\"nothing in this ladder "
           f"can be told apart from zero\"**. Real duration went unrewarded."),
        md("> 🔬 **For the quants.** Every leg is measured excess of BIL's *realised* "
           "total return, so the race spans the zero-rate decade and the 5% regime "
           "without a flat-rate fudge. Long-minus-short HAC *t*s: "
           f"SCHP−VTIP {R['diffs']['SCHP'][1]:+.2f}, TIP−VTIP {R['diffs']['TIP'][1]:+.2f}, "
           f"LTPZ−VTIP {R['diffs']['LTPZ'][1]:+.2f}. Bootstrap Sharpe CIs straddle zero "
           "for all four buckets."),
        md("## 3. So where did the roll-down go?\n\n"
           "Maybe the carry is there but buried under duration — the funds all rise and "
           "fall with the level of rates, and 2022 alone swamped a decade of coupon. So "
           "we take the duration out: hold the linker, short just enough of a "
           "**maturity-matched ordinary Treasury fund** to cancel the rate exposure, and "
           "look at what is left. That residual is the closest thing to a pure "
           "*inflation-linked-versus-nominal* carry you can build from listed funds."),
        code_with(
            [("S", R["sleeves"])],
            "for k in ('VTIP-SHY','SCHP-IEF','TIP-IEF','LTPZ-TLT'):\n"
            "    d = S[k]\n"
            "    print(f\"{k:10s} residual carry {d['gross']:+.2f}%/yr gross \"\n"
            "          f\"({d['net']:+.2f}%/yr after costs and borrow)   t = {d['tg']:+.2f}\")\n"
            "print('\\nA t-statistic needs to reach about 2 before this desk calls anything real.')"
        ),
        md(f"## 4. Positive everywhere, provable nowhere\n\n"
           f"Something *is* there — the residual is positive in all four buckets. But the "
           f"biggest *t* in the whole study is **+{R['sleeves']['VTIP-SHY']['tg']:.2f}**, "
           f"and it is the **shortest** bucket, not the longest. That is backwards for a "
           f"roll-down story: if the slope of the real curve were what you were being paid "
           f"for, the carry should grow as you extend. Instead it shrinks — "
           f"{R['sleeves']['VTIP-SHY']['gross']:+.2f}%/yr at the front end, "
           f"{R['sleeves']['LTPZ-TLT']['gross']:+.2f}%/yr at the long end."),
        md("## 5. And then you look at *when* it happened"),
        code_with(
            [("Y", R["years"])],
            "for y, v in Y.items():\n"
            "    bar = '#' * max(0, int(round(v * 6)))\n"
            "    tag = '   <-- the inflation shock lands' if y == 2021 else ''\n"
            "    print(f\"{y}  {v:+6.2f}%  {bar}{tag}\")"
        ),
        md(f"## 6. One year is the whole story\n\n"
           f"**2021 alone made +{R['years'][2021]:.2f}%** — four times the full-sample "
           f"average — and four of the eight years before it were *negative* (2013 is a "
           f"50-day part year, so treat that count as a tally of rows, not of full years). "
           f"Cut the sample around the 2021-2023 inflation shock and the pattern is "
           f"unmissable: the short sleeve earned "
           f"**{R['eras']['VTIP-SHY'][0][0]:+.2f}%/yr net across the 7.2 pre-shock "
           f"years**, **{R['eras']['VTIP-SHY'][1][0]:+.2f}%/yr during 2021-2023**, "
           f"and **{R['eras']['VTIP-SHY'][2][0]:+.2f}%/yr since**. Remove 2021 and the "
           f"whole result collapses to **{R['ex2021']['VTIP-SHY'][2]:+.2f}%/yr** "
           f"(*t* = {R['ex2021']['VTIP-SHY'][3]:+.2f}).\n\n"
           f"There is a reason, and it is not roll-down. Owning a linker and shorting an "
           f"ordinary bond is, mechanically, a bet that **inflation comes in higher than "
           f"the market had priced**. In 2021-2023 it did, spectacularly. That leg had to "
           f"pay, whatever the real curve's slope was doing — and a roll-down carry does "
           f"not switch itself on when CPI surprises."),
        md(f"> 🔬 **For the quants.** Roll-down in real yields and the realised-minus-"
           f"expected inflation term are **not separately identifiable** from fund total "
           f"returns; the residual is a long-breakeven position by construction. That "
           f"unidentifiability is precisely why the era cut, not the full-sample point "
           f"estimate, decides the verdict here. And note where the study's *only* "
           f"|*t*| ≥ 2 lives: the **gross** carry inside that shock window, "
           f"*t* = **+{R['eras_gross']['VTIP-SHY'][1][1]:.2f}** — one hit out of "
           f"{R['n_tests']} tests, on a sub-window chosen after the fact. It is the rival "
           f"hypothesis arriving on schedule, not the claim surviving."),
        md("## 7. Could you trade it anyway?\n\n"
           "The residual is a *spread*: long the linker, short the Treasury fund, every "
           "day. Shorting is not free — you pay a borrow fee on the short leg for as long "
           "as you hold it. Watch what a perfectly ordinary borrow rate does to the best "
           "number in the study."),
        code_with(
            [("B", R["borrow_sweep"])],
            "for bo, net, t, sh in B:\n"
            "    note = '   <- free shorting: not a thing' if bo == 0 else ''\n"
            "    note = '   <- base case' if bo == 30 else note\n"
            "    note = '   <- edge gone' if net < 0 else note\n"
            "    print(f\"borrow {bo:5.0f} bps/yr  ->  net {net:+.2f}%/yr  (t {t:+.2f}, \"\n"
            "          f\"Sharpe {sh:+.3f}){note}\")"
        ),
        md("## 8. Is the measuring stick broken? (live, offline synthetic)\n\n"
           "Before accepting a null, check the instrument. We build an artificial world "
           "where a **2%/yr residual carry really exists**, hand it to the same code, and "
           "see whether it finds it — then repeat with a world where there is nothing "
           "planted at all. Nothing here touches the real tape."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from tips_roll import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=949)[0])\n"
            "null    = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=949)[0])\n"
            "print('world with a real 2.00%%/yr carry : found %+.2f%%/yr, t = %+.2f'\n"
            "      % (planted['gross_ann']*100, planted['t_gross']))\n"
            "print('world with no carry at all       : found %+.2f%%/yr, t = %+.2f'\n"
            "      % (null['gross_ann']*100, null['t_gross']))"
        ),
        md("The detector finds a genuine 2%/yr carry at *t* well past 3, on a **shorter** "
           "sample than the real one, and stays quiet when there is nothing to find. The "
           "instrument works. The real tape simply has no roll-down carry in it."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Extending along the real curve paid nothing (excess "
           f"Sharpe falls from **{L['VTIP']['sharpe']:+.3f}** at the front to "
           f"**{L['LTPZ']['sharpe']:+.3f}** at the long end, every difference "
           f"insignificant). The duration-hedged residual is positive in all four buckets "
           f"and significant in none — best **full-sample** *t* "
           f"**+{R['sleeves']['VTIP-SHY']['tg']:.2f}** "
           f"gross, **+{R['sleeves']['VTIP-SHY']['tn']:.2f}** net — it *shrinks* as you "
           f"extend, and it is one inflation shock wearing a carry costume. Of the "
           f"**{R['n_tests']}** *t*s this study computes on the real tape, "
           f"**{R['n_over_2']}** reaches 2, and it is the gross carry inside the shock "
           f"window itself (Bonferroni bar ≈ {R['bonferroni']}).\n"
           f"- **Tradability — Mirage.** The best net figure anywhere is "
           f"**+{R['windows'][2][1]:.2f}%/yr** (504-day beta, *t* = "
           f"+{R['windows'][2][2]:.2f}); the base case pays "
           f"**{R['sleeves']['VTIP-SHY']['net']:+.2f}%/yr** — both from a sleeve short a "
           f"Treasury fund every day; it dies at about **1%/yr borrow**, and it dies "
           f"outright if you remove a single calendar year. Buying the long bucket as a "
           f"fund instead was worse: **{L['LTPZ']['cagr']:+.2f}%/yr** against cash's "
           f"**+{R['cash_cagr']:.2f}%**, for a **{L['LTPZ']['absdd']:.0f}%** drawdown.\n"
           f"- **What the tape does say.** The least-bad place on the linker curve was the "
           f"**front**. That is a statement about where risk went *unrewarded*, not an "
           f"edge you can harvest."),
    ]
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 949 — Riding the TIPS Curve — the teardown\n\n"
           "The excess-of-cash ladder race, the duration-hedged residual with a "
           "one-day-lagged rolling beta, Newey-West *t*s, block-bootstrap CIs, the "
           "2021-2023 era cut (gross **and** net), a one-year jackknife, "
           "cost/borrow/pairing/window sweeps, a multiplicity audit over every *t* here, "
           "and the live synthetic control.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`), "
           "window %s → %s, %d daily observations, as-of 2026-06-30. Daily **total-return** "
           "closes; the cash leg is BIL's realised total return (%.2f%%/yr over the window)."
           % (R["fp"], R["start"], R["end"], R["n_days"], R["cash_cagr"])),
        code("R = %r" % (R,)),
        md("## 1. Design, and the one execution lag\n\n"
           "Two engines share one convention.\n\n"
           "- **Ladder race.** Buy-and-hold funds, excess of BIL's actual total return. "
           "No signal, no turnover, no costs to charge — the expense ratio is already "
           "inside the NAV.\n"
           "- **Duration-hedged residual.** `resid_t = ex_tips_t − beta_{t−1} · "
           "ex_nom_t`, with `beta_{t−1}` from a trailing 252-day OLS ending at *t*−1. "
           "**That is the study's only execution lag**, and it is the only place anything "
           "could peek: the legs themselves have no timing rule.\n\n"
           "**Frictions.** One-way cost on |Δbeta| × NAV (the long leg is buy-and-hold "
           "and pays nothing); borrow on |beta| × NAV accrued daily. Base case 2 bps and "
           "30 bps/yr — **the borrow rate is a PROXY** and is swept below.\n\n"
           "**Identification, stated once and honoured throughout.** A long-linker / "
           "short-nominal residual is a *long-breakeven* position: E[resid] = roll-down in "
           "real yields **+** (realised inflation − breakeven at entry). Fund total returns "
           "cannot separate them. A positive residual is therefore evidence of *something* "
           "the duration hedge misses — not proof of roll-down."),
        md("> 💡 **In plain words.** We can measure what a linker earns beyond an ordinary "
           "bond of the same maturity, but we cannot tell whether that extra came from "
           "sliding down the real curve or from inflation simply arriving higher than "
           "expected. So we check *when* it arrived."),
        md("## 2. The ladder, excess-of-cash"),
        code(
            "print(f\"{'leg':6s}{'dur':>7s}{'exret':>9s}{'vol':>8s}{'exSh':>8s}\"\n"
            "      f\"{'HAC t':>8s}{'exDD':>9s}{'absCAGR':>10s}{'absDD':>9s}{'Sharpe 95% CI':>22s}\")\n"
            "for k in ('VTIP','SCHP','TIP','LTPZ'):\n"
            "    d = R['ladder'][k]\n"
            "    ci = f\"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]\"\n"
            "    print(f\"{k:6s}{d['dur']:>7s}{d['exret']:+8.2f}%{d['vol']:7.2f}%\"\n"
            "          f\"{d['sharpe']:+8.3f}{d['t']:+8.2f}{d['exdd']:+8.1f}%\"\n"
            "          f\"{d['cagr']:+9.2f}%{d['absdd']:+8.1f}%{ci:>22s}\")\n"
            "print()\n"
            "for k, (dif, t, gap) in R['diffs'].items():\n"
            "    print(f\"  {k:5s} - VTIP: {dif:+.2f}%/yr  HAC t {t:+.2f}  Sharpe gap {gap:+.3f}\")"
        ),
        md("Monotone the wrong way, and uniformly insignificant. Long TIPS returned "
           f"**{R['ladder']['LTPZ']['cagr']:+.2f}%/yr** absolute against the cash leg's "
           f"**+{R['cash_cagr']:.2f}%/yr**, for a **{R['ladder']['LTPZ']['absdd']:.0f}%** "
           "drawdown. Real duration was uncompensated on this window; the bootstrap CIs "
           "straddle zero for every bucket, so the defensible claim is *no reward*, not "
           "*negative reward*."),
        md("## 3. Duration-hedged residual carry\n\n"
           "Rolling 252-day beta, lagged one day. Net = gross − 2 bps one-way on |Δbeta| "
           "− 30 bps/yr borrow on |beta|."),
        code(
            "print(f\"{'sleeve':11s}{'beta':>6s}{'gross':>9s}{'t':>7s}{'net':>9s}\"\n"
            "      f\"{'t':>7s}{'netSh':>8s}{'vol':>8s}{'turnover':>10s}\")\n"
            "for k in ('VTIP-SHY','SCHP-IEF','TIP-IEF','LTPZ-TLT'):\n"
            "    d = R['sleeves'][k]\n"
            "    print(f\"{k:11s}{d['beta']:6.2f}{d['gross']:+8.2f}%{d['tg']:+7.2f}\"\n"
            "          f\"{d['net']:+8.2f}%{d['tn']:+7.2f}{d['sharpe']:+8.3f}\"\n"
            "          f\"{d['vol']:7.2f}%{d['turn']:9.1f}x\")\n"
            "g, glo, ghi, gneg = R['boot_gross']; n, nlo, nhi, nneg = R['boot_net']\n"
            "print(f\"\\nVTIP-SHY block bootstrap (2,000 draws, 21-day blocks):\")\n"
            "print(f\"  gross {g:+.2f}%/yr  95% CI [{glo:+.2f}%, {ghi:+.2f}%]  share<0 {gneg:.1f}%\")\n"
            "print(f\"  net   {n:+.2f}%/yr  95% CI [{nlo:+.2f}%, {nhi:+.2f}%]  share<0 {nneg:.1f}%\")\n"
            "print('\\nfull-sample OLS (IN-SAMPLE hedge, reference only):')\n"
            "for k, (a, b, t, r2) in R['static'].items():\n"
            "    print(f\"  {k:10s} alpha {a:+.2f}%/yr  beta {b:.3f}  HAC t {t:+.2f}  R2 {r2:.3f}\")"
        ),
        md("Positive in all four buckets, significant in none; the maximum |*t*| on the "
           f"**full sample** is **{R['t_max_full_sample']:.2f}**. The term structure "
           "of the residual **declines with duration**, which is the wrong shape for a "
           "slope-driven roll-down and the right shape for a short-dated inflation-accrual "
           "effect. The in-sample OLS alphas reproduce both the level and the ordering, so "
           "this is not an artefact of the rolling estimator."),
        md("> 💡 **In plain words.** If you were being paid for sliding down a sloped "
           "curve, the payment should grow the further out you sit. It does the opposite."),
        md("## 4. Era cut — the 2021-2023 inflation shock, **gross and net**\n\n"
           "The `pre-shock` era begins at the sleeve's **first day, 2013-10-21** — the "
           "252-day beta warmup eats the first year — so it is **1,813 days ≈ 7.2 years**, "
           "not eight. Gross is shown alongside net because that is where the study's only "
           "|*t*| ≥ 2 sits, and a net-only table would have hidden it."),
        code(
            "for tag, s, e, nd, yrs in R['era_spans']:\n"
            "    print(f\"  {tag:11s} {s} -> {e}  ({nd:4d}d, {yrs:.1f}y)\")\n"
            "print()\n"
            "print(f\"{'sleeve':11s}{'':>4s}{'pre-shock':>18s}{'shock 21-23':>18s}{'post 24+':>18s}\")\n"
            "for k in R['eras']:\n"
            "    for tag, src in (('gross', R['eras_gross']), ('net', R['eras'])):\n"
            "        (a, ta), (b, tb), (c, tc) = src[k]\n"
            "        hit = '   <== only |t|>=2 in the study' if abs(tb) >= 2 else ''\n"
            "        print(f\"{k:11s}{tag:>6s}{a:+10.2f}% (t{ta:+5.2f}){b:+9.2f}% \"\n"
            "              f\"(t{tb:+5.2f}){c:+9.2f}% (t{tc:+5.2f}){hit}\")\n"
            "print('\\njackknife -- drop 2021 entirely (RETURNS excised; the hedge beta is')\n"
            "print('NOT re-estimated without 2021, so early-2022 betas still see it):')\n"
            "for k, (g, tg, n, tn) in R['ex2021'].items():\n"
            "    print(f\"  {k:11s} gross {g:+.2f}%/yr (t {tg:+.2f})   net {n:+.2f}%/yr (t {tn:+.2f})\")"
        ),
        md(f"The **7.2** pre-shock years produced "
           f"**{R['eras_gross']['VTIP-SHY'][0][0]:+.2f}%/yr gross "
           f"(*t* = {R['eras_gross']['VTIP-SHY'][0][1]:+.2f})** and "
           f"**{R['eras']['VTIP-SHY'][0][0]:+.2f}%/yr net** on the headline sleeve — "
           f"nothing, with the net sign against the claim. Everything "
           f"the study found is in and after the shock, and removing **2021 alone** "
           f"(+{R['years'][2021]:.2f}% gross that year, against a +"
           f"{R['sleeves']['VTIP-SHY']['gross']:.2f}%/yr full-sample mean) leaves "
           f"**{R['ex2021']['VTIP-SHY'][2]:+.2f}%/yr, *t* = "
           f"{R['ex2021']['VTIP-SHY'][3]:+.2f}**, and turns three of the four sleeves "
           f"negative. This is the study's decisive result: the residual behaves exactly "
           f"like a long-breakeven leg meeting an inflation surprise, and not at all like "
           f"a structural roll-down.\n\n"
           f"**And the *t* = +{R['eras_gross']['VTIP-SHY'][1][1]:.2f} in the shock column "
           f"does not rescue it — it indicts it.** A roll-down carry is a property of a "
           f"sloped curve; it cannot switch on for exactly the three years CPI surprised "
           f"and stay dark for the other ten. §8 puts that number in its multiplicity "
           f"context."),
        md("## 5. Calendar years of the headline sleeve (VTIP − SHY, gross %)\n\n"
           "**2013 (n = 50, beta warmup) and 2026 (n = 123, as-of cut) are part years**, so "
           "the unweighted mean below is a tally of rows, not a return."),
        code(
            "for y, v in R['years'].items():\n"
            "    flag = '  <- 4x the sample mean' if y == 2021 else ''\n"
            "    if y in (2013, 2026):\n"
            "        flag = '  (PART year -- not a full 12 months)'\n"
            "    print(f\"  {y}: {v:+6.2f}%{flag}\")\n"
            "vals = [v for y, v in R['years'].items() if y < 2021]\n"
            "print(f\"\\npre-2021 rows: {len(vals)} (one of them a 50-day stub), of which \"\n"
            "      f\"negative: {sum(1 for v in vals if v < 0)}; \"\n"
            "      f\"unweighted mean {sum(vals)/len(vals):+.2f}%\")"
        ),
        md("## 6. Friction sweeps (VTIP − SHY)\n\n"
           "Borrow is a *level* charge — the sleeve is short every day — so it dominates "
           "trading cost. The borrow rate is a **PROXY**, hence the sweep."),
        code(
            "print('one-way cost sweep (borrow fixed at 30 bps/yr):')\n"
            "for c, net, t in R['cost_sweep']:\n"
            "    print(f\"  {c:5.1f} bps -> net {net:+.2f}%/yr  t {t:+.2f}\")\n"
            "print('\\nborrow sweep (one-way cost fixed at 2 bps):')\n"
            "for bo, net, t, sh in R['borrow_sweep']:\n"
            "    tag = '  <- base case' if bo == 30 else ('  <- edge extinguished' if net < 0 else '')\n"
            "    print(f\"  {bo:5.1f} bps/yr -> net {net:+.2f}%/yr  t {t:+.2f}  Sharpe {sh:+.3f}{tag}\")"
        ),
        md("## 7. Pairing and window robustness\n\n"
           "The linker↔nominal duration match is an **assumption** taken from sponsor fact "
           "sheets, so it gets swept; the hedge ratio itself is always estimated from "
           "returns, so the pairing only chooses the factor."),
        code(
            "for a, b, be, g, tg, n, tn in R['pairings']:\n"
            "    print(f\"  {a:5s} hedged with {b:4s}: beta {be:5.2f}  gross {g:+.2f}% \"\n"
            "          f\"(t {tg:+.2f})  net {n:+.2f}% (t {tn:+.2f})\")\n"
            "print('\\nbeta estimation window (VTIP-SHY):')\n"
            "for w, net, t in R['windows']:\n"
            "    print(f\"  {w:3d} days -> net {net:+.2f}%/yr  t {t:+.2f}\")"
        ),
        md("No pairing rescues the result and no window does either — the most flattering "
           f"choice (504-day beta) reaches *t* = {R['windows'][2][2]:+.2f} at "
           f"**+{R['windows'][2][1]:.2f}%/yr net**, which is the **largest net figure "
           f"anywhere in this study** (the headline row is the *base case*, not the best "
           "one). There is no specification here under which the residual clears the bar "
           "on the full sample."),
        md(f"## 8. The multiplicity audit — every real-tape *t*, ranked\n\n"
           f"The headline sleeve VTIP−SHY was chosen **after the fact** as the best of "
           f"four, so the study owes you the whole list rather than its favourite row. It "
           f"computes **{R['n_tests']}** HAC *t*s on the real tape. Top of that ranking:"),
        code_with(
            [("TMAX", R["tmax"]), ("N_TESTS", R["n_tests"]),
             ("N_OVER", R["n_over_2"]), ("BONF", R["bonferroni"]),
             ("T_FULL", R["t_max_full_sample"])],
            "for t, name in TMAX:\n"
            "    flag = '   <== CLEARS |t|>=2' if abs(t) >= 2.0 else ''\n"
            "    print(f\"  t {t:+5.2f}  {name}{flag}\")\n"
            "print(f\"\\n  {N_OVER} of {N_TESTS} statistics reach |t| >= 2; \"\n"
            "      f\"the Bonferroni bar for {N_TESTS} tests is |t| ~ {BONF:.1f}.\")\n"
            "print(f\"  On the FULL sample nothing exceeds |t| = {T_FULL:.2f}.\")"
        ),
        md(f"**{R['n_over_2']} of {R['n_tests']}** clears the nominal bar, against a "
           f"Bonferroni threshold of ≈**{R['bonferroni']}** — and that one is *gross*, "
           f"*post-hoc*, and confined to the three-year window the study already names as "
           f"the rival explanation. Charge the borrow the permanently-short leg actually "
           f"owes and it falls to +{R['eras']['VTIP-SHY'][1][1]:.2f}. No adjustment is "
           f"needed to reach the null here; what is needed is disclosure of how many "
           f"numbers were looked at, which is what this table is."),
        md("## 9. Synthetic control — is the detector powered and unbiased?\n\n"
           "*Live, offline, synthetic. Nothing below touches the real tape.* A planted "
           "2%/yr residual carry must be recovered with a decisive *t*; the null must stay "
           "quiet across seeds; and a panel with the **same** carry in every bucket must "
           "come back with a **flat** carry term structure — otherwise the real tape's "
           "declining term structure would be an estimator artefact."),
        code(
            "import os, sys\n"
            "import numpy as np\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from tips_roll import data, strategy as st\n"
            "pl = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=949)[0])\n"
            "nu = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=949)[0])\n"
            "print('planted 2.00%%/yr carry : recovered %+.2f%%/yr  HAC t %+.2f  (beta %.2f vs 0.85 planted)'\n"
            "      % (pl['gross_ann']*100, pl['t_gross'], pl['mean_beta']))\n"
            "print('null, no carry planted : recovered %+.2f%%/yr  HAC t %+.2f'\n"
            "      % (nu['gross_ann']*100, nu['t_gross']))\n"
            "ts = np.array([st.synthetic_detect(\n"
            "        data.synthetic_daily(signal_strength=0.0, seed=949+s)[0])['t_gross']\n"
            "    for s in range(8)])\n"
            "print('null across 8 seeds    : mean t %+.2f  sd %.2f  fires |t|>=2 on %d/8'\n"
            "      % (ts.mean(), ts.std(ddof=1), int((np.abs(ts) >= 2).sum())))\n"
            "panel, truth = data.synthetic_panel(signal_strength=1.0, seed=949)\n"
            "dl = st.synthetic_ladder_detect(panel, truth)\n"
            "rec = [dl['carries']['bucket_%d' % i]['gross_ann']*100 for i in range(truth['n_buckets'])]\n"
            "print('ladder panel, same carry in all 3 buckets: recovered '\n"
            "      + ', '.join('%+.2f%%' % c for c in rec)\n"
            "      + '  (planted %+.2f%%) -> flat, as it should be' % (dl['planted_carry_ann']*100,))"
        ),
        md("Powered (*t* > 3 on a planted 2%/yr carry, on a **shorter** sample than the "
           "real one), unbiased (0/8 false positives on the null), and free of a spurious "
           "term-structure tilt. Had a real-yield roll-down carry of the advertised size "
           "been present in the tape, this machinery would have stamped it."),
        md(f"## Verdict\n\n"
           f"**Signal — None.** Both forms of the claim fail. Extending along the real "
           f"curve produced a *monotonically falling* excess Sharpe "
           f"({R['ladder']['VTIP']['sharpe']:+.3f} → {R['ladder']['LTPZ']['sharpe']:+.3f}) "
           f"with every long-minus-short difference insignificant and every Sharpe CI "
           f"straddling zero. The duration-hedged residual is positive in all four buckets "
           f"and significant in none (max **full-sample** HAC *t* "
           f"**+{R['sleeves']['VTIP-SHY']['tg']:.2f}** gross, "
           f"**+{R['sleeves']['VTIP-SHY']['tn']:.2f}** net; bootstrap CI "
           f"[{R['boot_net'][1]:+.2f}%, {R['boot_net'][2]:+.2f}%]), it declines with "
           f"duration, and it is entirely post-2020 "
           f"({R['eras_gross']['VTIP-SHY'][0][0]:+.2f}%/yr gross / "
           f"{R['eras']['VTIP-SHY'][0][0]:+.2f}%/yr net across the 7.2 pre-shock years, "
           f"{R['eras']['VTIP-SHY'][1][0]:+.2f}%/yr in 2021-2023, "
           f"{R['ex2021']['VTIP-SHY'][2]:+.2f}%/yr ex-2021). Given that the residual is by "
           f"construction a long-breakeven leg, the parsimonious reading is an inflation "
           f"surprise, not roll-down — and the two are not separately identifiable here, "
           f"which is itself a reason not to award a green stamp.\n\n"
           f"**The one *t* ≥ 2, disclosed.** Exactly "
           f"{R['n_over_2']} of the {R['n_tests']} statistics computed on the real tape "
           f"reaches 2: **+{R['eras_gross']['VTIP-SHY'][1][1]:.2f}**, the gross carry of "
           f"the best-of-four sleeve inside the hand-picked shock window (§8). Against a "
           f"Bonferroni bar of ≈{R['bonferroni']}, and landing exactly where the rival "
           f"hypothesis says it should, it does not move the stamp.\n\n"
           f"**Tradability — Mirage.** Best net figure anywhere "
           f"+{R['windows'][2][1]:.2f}%/yr (504-day beta, *t* = "
           f"+{R['windows'][2][2]:.2f}); base case "
           f"{R['sleeves']['VTIP-SHY']['net']:+.2f}%/yr at "
           f"{R['sleeves']['VTIP-SHY']['vol']:.2f}% vol, from a permanently short spread "
           f"that turns negative at ~{R['borrow_sweep'][4][0]:.0f} bps/yr borrow and "
           f"survives no single-year jackknife. The long-only version — just buying LTPZ — "
           f"under-earned T-bills by {R['cash_cagr'] - R['ladder']['LTPZ']['cagr']:.2f} pp/yr "
           f"while drawing down {R['ladder']['LTPZ']['absdd']:.0f}%.\n\n"
           f"**Sample caveat, named on the Signal axis.** VTIP's 2012-10 inception confines "
           f"the whole study to the post-GFC real-rate regime — no 2008-2009 TIPS liquidity "
           f"dislocation, no pre-2004 illiquidity premium. What is measured here is a "
           f"13.5-year window that happens to contain exactly one inflation shock."),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, builder in [("01_for_the_curious", build_curious),
                          ("02_for_the_quants", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
