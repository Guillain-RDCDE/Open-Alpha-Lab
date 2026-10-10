"""Notebook builder for Study 1025 — Don't Look.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

(from this ``notebooks/`` directory). Both notebooks walk the seven desk beats
(METHODOLOGY.md) at two altitudes. The code cells run the real, SHA-pinned tapes directly
(``quantlab.bundled`` — offline), with smaller bootstraps than ``verify.py`` so they stay fast.
The **headline numbers quoted in prose** live in ONE place — the ``REAL`` dict below, which
mirrors ``docs/results.md`` — and the verdict paragraphs are read from ``docs/results.md`` at
build time, so a re-run of ``examples/verify.py`` followed by a rebuild cannot leave them
disagreeing.
"""

from __future__ import annotations

import os
import re

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "docs", "results.md")

# --------------------------------------------------------------------------- #
# The real run — mirrors docs/results.md (as-of 2018-11-30). Update after verify.py.
# --------------------------------------------------------------------------- #
REAL = {
    "AS_OF": "2018-11-30",
    "FP_M": "b5226282e9da",
    "FP_D": "f8b91faaa94b",
    "SIGNAL": "Weak",
    "TRAD": "Fragile",
    "BE": "30 months (2.5 years)",
    "BE_CI": "2.6 months to 9.4 years",
    "BE_BILLS": "17 months",
    "BE_BILLS_CI": "4.2 months to 3.8 years",
    "BE_IID": "11.6 months",
    "BE_BT": "23 months",
    "BE_PRE": "11 months",
    "BE_POST": "43 months",
    "BE_REAL": "16 months",
    "P_SHORT": "< 0.001",
    "P_LONG": "0.044",
    "LAMBDA_12": "1.79",
    "LOSS_1D": "47%",
    "LOSS_1M": "37%",
    "LOSS_12M": "25%",
    "LOSS_120M": "5%",
    "COST_D": "11.5%",
    "T_D": "5.8",
    "COST_W": "7.8%",
    "T_W": "3.9",
    "COST_M": "0.9%",
    "COST_Q": "1.2%",
    "COST_Y": "1.1%",
    "P_MQY": "0.16-0.21",
}

BADGE = {"Real": "2ea44f", "Weak": "dab617", "Mixed": "dab617", "None": "c0392b",
         "Investable": "2ea44f", "Fragile": "dab617", "Mirage": "c0392b"}


def verdict_paragraphs() -> tuple[str, str, str]:
    """Signal / Tradability paragraphs and the one-sentence line, read from results.md."""
    with open(RESULTS, encoding="utf-8") as fh:
        txt = fh.read()
    sig = re.search(r"^\*\*Signal: .*$", txt, re.M).group(0)
    trd = re.search(r"^\*\*Tradability: .*$", txt, re.M).group(0)
    one = re.search(r"^> \*\*In one sentence:\*\* .*$", txt, re.M).group(0)
    return sig, trd, one


def fill(text: str) -> str:
    for k, v in REAL.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def md(text):
    return new_markdown_cell(fill(text.strip("\n").rstrip()))


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


def badges():
    s, t = REAL["SIGNAL"], REAL["TRAD"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{BADGE[s]}"
            f"?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{BADGE[t]}"
            f"?style=flat-square)")


BOOT = r"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
warnings.filterwarnings("ignore", category=RuntimeWarning)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.dpi"] = 80
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from dontlook import data, strategy as st

m = data.load_monthly()
X = m[["stock", "bond", "bill"]]
HAVE_DAILY = data.have_daily()
d = data.load_daily() if HAVE_DAILY else None
H = st.HORIZONS_M
print(f"as-of {data.AS_OF} | monthly {X.index[0]:%Y-%m} -> {X.index[-1]:%Y-%m} "
      f"({len(X)} months, fingerprint {data.fingerprint(m)})")
if HAVE_DAILY:
    print(f"daily S&P 500 PRICE index {d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d} "
          f"({len(d)} sessions, fingerprint {data.fingerprint(d)})")
C = {"stock": "#c0392b", "bond": "#1f6feb", "bill": "#8b949e"}
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    sig, trd, one = verdict_paragraphs()
    cells = [
        md(r"""
# Don't Look 🙈
### Does checking your portfolio less really keep you invested, and what does checking cost?

""" + badges() + r"""

A famous explanation of why stocks pay so much more than bonds is a story about *feelings*.
Losses hurt about twice as much as gains please, and the more often you look, the more often
you see a loss. Benartzi and Thaler worked out that someone who looks once a year would be
exactly on the fence between stocks and bonds. We checked that number on ninety-two years of
data, and then asked the practical question: what does looking too often actually *cost*?

> 📓 **This is the plain-language layer.** The prospect-theory machinery, the bootstrap and
> every robustness sweep are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** Every chart below is generated by the code beside it, on the
> real (frozen, fingerprinted) tapes.
"""),
        code(BOOT),
        md(r"""
## Beat 0 · Verdict (real tape)

*Numbers from the pinned run in [`docs/results.md`](../docs/results.md), as-of {{AS_OF}}.*

| Question | Answer |
|---|---|
| Do you see fewer losses if you look less often? | Yes, from {{LOSS_1D}} of days to {{LOSS_12M}} of years and {{LOSS_120M}} of decades. |
| Does a loss-averse investor switch from bonds to stocks as they look less? | Yes, and the switch is statistically real. |
| At about one year, as the famous paper says? | The data say {{BE}}, somewhere between {{BE_CI}}. One year is what you get if you pretend months are independent ({{BE_IID}}). |
| Does looking too often cost money? | Only if you *act* on daily or weekly moves: about {{COST_D}} a year for a daily panicker. A yearly one loses about {{COST_Y}}, which is not distinguishable from zero. |
"""),
        md(r"""
## 1 · The claim

> *"The equity premium is the price of looking too often."*

Shlomo Benartzi and Richard Thaler (1995) put two facts about people together:

1. **Loss aversion.** Losing \$100 hurts roughly as much as winning \$225 pleases (Tversky &
   Kahneman 1992).
2. **Myopia.** People keep score often, and every check is a fresh chance to feel a loss.

Stocks rise on average, but on any single day they are almost a coin flip. Look daily and you
spend half your life in pain. Look once a decade and you almost never see red. With the
Tversky-Kahneman numbers, they found an investor would be *indifferent* between stocks and
bonds if they looked about **once a year**. Experiments backed it up: people given less
frequent feedback took more risk (Thaler, Tversky, Kahneman & Schwartz 1997; Gneezy &
Potters 1997). The folk version is advice you have probably heard: *check your portfolio less
and you'll stay invested.*
"""),
        md(r"""
## 2 · So what?

If it's true, two big things follow:

- **One of finance's great puzzles becomes psychology.** Stocks have beaten bonds by several
  percent a year for a century, far more than standard risk models can justify. On this account
  investors demand that premium because they *watch too much*.
- **A free improvement for savers.** If looking less makes people hold more stocks, then the
  app notification is costing them money, and *not looking* is worth real return.

Both deserve checking against the data.
"""),
        md(r"""
## 3 · How we'd know

We fixed the tests **before** running them on the real data:

- **Is the flip real?** A loss-averse investor must clearly prefer bonds at a one-month look
  and stocks at a ten-year look. "Clearly" means under 5% odds of a fluke, measured by
  reshuffling history in two-year blocks.
- **Is it about a year?** The range of plausible break-even horizons must sit between 3 and 36
  months and include 12. If the flip is real but the range is wider, the stamp is **Weak**.
- **Does looking cost money?** We simulate an investor who goes to cash after seeing a loss and
  comes back after a gain, with realistic trading costs and no peeking ahead, and compare them
  with someone who just holds. If no checking frequency loses money with confidence, the advice
  is a **Mirage**.
"""),
        md(r"""
## 4 · The teardown

### 4.1 How often do you *see* red?
"""),
        code(r"""
L = st.loss_probability_curve(X, H)
fig, ax = plt.subplots(figsize=(10.5, 5.2))
if HAVE_DAILY:
    Ld = st.loss_probability_curve(d[["stock"]], st.HORIZONS_D)
    ax.plot(np.array(st.HORIZONS_D) / 21.0, Ld["stock"] * 100, "o--", color=C["stock"],
            alpha=0.6, label="stocks, daily tape (S&P 500 price, 1990-2018)")
for c, lbl in (("stock", "stocks"), ("bond", "bonds (AAA, 20y)"), ("bill", "bills")):
    ax.plot(L.index, L[c] * 100, "o-", lw=2.2, color=C[c], label=lbl + ", 1926-2018")
ax.set_xscale("log")
ax.set_xticks([0.05, 0.25, 1, 3, 12, 36, 120])
ax.set_xticklabels(["1 day", "1 week", "1 month", "3 mo", "1 year", "3 years", "10 years"])
ax.set_ylabel("% of periods that end in a loss")
ax.set_title("look less often, see fewer losses", fontsize=11)
ax.legend(fontsize=8); plt.show()
print(L.loc[[1, 12, 60, 120]].round(3).to_string())
"""),
        md(r"""
This part of the story is simply true. Check daily and stocks are down on about {{LOSS_1D}} of
days. Check yearly and it's {{LOSS_12M}}. Check once a decade and it's {{LOSS_120M}}. It is also
**mechanical**: anything that drifts up a little while jiggling a lot looks like that.

### 4.2 How does a loss-averse investor *feel* about each asset?

We score every holding period the way prospect theory does: gains count, losses count
**2.25 times**, and rare extreme outcomes get extra weight. A positive score means the investor
likes the asset over that horizon.
"""),
        code(r"""
P = st.pt_curve(X, H)
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for c, lbl in (("stock", "stocks"), ("bond", "bonds"), ("bill", "bills")):
    ax.plot(P.index, P[c], "o-", lw=2.2, color=C[c], label=lbl)
ax.axhline(0, color="k", lw=1)
ax.set_xscale("log"); ax.set_xticks([1, 3, 12, 36, 120])
ax.set_xticklabels(["1 month", "3 mo", "1 year", "3 years", "10 years"])
ax.set_ylabel("prospect-theory score (higher = more attractive)")
ax.set_title("short looks: stocks feel worst. long looks: stocks feel best", fontsize=11)
ax.legend(fontsize=9); plt.show()
gap = (P["stock"] - P["bond"]).to_numpy()
print(f"stocks beat bonds for this investor from about {st.break_even(H, gap):.0f} months on")
"""),
        md(r"""
The crossing point is the "equilibrium" horizon. Benartzi and Thaler put it at about a year. On
the full 1926-2018 record we get **{{BE}}**.

### 4.3 How sure can we be about that crossing point?

Ninety-two years sounds like a lot, but they contain only nine separate decades. We replay
history many times, reshuffled in two-year chunks, and recompute the crossing each time.
"""),
        code(r"""
boot = st.bootstrap_curves(X, H, n_boot=300, seed=1025)
s = st.summarise_gap(P, boot, "stock", "bond")
draws = np.where(np.isfinite(s["be_draws"]), s["be_draws"], 150)
fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.hist(np.log10(draws), bins=30, color="#1f6feb", alpha=0.8)
ax.axvline(np.log10(12), color="k", ls="--", lw=2, label="Benartzi-Thaler: about 1 year")
ax.axvline(np.log10(s["break_even"]), color=C["stock"], lw=2.5, label="this tape")
ax.set_xticks(np.log10([1, 3, 12, 36, 120, 150]))
ax.set_xticklabels(["1 mo", "3 mo", "1 yr", "3 yrs", "10 yrs", "never"])
ax.set_xlabel("break-even horizon in each replay of history")
ax.set_ylabel("replays"); ax.legend(fontsize=9); plt.show()
print(f"95% range: {s['be_lo']:.1f} to {s['be_hi']:.1f} months")
"""),
        md(r"""
The replays scatter from a few months to nearly a decade. The data **can't tell one year apart
from three**, so the famous "about one year" can't be confirmed on this record.

> 🔬 **For the quants.** Circular block bootstrap of the joint monthly (stock, bond, bill)
> vector, 24-month blocks. Percentile intervals and two-sided p-values. The flip itself *is*
> significant: p {{P_SHORT}} at one month, p {{P_LONG}} at ten years.

### 4.4 Where "one year" comes from

Shuffle the months **independently**, as if each month knew nothing about the last, and the
crossing lands at **{{BE_IID}}**, right on Benartzi and Thaler's number. Real markets have runs,
though. A bad year is often followed by another (1929-32 is the extreme case). That makes one-
and two-year looks scarier than independent months would suggest, and it pushes the crossing out.
"""),
        code(r"""
rows = []
for meth, lbl in (("overlapping", "real sequence of returns"),
                  ("iid", "months shuffled independently")):
    Pm = st.pt_curve(X, H, method=meth)
    rows.append((lbl, st.break_even(H, (Pm["stock"] - Pm["bond"]).to_numpy())))
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[C["stock"], "#8b949e"])
ax.axvline(12, color="k", ls="--", lw=2, label="1 year")
ax.set_xlabel("break-even horizon vs bonds (months)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
for lbl, v in rows:
    print(f"{lbl:32s} {v:5.1f} months")
"""),
        md(r"""
### 4.5 What does *acting* on what you see cost?

Looking itself is free. The cost comes from **reacting**. Our panicky investor checks at a fixed
rhythm. After a loss they move to cash for the next period, and after a gain they move back into
stocks. They pay 0.05% each time they switch, and they can only act on what they have already
seen.
"""),
        code(r"""
fig, ax = plt.subplots(figsize=(10.5, 5.2))
if HAVE_DAILY:
    for f, col in (("D", "#c0392b"), ("W", "#e67e22"), ("M", "#1f6feb")):
        sw = st.myopic_switcher(d["stock"], d["bill"], f, cost_bps=5.0)
        ax.plot(sw.index, np.cumprod(1 + sw["net"]), lw=1.8, color=col,
                label=f"checks {st.FREQ_LABEL[f]}, reacts to losses")
    ax.plot(sw.index, np.cumprod(1 + sw["bh"]), color="k", lw=2.4,
            label="never reacts (buy and hold)")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (S&P 500 price index, log scale)")
    ax.legend(fontsize=9); ax.set_title("1990-2018, net of 5 bp per switch", fontsize=11)
    plt.show()
t = st.discipline_table(X, ["M", "Q", "Y"], 12, cost_bps=5.0, n_boot=300)
print("1926-2018, total return — annual return given up by the reacting investor:")
print(t[["d_cagr_net", "t_hac", "p_ret", "time_invested"]].round(3).to_string())
"""),
        md(r"""
The daily reactor is a disaster. They switch over a hundred times a year and give up about
**{{COST_D}} a year**. The weekly one gives up about {{COST_W}}. Once the rhythm is monthly or
slower, the damage shrinks to about **1% a year**, and over ninety years that is not
distinguishable from luck (p {{P_MQY}}). The slow reactor also spends less time in stocks and
carries less risk, so per unit of risk it does no worse.
"""),
        md(r"""
## 5 · The verdict

""" + sig + "\n\n" + trd + "\n\n" + one),
        md(r"""
## 6 · Could you trade it?

There's nothing to trade here, only a habit to avoid.

- **"Don't look" earns the equity premium, nothing more.** Buy-and-hold collects the
  premium stocks have always paid. Not looking keeps you from throwing part of it away, and it
  adds no edge on top.
- **The habit worth breaking is reacting to short-term noise.** Selling after a red day or a red
  week is where the measured damage is, about {{COST_D}} and {{COST_W}} a year in this sample.
- **A yearly glance is cheap.** If you rebalance or react once a year, the cost on this record
  is about {{COST_Y}} a year and statistically indistinguishable from zero.
"""),
        md(r"""
## 7 · Going further 🚪

- **A better bond leg.** Our bond is built from a corporate yield. A true Treasury total-return
  series would be the cleaner Benartzi-Thaler comparison.
- **Other countries.** The US had an unusually good century for stocks. A loss-averse investor
  elsewhere may never have found stocks attractive at any horizon.
- **Other reactions.** Selling half instead of everything, or a one-period time-out, would cost
  a different amount. The rule here was fixed in advance on purpose, so a fork that tries
  others should correct for trying many.
- **The quant notebook** has the λ × α sweep, the pre/post-1970 split and the real-terms check.
  The break-even moves a lot with all of them, which is the point.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    sig, trd, one = verdict_paragraphs()
    cells = [
        md(r"""
# Don't Look — a quantitative teardown 🔬
### CPT on empirical horizon distributions · break-even with block-bootstrap CI · the cost of myopia

""" + badges() + r"""

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). **§4** holds
the load-bearing result: the stocks-vs-bonds break-even horizon and its circular-block-bootstrap
interval. **§5** shows why the one-year figure is an i.i.d. artefact. **§7** prices the myopic
behaviour with one lag, costs, gross and net, excess-vs-excess Sharpe, Newey-West and a paired
bootstrap.

> ⚠️ **Not investment advice.** Real tapes from `quantlab.bundled` (SHA-256 pinned, offline),
> as-of {{AS_OF}}; monthly fingerprint `{{FP_M}}`, daily `{{FP_D}}`. Headline numbers quoted in
> prose come from [`docs/results.md`](../docs/results.md). Executed cells use smaller
> bootstraps than `verify.py`, so their intervals wobble slightly around the pinned ones.
>
> 💡 **The `💡 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        md(r"""
## Verdict, up front (real tape)

""" + sig + "\n\n" + trd + r"""

> 💡 **In plain words.** The horizon matters, but nobody can say from this record that it
> matters at "one year" rather than at three. Not looking protects you from one specific
> mistake and does nothing else for you.
"""),
        md(r"""
## 1 · Hypotheses and the pre-registered rule

- **H₁ (flip).** CPT gap (stocks − bonds) < 0 at h = 1 month and > 0 at h = 120 months, each
  with two-sided block-bootstrap p < 0.05.
- **H₂ (calibration).** 95% bootstrap CI of the break-even ⊂ [3, 36] months and ∋ 12.
- **H₃ (regimes).** Finite break-even point estimates pre- and post-1970.
- **Signal:** None if ¬H₁; Real if H₁ ∧ H₂ ∧ H₃; Mixed if H₁ ∧ H₂ ∧ ¬H₃; Weak if H₁ ∧ ¬H₂.
- **Tradability** (switcher vs buy-and-hold, 5 bp one-way, clocks D/W/M/Q/Y): Investable if
  ΔSharpe > 0 with p < 0.05 at *every* clock; Fragile if the net return gap is > 0 with p < 0.05
  at ≥ 1 clock; else Mirage.

The rule is `strategy.verdict`, unit-tested in both directions in `tests/test_strategy.py`.
"""),
        md(r"""
## 2 · Data: the bond leg is constructed, so check the construction

Moody's AAA (monthly-average yield) → 20-year constant-maturity par bond,
`r ≈ y₋₁/12 − D·Δy + ½·C·Δy²`. Compare with exact par-bond repricing:
"""),
        code(r"""
from quantlab import bundled
aaa = bundled.load_arch("default")["AAA"]
a = data.bond_total_return(aaa, method="duration")
b = data.bond_total_return(aaa, method="reprice")
print(f"max |duration - reprice| = {(a - b).abs().max() * 1e4:.1f} bp/month, "
      f"corr {np.corrcoef(a, b)[0, 1]:.6f}")
g = (1 + X).prod() ** (12 / len(X)) - 1
print("CAGR 1926-2018:", g.round(4).to_dict())
print("ann. vol      :", (X.std() * np.sqrt(12)).round(4).to_dict())
pd.DataFrame(data.provenance())
"""),
        md(r"""
> 💡 **In plain words.** The shortcut formula and the exact bond-pricing formula agree to a
> small fraction of a percent a month, so the approximation isn't what drives anything below.
> The *inputs* are softer: an averaged yield, and a corporate rather than a Treasury bond.

## 3 · Loss probabilities with block-bootstrap bands
"""),
        code(r"""
boot = st.bootstrap_curves(X, H, n_boot=400, seed=1025)
band = st.loss_band(boot)
L = st.loss_probability_curve(X, H)
fig, ax = plt.subplots(figsize=(10.5, 5))
for c in ("stock", "bond", "bill"):
    ax.plot(L.index, L[c] * 100, "o-", color=C[c], lw=2, label=c)
    ax.fill_between(L.index, band[c]["lo"] * 100, band[c]["hi"] * 100, color=C[c], alpha=0.15)
ax.set_xscale("log"); ax.set_xlabel("evaluation horizon (months)")
ax.set_ylabel("P(loss) %"); ax.legend(); plt.show()
out = pd.concat({c: pd.concat([L[c], band[c]], axis=1) for c in ("stock", "bond")}, axis=1)
out.loc[[1, 12, 60, 120]].round(3)
"""),
        md(r"""
## 4 · CPT value by horizon and the break-even

TK92 value `x^0.88`, `−2.25(−x)^0.88`, with rank-dependent weights (γ = 0.61, δ = 0.69) on the
empirical distribution of overlapping h-month returns. The reference point is a zero nominal
return.
"""),
        code(r"""
P = st.pt_curve(X, H)
sb = st.summarise_gap(P, boot, "stock", "bond")
sf = st.summarise_gap(P, boot, "stock", "bill")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, s, lbl in ((axes[0], sb, "stocks - bonds"), (axes[1], sf, "stocks - bills")):
    t = s["table"]
    ax.plot(t.index, t["gap"], "o-", color=C["stock"], lw=2)
    ax.fill_between(t.index, t["gap_lo"], t["gap_hi"], color=C["stock"], alpha=0.15)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(12, color="k", ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("evaluation horizon (months)")
    ax.set_title(f"CPT gap {lbl}; break-even {s['break_even']:.1f} mo "
                 f"[{s['be_lo']:.1f}, {s['be_hi']:.1f}]", fontsize=10)
plt.tight_layout(); plt.show()
sb["table"].loc[[1, 3, 6, 12, 24, 36, 60, 120]].round(4)
"""),
        md(r"""
> 💡 **In plain words.** The shaded band is the range of history-replays. At one month it sits
> wholly below zero (bonds win). At ten years it is just above zero (stocks win). In between,
> where the crossing lives, it straddles zero for years, which is why the crossing can't be
> pinned down.
"""),
        code(r"""
fin = np.isfinite(sb["be_draws"])
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.hist(np.log10(sb["be_draws"][fin]), bins=35, color="#1f6feb", alpha=0.8)
ax.axvline(np.log10(12), color="k", ls="--", lw=2, label="12 months")
ax.set_xticks(np.log10([1, 3, 6, 12, 24, 60, 120]))
ax.set_xticklabels([1, 3, 6, 12, 24, 60, 120]); ax.set_xlabel("break-even (months, log)")
ax.legend(); plt.show()
print(f"finite in {fin.mean():.0%} of resamples; in [6, 24] months in "
      f"{np.mean((sb['be_draws'] >= 6) & (sb['be_draws'] <= 24)):.0%}")
print(f"bootstrap median {sb['be_median']:.1f} months")
"""),
        md(r"""
## 5 · Three estimators: why "one year" is the i.i.d. answer
"""),
        code(r"""
rows = []
for meth in ("overlapping", "nonoverlapping", "iid"):
    for lbl, df in (("1926-2018", X), ("1926-1990 (B&T)", X[X.index < "1991-01-01"])):
        Pm = st.pt_curve(df, H, method=meth)
        rows.append({"estimator": meth, "window": lbl,
                     "be_bonds": st.break_even(H, (Pm["stock"] - Pm["bond"]).to_numpy()),
                     "be_bills": st.break_even(H, (Pm["stock"] - Pm["bill"]).to_numpy())})
est = pd.DataFrame(rows)
vr = {}
for c in ("stock", "bond"):
    l1 = np.log1p(X[c].to_numpy()).var()
    vr[c] = [np.log1p(st.horizon_returns(X[c].to_numpy(), k)).var() / (k * l1) for k in H]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for c in vr:
    ax.plot(H, vr[c], "o-", color=C[c], lw=2, label=c)
ax.axhline(1, color="k", ls="--", lw=1.5, label="i.i.d.")
ax.set_xscale("log"); ax.set_xlabel("horizon (months)"); ax.set_ylabel("variance ratio")
ax.legend(); plt.show()
est.round(1)
"""),
        md(r"""
> 💡 **In plain words.** Stocks swing *more* over one to two years than independent months
> would imply (runs of bad years), and *less* over ten (mean reversion). Shuffling months
> independently erases both, and the one-to-two-year excess is exactly where the crossing sits.
> That is how an i.i.d. calculation lands near {{BE_IID}} while the real sequence gives {{BE}}.
> The bond leg's huge ratios are partly a by-product of building it from a monthly-*averaged*
> yield.

## 6 · Robustness: λ, α, weighting, regimes, real terms, maturity
"""),
        code(r"""
sw_w = st.param_sweep(X, weighting=True)
sw_n = st.param_sweep(X, weighting=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, sw, ttl in ((axes[0], sw_w, "with TK92 weighting"), (axes[1], sw_n, "no weighting")):
    im = ax.imshow(np.log10(sw.to_numpy()), cmap="viridis", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(sw.shape[1])); ax.set_xticklabels(sw.columns)
    ax.set_yticks(range(sw.shape[0])); ax.set_yticklabels(sw.index)
    ax.set_xlabel("alpha"); ax.set_ylabel("lambda"); ax.set_title(ttl, fontsize=10)
    for i in range(sw.shape[0]):
        for j in range(sw.shape[1]):
            ax.text(j, i, f"{sw.iloc[i, j]:.0f}", ha="center", va="center", color="w",
                    fontsize=8)
plt.colorbar(im, ax=axes, label="log10 break-even (months)"); plt.show()
print(f"lambda making an ANNUAL evaluator indifferent: {st.break_even_lambda(X, 12):.2f}")
"""),
        code(r"""
R = data.to_real(m)
cases = {"pre-1970": X[X.index < data.SPLIT], "post-1970": X[X.index >= data.SPLIT],
         "1957-2018 nominal": X[X.index >= R.index[0]], "1957-2018 real": R}
rows = []
for k, df in cases.items():
    Pr = st.pt_curve(df, H)
    br = st.bootstrap_curves(df, H, n_boot=200, seed=1026)
    s = st.summarise_gap(Pr, br, "stock", "bond")
    rows.append({"sample": k, "months": len(df), "be": s["break_even"], "lo": s["be_lo"],
                 "hi": s["be_hi"], "finite_share": s["share_finite"]})
mats = []
for mat in (5.0, 10.0, 20.0, 30.0):
    mm = data.load_monthly(maturity=mat)[["stock", "bond", "bill"]]
    Pm = st.pt_curve(mm, H)
    mats.append({"maturity": mat, "bond_vol": mm["bond"].std() * np.sqrt(12),
                 "be": st.break_even(H, (Pm["stock"] - Pm["bond"]).to_numpy())})
display(pd.DataFrame(rows).round(2))
pd.DataFrame(mats).round(3)
"""),
        md(r"""
> 💡 **In plain words.** Every knob moves the answer by years. Pre-1970 the crossing sits near a
> year ({{BE_PRE}}). Post-1970 it is {{BE_POST}}. In real terms it is {{BE_REAL}}. Every
> sub-sample's interval runs to "never within ten years". Loss aversion λ does most of the work.
> Without it (λ = 1) stocks win at any horizon.

## 7 · The discipline question: the myopic switcher

Rule (fixed before the run): at each period end, look at the stock market's period return.
After a loss, hold bills next period. After a gain, hold stocks. **One lag**: the period's sign
is known at its last close and sets the position from the next observation. Costs: 5 bp × 1×NAV
one-way per switch. The B&H-minus-switcher net return gap gets a Newey-West t and a paired
circular-block bootstrap (block ≥ 2 × the evaluation period). ΔSharpe is excess-of-bills against
excess-of-bills.
"""),
        code(r"""
tabs = []
if HAVE_DAILY:
    td = st.discipline_table(d, ["D", "W"], 252, cost_bps=5.0, n_boot=400, base_block=21)
    td["tape"] = "daily S&P price 1990-2018"
    tabs.append(td)
tm = st.discipline_table(X, ["M", "Q", "Y"], 12, cost_bps=5.0, n_boot=400)
tm["tape"] = "FF total return 1926-2018"
tabs.append(tm)
T = pd.concat(tabs)
cols = ["tape", "bh_cagr", "my_cagr_gross", "my_cagr_net", "d_cagr_net", "t_hac", "p_ret",
        "bh_sharpe", "my_sharpe_net", "p_sharpe", "time_invested", "switches_per_year"]
fig, ax = plt.subplots(figsize=(10, 4.6))
colors = ["#c0392b" if p < 0.05 else "#8b949e" for p in T["p_ret"]]
ax.bar([st.FREQ_LABEL[f] for f in T.index], T["d_cagr_net"] * 100, color=colors)
ax.axhline(0, color="k", lw=1)
ax.set_ylabel("CAGR given up vs buy-and-hold (pp/yr, net)")
ax.set_title("red = significant at 5% (paired block bootstrap)", fontsize=10); plt.show()
T[cols].round(3)
"""),
        code(r"""
if HAVE_DAILY:
    cs = st.cost_sweep(d, "D", 252)
    print("daily checker, cost sweep (S&P price 1990-2018):")
    print(cs.round(4).to_string())
cs_m = st.cost_sweep(X, "M", 12)
print("\nmonthly checker, cost sweep (FF 1926-2018):")
print(cs_m.round(4).to_string())
"""),
        md(r"""
> 💡 **In plain words.** The daily panicker loses about {{COST_D}} a year (t ≈ {{T_D}}). Part
> of that is costs and part is that daily moves in this period tended to reverse, so they sold
> lows and bought highs. Checking monthly to yearly, the loss is about 1% a year with p
> {{P_MQY}}, and the slow switcher's Sharpe is no worse, because it sits out a third of the time
> and gives up premium it wasn't being paid risk for. On the daily tape the price-only S&P
> *understates* the daily cost: buy-and-hold misses dividends while the switcher earns bills.

## 8 · Synthetic control: machinery proof, not market evidence
"""),
        code(r"""
rows = []
for s_ in (1.0, 0.5, 0.0):
    dfs, tr = data.synthetic_monthly(n_months=6000, signal_strength=s_, seed=1025)
    Ps = st.pt_curve(dfs, H)
    rows.append({"signal_strength": s_,
                 "premium_pa": tr["equity_premium_monthly"] * 12,
                 "be_bonds": st.break_even(H, (Ps["stock"] - Ps["bond"]).to_numpy()),
                 "be_bills": st.break_even(H, (Ps["stock"] - Ps["bill"]).to_numpy())})
fin = []
for seed in range(30):
    dfs, _ = data.synthetic_monthly(n_months=len(X), signal_strength=1.0, seed=seed)
    Ps = st.pt_curve(dfs, H, cols=["stock", "bill"])
    fin.append(st.break_even(H, (Ps["stock"] - Ps["bill"]).to_numpy()))
fin = np.array(fin)
print("planted premium, 92-year samples: break-even vs bills, 10/50/90th pct:",
      np.round(np.percentile(fin[np.isfinite(fin)], [10, 50, 90]), 1))
pd.DataFrame(rows).round(3)
"""),
        md(r"""
> 💡 **In plain words.** With no premium planted, no horizon makes stocks attractive, as it
> should. With a premium of known size, the method finds a crossing near a year when given 500
> years of data. Given 92 years of the *same* world, it scatters widely. The wide interval on
> the real tape is what this much data can resolve. The method isn't at fault.

## 9 · Going further

- **Treasury bond leg.** A true long-Treasury total-return series is the clean B&T comparison.
  The AAA construction adds a credit spread and averaging smoothness.
- **Portfolio framing.** B&T also solved for the CPT-optimal stock/bond *mix* at each horizon.
  Adding that optimisation is a natural fork, and it inherits the same interval problem.
- **International tapes.** The US equity century is a survivor. A loss-averse investor in a
  market that suffered a long drawdown may never have reached a finite break-even.
- **Other switching rules** (partial de-risking, time-outs, reference points that move with the
  last peak) would cost different amounts. Test them as a family, with a Reality Check, so a
  rule isn't picked for its result.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


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
