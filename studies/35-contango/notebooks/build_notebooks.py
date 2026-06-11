"""Generate the two narrative notebooks for Study 35 (Contango) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE: the **real** energy roll-yield tape is read from the cached ETF pairs
(_cache/energy_carry_etfs.parquet — USO/USL, UNG/UNL; populate once with `examples/verify.py --fetch`), and
the cross-sectional bucket **machinery** is proved on the seeded synthetic commodity panel. Both notebooks
walk the SAME seven desk beats. Contango is the commodity sibling of Study 27 (FX carry).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../.."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from contango import data, strategy, costs, extension, energy

# Real energy tape: front-month vs 12-month-laddered ETF pairs (USO/USL, UNG/UNL). Their return gap IS the
# term-structure roll — no paid feed needed. Populate once with: python examples/verify.py --fetch
prices = energy.load_pairs()
HAVE_TAPE = not prices.empty
print("real energy tape:", "loaded" if HAVE_TAPE else "cache absent (run verify.py --fetch)",
      "" if not HAVE_TAPE else f"{prices.index.min().date()} -> {prices.index.max().date()}")
# Offline synthetic control: a 12-commodity panel where roll yield predicts return (the bucket machinery) + a null.
r, ry, truth = data.synthetic_term_structure(carry_strength=0.9, seed=35)
r0, ry0, _ = data.synthetic_term_structure(carry_strength=0.0, seed=35)
"""

# Frozen headline numbers (real tape: as-of 2026-06-05, fp 92a7674a430b; synthetic: fp b502aaa6304f).
R = dict(
    uso="-76", usl="+4", wti_gap="80", wti_drag="5.1", wti_t="1.53",
    ung="-99", unl="-88", gas_gap="11", gas_drag="8.9", gas_t="1.75",
    wti_sh="+0.35", wti_cagr="+6.4", wti_dd="-76", wti_skew="+0.71", wti_bt="1.45", uso_long_sh="-0.01", uso_long_dd="-98",
    gas_sh="+0.04", gas_bt="0.17", ung_long_sh="-0.35", ung_long_dd="-99",
    combo_sh="+0.16", combo_cagr="+0.4", combo_dd="-83", combo_t="0.66", combo_net="+0.12", turn="14",
    syn_sh="1.86", syn_hml="27.6", syn_null="-0.28", syn_turn="0.19", syn_be="160",
    blend="2.03", blend_carry="1.80", blend_mom="1.43", blend_corr="+0.27",
    fp="92a7674a430b", syn_fp="b502aaa6304f", asof="2026-06-05",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-cf222e?style=flat-square)\n"
    "![Real-tape run?: Done](https://img.shields.io/badge/Real--tape_run%3F-Done-2ea44f?style=flat-square)\n\n"
)


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


# --- shared code cells (execute live on the real cached tape) -------------------------------------------
BLEED_CODE = (
    "if HAVE_TAPE:\n"
    "    bt = energy.bleed_table(prices)\n"
    "    print(bt[['front_total_pct','lad_total_pct','gap_pct','ann_drag_pct','weeks_in_contango_pct','drag_hac_t']].round(2))\n"
    "    fig, ax = plt.subplots()\n"
    "    for cmd,(front,lad) in energy.PAIRS.items():\n"
    "        sub = prices[[front,lad]].dropna(); base = sub/sub.iloc[0]\n"
    "        ax.plot(base.index, base[front], lw=1.1, label=f'{front} (front, {cmd})')\n"
    "        ax.plot(base.index, base[lad], lw=1.1, ls='--', label=f'{lad} (12-mo laddered, {cmd})')\n"
    "    ax.set_yscale('log'); ax.legend(fontsize=8); ax.set_title('The contango bleed: front-month vs laddered, same underlying')\n"
    "else:\n"
    "    print('cache absent — run: python examples/verify.py --fetch')"
)

TIMING_CODE = (
    "if HAVE_TAPE:\n"
    "    rows = {}\n"
    "    for cmd in list(energy.PAIRS) + [None]:\n"
    "        s = energy.book_summary(prices, commodity=cmd)\n"
    "        rows['WTI+GAS combo' if cmd is None else cmd] = {k: s[k] for k in ('sharpe','cagr','max_drawdown','skew','hac_t','turnover_per_yr')}\n"
    "    print(pd.DataFrame(rows).T.round(3))\n"
    "    for cmd in energy.PAIRS:\n"
    "        al = energy.summary(energy.always_long_front(prices, cmd))\n"
    "        print(f\"  always-long {energy.PAIRS[cmd][0]}: Sharpe {al['sharpe']:+.2f}  CAGR {al['cagr']*100:+.1f}%  maxDD {al['max_drawdown']*100:+.0f}%\")\n"
    "else:\n"
    "    print('cache absent — run: python examples/verify.py --fetch')"
)

SYN_CODE = (
    "for label, rr, yy in [('carry panel', r, ry), ('null (disconnected)', r0, ry0)]:\n"
    "    s = strategy.summary(strategy.book_returns(rr, yy, cost_bps=0.0))\n"
    "    pb = strategy.carry_premium_by_bucket(rr, yy)\n"
    "    print(f\"{label:22} gross Sharpe {s['sharpe']:+6.2f}  H-L {pb['hml_ann_pct']:+6.1f}%/yr  turnover/wk {strategy.turnover(yy):.3f}\")"
)

BLEND_CODE = (
    "c = extension.combine(r, ry, cost_bps=5.0)\n"
    "print(f\"carry {c['carry_sharpe']:.2f}   momentum {c['momentum_sharpe']:.2f}   blend {c['blend_sharpe']:.2f}   (leg corr {c['correlation']:+.2f})\")\n"
    "eqc=(1+c['carry']).cumprod(); eqm=(1+c['momentum']).cumprod(); eqb=(1+c['blend']).cumprod()\n"
    "fig, ax = plt.subplots()\n"
    "ax.plot(eqc.index, eqc.values, color='#2ea44f', lw=1.0, label=f\"carry ({c['carry_sharpe']:.2f})\")\n"
    "ax.plot(eqm.index, eqm.values, color='#8b949e', lw=1.0, label=f\"momentum ({c['momentum_sharpe']:.2f})\")\n"
    "ax.plot(eqb.index, eqb.values, color='#c0392b', lw=1.3, label=f\"50/50 blend ({c['blend_sharpe']:.2f})\")\n"
    "ax.legend(); ax.set_title('Machinery: lowly-correlated legs, the blend beats either standalone')"
)


def build_curious():
    cells = [
        md(
            "# Contango 🛢️\n"
            "### The roll yield is real — and it quietly destroyed the most popular oil ETF. Can you trade it? Barely.\n\n"
            + BADGES +
            "A commodity future doesn't just track the spot price — as your long position rolls toward expiry it "
            "slides along the **term-structure curve**. If the curve is **backwardated** (front dearer than the "
            "deferred contract) you roll *up* and bank a positive **roll yield**; if it's in **contango** (front "
            "cheaper) you roll *down* and pay a tax. You don't need a fancy futures feed to see this: the "
            "front-month oil ETF **USO** and the 12-month-laddered **USL** track the *same* crude and differ only "
            "in *where on the curve* they sit — so the gap between them **is** the roll.\n\n"
            "> 📓 **Plain-language layer.** The bleed table, the curve-timing book and the carry+momentum blend "
            "are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ✅ **Real run, no paid data.** Numbers below are live from the cached ETF pairs (USO/USL, UNG/UNL); "
            "the bucket machinery is proved on a synthetic control. Not investment advice. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🛢️\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            f"| Is the roll yield real? | 🟩 **Yes, and it's huge.** On the real tape the front-month **USO** lost "
            f"**{R['uso']}%** while the laddered **USL**, on the *same* crude, was **{R['usl']}%** — an "
            f"**{R['wti_gap']}-point** contango tax (gas UNG is **{R['ung']}%**). |\n"
            f"| Could you trade it? | 🟥 **Barely.** Timing the front by the curve points the right way (WTI Sharpe "
            f"**{R['wti_sh']}** vs −0.01 buy-and-hold) but the combined book is statistically flat "
            f"(**{R['combo_sh']}**, HAC *t* **{R['combo_t']}**) with an **{R['combo_dd']}%** drawdown. |\n"
            "| Did we measure the real tape? | 🟩 **Yes.** Straight from liquid ETFs — no FRED, no EIA, no paid "
            "curve feed. |\n\n"
            "> Desk shorthand: **Signal `REAL` · Tradability `MIRAGE` · Real-tape run? `DONE`** — the cost is "
            "real and brutal; harvesting it as positive carry is a mirage. The value of the signal is "
            "*defensive*: don't be the sucker holding the front-month in contango."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Commodity carry / roll yield (Kakushadze–Serur §9.1, §9.4; lineage Gorton–Rouwenhorst 2006, "
            "Erb–Harvey 2006; Koijen et al. 2018): a backwardated curve *pays you to hold the future*, a "
            "contangoed one taxes you. So the front-month should systematically bleed against a contract further "
            "out whenever the curve is in contango. The cleanest real-world witness is the pair of oil ETFs on "
            "the same barrel — front-month USO vs 12-month-laddered USL."
        ),
        md("### The bleed, on the real tape"),
        code(BLEED_CODE),

        md(
            "## 2 · So what? 💰\n\n"
            "USO is one of the most-traded commodity ETFs on Earth, and it has bled three-quarters of its value to "
            "the **roll** — not to the oil price (USL, on the same oil, was roughly flat). This is the single "
            "clearest demonstration that the term structure, not spot, drives commodity returns over time "
            "(Erb–Harvey 2006). For a desk it means two things: (1) there's a real carry premium in the curve, and "
            "(2) naively holding the front-month is a slow-motion disaster. The question is whether you can turn "
            "(1) into money without being wrecked by the volatility — the `MIRAGE` half of the verdict."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "1. **Real?** Does the front-month bleed against the ladder in contango, and roll *up* in "
            "backwardation? (the bleed table)\n"
            "2. **Tradable?** A curve-timing book — long the front only when backwardated, short it in contango — "
            "vs simply holding it. Sharpe, drawdown, and a Newey–West *t*.\n"
            "3. **Diversifiable?** Does adding a momentum sleeve lift the combined Sharpe? (beat 7, machinery)\n\n"
            "**Mirage line** (pre-registered): if the real roll spread is statistically indistinguishable from "
            "zero (HAC *t* < 2), or only the liquid contracts (which carry least) are tradable, the *tradable* "
            "signal drops to `WEAK`/`MIRAGE`."
        ),

        md("## 4 · The teardown 🔧\n\n### 4a · Time the curve, or just hold the front?"),
        code(TIMING_CODE),
        md(
            f"> 💡 **In plain words.** Knowing the curve *matters*: timing WTI earns Sharpe **{R['wti_sh']}** with "
            f"**positive** skew and dodges USO's **{R['uso_long_dd']}%** drawdown. But as a standalone edge it "
            f"vanishes — the combined book is **{R['combo_sh']}** at HAC *t* **{R['combo_t']}** (indistinguishable "
            "from zero) with an enormous drawdown. Two liquid energy curves are too few and too crash-prone to "
            "harvest carry cleanly."
        ),

        md("### 4b · Why the machinery is trustworthy — the synthetic control"),
        code(SYN_CODE),
        md(
            f"> 💡 **In plain words.** On a 12-commodity panel where roll yield *truly* predicts return, the "
            f"cross-sectional carry book recovers a **+{R['syn_hml']}%/yr** spread (gross Sharpe **{R['syn_sh']}**); "
            f"on a disconnected null it earns nothing (**{R['syn_null']}**). The apparatus measures carry where it "
            "exists — so when it finds only a mirage on the two real energy curves, that's the *market*, not a bug."
        ),

        md("## 5 · The verdict 🧾\n\n"
           f"- **Signal `REAL`** — the contango bleed is real and enormous (USO {R['uso']}% vs USL {R['usl']}%; "
           f"+{R['wti_drag']}%/yr WTI, +{R['gas_drag']}%/yr gas).\n"
           f"- **Tradability `MIRAGE`** — the curve-timing book is statistically flat (combo {R['combo_sh']}, HAC "
           f"*t* {R['combo_t']}) with an {R['combo_dd']}% drawdown; cost isn't the killer, concentration and the "
           "crash tail are.\n"
           "- **Real-tape run `DONE`** — measured straight from liquid ETFs, fingerprinted and as-of pinned.\n\n"
           "> **The commodity sibling of Steamroller.** A real premium you mostly can't keep: the signal's worth "
           "is knowing *not* to hold the bleeding front-month."),

        md("## 6 · Could you trade it? 💸\n\n"
           "- **The cost is real and one-directional.** Holding USO in contango bleeds ~5%/yr (gas ~9%/yr). The "
           "first, biggest win is simply *not* doing that.\n"
           "- **Timing it doesn't clear the noise.** The curve-timing book is the right idea and beats buy-and-"
           "hold, but its Sharpe is statistically zero and it still draws down >80% on two volatile names.\n"
           "- **Cost isn't the constraint.** A 10 bp round-trip barely moves the combined book "
           f"({R['combo_sh']} → {R['combo_net']}); the crash tail and two-name concentration are.\n"
           "- **The honest move:** treat roll yield as a *risk to avoid* (ladder your exposure, like USL) rather "
           "than an alpha to lever."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — \"diversify the carry with momentum\" ([`../docs/extension.md`](../docs/extension.md))\n"
            "Carry and time-series momentum are the two classic commodity premia and are *lowly correlated* "
            "(Koijen et al. 2018). On the bucket machinery a 50/50 blend beats the stronger leg:"
        ),
        code(BLEND_CODE),
        md(
            f"> 💡 **In plain words.** Leg correlation is just **{R['blend_corr']}**, so the blend Sharpe "
            f"(**{R['blend']}**) beats either leg (carry {R['blend_carry']}, momentum {R['blend_mom']}). On this "
            "desk the edge is *diversification*, not prediction — echoing [Trade-Winds](../../31-trade-winds/).\n\n"
            "### Other forks\n"
            "- **Add more curves** — a broader basket (metals, ags via laddered ETFs) might revive the cross-section.\n"
            "- **Vol-target the timing book** — does constant-risk sizing tame the −80% drawdown (cf. Study 27)?\n"
            "- **Trade the calendar spread directly** — long-deferred / short-front as a cleaner roll-yield express.\n\n"
            "PRs welcome — add curves, vol-target, or trade the spread directly."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Contango — a quantitative teardown 🔬\n"
            "### The real energy bleed · the curve-timing book vs buy-and-hold · the bucket machinery on a control · the carry+momentum blend\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven beats, "
            "every claim with its number.* The steelman: the commodity roll yield is a real, documented carry "
            "premium (Gorton–Rouwenhorst 2006; Erb–Harvey 2006; Koijen et al. 2018). On the real energy tape we "
            f"find the premium is **`REAL`** and economically huge (USO **{R['uso']}%** vs USL **{R['usl']}%**), "
            f"but **`MIRAGE`** to harvest (curve-timing combo Sharpe **{R['combo_sh']}**, HAC *t* **{R['combo_t']}**, "
            f"drawdown **{R['combo_dd']}%**). The bucket machinery is validated on a synthetic control.\n\n"
            "> ✅ **Not investment advice.** The real tape is the cached front/laddered ETF pairs (no paid feed); "
            "the cross-sectional bucket apparatus executes on a synthetic roll-yield panel. Sources in "
            "[`../docs/references.md`](../docs/references.md); full numbers in [`../docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — is the roll yield real? | 🟢 `REAL` | Real tape: USO **{R['uso']}%** vs USL "
            f"**{R['usl']}%** (gap {R['wti_gap']} pts); roll drag **+{R['wti_drag']}%/yr** WTI, **+{R['gas_drag']}%/yr** gas. |\n"
            f"| **Tradability** | 🔴 `MIRAGE` | Curve-timing combo Sharpe **{R['combo_sh']}**, HAC *t* **{R['combo_t']}** "
            f"(≈0), drawdown **{R['combo_dd']}%**; cost isn't the killer. |\n"
            "| **Real-tape run?** | 🟢 `Done` | Measured from liquid ETF pairs — no FRED, no EIA, no paid curve. |\n\n"
            "> **In one sentence:** a real, enormous commodity roll-yield cost (the USO bleed) that you mostly "
            f"can't harvest — the timing book is statistically flat with an {R['combo_dd']}% drawdown — so the "
            "signal's value is defensive.\n\n"
            f"*(Real numbers as-of {R['asof']}, fingerprint `{R['fp']}`; synthetic control fp `{R['syn_fp']}`.)*"
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "A long futures position earns the roll $y_{i,t}$ = the curve slope (backwardation $>0$, contango "
            "$<0$). We observe it without a curve feed: for each energy commodity, the **front-month** ETF and the "
            "**12-month-laddered** ETF on the *same* underlying differ only in curve placement, so the weekly "
            "return spread $r^{\\text{lad}}-r^{\\text{front}}$ is the realized roll cost borne by the front. The "
            "bucket book (long most-backwardated, short most-contangoed) is the cross-sectional version, validated "
            "on the synthetic panel."
        ),
        md("### The realized roll cost, per energy commodity"),
        code(BLEED_CODE),
        md(
            f"> 💡 **In plain words.** The front-month bleeds against the ladder **{R['wti_drag']}%/yr** in crude and "
            f"**{R['gas_drag']}%/yr** in gas, in contango 53–56% of weeks. The weekly spread's HAC *t* is "
            f"+1.5–1.8 — noisy week-to-week, but the cumulative tax (an {R['wti_gap']}-point USO/USL gap) is "
            "overwhelming and never in doubt."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Gorton–Rouwenhorst (2006) show the commodity basket's premium is dominated by the **roll return**; "
            "Erb–Harvey (2006) show the cross-section is explained far more by term structure than spot; Koijen et "
            "al. (2018) generalise 'carry' across asset classes. The real energy tape confirms the *force* "
            "spectacularly (USO's bleed) and lets us ask the only open question: can you harvest it on the "
            "contracts liquid enough to actually trade? Beats 4–6 answer no."
        ),

        md(
            "## Beat 3 · Pre-registered protocol\n\n"
            "1. **Real?** `energy.bleed_table` on the ETF pairs (drag, % weeks in contango, HAC *t*).\n"
            "2. **Tradable?** `energy.book_summary` — the curve-timing book per commodity and combined, vs "
            "`energy.always_long_front`; Sharpe, drawdown, HAC *t*, turnover, net of cost.\n"
            "3. **Diversifiable?** `extension.combine` on the synthetic control — carry + momentum blend.\n\n"
            "**Mirage line:** real roll spread with HAC *t* < 2, or a premium tradable only in the (few, illiquid) "
            "right contracts ⇒ `MIRAGE`. The combined timing book lands at HAC *t* ≈ 0.7 — inside the mirage."
        ),

        md("## Beat 4 · The teardown\n\n### 4a · The curve-timing book — right sign, no edge"),
        code(TIMING_CODE),
        md(
            f"> 💡 **In plain words.** Timing WTI by the curve (Sharpe **{R['wti_sh']}**, positive skew) trounces "
            f"holding USO (**{R['uso_long_sh']}**, **{R['uso_long_dd']}%** drawdown) — the signal is real. But gas "
            f"timing earns nothing (**{R['gas_sh']}**, *t* {R['gas_bt']}) and the combined book is **{R['combo_sh']}** "
            f"at HAC *t* **{R['combo_t']}** with an **{R['combo_dd']}%** drawdown. Statistically flat."
        ),

        md("### 4b · The bucket machinery, proved on a control (and flat on a null)"),
        code(SYN_CODE),
        md(
            f"> 💡 **In plain words.** Where a broad cross-section of roll yields exists, the bucket book harvests "
            f"a **+{R['syn_hml']}%/yr** spread (Sharpe **{R['syn_sh']}**) and the null is flat (**{R['syn_null']}**). "
            "The apparatus works; the real energy tape just doesn't offer enough liquid curves to deploy it — so "
            "we time the two we have, and that's the mirage."
        ),

        md("### 4c · Cost is not the binding constraint"),
        code(
            "if HAVE_TAPE:\n"
            "    for c in (0, 5, 10, 25):\n"
            "        s = energy.book_summary(prices, commodity=None, cost_bps=c)\n"
            "        print(f'  combo @{c:>2}bp  Sharpe {s[\"sharpe\"]:+.2f}  CAGR {s[\"cagr\"]*100:+.1f}%')\n"
            "else:\n"
            "    print('cache absent — run: python examples/verify.py --fetch')"
        ),
        md(
            f"> 💡 **In plain words.** The book turns over ~{R['turn']}×/yr, yet a 10 bp round-trip only nicks it "
            f"({R['combo_sh']} → {R['combo_net']}). Costs don't kill this book — the crash-prone, two-name "
            "concentration does. That's the `MIRAGE`."
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`REAL`** (beat 1): USO {R['uso']}% vs USL {R['usl']}%; roll drag +{R['wti_drag']}%/yr WTI, "
           f"+{R['gas_drag']}%/yr gas — one of the largest, most durable costs in commodities.\n"
           f"- **`MIRAGE`** (4a/4c): curve-timing combo Sharpe {R['combo_sh']}, HAC *t* {R['combo_t']}, drawdown "
           f"{R['combo_dd']}%; not a tradable positive-carry edge on the liquid energy curves.\n"
           "- **Real-tape `DONE`** (beat 3): measured from cached ETF pairs, fingerprinted & as-of pinned.\n\n"
           "> **Signal `REAL` · Tradability `MIRAGE` · Real-tape run? `DONE`** — the commodity sibling of Steamroller."),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **The defensive trade is the real one.** Avoiding front-month contango (ladder your exposure, like "
           "USL) saves ~5%/yr in crude, ~9%/yr in gas. That's the harvestable part.\n"
           "- **The offensive trade is a mirage.** Timing the curve is directionally right but statistically flat "
           "and deeply drawdown-prone on two volatile names.\n"
           "- **Capacity & the illiquidity tilt.** A broader carry cross-section lives in smaller, less-liquid "
           "contracts; the deeply liquid energy curves carry the premium *and* the crash. Same tension as Slingshot.\n"
           "- **Honest fix:** diversify the machinery (beat 7); don't lever two energy names."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — the carry+momentum blend ([`../docs/extension.md`](../docs/extension.md))\n"
            "Carry and time-series momentum are the two classic commodity premia and are lowly correlated "
            "(Koijen et al. 2018). On the bucket machinery a 50/50 risk blend beats the stronger leg:"
        ),
        code(BLEND_CODE),
        md(
            f"> 💡 **In plain words.** Leg correlation **{R['blend_corr']}**; the blend Sharpe (**{R['blend']}**) "
            f"beats either leg (carry {R['blend_carry']}, momentum {R['blend_mom']}). The institutional answer to "
            "carry's crash isn't leverage; it's a lowly-correlated sleeve. Echoes [Trade-Winds](../../31-trade-winds/).\n\n"
            "### 7b · Other forks\n"
            "- **Broaden the curve set** — laddered ETFs in metals/ags to rebuild a tradable cross-section.\n"
            "- **Vol-target the timing book** — does constant-risk sizing tame the −80% drawdown (cf. Study 27)?\n"
            "- **Trade the calendar spread** — long-deferred / short-front as a purer roll-yield express.\n\n"
            "PRs welcome — broaden the curves, vol-target, or trade the spread directly."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
