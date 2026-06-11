"""Generate the two narrative notebooks for Study 04 (Social-Oracle) from source.

Like Studies 01–03, the notebooks are a *generated artefact*: edit the cell text
here, rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE on the **seeded synthetic universe** — a toy panel with
a deliberately wired pump-and-fade — so the cells prove the *machinery*, deterministically
and with no network. The **verdict numbers are real**: the fingerprinted run on 1,468
r/WallStreetBets surges lives in ../docs/results_wsb.md (built by examples/verify_wsb.py)
and both heroes quote it from there, exactly like Studies 31–39. Never let a synthetic
cell output masquerade as the real tape.

Both notebooks follow the SAME seven desk beats (see ../../../METHODOLOGY.md):
  0 Verdict · 1 The Claim · 2 So What? · 3 How We'd Know · 4 The Teardown ·
  5 The Verdict · 6 Could You Trade It? · 7 Going Further
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))  # study root (social_oracle/ lives there)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from social_oracle import data, mentions, eventstudy, benchmark, backtest, robustness

# SYNTHETIC CONTROL — the cells below run a seeded toy universe with a pump-and-fade
# wired in, to prove the machinery offline and deterministically. The REAL numbers
# (1,468 r/WallStreetBets surges) live in ../docs/results_wsb.md, built by
# examples/verify_wsb.py — that run, not these cells, is the study's verdict.
panel, feed = data.synthetic_panel(seed=0)
events, coverage = mentions.to_events(feed, panel)
print(f"SYNTHETIC control: {len(panel)} names, {len(feed)} mentions -> {len(events)} clean events")
print("coverage:", coverage)
"""

# Real-tape numbers quoted in the heroes and beat-5 cells — from ../docs/results_wsb.md
# (examples/verify_wsb.py, as-of 2026-06-01, price fingerprint 1a11c294eeba).
R = dict(
    fp="1a11c294eeba", asof="2026-06-01", n_events="1,468", n_names="224",
    span="2021-01-11 → 2025-12-29", missing="42",
    ex1="+0.08", p1="0.23", ex5="+0.05", p5="0.40", ex21="-0.66", p21="0.94",
    boot21_p="0.85", boot21_lo="-1.94", boot21_hi="+0.62", boot5_p="0.45",
    hot21="-1.06", hot21_p="0.97", pos21="45.7", pos21_rand="51.4",
    gross="+0.72", abn="+0.05", med_net="-1.3", win="45.1", sleeve_sh="-0.006",
    sleeve_dd="-84", sleeve_tot="-44", be_halfspread="25",
)

BANNER = (
    "> 🧪 **What executes vs what's real.** The code cells below run the **synthetic "
    "control** — a seeded toy universe with a pump-and-fade deliberately wired in, so the "
    "machinery is proven offline and deterministically. The **real numbers** — the "
    f"fingerprinted run on **{R['n_events']}** r/WallStreetBets surges (as-of {R['asof']}, "
    f"fp `{R['fp']}`) — live in **[docs/results_wsb.md](../docs/results_wsb.md)** and are "
    "quoted, sourced, wherever a verdict is called. Don't read a synthetic cell output as "
    "the market."
)


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Social-Oracle 🔮\n"
            "### Does buying what a viral stock guru mentions actually pay? Tested honestly, "
            "in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Pump--and--fade: Directional only](https://img.shields.io/badge/Pump--and--fade-Directional_only-8b949e?style=flat-square)\n\n"
            "Every cycle a retail-investing folk hero goes viral, and within weeks GitHub fills "
            "with bots that scrape her posts, pull out the `$TICKERS`, and score them as buy "
            "signals — *her timeline front-runs the market, just buy what she mentions.* On the "
            f"real tape — **{R['n_events']} r/WallStreetBets surges** "
            "([docs/results_wsb.md](../docs/results_wsb.md)) — a mention carries **no abnormal "
            f"edge over a random day** (excess {R['ex1']}% / {R['ex5']}% / {R['ex21']}% at "
            f"1d/1wk/1mo, p = {R['p1']}/{R['p5']}/{R['p21']}), the gross 'gain' is market beta "
            "the costs erase, and the month-ahead drift points *down* without ever reaching "
            "significance. It's a pump you're late to, dressed as a signal.\n\n"
            "> 📓 **This is the plain-language layer.** Want the statistics, the microstructure "
            "and the capacity maths? That's the companion notebook, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** An educational, reproducible research tool: every "
            "chart below is generated by the code beside it. It tests a *phenomenon*, not a "
            "person. House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n\n"
            + BANNER
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict (real tape) 🎯\n\n"
            f"From the fingerprinted run on **{R['n_events']}** WSB surges, {R['span']} "
            "([docs/results_wsb.md](../docs/results_wsb.md)):\n\n"
            "| What we asked | The real-tape answer |\n"
            "|---|---|\n"
            f"| Does a mention beat a random day? | ❌ **No** — excess {R['ex1']}% at 1d "
            f"(p = {R['p1']}), {R['ex5']}% at 1wk (p = {R['p5']}), {R['ex21']}% at 1mo "
            f"(p = {R['p21']}). |\n"
            f"| Does \"mentioned\" beat \"already hot\"? | ❌ **No** — by a month the mention does "
            f"*{R['hot21']}%* relative to a name that was simply already running. |\n"
            f"| Does the pop reverse? | ⚪ **It points down, but not significantly** — the clustered "
            f"bootstrap p that the 1-mo excess ≤ 0 is {R['boot21_p']}, CI "
            f"[{R['boot21_lo']}%, {R['boot21_hi']}%] straddling zero. A direction, not a finding. |\n"
            f"| Could a follower trade it? | ❌ **Not after costs** — median trade {R['med_net']}%, "
            f"sleeve Sharpe {R['sleeve_sh']}, max drawdown {R['sleeve_dd']}%, and {R['missing']} of "
            "the most-viral names literally delisted. |\n\n"
            "> Desk shorthand: **Signal `NONE` · Tradability `MIRAGE` · Pump-and-fade "
            "`DIRECTIONAL ONLY`** — the cells below show the *method* that earned those stamps, "
            "on the synthetic control."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "A viral persona (here *白毛股神* **Serenity**, @aleabitoreddit) gets her posts "
            "scraped into cashtag signals by a wave of open-source repos. The strongest version "
            "of the claim: *a public mention is, on average, followed by a gain you could have "
            "captured.* We test that — on **abnormal** returns (the name minus the market), so a "
            "rising tide that lifted everything doesn't count as a call."
        ),
        code(
            "es = eventstudy.event_study(panel, events, horizon=21, pre=5)\n"
            "m = es['matrix'].mean()\n"
            "plt.axvline(0, color='0.7', lw=1); plt.axhline(0, color='0.7', lw=1)\n"
            "plt.plot(m.index, 100*m.values, lw=2)\n"
            "plt.title(f\"Average abnormal path around a mention  (n={es['n_events']} events)\")\n"
            "plt.xlabel('trading days from the mention'); plt.ylabel('cumulative abnormal return (%)')\n"
            "plt.show()"
        ),
        md(
            "See the shape? It climbs **into** day 0 (the run-up the tweet is chasing) and "
            "**bleeds** afterwards. The follower enters on the right-hand slope."
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "If attention really paid, anyone with an API key would have free alpha. If it "
            "*doesn't* — if it's a late, fading pop on names too thin to exit — then thousands "
            "of followers are buying **negatively-skewed attention beta** and calling it skill.\n\n"
            "> The lesson Studies 02–03 keep finding from new angles: *a pattern can be obvious "
            "to the eye and still be the opposite of an edge once you ask 'more than a random "
            "day? more than the momentum it already had? net of what I'd actually pay?'*"
        ),

        md(
            "## 3 · How we'd know 🔍\n\n"
            "Two controls stacked, because these names are volatile by selection:\n\n"
            "1. Did it beat a **random day** in the same universe? (the desk's usual yardstick)\n"
            "2. Did the **mention** beat a name that was simply **already hot**? (momentum is "
            "the confound — attention follows performance)\n\n"
            "Plus the **fade**: we trace the abnormal return day by day and watch it reverse."
        ),
        code(
            "print('random-day null:')\n"
            "display(benchmark.conditional_vs_unconditional(panel, events, n_iter=800).round(4))\n"
            "print('\\nmomentum control (mention vs already-hot):')\n"
            "hot = mentions.hot_streak_events(panel)\n"
            "display(benchmark.excess_vs_alternative(panel, events, hot, n_iter=800).round(4))"
        ),

        md(
            "## 4 · The teardown 🔬\n\n"
            "The fade, in one table — mean abnormal return at each horizon. A peak that "
            "reverses is the follower's whole problem."
        ),
        code(
            "robustness.fade_curve(panel, events)"
        ),

        md(
            "## 5 · The verdict ⚖️\n\n"
            "On the synthetic the method behaves exactly as designed: the mention path runs up "
            "into the tweet, fades after, and **loses** to both a random day and a hot streak — "
            "the machinery catches a wired-in pump-and-fade.\n\n"
            "On the **real tape** ([docs/results_wsb.md](../docs/results_wsb.md)) the same "
            f"gauntlet finds *nothing to catch*: excess over a random day {R['ex1']}% / "
            f"{R['ex5']}% / {R['ex21']}% at 1d/1wk/1mo (p = {R['p1']}/{R['p5']}/{R['p21']}), and "
            "the month-ahead fade, while pointing down everywhere, never clears the clustered "
            f"bootstrap (p = {R['boot21_p']}). **Signal `NONE` · Tradability `MIRAGE` · "
            "Pump-and-fade `DIRECTIONAL ONLY`.** The next beat is why a follower loses even the "
            "flicker."
        ),

        md(
            "## 6 · Could you trade it? 🏦\n\n"
            "You read the post *after* it's public — so your entry is the **next open**, past "
            "the pop. The names are \\$1–3 micro-caps with brutal spreads. Charge that, twice, "
            "and watch the mean trade sink as the spread widens."
        ),
        code(
            "res = backtest.run(panel, events, hold_days=10)\n"
            "print({k: (round(v,4) if isinstance(v,float) else v) for k,v in res.stats.items()})\n"
            "backtest.cost_sweep(panel, events)"
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **The inversion:** if the drift points down, the side that *might* pay is being "
            "**early or short the fade**, not the late follower — same punchline as Fear-Gauge "
            "(*sell* the fear, don't buy it). On the real tape the fade is directional, not "
            "significant — so even the inversion is unproven, and shorting \\$1–3 names costs "
            "borrow.\n"
            "- **Bring another feed:** the real WSB run is [docs/results_wsb.md](../docs/results_wsb.md) "
            "(rebuild via `examples/verify_wsb.py`); the same `data.load_feed('mentions.csv')` "
            "swap works for any guru's timeline you can export.\n"
            "- **Conviction & first-mention:** does a high-score or first-ever mention behave "
            "differently?\n\n"
            "The deep version — t-stats, clustering bootstrap, name jackknife, capacity — is in "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Social-Oracle — a quantitative teardown 🔬\n"
            "### Mention event study · abnormal-return CARs · random-day null · momentum "
            "control · the fade · clustering bootstrap · name jackknife · micro-cap capacity\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Pump--and--fade: Directional only](https://img.shields.io/badge/Pump--and--fade-Directional_only-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We take the "
            "social-trading claim seriously, then ask whether a public mention is anything more "
            "than a **small, late attention bump** once you net out a random day, the "
            "momentum the name already had, and realistic micro-cap costs.\n\n"
            "> ⚠️ **Not investment advice.** The verdict rests on the fingerprinted run on "
            f"**{R['n_events']}** r/WallStreetBets surges (CC-BY `youyanggu/yolostocks-data`, "
            f"as-of {R['asof']}, fp `{R['fp']}`) in "
            "[`docs/results_wsb.md`](../docs/results_wsb.md): abnormal-return event study with a "
            "random-day null, a momentum control, a calendar-block clustering bootstrap and a "
            "name jackknife — references in [`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition — "
            "so this notebook still reads even if you skim the maths. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md).\n\n"
            + BANNER
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict (real tape)\n\n"
            f"From [docs/results_wsb.md](../docs/results_wsb.md) — {R['n_events']} events, "
            f"{R['n_names']} names, {R['span']}:\n\n"
            "| Axis | Stamp | The decisive numbers |\n"
            "|---|---|---|\n"
            f"| **Signal** | 🔴 `NONE` | Excess vs the random-day null {R['ex1']}% / {R['ex5']}% / "
            f"{R['ex21']}% at h = 1/5/21 (p_greater = {R['p1']}/{R['p5']}/{R['p21']}); clustered "
            f"bootstrap CIs straddle zero at every horizon. |\n"
            f"| **Tradability** | 🔴 `MIRAGE` | Gross {R['gross']}%/trade is beta ({R['abn']}% "
            f"abnormal); net hits zero at a {R['be_halfspread']} bps half-spread; median trade "
            f"{R['med_net']}%; sleeve Sharpe {R['sleeve_sh']}, max DD {R['sleeve_dd']}%. |\n"
            f"| **Pump-and-fade?** | ⚪ `DIRECTIONAL ONLY` | Month-ahead excess {R['ex21']}%, "
            f"up-share {R['pos21']}% vs {R['pos21_rand']}% random, {R['hot21']}% vs already-hot — "
            f"all pointing down, none significant (p_excess≤0 = {R['boot21_p']} at h=21). |\n"
        ),

        md(
            "## 1 · The claim, as testable hypotheses\n\n"
            "H₁: E[CAR_{0→h} | mention] > 0 on **abnormal** returns (name − market), with "
            "clustering-robust t > 2, h ∈ {1,5,21}.\n"
            "H₁′ (the sharp one): that excess survives a **momentum control** and realistic "
            "**micro-cap costs**.\n"
            "H₀: forward abnormal return ≈ 0, or fully explained by prior momentum."
        ),
        code(
            "es = eventstudy.event_study(panel, events, horizon=21, pre=5)\n"
            "es['summary'].loc[[-5,-1,0,1,5,10,21]]"
        ),

        md(
            "## 3–4 · The random-day null\n\n"
            "`p_greater` = P(a random basket of the same size, drawn from every (name, day) in "
            "the universe, beats the mention basket). Small ⇒ the mention adds something."
        ),
        code(
            "benchmark.conditional_vs_unconditional(panel, events, horizons=(1,5,21), n_iter=2000)"
        ),

        md(
            "## 4 · The momentum control — does 'mentioned' beat 'already hot'?\n\n"
            "The confound that makes this study hard: attention follows performance, so a "
            "mention rides an existing run. We pit mentions against hot-streak events (a name "
            "in its top-decile trailing return) on the same forward abnormal returns. **A "
            "mention that clears the random-day null but not this is a momentum sensor.**"
        ),
        code(
            "hot = mentions.hot_streak_events(panel)\n"
            "benchmark.excess_vs_alternative(panel, events, hot, horizons=(1,5,21), n_iter=2000)"
        ),

        md(
            "## 4 · The fade, clustering, and concentration\n\n"
            "**(a) The fade** — mean abnormal CAR by horizon; a peak that reverses is the tell."
        ),
        code(
            "robustness.fade_curve(panel, events)"
        ),
        md(
            "**(b) Clustering** — mentions arrive in hype waves; a meme week is one bet, not "
            "thirty. A calendar-block bootstrap gives the honest CI on the excess."
        ),
        code(
            "{k: round(v,4) for k,v in robustness.block_bootstrap_excess(panel, events, horizon=5, n_iter=2000).items()}"
        ),
        md(
            "**(c) Concentration** — drop the most-mentioned names one at a time. If the excess "
            "collapses, you found a stock, not a skill."
        ),
        code(
            "robustness.name_jackknife(panel, events, horizon=5, top=3)"
        ),

        md(
            "## 5 · The verdict, with the numbers\n\n"
            "The decisive cells are the random-day `p_greater`, the momentum `gap` and its "
            "p-value, the block-bootstrap `p_excess_le_0`, the fade-curve peak-vs-month and the "
            "jackknife swing. On the **real tape** ([docs/results_wsb.md](../docs/results_wsb.md)) "
            "they read:\n\n"
            f"- random-day null: excess {R['ex1']}% / {R['ex5']}% / {R['ex21']}% at h = 1/5/21, "
            f"p_greater = {R['p1']}/{R['p5']}/{R['p21']} — never significant;\n"
            f"- momentum control: {R['hot21']}% vs an already-hot name by a month "
            f"(p = {R['hot21_p']});\n"
            f"- clustered bootstrap at h=21: mean {R['ex21']}%, CI [{R['boot21_lo']}%, "
            f"{R['boot21_hi']}%], p_excess≤0 = {R['boot21_p']} — the fade is a direction, not a "
            "finding;\n"
            "- jackknife: flat — no single name is carrying (or hiding) anything.\n\n"
            "**Signal `NONE`** — the bump never clears either null — and **Tradability `MIRAGE` "
            "regardless** once beat 6 charges costs. The pump-and-fade shape the synthetic cells "
            "above display so cleanly is, on the real tape, only **directional**."
        ),

        md(
            "## 6 · Could you trade it — costs and capacity\n\n"
            "Enter at the next open (you saw the tweet when everyone did), hold a fixed window, "
            "charge a micro-cap spread twice. Then ask how much size the names can even absorb "
            "before your own order is the move."
        ),
        code(
            "res = backtest.run(panel, events, hold_days=10)\n"
            "print({k:(round(v,4) if isinstance(v,float) else v) for k,v in res.stats.items()})\n"
            "print('\\ncost sweep (half-spread bps -> mean net trade):')\n"
            "display(backtest.cost_sweep(panel, events))\n"
            "print('capacity at a nominal 50bp edge:')\n"
            "backtest.capacity(panel, events, edge_bps=50.0)"
        ),
        md(
            "**The data-mining check.** Hold period, lookback, cooldown — try enough knobs and "
            "one cell shines. Deflate the best Sharpe for the number of configs tried."
        ),
        code(
            "import itertools\n"
            "rows = []\n"
            "for hold in (3,5,10,21):\n"
            "    r = backtest.run(panel, events, hold_days=hold)\n"
            "    rows.append({'hold': hold, 'sharpe': r.stats.get('sleeve_sharpe', float('nan')),\n"
            "                 'n': r.stats['n_trades']})\n"
            "scan = pd.DataFrame(rows).sort_values('sharpe', ascending=False)\n"
            "best = scan.iloc[0]\n"
            "dsr = robustness.deflated_sharpe(best.sharpe, n_trials=len(scan), n_obs=int(best.n))\n"
            "print(f\"best hold={int(best.hold)}d Sharpe={best.sharpe:.2f}; deflated over {len(scan)} configs: {dsr:.3f}\")\n"
            "scan"
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Short the fade** — the month-ahead drift points down but isn't significant "
            "([docs/results_wsb.md](../docs/results_wsb.md)); test the inverted trade directly, "
            "net of micro-cap borrow, before believing it.\n"
            "- **Beta-estimated abnormal return** to replace the β=1 market adjustment.\n"
            "- **Conviction / first-mention / pile-on** splits of the feed.\n"
            f"- **A survivorship-clean feed** — the real run drops {R['missing']} delisted names "
            "for lack of a price history; recovering them would make the month-ahead numbers "
            "*worse*, not better, and might turn the directional fade into a finding.\n\n"
            "Engine: [`../../../quantlab/`](../../../quantlab/). Method: "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md)."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"wrote {name}  ({len(nb.cells)} cells)")


def main():
    build_curious()
    build_quants()


if __name__ == "__main__":
    main()
