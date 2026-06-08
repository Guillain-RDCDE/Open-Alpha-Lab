"""Generate the two narrative notebooks for Study 04 (Social-Oracle) from source.

Like Studies 01–03, the notebooks are a *generated artefact*: edit the cell text
here, rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Unlike Studies 01–03, this study ships **no live dataset** (a mention feed is
third-party and licence-encumbered), so both notebooks run on the **offline synthetic
universe** — a toy panel with a deliberately mild pump-and-fade baked in. They are a
*worked method*, not a live verdict: every cell is exactly the code you'd run on a
real feed, and the last beat shows the one-line swap to `data.load_feed(...)`. No
network needed, ever.

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

# No live feed ships with this study: we run the *method* on a synthetic universe
# with a baked-in pump-and-fade. Swap the next line for data.load_feed('mentions.csv')
# + data.build_panel(...) to run it for real.
panel, feed = data.synthetic_panel(seed=0)
events, coverage = mentions.to_events(feed, panel)
print(f"{len(panel)} names, {len(feed)} mentions -> {len(events)} clean events")
print("coverage:", coverage)
"""


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
            "# Does following a viral stock guru actually pay? 🔮\n"
            "### \"Buy what the oracle tweets\" — tested honestly, in plain English\n\n"
            "Every cycle a retail-investing folk hero goes viral, and within weeks GitHub "
            "fills with bots that scrape her posts, pull out the `$TICKERS`, and score them as "
            "buy signals. The pitch writes itself: *her timeline front-runs the market — just "
            "buy what she mentions.*\n\n"
            "This is the cousin of our [Falling-Knife](../../02-falling-knife/) and "
            "[Fear-Gauge](../../03-fear-gauge/) studies, but the trigger isn't a price or a "
            "fear gauge — it's **attention**. And it has the same twist as Fear-Gauge: there "
            "might be a *real* little bump there, and that's exactly the trap.\n\n"
            "> ⚠️ **Not investment advice, and not about a person** — it tests the *phenomenon* "
            "of social trading. No live feed ships here (those are scraped and licence-bound), "
            "so we run the **method** on a synthetic universe with a known pump-and-fade. The "
            "code is exactly what you'd run on a real feed.\n\n"
            "*Follows the desk's seven beats ([METHODOLOGY.md](../../../METHODOLOGY.md)). The "
            "rigorous version is the companion,* "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer (the desk's prior) |\n"
            "|---|---|\n"
            "| Does a name move after she mentions it? | ⚠️ **A little, briefly** — attention "
            "does nudge price. But look *left* of the tweet: most of the move already happened. |\n"
            "| Is that bump a free edge? | ❌ **No** — it *fades*, and you only see the tweet "
            "*after* the pop, so you buy the start of the reversal. |\n"
            "| Does \"mentioned\" beat \"already hot\"? | ❓ **Barely / not really** — strip out "
            "the momentum the name already had and most of the signal goes with it. |\n"
            "| Could a follower actually trade it? | ❌ **Not after costs** — these are \\$1–3 "
            "names with huge spreads and tiny capacity. |\n\n"
            "> Desk shorthand: **Signal `WEAK` · Tradability `MIRAGE`** — let's see the method "
            "earn them."
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
            "into the tweet, fades after, and **loses** to both a random day and a hot streak. "
            "On a real feed the literature prior (attention → small pop → reversal) says expect "
            "**Signal `WEAK`**, **Tradability `MIRAGE`**. The next beat is why a follower can't "
            "even keep the `WEAK` part."
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
            "- **The inversion:** if it's a pop-and-fade, the side that *might* pay is being "
            "**early or short the fade**, not the late follower — same punchline as Fear-Gauge "
            "(*sell* the fear, don't buy it).\n"
            "- **Bring a real feed:** the whole study is one `data.load_feed('mentions.csv')` "
            "away from live numbers.\n"
            "- **Conviction & first-mention:** does a high-score or first-ever mention behave "
            "differently?\n\n"
            "The deep version — t-stats, clustering bootstrap, name jackknife, capacity — is in "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Social-Oracle — the teardown 🔮🔬\n"
            "### Mention event study: random-day null, the momentum control, the fade, "
            "clustering, concentration, and micro-cap capacity\n\n"
            "The rigorous companion to [`01_for_the_curious.ipynb`](01_for_the_curious.ipynb). "
            "Same seven beats, full method. Thesis: a public mention is a **small, late, "
            "reversing attention bump** — plausibly real (`WEAK`) but uninvestable (`MIRAGE`) "
            "once you pay micro-cap costs and enter after the pop.\n\n"
            "> ⚠️ **Not investment advice; tests a phenomenon, not a person.** No live feed "
            "ships — we run the method on a synthetic universe with a known pump-and-fade, so "
            "the *machinery* is validated end-to-end. Swap in `data.load_feed(...)` for a real "
            "verdict. Fixed seeds; no network."
        ),
        code(BOOT),

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
            "Collate the decisive cells: random-day `p_greater`, the momentum `gap` and its "
            "p-value, the block-bootstrap `p_excess_le_0`, the fade-curve peak-vs-month, the "
            "jackknife swing. On a real feed these fill the README's beat-5 stamps. Expected "
            "shape: **Signal `WEAK`** (a real but small, run-up-contaminated bump) → "
            "**Tradability `MIRAGE`** once beat 6 charges costs."
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
            "- **Short the fade** — test the inverted trade directly, net of micro-cap borrow.\n"
            "- **Beta-estimated abnormal return** to replace the β=1 market adjustment.\n"
            "- **Conviction / first-mention / pile-on** splits of the feed.\n"
            "- **A real, survivorship-clean feed** — the one input that turns this prior into a "
            "verdict.\n\n"
            "Engine: [`../../../quantlab/`](../../../quantlab/). Method: "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def main():
    targets = {
        "01_for_the_curious.ipynb": build_curious(),
        "02_for_the_quants.ipynb": build_quants(),
    }
    for fname, nb in targets.items():
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {fname}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
