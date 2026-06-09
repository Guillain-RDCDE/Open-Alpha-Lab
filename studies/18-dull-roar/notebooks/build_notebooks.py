"""Generate the two narrative notebooks for Study 18 (Dull-Roar) from source.

Like every study, the notebooks are a *generated artefact*: edit the cell text here, rebuild the
skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic universe** — a single-factor cross-section with a
*baked-in* flat security-market line (low-beta names carry positive alpha) — because the cached real
panel is git-ignored and the desk's reproducible core must run with no network. The synthetic is where
the machinery is *provable*: the anomaly is real by construction, the diagnostics recover it, and a
fair-CAPM null kills every leg. That is exactly the point — it proves the code, so the **real verdict**
(current S&P 500, quoted from [`docs/results.md`](../docs/results.md), produced by `examples/verify.py`)
is a fact about the market: that the celebrated anomaly has *decayed to absent* on the modern,
investable tape. Both notebooks follow the SAME seven desk beats (see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (dull_roar/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from dull_roar import data, sort, strategy, decompose, extension

# Offline synthetic cross-section: a single-factor universe with a BAKED-IN flat security-market line
# (low-beta names carry +alpha, high-beta -alpha -- the Frazzini-Pedersen mechanism), idio vol scaling
# with beta so a total-vol sort is a clean beta sort. The NULL (sml_slope=0) is textbook CAPM: alpha=0
# everywhere, so a vol sort must earn nothing. The real S&P 500 verdict is in ../docs/results.md.
panel, market, truth = data.synthetic_panel(seed=18)                       # the anomaly tape
panel0, market0, _   = data.synthetic_panel(sml_slope=0.0, seed=18)        # the fair-CAPM null
print(f"{truth.n_stocks} stocks x {truth.n_bars} days | baked low-minus-high alpha spread "
      f"{truth.annual_sml_alpha_spread:+.1%}/yr | null per-stock Sharpe {truth.null_stock_sharpe:+.2f}")
"""

# Real headline numbers (from docs/results.md via examples/verify.py; the cells below EXECUTE on the
# synthetic tape). Current S&P 500, 263 names, 2010-2026, as-of 2026-06-01, fingerprint 12af7604edc8.
R = dict(
    n_stocks="263", lo="2010", hi="2026",
    sml_slope="-0.07", low_buck="+0.63", high_buck="+0.59",
    mkt_sr="+1.03", low_sr="+0.93", sr_gain="-0.10",
    bab_alpha="-2.2", bab_t="-0.4", bab_sr="-0.10",
    low_beta="0.59", low_alpha="+1.6", low_alpha_t="+0.8", excess_b1="+2.1",
    leg_low="+1.6", leg_low_t="+0.8", leg_high="+7.2", leg_high_t="+2.3", share_short="81",
    boot_lo="-0.44", boot_hi="+0.21", boot_pneg="73",
    breakeven="0", dd_mkt="-38", dd_low="-34", turn="2.1",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free alpha?: Beta-tilt](https://img.shields.io/badge/Free_alpha%3F-Beta--tilt-8b949e?style=flat-square)\n\n"
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
            "# Dull-Roar 🐢\n"
            "### \"Boring stocks quietly beat exciting ones\" — the most-cited free lunch in finance. Can you actually eat it?\n\n"
            + BADGES +
            "Here's one of the great counter-intuitions in markets, and it's *almost* true. Sort every "
            "stock by how wild it's been and — the textbooks say — the **calm** ones go on to earn a "
            "*better risk-adjusted return* than the **wild** ones. More risk is supposed to mean more "
            "reward; the low-volatility anomaly says the opposite, and decades of academic data back it "
            "(Ang–Hodrick–Xing–Zhang 2006; Baker–Bradley–Wurgler 2011; Frazzini–Pedersen 2014). It's the "
            "idea behind every \"min-vol\" fund you've seen advertised.\n\n"
            "This is the first idea we've pulled from a *book* — Kakushadze & Serur's *151 Trading "
            "Strategies*, strategy #3.4 — and it's a perfect desk specimen: a real, famous effect with a "
            "much bigger promise bolted on top. We'll do two things. First, **prove our machinery works** "
            "on a synthetic market where we *secretly bake the anomaly in* — so you can watch the tools "
            "find it, and watch them find *nothing* when we switch it off. Then we point the exact same "
            "tools at the **real S&P 500** — and discover the calm-beats-wild edge has quietly gone "
            "missing.\n\n"
            "> 📓 **This is the plain-language layer.** Want the beta-neutral BAB regression, the HAC "
            "*t*-stats and the borrow-cost break-even? That's the companion notebook, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** An educational, reproducible research tool: every chart below "
            "is generated by the code beside it. The reproducible core runs on a **synthetic** universe "
            "where we *bake in* a known anomaly — so the real-market numbers (quoted from "
            "[`../docs/results.md`](../docs/results.md)) are a measurement, not a hope. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Do calm stocks *really* out-earn wild ones, risk-adjusted? | 🟡 **In theory, yes; lately, "
            "no.** Decades of academic data say yes — but on the **modern S&P 500** the line is flat. |\n"
            "| Is the clean version tradable? | 🚫 **No.** The proper, beta-neutral form needs you to "
            "**short** the wild stocks — the very names that are expensive or impossible to borrow. |\n"
            "| What about just *buying* the calm stocks? | ⚪ **That's mostly low beta** — you're running "
            "less market risk, not finding free money. Lever it back up and the 'edge' fades. |\n"
            "| So is it a free lunch? | 🚫 **No** — the one thing that survives is a *gentler ride* "
            "(shallower drawdowns), which is exactly what real min-vol funds quietly deliver. |\n\n"
            "> Desk shorthand: **Signal `WEAK` · Tradability `MIRAGE` · Free alpha? `BETA-TILT`.** A real "
            "mechanism, an unharvestable promise. Let's see how we got there."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Stated at full strength: **risk is not rewarded the way the textbook says.** Rank stocks by "
            "past volatility, and the calm decile delivers a *higher Sharpe* than the wild decile. The "
            "leading explanation (Frazzini–Pedersen 2014) is elegant: lots of investors can't or won't "
            "use leverage, so to chase returns they pile into high-volatility, high-beta stocks instead — "
            "overpricing them — while the boring low-beta names are left cheap. The **security-market "
            "line** — the supposed straight-line trade-off between risk and reward — comes out *too "
            "flat*.\n\n"
            "Let's make that concrete. On our synthetic market we've baked exactly this in. Sort the "
            "universe into ten volatility buckets and look at the Sharpe of each:"
        ),
        code(
            "sml = sort.security_market_line(panel)\n"
            "tbl = sml['table']\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(tbl.index, tbl['sharpe'].values, color='#2ea44f', alpha=.85)\n"
            "ax.set_xlabel('volatility bucket  (0 = calmest, 9 = wildest)'); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title(f\"The baked anomaly: calm earns more per unit risk \"\n"
            "             f\"(slope {sml['sharpe_vol_slope']:+.2f})\")\n"
            "plt.show()\n"
            "print(f\"calmest-decile Sharpe {sml['low_bucket_sharpe']:+.2f}  vs  wildest "
            "{sml['high_bucket_sharpe']:+.2f}\")"
        ),
        md(
            "A clean downward staircase: the calmer the bucket, the better the risk-adjusted return. "
            "*That* is the anomaly — and the whole question is whether it's real in the wild, and whether "
            "you could ever put it in your pocket."
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "If it holds, it breaks the first rule everyone learns: that to earn more you must risk more. "
            "You'd build a portfolio of *sleepy* stocks and beat the market on a risk-adjusted basis — "
            "higher Sharpe, shallower crashes, less stress. It's the closest thing to a free lunch the "
            "equity market offers, and it's why \"minimum-volatility\" and \"low-beta\" funds gather "
            "hundreds of billions of dollars. If the calm-beats-wild gap is real *and* harvestable, it's "
            "one of the most valuable facts in investing. Two very big 'if's."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "Five checks, decided up front:\n\n"
            "1. **Is the security-market line really too flat?** Sort by vol; does Sharpe fall as vol "
            "rises?\n"
            "2. **Does a clean (beta-neutral) long-short earn a real alpha** — return the market can't "
            "explain?\n"
            "3. **Is buying the calm decile actually alpha, or just *less beta*?** Lever it to the "
            "market's risk and see what's left.\n"
            "4. **Where does the effect live** — in the calm stocks you can buy, or the wild ones you'd "
            "have to *short*?\n"
            "5. **Could you short them cheaply?** Charge a realistic borrow fee on the wild leg.\n\n"
            "**The null:** on a fair-CAPM market (anomaly switched off), every one of these must come up "
            "empty. **What would make us cry 'mirage':** the edge needs the unshortable wild leg, *or* "
            "it's just a low-beta tilt, *or* it's simply absent on the real modern tape."
        ),

        md(
            "## 4 · The teardown 🔧\n\n"
            "### 4a · First, prove the tools — find the anomaly we hid, then watch it vanish on the null\n"
            "Same sort, two synthetic markets: the one with the baked anomaly, and a fair-CAPM **null** "
            "where risk *is* rewarded normally. If our tools are honest, the staircase should appear in "
            "one and flatten in the other."
        ),
        code(
            "s1 = sort.security_market_line(panel)['table']['sharpe']\n"
            "s0 = sort.security_market_line(panel0)['table']['sharpe']\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(s1.index, s1.values, 'o-', color='#2ea44f', label='anomaly baked in')\n"
            "ax.plot(s0.index, s0.values, 's--', color='#8b949e', label='fair-CAPM null')\n"
            "ax.set_xlabel('volatility bucket (0 = calmest)'); ax.set_ylabel('Sharpe'); ax.legend()\n"
            "ax.set_title('Tools work: a clear tilt with the anomaly, flat without it')\n"
            "plt.show()"
        ),

        md(
            "### 4b · The clean way to harvest it — and the trap in the naive way\n"
            "The obvious trade is dollar-neutral: buy the calm decile, short the wild one. But calm "
            "stocks are *low-beta* and wild stocks *high-beta*, so that book is secretly **short the "
            "market** — in a rising market the beta drag alone can sink it. The fix (Frazzini–Pedersen) is "
            "to **beta-neutralise**: scale each leg so the bet is purely calm-vs-wild, not down-on-the-"
            "market. Watch the difference on the anomaly tape:"
        ),
        code(
            "b = strategy.build(panel, market=market, cost_bps=1.0)\n"
            "naive = (b['low_only'] - b['high'])                      # dollar-neutral: structurally short beta\n"
            "bab   = decompose.beta_neutral_bab(panel, market=market, cost_bps=1.0)\n"
            "eq_naive = (1+naive).cumprod(); eq_bab = (1+bab['stream']).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq_naive.index, eq_naive.values, color='#c0392b', label='naive dollar-neutral (short-the-market drag)')\n"
            "ax.plot(eq_bab.index, eq_bab.values, color='#2ea44f', label='beta-neutral BAB (the clean harvest)')\n"
            "ax.set_ylabel('growth of $1'); ax.legend(); ax.set_title('Beta-neutralising lifts the real signal out of the beta noise')\n"
            "plt.show()\n"
            "print(f\"beta-neutral BAB alpha {bab['alpha_ann_pct']:+.1f}%/yr (t={bab['alpha_t']:+.1f}), \"\n"
            "      f\"Sharpe {bab['sharpe']:+.2f} -- a real, market-independent edge on the baked tape.\")"
        ),

        md(
            "### 4c · Is *buying* the calm stocks free money — or just less risk?\n"
            "A min-vol fund is long-only. Its higher Sharpe could be genuine alpha, or it could simply be "
            "that it runs *less market risk* — a thing you could copy by holding the index and some cash. "
            "The test: measure the calm book's beta, then lever it back to the market's risk and see if "
            "any excess return survives."
        ),
        code(
            "tilt = decompose.beta_tilt_test(panel, market=market, cost_bps=1.0)\n"
            "print(f\"calm book beta = {tilt['low_beta']:.2f}  (well below 1 -> it runs less market risk)\")\n"
            "print(f\"Sharpe vs market: {tilt['market_sharpe']:+.2f} -> {tilt['low_only_sharpe']:+.2f} \"\n"
            "      f\"(gain {tilt['sharpe_gain']:+.2f})\")\n"
            "print(f\"but levered back to the market's risk, the *excess* return is only \"\n"
            "      f\"{tilt['excess_cagr_at_beta1_pct']:+.1f}%/yr -- the smaller, honest number.\")"
        ),
        md(
            "On the synthetic tape there's still real alpha left over (we baked it in). Hold that thought "
            "— on the *real* market this is exactly where the story turns."
        ),

        md(
            "### 4d · The null collapses\n"
            "Switch the anomaly off (fair-CAPM market) and re-run the clean harvest. If our machine is "
            "honest, the alpha must vanish."
        ),
        code(
            "bab0 = decompose.beta_neutral_bab(panel0, market=market0, cost_bps=1.0)\n"
            "print(f\"anomaly tape : beta-neutral alpha {decompose.beta_neutral_bab(panel, market=market)['alpha_ann_pct']:+.1f}%/yr\")\n"
            "print(f\"fair-CAPM null: beta-neutral alpha {bab0['alpha_ann_pct']:+.1f}%/yr (t={bab0['alpha_t']:+.1f}) -- gone, as it must be.\")"
        ),

        md(
            "### 4e · Now the real market — and the twist\n"
            "Tools proven, we point them at the **current S&P 500** (2010 onward). Here's what the same "
            "sort finds, quoted from [`../docs/results.md`](../docs/results.md):\n\n"
            f"- The security-market line is **flat** — slope **{R['sml_slope']}**, calm-decile Sharpe "
            f"**{R['low_buck']}** vs wild-decile **{R['high_buck']}**. The staircase is gone.\n"
            f"- The clean beta-neutral long-short earns **{R['bab_alpha']}%/yr** (t = **{R['bab_t']}**) — "
            "indistinguishable from zero, and if anything *negative*.\n"
            f"- The calm book actually **trailed** the market (Sharpe {R['low_sr']} vs {R['mkt_sr']}).\n"
            f"- Astonishingly, **{R['share_short']}%** of what little spread exists sits in the *wild* "
            f"leg — whose alpha was **+{R['leg_high'].lstrip('+')}%** (t {R['leg_high_t']}). Over this "
            "decade the exciting stocks *won*.\n\n"
            "The celebrated anomaly didn't survive contact with the modern large-cap tape."
        ),

        md(
            "## 5 · The verdict 🧾\n\n"
            "- **The mechanism is real** — our control recovers a beta-neutral alpha at *t* ≈ 5.6, and "
            "the null is flat. The tools are sound.\n"
            f"- **But on the real S&P 500 it's absent** — flat SML, a {R['bab_alpha']}%/yr beta-neutral "
            f"alpha (t = {R['bab_t']}). Real in the long-run literature, **decayed-to-gone** in the "
            "sample you'd actually trade.\n"
            "- **What's left is a low-beta tilt**, not free alpha.\n\n"
            "> **Signal `WEAK` · Tradability `MIRAGE` · Free alpha? `BETA-TILT`.** Why `WEAK` and not a "
            "flat `NONE`? Because the long-run effect and our control are unambiguously real — this "
            "modern, survivorship-biased window is where it fails, which is a fact about its *fragility*, "
            "not a clean refutation."
        ),

        md(
            "## 6 · Could you trade it? 💸\n\n"
            "Three walls, any one fatal:\n\n"
            f"- **You'd have to short the wild stocks.** The clean edge lives in the high-vol leg "
            f"({R['share_short']}% of it), and those names — small, lottery-like, heavily shorted — are "
            "exactly the expensive-or-impossible-to-borrow ones. On the real tape the dollar-neutral "
            f"book is under water before the borrow even costs anything (break-even **{R['breakeven']} "
            "bps**).\n"
            "- **Buying calm is just low beta.** The long-only book's lift is mostly *less market risk* "
            f"(beta {R['low_beta']}); levered to the market's risk it adds a thin, insignificant "
            f"**{R['excess_b1']}%/yr** (t {R['low_alpha_t']}).\n"
            f"- **It was absent when you'd have run it.** Over 2010–2026 the calm book trailed.\n\n"
            "The one durable consolation is the *ride*, not the return:"
        ),
        code(
            "b = strategy.build(panel, market=market, cost_bps=1.0)\n"
            "def dd(x): e=(1+x).cumprod(); return e/e.cummax()-1\n"
            "fig, ax = plt.subplots()\n"
            "ax.fill_between(b['market'].index, dd(b['market']).values, 0, alpha=.4, color='#c0392b', label='market')\n"
            "ax.fill_between(b['low_only'].index, dd(b['low_only']).values, 0, alpha=.6, color='#2ea44f', label='low-vol (calm) book')\n"
            "ax.set_ylabel('drawdown'); ax.legend(); ax.set_title('The one durable benefit: calm stocks crash less hard')\n"
            "plt.show()\n"
            "print('(Synthetic tape shown. On the REAL S&P 500 the worst drawdown goes "
            f"{R['dd_mkt']}% -> {R['dd_low']}% -- the gentler ride min-vol funds actually sell.)')"
        ),
        md(
            "> Tradability **`MIRAGE`** — the tradable anomaly needs a short you can't afford. \"Free "
            "alpha?\" **`BETA-TILT`** — buying calm is buying less beta, dressed up."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **We built the salvage case — \"what if you only go long?\"** The one slice needing no "
            "short is the long-only calm book. Worked complement (beat 7 of the quants notebook + "
            "[`../docs/extension.md`](../docs/extension.md)): on the real tape it's a thin, insignificant "
            "defensive tilt across *every* lookback and decile width we tried — lower beta, shallower "
            "drawdowns, no alpha. The salvage confirms the verdict.\n"
            "- **Go back further, and include the dead.** Our sample is *current* S&P 500 since 2010 — a "
            "survivorship-biased, high-beta-friendly decade. The long-run anomaly lives in 1926-on, "
            "all-cap, *delisted-included* data. A fork that loads CRSP-style history could show the effect "
            "alive where ours is absent — and date when it faded.\n"
            "- **Control for value & quality.** Much of low-vol's academic alpha overlaps cheap, "
            "profitable, defensive names. Does any survive once you net those out?\n\n"
            "PRs welcome — resurrect the anomaly on honest long-run data, or pin down exactly when the "
            "crowd competed it away."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Dull-Roar — a quantitative teardown 🔬\n"
            "### Security-market-line slope · beta-neutral BAB alpha (HAC) · the beta-tilt counter · leg attribution · borrow break-even · the fair-CAPM null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* The steelman is the low-volatility "
            "anomaly (Kakushadze–Serur *151 Trading Strategies* §3.4; Ang et al. 2006; Frazzini–Pedersen "
            "2014): sort on trailing vol and the calm decile out-earns the wild one risk-adjusted, i.e. "
            "the security-market line is too flat. We prove the apparatus on a synthetic universe with a "
            "baked flat SML, then read the real verdict off the modern S&P 500.\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core executes on a synthetic single-factor "
            "cross-section with a known anomaly (and a fair-CAPM null); the real S&P 500 run is in "
            "[`../docs/results.md`](../docs/results.md) via `examples/verify.py`, sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            "| **Signal** — is the effect real? | 🟡 `WEAK` | The mechanism is real (synthetic control: "
            "beta-neutral BAB alpha **+7.4%/yr**, HAC *t* ≈ **5.6**; null flat) — but on the current S&P "
            f"500 the SML slope is **{R['sml_slope']}** and the BAB alpha is **{R['bab_alpha']}%/yr** "
            f"(*t* = **{R['bab_t']}**), a statistical zero. Real in theory, absent in the modern sample. |\n"
            "| **Tradability** | 🔴 `MIRAGE` | The clean spread needs shorting the high-vol leg "
            f"(**{R['share_short']}%** of the effect sits there); the dollar-neutral book's borrow "
            f"break-even is **{R['breakeven']} bps/yr**. |\n"
            "| **Free alpha?** | ⚪ `BETA-TILT` | The long-only book is low beta "
            f"(**{R['low_beta']}**); its CAPM alpha is **{R['low_alpha']}%/yr** (*t* = {R['low_alpha_t']}) "
            "— less market risk, not free money. |\n\n"
            "> **In one sentence:** the most-cited anomaly in finance is real in the long-run literature "
            "and on our control, but on the modern investable cross-section it reduces to a low-beta "
            "defensive tilt whose textbook long-short form is fragile to exactly the short-leg borrow it "
            "can't avoid.\n\n"
            "*(This notebook executes on the synthetic tape — where the anomaly is provable and the null "
            "is clean. The real S&P 500 numbers that set the stamps are in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, stated precisely\n\n"
            "Single-factor returns $r_{i,t} = \\alpha_i + \\beta_i r^m_t + \\nu_i \\varepsilon_{i,t}$. CAPM "
            "says $\\alpha_i = 0$ for all $i$. The low-vol anomaly is the empirical finding that the "
            "realized SML is **too flat**: low-$\\beta$ names earn $\\alpha_i > 0$ and high-$\\beta$ names "
            "$\\alpha_i < 0$. We encode it as $\\alpha_i = -s\\,(\\beta_i - \\bar\\beta)$ with slope "
            "$s>0$, and set idiosyncratic vol $\\nu_i = \\ell\\,\\beta_i$ so total vol $\\sigma_i = "
            "\\beta_i\\sqrt{\\sigma_m^2+\\ell^2}$ is monotone in $\\beta$ — a vol sort is a $\\beta$ sort, "
            "and under the null ($s=0$) every stock shares one Sharpe (a genuinely flat control)."
        ),
        code(
            "print(f\"baked SML slope s -> low-minus-high alpha spread {truth.annual_sml_alpha_spread:+.1%}/yr\")\n"
            "print(f\"null per-stock Sharpe (betas cancel): {truth.null_stock_sharpe:+.2f} -- identical across the cross-section\")\n"
            "vols = panel.std()*np.sqrt(252)\n"
            "print(f\"cross-section vol range: {vols.min():.0%} .. {vols.max():.0%} (plenty to sort on)\")"
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "If the SML is flat, a *beta-neutral* long-short (long low-$\\beta$ levered up, short "
            "high-$\\beta$ levered down) earns positive alpha with ~zero market exposure — a higher-Sharpe "
            "portfolio not spanned by the market (Frazzini–Pedersen's BAB). The open questions are exactly "
            "three: **(i)** is the realized alpha significant under autocorrelation-robust inference; "
            "**(ii)** how much of the *long-only* version is just a sub-1 beta a cash blend replicates; "
            "**(iii)** does the harvest survive the borrow on the short leg. Beats 4–6 answer all three — "
            "on the synthetic control *and* the real tape, which disagree in an instructive way."
        ),

        md(
            "## Beat 3 · How we'd know — the pre-registered protocol\n\n"
            "1. **SML shape** (`sort.security_market_line`): cross-sectional OLS of per-stock Sharpe on "
            "vol; slope $<0$ ⇔ anomaly. Pre-registered: clearly negative on the baked tape, ≈0 on null.\n"
            "2. **Beta-neutral alpha** (`decompose.beta_neutral_bab`): CAPM regression of the BAB factor; "
            "**Newey–West** intercept *t*. Real ⇔ $\\alpha>0$, $|t|>2$.\n"
            "3. **Beta-tilt counter** (`decompose.beta_tilt_test`): lever the long-only book to $\\beta=1$ "
            "and read the residual excess CAGR — the part that is *not* a market-exposure choice.\n"
            "4. **Leg attribution** (`decompose.leg_attribution`): each leg's CAPM alpha; how much sits in "
            "the unshortable high-vol leg.\n"
            "5. **Borrow break-even** (`extension.borrow_breakeven`): the short-leg stock-loan fee at "
            "which the dollar-neutral Sharpe hits zero.\n"
            "6. **Null:** every leg re-run on the fair-CAPM tape must collapse.\n\n"
            "**Mirage line:** the alpha needs the short leg, *or* the long-only excess is just beta, *or* "
            "the real tape shows no significant beta-neutral alpha."
        ),

        md(
            "## Beat 4 · The teardown\n\n"
            "### 4a · The SML slope — anomaly vs null\n"
            "The cross-sectional Sharpe-on-vol regression, both tapes."
        ),
        code(
            "for label, p in [('anomaly', panel), ('fair-CAPM null', panel0)]:\n"
            "    s = sort.security_market_line(p)\n"
            "    print(f\"{label:16s}: Sharpe~vol slope {s['sharpe_vol_slope']:+.2f} | \"\n"
            "          f\"low/high bucket Sharpe {s['low_bucket_sharpe']:+.2f}/{s['high_bucket_sharpe']:+.2f} | \"\n"
            "          f\"anomaly_present={s['anomaly_present']}\")\n"
            "display(sort.security_market_line(panel)['table'].round(3))"
        ),

        md(
            "### 4b · The beta-neutral BAB alpha with Newey–West errors\n"
            "Regress the beta-neutralised long-short on the market; the intercept is the part the market "
            "can't span. HAC because daily returns are autocorrelated and heteroskedastic."
        ),
        code(
            "bab = decompose.beta_neutral_bab(panel, market=market, cost_bps=1.0)\n"
            "print(f\"beta-neutral BAB: alpha {bab['alpha_ann_pct']:+.2f}%/yr | HAC t = {bab['alpha_t']:+.2f} | \"\n"
            "      f\"realized beta {bab['beta']:+.2f} | Sharpe {bab['sharpe']:+.2f}\")\n"
            "print(f\"leg betas: low {bab['beta_low_leg']:.2f} / high {bab['beta_high_leg']:.2f} \"\n"
            "      f\"(levered to neutralise the structural short-the-market drag)\")"
        ),
        md(
            "> 💡 **In plain words.** A portfolio with positive alpha and ~zero beta is earning a return "
            "you can't get by dialing the market up or down — the textbook signature of a real anomaly. "
            "On the baked tape it's there at *t* ≈ 5.6; the question is whether the *market* agrees."
        ),

        md(
            "### 4c · The honest counter — is the long-only book just low beta?\n"
            "A min-vol fund is long-only, so its Sharpe lift could be alpha or just less market risk. "
            "Lever it to $\\beta=1$ and price what's left."
        ),
        code(
            "tilt = decompose.beta_tilt_test(panel, market=market, cost_bps=1.0)\n"
            "print(f\"long-only beta {tilt['low_beta']:.2f} | CAPM alpha {tilt['alpha_ann_pct']:+.2f}%/yr \"\n"
            "      f\"(t={tilt['alpha_t']:+.2f})\")\n"
            "print(f\"Sharpe gain over market {tilt['sharpe_gain']:+.2f}; excess CAGR levered to beta 1 \"\n"
            "      f\"{tilt['excess_cagr_at_beta1_pct']:+.2f}%/yr\")"
        ),

        md(
            "### 4d · Leg attribution — where does the alpha live?\n"
            "Split the spread into the long (calm) leg and the short (wild) leg. The literature (Ang et "
            "al.) says the action concentrates in the high-vol short — the one you can't cheaply borrow."
        ),
        code(
            "legs = decompose.leg_attribution(panel, market=market, cost_bps=1.0)\n"
            "print(f\"calm leg alpha {legs['low_leg_alpha_pct']:+.2f}% (t{legs['low_leg_alpha_t']:+.1f}), beta {legs['low_leg_beta']:.2f}\")\n"
            "print(f\"wild leg alpha {legs['high_leg_alpha_pct']:+.2f}% (t{legs['high_leg_alpha_t']:+.1f}), beta {legs['high_leg_beta']:.2f}\")\n"
            "print(f\"share of the spread sitting in the (short) wild leg: {legs['share_in_short_leg']:.0%}\")"
        ),

        md(
            "### 4e · The borrow break-even\n"
            "The dollar-neutral book's net Sharpe as the annual stock-loan fee on the wild leg rises — "
            "and the fee at which it crosses zero."
        ),
        code(
            "sw = strategy.borrow_sweep(panel, market=market)\n"
            "be = extension.borrow_breakeven(panel, market=market)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(sw.index, sw['long_short_sharpe'].values, 'o-'); ax.axhline(0, ls=':', c='grey')\n"
            "ax.set_xlabel('short-leg borrow fee (bps/yr)'); ax.set_ylabel('dollar-neutral net Sharpe')\n"
            "ax.set_title(f\"Break-even borrow ~ {be['breakeven_bps']:.0f} bps/yr\")\n"
            "plt.show()\n"
            "print(f\"break-even borrow fee: {be['breakeven_bps']:.0f} bps/yr \"\n"
            "      f\"(lottery-stock borrows routinely exceed this)\")"
        ),

        md(
            "### 4f · The null collapses\n"
            "Every leg, re-run on the fair-CAPM tape. No baked alpha ⇒ nothing to find."
        ),
        code(
            "bab0 = decompose.beta_neutral_bab(panel0, market=market0, cost_bps=1.0)\n"
            "tilt0 = decompose.beta_tilt_test(panel0, market=market0, cost_bps=1.0)\n"
            "sml0 = sort.security_market_line(panel0)\n"
            "print(f\"null: SML slope {sml0['sharpe_vol_slope']:+.2f} | BAB alpha {bab0['alpha_ann_pct']:+.2f}% \"\n"
            "      f\"(t={bab0['alpha_t']:+.2f}) | long-only alpha {tilt0['alpha_ann_pct']:+.2f}% (t={tilt0['alpha_t']:+.2f})\")\n"
            "print('every number ~0: the apparatus measures the effect, not itself.')"
        ),

        md(
            "## Beat 5 · The verdict\n\n"
            "- **Apparatus sound** (4a–4b, 4f): the baked anomaly is recovered (BAB alpha +7.4%/yr, HAC "
            "*t* ≈ 5.6) and the null is flat.\n"
            f"- **Real tape disagrees** (`../docs/results.md`): SML slope {R['sml_slope']}, beta-neutral "
            f"alpha {R['bab_alpha']}%/yr (*t* = {R['bab_t']}), the calm book *trailing* the market — "
            f"and **{R['share_short']}%** of the spread in a high-vol leg whose own alpha was "
            f"**+{R['leg_high'].lstrip('+')}%** (*t* {R['leg_high_t']}).\n"
            "- **What remains** is a low-beta tilt, no significant long-only alpha.\n\n"
            "> **Signal `WEAK`** — real mechanism, real long-run literature, absent on the modern "
            "survivorship-biased large-cap sample. Not a refutation of the anomaly; a measurement of its "
            "fragility."
        ),

        md(
            "## Beat 6 · Could you trade it?\n\n"
            "The three usual killers all land:\n\n"
            f"- **The short leg.** {R['share_short']}% of the (real-tape) spread sits in the high-vol "
            "names — small, lottery-like, heavily-shorted, expensive-or-impossible to borrow. The "
            f"dollar-neutral book's break-even borrow is **{R['breakeven']} bps/yr**: it doesn't survive "
            "*any* realistic fee.\n"
            f"- **Capacity/decay.** The effect has decayed post-publication in US large caps — the modern "
            "window shows nothing to crowd into.\n"
            f"- **It's beta.** The long-only survivor is a {R['low_beta']}-beta defensive book; levered to "
            f"the market's risk its excess is **{R['excess_b1']}%/yr** (*t* {R['low_alpha_t']}).\n\n"
            "The single durable artefact is a shallower drawdown "
            f"(real: {R['dd_mkt']}% → {R['dd_low']}%) — the genuine, modest service min-vol products sell. "
            "Tradability **`MIRAGE`**; \"free alpha?\" **`BETA-TILT`**."
        ),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — the no-shorting test\n"
            "The headline pins one caveat: the clean harvest needs the unshortable wild leg. So make the "
            "constraint real — forbid shorting entirely and keep only the long-only calm book — and ask "
            "what a real-money, long-only investor actually banks. The edge splits cleanly: the "
            "**defensive** (no-short) slice vs the **short-the-wild** slice."
        ),
        code(
            "sd = extension.shorting_decomposition(panel, market=market)\n"
            "print(f\"full beta-neutral alpha {sd['alpha_full']:+.2f}% = defensive {sd['alpha_defensive']:+.2f}% \"\n"
            "      f\"+ short-the-wild {sd['alpha_short']:+.2f}%  (defensive share {sd['share_defensive']:.0%})\")\n"
            "print(f\"defensive slice needs no borrow; drawdown {sd['market_maxdd']:.0%} -> {sd['low_only_maxdd']:.0%}\")\n"
            "# robustness on the synthetic: the split holds across lookback / decile width\n"
            "for lb, dec in [(126,0.1),(252,0.1),(126,0.2)]:\n"
            "    s = extension.shorting_decomposition(panel, market=market, lookback=lb, decile_frac=dec)\n"
            "    print(f\"  lookback {lb}, decile {dec:.0%}: defensive share {s['share_defensive']:+.0%}\")"
        ),
        md(
            "**The result — and on the real tape it's decisive.** On the synthetic the defensive slice is "
            "real but the *smaller* part; on the **real S&P 500** (full run in "
            "[`../docs/extension.md`](../docs/extension.md), `examples/extension.py`) the no-shorting "
            "investor is left with a thin, insignificant defensive tilt across *every* lookback and decile "
            "width — lower beta, shallower drawdowns, no alpha — while the dollar-neutral book is under "
            "water before the borrow even costs anything. The constraint that a real investor *actually "
            "faces* (no cheap short) is exactly the one that turns the famous anomaly into a defensive "
            "tilt. The worked complement doesn't rescue `MIRAGE` — it *earns* it.\n\n"
            "### 7b · Other forks\n"
            "- **Long-run, delisted-inclusive data.** Our universe is current S&P 500 since 2010 — "
            "survivorship-biased and high-beta-friendly. A CRSP-style 1926-on, all-cap panel is where the "
            "academic effect lives; a fork could show it alive and *date its decay*.\n"
            "- **Factor controls.** Net out value, quality and the dividend tilt — how much low-vol alpha "
            "is independent of them?\n"
            "- **Real BAB financing.** Frazzini–Pedersen's own construction needs leverage on the low-beta "
            "leg; price it with realistic financing and margin, à la Study 16's leverage honesty.\n\n"
            "PRs welcome — resurrect the anomaly on honest long-run data, or find the market where its "
            "tradable form still breathes."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def main():
    nbf.write(build_curious(), os.path.join(HERE, "01_for_the_curious.ipynb"))
    nbf.write(build_quants(), os.path.join(HERE, "02_for_the_quants.ipynb"))
    print("wrote 01_for_the_curious.ipynb and 02_for_the_quants.ipynb")


if __name__ == "__main__":
    main()
