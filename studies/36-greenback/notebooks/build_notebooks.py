"""Generate the two narrative notebooks for Study 36 (Greenback) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed cells run on BOTH (a) the seeded synthetic FX panel — the controlled machinery proof — and
(b) the **real G10 tape**, read OFFLINE from the two `_cache/g10_*.parquet` files (OECD 3-month short rates
+ yfinance FX, a USD-funded monthly book, 2001–2024, as-of 2024-01-31). The headline real numbers are in
../docs/results.md (fingerprint ef7450ae792e). Both notebooks walk the SAME seven desk beats. Greenback is
the dollar-carry / carry⊕momentum-combo companion to Study 27 (Steamroller).
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
from greenback import data, strategy, costs, extension

# (1) Offline synthetic control: a currency panel with a carry premium + risk-off crashes + an independent
# PROFITABLE trend for the momentum sleeve (the machinery) + a full-UIRP carry null.
xr, rd, truth = data.synthetic_fx(carry_strength=0.9, trend_strength=0.35, seed=36)
x0, r0, _ = data.synthetic_fx(carry_strength=0.0, trend_strength=0.35, seed=36)
print(f"synthetic control: {truth.n_ccy} currencies x {truth.n_months} months, "
      f"carry_strength {truth.carry_strength} (null=0), trend_strength {truth.trend_strength}")

# (2) The REAL G10 tape, read offline from cache (OECD 3-month short rates + yfinance FX, USD-funded
# monthly, as-of 2024-01-31). build_carry_panel does the carry math; load_real_panel pins the as-of.
REAL = data.load_real_panel()
if REAL is not None:
    rxr, rrd = REAL
    print(f"real G10 tape: {rxr.index.min().date()} -> {rxr.index.max().date()} "
          f"({len(rxr)} months), currencies {list(rxr.columns)}")
"""

# Synthetic-CONTROL numbers (seed 36, net @10 bp) — these match the executed control cells.
R = dict(
    hml="5.3", null_hml="0.8",
    carry_sh="1.18", carry_skew="-1.55", carry_dd="-60", carry_worst="-13.7",
    dollar_sh="0.17", mom_sh="1.53", mom_dd="-18",
    combo_sh="1.69", combo_dd="-20", corr="0.26",
    combo_worst5="-9.3", carry_worst5="-11.0",
    mom_be="14.6",
    c0="1.75", c10="1.69", c25="1.59", c50="1.44", c100="1.13",
)

# REAL G10-tape numbers (OECD short rates + yfinance FX, USD-funded monthly, 2001–2024, as-of 2024-01-31,
# fingerprint ef7450ae792e) — these match the executed real-tape cells.
G = dict(
    fp="ef7450ae792e", asof="2024-01-31", span="2001–2024",
    hml="3.0", carry_sh="+0.22", carry_skew="-0.70", carry_dd="-27", carry_worst="-10.6",
    carry_t="1.0",
    ci_lo="-0.17", ci_hi="+0.69", frac_neg="14",
    dollar_sh="+0.17", mom_sh="-0.14",
    combo_sh="+0.06", combo_dd="-26", combo_skew="-0.57", combo_worst="-6.4", corr="+0.05",
    combo_worst5="-3.0", carry_worst5="-7.6",
    to_carry="0.55", to_mom="4.42", carry_be="13",
    c0="+0.17", c10="+0.06", c25="-0.10",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Combo diversifies the crash?: Partial](https://img.shields.io/badge/Combo_diversifies_the_crash%3F-Partial-dab617?style=flat-square)\n\n"
)

REALNOTE = (
    "> ✅ **Real run · offline from cache · as-of {asof} · fingerprint `{fp}`.** This notebook executes on "
    "BOTH the synthetic control (the machinery proof) AND the **real G10 tape** — OECD 3-month short rates "
    "+ yfinance FX, a USD-funded monthly book over {span}, read offline from the two `_cache/g10_*.parquet` "
    "files. Headline numbers in [`../docs/results.md`](../docs/results.md).".format(
        asof=G['asof'], fp=G['fp'], span=G['span'])
)


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Greenback 💵\n"
            "### \"Borrow cheap, lend dear\" earns a premium the literature calls real — our 23-year tape reads it weak — and rents you a spot in front of a steamroller. The fix isn't a stop-loss; it's a second, decorrelated trade.\n\n"
            + BADGES +
            "This builds on [Steamroller](../../27-steamroller/), which showed the FX **carry** premium is "
            "real but crash-prone, and that vol-targeting *can't* dodge the crash. Greenback asks the next "
            "question — the one carry desks actually trade: pair carry with its natural complement, "
            "**momentum** (ride the trend), and add a **dollar-carry** tilt (long/short the USD basket by "
            "the average rate gap). Because carry and momentum pay at *different times*, the **combo** earns "
            "a higher Sharpe than either alone — and momentum tends to lean the other way exactly when "
            "carry is being run over, cushioning the crash.\n\n"
            "> 📓 **Plain-language layer.** The three sleeves, the combo uplift, the leg correlation and the "
            "crash cushion are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Charts below are generated by the code beside them; the study "
            "runs on BOTH a **synthetic** machinery control and the **real G10 tape** (offline from cache — "
            "see [`../docs/results.md`](../docs/results.md)). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(REALNOTE),

        md(
            "## The answer first 💵 (on the real 2001–2024 G10 tape)\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Do high-rate currencies out-earn? | 🟨 **The literature says yes; this sample alone reads "
            "weak.** Carry high-minus-low **+{hml}%/yr**; carry Sharpe **{cs}** — but that is only *t* ≈ 1.0 "
            "over 22 years, and the bootstrap 95% CI **[{lo}, {hi}]** spans zero ({fn}% of resamples "
            "negative). |\n"
            "| Is there a steamroller? | 🟨 **Yes.** Carry skew **{csk}**, worst month **{cw}%** (Oct-2008), "
            "max drawdown **{cdd}%** — the negative-skew crash carry can't escape. |\n"
            "| Did FX momentum work? | 🟥 **No.** Momentum Sharpe **{ms}** over 2001–2024 — cross-sectional "
            "FX momentum decayed to *negative* this sample. |\n"
            "| Did the carry⊕momentum combo help? | 🟨 **Partly.** It *cushions* the crash (legs correlate "
            "just **{corr}**; worst month **{cw}% → {kw}%**), but combo Sharpe **{cob}** can't beat carry "
            "**{cs}** while momentum loses. |\n\n"
            "> Desk shorthand: **Signal `WEAK` · Tradability `FRAGILE` · Combo diversifies the crash? "
            "`PARTIAL`** — the stamp leans on three decades of literature for carry's existence; this "
            "23-year sample alone reads weak. Its crash is *dulled* by diversification but its Sharpe not "
            "lifted, because this sample's momentum had no edge to lend.".format(
                hml=G['hml'], cs=G['carry_sh'], lo=G['ci_lo'], hi=G['ci_hi'], fn=G['frac_neg'],
                csk=G['carry_skew'], cw=G['carry_worst'], cdd=G['carry_dd'], ms=G['mom_sh'],
                corr=G['corr'], kw=G['combo_worst'], cob=G['combo_sh'])
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "The FX carry trade and its complements (Kakushadze-Serur §8.3 dollar carry, §8.4 momentum+carry):\n\n"
            "1. **Carry** — long the high-short-rate currencies, short the low-rate ones, dollar-neutral. "
            "Uncovered interest-rate parity fails, so the rate gap is a paid premium.\n"
            "2. **Dollar carry** — long the whole basket vs USD when foreign rates are high vs the dollar, "
            "short it when they're low (the LRV *dollar factor*).\n"
            "3. **Momentum** — long the currencies trending up, short those trending down.\n"
            "4. **The combo** — blend carry and momentum; the believer's case (Asness-Moskowitz-Pedersen "
            "2013; Koijen et al 2018) is that they pay at different times, so the blend beats either leg."
        ),
        code(
            "carry = strategy.carry_returns(xr, rd, cost_bps=10.0)         # vol-scaled, net @10bp\n"
            "eq = (1+carry).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, color='#2ea44f', lw=1.1)\n"
            "ax.set_title('Carry on a panel where high rates truly out-earn — steady gains, then a crash')\n"
            "s = strategy.summary(carry); print(f\"synthetic carry Sharpe {s['sharpe']:.2f}, skew {s['skew']:.2f}, max DD {s['max_drawdown']*100:.0f}%\")"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "A real, self-financed premium that diversifies a stock/bond book is worth a lot — carry and "
            "trend are the backbone of every macro and CTA desk. But carry's catch is famous: it drips "
            "nickels and then loses a fortune all at once (1998, 2008). The interesting question isn't "
            "whether carry is real (it is) — it's whether you can hold it through the steamroller. The "
            "believers' answer is *diversification*: pair it with momentum, which tends to be short the "
            "crashing currencies just as carry is long them."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "1. **Real?** Carry premium by rate bucket and gross Sharpe on the panel (flat on a full-UIRP null).\n"
            "2. **Crash?** The carry book's skew, worst month and drawdown.\n"
            "3. **Does the combo help?** Combo Sharpe vs either leg, the leg correlation, and the combo's "
            "drawdown vs carry's.\n\n"
            "**Mirage line:** if the combo *didn't* beat the better leg, or the legs were highly correlated, "
            "the diversification story would be a mirage. (On the real tape, a further fragility line is the "
            "carry crash deepening rather than dulling.)"
        ),

        md("## 4 · The teardown 🔧\n\n### 4a · The premium is real where carry is real (control vs null)"),
        code(
            "for label, (xx, rr) in [('carry panel', (xr, rd)), ('full-UIRP null', (x0, r0))]:\n"
            "    pb = strategy.carry_premium_by_bucket(xx, rr)\n"
            "    s = strategy.summary(strategy.carry_returns(xx, rr, cost_bps=0.0, target_vol_ann=None))\n"
            "    print(f\"{label:16} high-minus-low {pb['hml_ann_pct']:+5.1f}%/yr   carry Sharpe {s['sharpe']:+.2f}\")"
        ),
        md(
            "### 4b · The three sleeves and the combo\n"
            "Net @10 bp on the synthetic control — carry crashes, momentum doesn't, and the combo beats both:"
        ),
        code(
            "carry = strategy.carry_returns(xr, rd, cost_bps=10.0)\n"
            "mom   = strategy.momentum_returns(xr, cost_bps=10.0)\n"
            "combo = strategy.combine(carry, mom)\n"
            "dollar = strategy.dollar_carry_returns(xr, rd, cost_bps=10.0)\n"
            "rows = {}\n"
            "for nm, r in [('carry', carry), ('dollar-carry', dollar), ('momentum', mom), ('COMBO', combo)]:\n"
            "    s = strategy.summary(r); rows[nm] = {'Sharpe': s['sharpe'], 'ret%': s['ann_return']*100, 'skew': s['skew'], 'maxDD%': s['max_drawdown']*100}\n"
            "display(pd.DataFrame(rows).T.round(2))\n"
            "for nm, r, c in [('carry', carry, '#c0392b'), ('momentum', mom, '#8b949e'), ('combo', combo, '#2ea44f')]:\n"
            "    plt.plot((1+r).cumprod(), label=nm, color=c, lw=1.1)\n"
            "plt.legend(); plt.title('Carry crashes, momentum cushions — the combo grows steadier than either leg'); plt.yscale('log')"
        ),

        md("### 4c · The same books on the REAL G10 tape (2001–2024)\n"
           "The synthetic above shows the machinery *with a winning momentum leg*. Here is what actually "
           "happened on the real tape — carry pays thinly with its crash, but momentum **lost** money this "
           "sample, so the combo cushions the crash without beating carry on Sharpe:"),
        code(
            "rcarry = strategy.carry_returns(rxr, rrd, cost_bps=10.0)\n"
            "rmom   = strategy.momentum_returns(rxr, cost_bps=10.0)\n"
            "rcombo = strategy.combine(rcarry, rmom)\n"
            "rdollar= strategy.dollar_carry_returns(rxr, rrd, cost_bps=10.0)\n"
            "rows = {}\n"
            "for nm, r in [('carry', rcarry), ('dollar-carry', rdollar), ('momentum', rmom), ('COMBO', rcombo)]:\n"
            "    s = strategy.summary(r); rows[nm] = {'Sharpe': s['sharpe'], 'ret%': s['ann_return']*100, 'skew': s['skew'], 'maxDD%': s['max_drawdown']*100}\n"
            "display(pd.DataFrame(rows).T.round(2))\n"
            "rcc = extension.carry_crash(rxr, rrd, cost_bps=10.0)\n"
            "print(f\"real: carry skew {rcc['carry_skew']:+.2f}, worst month {rcc['carry_worst_month_pct']:+.1f}%; \"\n"
            "      f\"combo cushions to worst month {rcc['combo_worst_month_pct']:+.1f}% (leg corr +0.05)\")\n"
            "for nm, r, c in [('carry', rcarry, '#c0392b'), ('momentum', rmom, '#8b949e'), ('combo', rcombo, '#2ea44f')]:\n"
            "    plt.plot((1+r).cumprod(), label=nm, color=c, lw=1.1)\n"
            "plt.legend(); plt.title('Real G10 tape (2001–2024): carry pays thinly with its crash; momentum lost; combo cushions')"
        ),

        md("## 5 · The verdict 🧾 (on the real tape)\n\n"
           f"- **Signal `WEAK`** — carry high-minus-low **+{G['hml']}%/yr**, carry Sharpe **{G['carry_sh']}** "
           f"at only Lo *t* ≈ {G['carry_t']}, bootstrap CI [{G['ci_lo']}, {G['ci_hi']}] spanning zero — this "
           "sample can't certify the premium; three decades of literature (LRV 2011; Menkhoff et al. 2012) "
           f"carry the existence case. The steamroller, at least, is unmistakable (skew **{G['carry_skew']}**, "
           f"worst month **{G['carry_worst']}%** in Oct-2008).\n"
           f"- **Tradability `FRAGILE`** — thin and cost-sensitive: carry break-even only **≈{G['carry_be']} "
           f"bp**, and the crash never leaves.\n"
           f"- **Combo diversifies the crash? `PARTIAL`** — legs decorrelate (**{G['corr']}**) so the combo "
           f"cushions the crash (worst month **{G['carry_worst']}% → {G['combo_worst']}%**), but combo Sharpe "
           f"**{G['combo_sh']}** can't beat carry **{G['carry_sh']}** because momentum lost (**{G['mom_sh']}**) "
           "this sample.\n\n"
           "> **The honest fix isn't a stop — it's a second trade.** Study 27 showed vol-targeting can't "
           "dodge the carry crash. Pairing carry with decorrelated momentum *does* cushion it — but only "
           "lifts the Sharpe when momentum itself has an edge, which on 2001–2024 it didn't."),

        md("## 6 · Could you trade it? 💸\n\n"
           f"- **Thin and cost-sensitive.** Carry turns over slowly (**{G['to_carry']}×/yr**) but its "
           f"break-even is only **≈{G['carry_be']} bp**; the combo's net Sharpe decays **{G['c0']} (0bp) → "
           f"{G['c10']} (10bp) → {G['c25']} (25bp)** — cost bites fast.\n"
           "- **The threat is the crash *and* the thin edge.** Even diversified, the combo keeps a negative "
           "skew — you must be *willing and able* to hold a crash-prone book through its worst months.\n"
           "- **Capacity** is large (G10 FX is the deepest market on earth); the binding constraint is the "
           "thin, crash-prone edge, not liquidity. Hence `FRAGILE`, not `INVESTABLE`."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — the carry⊕momentum diversification ([`../docs/extension.md`](../docs/extension.md))\n"
            "The headline of the study, made explicit on the real tape:\n\n"
            f"- The legs decorrelate (**{G['corr']}**) so the combo *cushions* the steamroller: worst month "
            f"**{G['carry_worst']}% → {G['combo_worst']}%**, and in carry's worst 5 months the combo lost "
            f"**{G['combo_worst5']}%** vs carry **{G['carry_worst5']}%**.\n"
            f"- But the combo Sharpe **{G['combo_sh']}** can't beat carry **{G['carry_sh']}**, because FX "
            f"momentum lost money (**{G['mom_sh']}**) over 2001–2024 — a losing leg can't lift the blend.\n"
            "- On the synthetic control (where momentum is profitable *by construction*) the combo DOES beat "
            f"both legs (**{R['combo_sh']}** vs **{R['carry_sh']}**/**{R['mom_sh']}**) — so the mechanism is "
            "real; the real-tape `PARTIAL` is about *this sample's* dead momentum, not the combo idea.\n\n"
            "### Other forks\n"
            "- **Optimal combo weight** — sweep `w_carry` and risk-parity vs the realised crisis correlation.\n"
            "- **Dollar-carry as a third leg** — add the LRV dollar factor (Sharpe +0.17 on the real tape; "
            "note its cost model under-charges the basket's turnover, so that number is if anything "
            "flattered).\n"
            "- **Crisis-conditional tilt** — lean toward momentum when global FX vol is rising.\n\n"
            "PRs welcome."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Greenback — a quantitative teardown 🔬\n"
            "### The carry premium by bucket · three sleeves · the combo's Sharpe uplift · the leg correlation · the negative-skew crash\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The steelman: FX carry is a real premium (§8.3/§8.4 of "
            "Kakushadze-Serur), and the **carry⊕momentum combo** beats either leg because the two "
            "decorrelate. The synthetic control confirms the machinery (carry Sharpe +1.18, null flat, "
            "combo +1.69 with a winning momentum leg). On the **real G10 tape (2001–2024)** carry reads "
            "`WEAK` (Sharpe +0.22, Lo *t* ≈ 1.0, bootstrap CI spanning zero — the literature, not this "
            "sample, carries the existence case) with its `FRAGILE` steamroller (skew −0.70, worst month "
            "−10.6% in Oct-2008), FX momentum **decayed to −0.14**, and so the combo `PARTIAL`-ly delivers: "
            "it cushions the crash (decorrelated legs +0.05) but can't out-Sharpe carry — building on Study "
            "27, not repeating it.\n\n"
            "> ⚠️ **Not investment advice.** Executes on BOTH a synthetic control and the **real G10 tape** "
            "(offline from cache); the fingerprinted real numbers are in "
            "[`../docs/results.md`](../docs/results.md), sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),
        md(REALNOTE),

        md(
            "## Beat 0 · Verdict (on the real 2001–2024 G10 tape)\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — carry real? | 🟡 `WEAK` | High-minus-low **+{G['hml']}%/yr**, carry Sharpe "
            f"**{G['carry_sh']}** — but Lo *t* ≈ **{G['carry_t']}** and the bootstrap CI **[{G['ci_lo']}, "
            f"{G['ci_hi']}]** spans zero ({G['frac_neg']}% of resamples negative): this 23-year sample alone "
            "can't certify the premium; the stamp's existence case is the literature's three decades. |\n"
            f"| **Tradability** | 🟡 `FRAGILE` | Carry skew **{G['carry_skew']}**, worst month "
            f"**{G['carry_worst']}%**, break-even only **≈{G['carry_be']} bp**; thin and crash-prone. |\n"
            f"| **Combo diversifies the crash?** | 🟡 `PARTIAL` | Legs decorrelate (**{G['corr']}**) so the "
            f"combo cushions the crash (worst month **{G['carry_worst']}% → {G['combo_worst']}%**), but combo "
            f"Sharpe **{G['combo_sh']}** < carry **{G['carry_sh']}** because momentum lost (**{G['mom_sh']}**). |\n\n"
            "> **In one sentence:** an FX carry premium the literature calls real but this 23-year sample "
            "stamps `WEAK` (*t* ≈ 1.0, CI spanning zero), with a brutal negative-skew crash, *cushioned* "
            "(not out-Sharpe'd) by blending in decorrelated momentum — because over 2001–2024 FX momentum "
            "itself lost money, the combo dulls the jump without lifting the Sharpe.\n\n"
            "*(Both the synthetic control and the real G10 tape execute below; fingerprinted real numbers in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "Carry weight $w^{c}_{i,t}\\propto \\text{sign of rank}(r_{i}-\\bar r)$ — long the top, short the "
            "bottom rate-differential quantile, dollar-neutral, lagged. Momentum weight "
            "$w^{m}_{i,t}\\propto \\text{rank of }\\big(\\prod_{k=1}^{12}(1+x_{i,t-k})-1\\big)$. Both legs are "
            "vol-scaled to a common target, then the **combo** is $\\tfrac12 c_t+\\tfrac12 m_t$. Claim: "
            "$\\text{Sharpe}(\\text{combo})>\\max(\\text{Sharpe}(c),\\text{Sharpe}(m))$ and "
            "$\\text{corr}(c,m)$ is low. Null: full UIRP ⇒ no carry premium (the spot exactly offsets the "
            "rate gap).\n\n"
            "Two construction notes, stated plainly: the momentum sleeve is a **12-month trend** — the "
            "trailing 12-month return with *no* skip month (a 12-0 signal, not the academic 12-1 "
            "convention that drops the most recent month); and the rate-bucket table below is a same-month "
            "*descriptive* sort (unlagged), unlike the tradable sleeves whose weights are lagged one month."
        ),
        code(
            "for label, (xx, rr) in [('carry panel', (xr, rd)), ('null', (x0, r0))]:\n"
            "    pb = strategy.carry_premium_by_bucket(xx, rr)\n"
            "    s = strategy.summary(strategy.carry_returns(xx, rr, cost_bps=0.0, target_vol_ann=None))\n"
            "    print(f\"{label:12} high-minus-low {pb['hml_ann_pct']:+5.1f}%/yr   carry Sharpe {s['sharpe']:+.2f}   skew {s['skew']:+.2f}\")"
        ),
        md(
            "> 💡 **In plain words.** When carry is real the high-rate bucket out-earns the low-rate bucket "
            f"(**+{R['hml']}%/yr**) and the carry book makes money with a sharp negative skew; under full "
            f"UIRP it's flat (**+{R['null_hml']}%/yr**). The apparatus sees carry only when it's there — so "
            "the real-tape result is about the *market*."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Lustig-Roussanov-Verdelhan (2011) show carry and the dollar factor are *priced* currency risk "
            "premia; Asness-Moskowitz-Pedersen (2013) and Koijen et al (2018) show carry/value and momentum "
            "are negatively correlated *everywhere*, so combining them is the single highest-Sharpe move in "
            "factor investing. The prediction: the legs decorrelate and momentum leans *against* the carry "
            "crash. Beats 4-7 test that head-on — and find the decorrelation holds on the real tape, but the "
            "Sharpe uplift needs a *winning* momentum leg, which 2001–2024 didn't provide."
        ),

        md(
            "## Beat 3 · Protocol\n\n"
            "1. **Real?** `carry_premium_by_bucket` + `carry_returns` (gross) on control vs null.\n"
            "2. **Crash?** the carry book's `skew`, worst month, `max_drawdown` (`extension.carry_crash`).\n"
            "3. **Combo?** `extension.diversification` — combo Sharpe vs legs, the leg correlation.\n"
            "4. **Costs?** `costs.cost_sweep`, `costs.breakeven_cost_bps`.\n\n"
            "**Mirage line:** combo fails to beat the better leg, or the legs are highly correlated — then "
            "the diversification story is a mirage."
        ),

        md("## Beat 4 · The teardown\n\n### 4a · The three sleeves (net @10 bp)"),
        code(
            "carry = strategy.carry_returns(xr, rd, cost_bps=10.0)\n"
            "mom   = strategy.momentum_returns(xr, cost_bps=10.0)\n"
            "combo = strategy.combine(carry, mom)\n"
            "dollar = strategy.dollar_carry_returns(xr, rd, cost_bps=10.0)\n"
            "rows = {}\n"
            "for nm, r in [('carry', carry), ('dollar-carry', dollar), ('momentum', mom), ('COMBO', combo)]:\n"
            "    s = strategy.summary(r)\n"
            "    rows[nm] = {'Sharpe': s['sharpe'], 'ret%': s['ann_return']*100, 'vol%': s['vol_ann']*100, 'skew': s['skew'], 'maxDD%': s['max_drawdown']*100}\n"
            "display(pd.DataFrame(rows).T.round(2))"
        ),
        md(
            "> 💡 **In plain words.** Carry and momentum both earn — but carry crashes (skew −1.55, DD −60%) "
            "and momentum doesn't (DD −18%). The dollar-carry tilt is weak *on this synthetic* because the "
            "average rate gap is near-constant by construction; on the real tape, where the USD rate cycles, "
            "it's a genuine separate premium (Sharpe +0.17 — shown in 4d)."
        ),

        md("### 4b · The combo beats the legs, and the legs decorrelate (the REAL diversification call)"),
        code(
            "d = extension.diversification(xr, rd, cost_bps=10.0)\n"
            "print(f\"carry Sharpe {d['carry_sharpe']:+.2f}  momentum {d['momentum_sharpe']:+.2f}  combo {d['combo_sharpe']:+.2f}\")\n"
            "print(f\"combo beats both legs: {d['combo_beats_legs']}   carry-momentum correlation {d['leg_correlation']:+.2f}\")\n"
            "fig, ax = plt.subplots()\n"
            "for nm, r, c in [('carry', carry, '#c0392b'), ('momentum', mom, '#8b949e'), ('combo', combo, '#2ea44f')]:\n"
            "    ax.plot((1+r).cumprod(), label=nm, color=c, lw=1.1)\n"
            "ax.legend(); ax.set_yscale('log'); ax.set_title('Equal-risk carry+momentum combo grows steadier than either leg')"
        ),

        md("### 4c · The steamroller, and how momentum cushions it (the FRAGILE call)"),
        code(
            "cc = extension.carry_crash(xr, rd, cost_bps=10.0)\n"
            "print(f\"carry: skew {cc['carry_skew']:+.2f}, worst month {cc['carry_worst_month_pct']:+.1f}%, max DD {cc['carry_max_dd_pct']:.0f}%\")\n"
            "print(f\"combo: skew {cc['combo_skew']:+.2f}, worst month {cc['combo_worst_month_pct']:+.1f}%, max DD {cc['combo_max_dd_pct']:.0f}%\")\n"
            "print(f\"in carry's worst 5 months, combo lost {cc['combo_in_carry_worst5_pct']:+.1f}% vs carry {cc['carry_worst5_mean_pct']:+.1f}%\")\n"
            "fig, ax = plt.subplots()\n"
            "for nm, r, c in [('carry', carry, '#c0392b'), ('combo', combo, '#2ea44f')]:\n"
            "    eq = (1+r).cumprod(); ax.plot(eq/eq.cummax()-1, label=nm, color=c, lw=1.0)\n"
            "ax.legend(); ax.set_title('Drawdown: the combo (green) is dragged far less deep into the steamroller')"
        ),
        md(
            "> 💡 **In plain words.** Diversification is the honest fix Study 27's vol-overlay couldn't "
            "provide: momentum is leaning *against* the crashing carry trade, so the combo's drawdown is "
            f"**{R['combo_dd']}%** versus carry's **{R['carry_dd']}%**. But the combo's skew is still "
            "negative — the jump is *dulled*, not gone. That's exactly why tradability is `FRAGILE`."
        ),

        md("### 4d · The same books on the REAL G10 tape (2001–2024) — the earned verdict\n"
           "Synthetic 4a–4c showed the machinery *with a winning momentum leg by construction*. Now the real "
           "tape: carry pays thinly with its crash, momentum **lost** money, and the combo cushions the "
           "crash without out-Sharpe-ing carry. Both books from the same functions, no re-fit."),
        code(
            "rcarry = strategy.carry_returns(rxr, rrd, cost_bps=10.0)\n"
            "rmom   = strategy.momentum_returns(rxr, cost_bps=10.0)\n"
            "rcombo = strategy.combine(rcarry, rmom)\n"
            "rdollar= strategy.dollar_carry_returns(rxr, rrd, cost_bps=10.0)\n"
            "rows = {}\n"
            "for nm, r in [('carry', rcarry), ('dollar-carry', rdollar), ('momentum', rmom), ('COMBO', rcombo)]:\n"
            "    s = strategy.summary(r)\n"
            "    rows[nm] = {'Sharpe': s['sharpe'], 'ret%': s['ann_return']*100, 'vol%': s['vol_ann']*100, 'skew': s['skew'], 'maxDD%': s['max_drawdown']*100}\n"
            "display(pd.DataFrame(rows).T.round(2))\n"
            "rd_ = extension.diversification(rxr, rrd, cost_bps=10.0)\n"
            "rcc = extension.carry_crash(rxr, rrd, cost_bps=10.0)\n"
            "print(f\"real carry premium (high-minus-low): {strategy.carry_premium_by_bucket(rxr, rrd)['hml_ann_pct']:+.1f}%/yr\")\n"
            "print(f\"leg correlation {rd_['leg_correlation']:+.2f}; combo beats best leg: {rd_['combo_beats_legs']}\")\n"
            "print(f\"crash cushion — carry worst month {rcc['carry_worst_month_pct']:+.1f}% vs combo {rcc['combo_worst_month_pct']:+.1f}%; \"\n"
            "      f\"in carry's worst 5 months combo {rcc['combo_in_carry_worst5_pct']:+.1f}% vs carry {rcc['carry_worst5_mean_pct']:+.1f}%\")\n"
            "fig, ax = plt.subplots()\n"
            "for nm, r, c in [('carry', rcarry, '#c0392b'), ('momentum', rmom, '#8b949e'), ('combo', rcombo, '#2ea44f')]:\n"
            "    ax.plot((1+r).cumprod(), label=nm, color=c, lw=1.1)\n"
            "ax.legend(); ax.set_title('Real G10 tape: carry pays thinly with its crash, momentum lost, combo cushions but does not out-Sharpe')"
        ),
        md(
            "> 💡 **In plain words.** The decorrelation thesis *holds* on the real tape (leg corr +0.05, the "
            "combo's worst month halves) — the diversification is real. But the Sharpe uplift needs a "
            "*winning* momentum leg, and 2001–2024 FX momentum lost money, so the combo cushions the crash "
            "without beating carry. That is the honest `PARTIAL`."
        ),

        md("## Beat 5 · The verdict (real tape)\n\n"
           f"- **`WEAK`** (4d): high-minus-low +{G['hml']}%/yr, carry Sharpe {G['carry_sh']} — but Lo *t* ≈ "
           f"{G['carry_t']} and bootstrap CI [{G['ci_lo']}, {G['ci_hi']}] spanning zero: below the desk's "
           "robust-inference bar. The stamp leans on three decades of literature for existence; this sample "
           "alone reads weak.\n"
           f"- **`FRAGILE`** (4d): carry skew {G['carry_skew']}, worst month {G['carry_worst']}%, break-even ≈{G['carry_be']} bp; crash + thin edge.\n"
           f"- **Combo diversifies the crash? `PARTIAL`**: legs decorrelate ({G['corr']}), combo cushions ({G['carry_worst']}% → {G['combo_worst']}% worst month) but Sharpe {G['combo_sh']} < carry {G['carry_sh']} (momentum {G['mom_sh']}).\n\n"
           "> **Signal `WEAK` · Tradability `FRAGILE` · Combo diversifies the crash? `PARTIAL`** — the "
           "dollar-carry / carry⊕momentum-combo companion to Study 27."),

        md("## Beat 6 · Could you trade it?\n\n"
           f"- **Thin and cost-sensitive.** Carry turns over slowly ({G['to_carry']}×/yr) but break-even is "
           f"only ≈{G['carry_be']} bp; the combo's net Sharpe decays {G['c0']} → {G['c10']} → {G['c25']} "
           "from 0 to 25 bp — cost bites fast.\n"
           "- **The binding risk is the crash *and* the thin edge** (G10 FX capacity itself is the deepest "
           "market on earth). Even diversified, the combo keeps negative skew.\n"
           "- **The real fragility line** (confirmed real tape): the steamroller hits in Oct-2008 (−10.6%) "
           "and Mar-2020 (−9.1%) — the GFC and COVID risk-offs."),

        md("## Beat 7 · Going further\n\n"
           "### 7a · Worked complement — the carry⊕momentum diversification ([`../docs/extension.md`](../docs/extension.md))\n"
           "The cost sweep below confirms (on the synthetic control) the combo's cost behaviour; the "
           "diversification itself (beats 4b/4d) is the complement. On the real tape the decorrelation holds "
           "but momentum's lost edge makes the combo `PARTIAL`."),
        code(
            "print('SYNTHETIC control combo cost sweep:')\n"
            "cs = costs.cost_sweep(xr, rd, which='combo')\n"
            "display(cs.round(3))\n"
            "print('REAL G10 combo cost sweep:')\n"
            "rcs = costs.cost_sweep(rxr, rrd, which='combo')\n"
            "display(rcs.round(3))\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#8b949e', label='synthetic control')\n"
            "ax.plot(rcs.index, rcs['sharpe'], marker='s', color='#2ea44f', label='real G10')\n"
            "ax.legend(); ax.set_xlabel('cost (bp per unit traded)'); ax.set_ylabel('net Sharpe'); ax.set_title('Combo cost decay — real G10 edge is thin, crossing zero by ~25 bp')\n"
            "print('real carry break-even:', round(costs.breakeven_cost_bps(rxr, rrd, which='carry'), 1), 'bp')"
        ),
        md(
            "### 7b · Other forks\n"
            "- **Optimal combo weight** — sweep `w_carry`, risk-parity vs the realised crisis correlation.\n"
            "- **Dollar-carry as a third leg** — the LRV dollar factor earns Sharpe +0.17 on the real tape "
            "(its cost term under-charges the basket's turnover by roughly the basket width, so that "
            "non-headline number is if anything flattered).\n"
            "- **Crisis-conditional tilt** — lean toward momentum when global FX vol (Brunnermeier-Nagel-"
            "Pedersen) is rising.\n\n"
            "**The result.** On the synthetic control the carry premium is recovered and the combo beats "
            "both legs *with a winning momentum leg* — the machinery proof. On the **real G10 tape "
            "(2001–2024)** carry reads `WEAK` (*t* ≈ 1.0, CI spanning zero — the literature carries the "
            "existence case) with its `FRAGILE` steamroller, FX momentum decayed to negative, and the "
            "combo `PARTIAL`-ly delivers — cushioning the crash (decorrelated legs) without lifting the "
            "Sharpe. Exactly the §8.3/§8.4 thesis, building on Study 27. Fingerprinted writeups in "
            "[`../docs/results.md`](../docs/results.md) and [`../docs/extension.md`](../docs/extension.md)."
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
