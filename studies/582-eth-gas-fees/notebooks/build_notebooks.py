"""Generate the two narrative notebooks for Study 582 (ETH-Gas-Fees).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; there is NO real gas tape (synthetic-only study), so every
signal-side cell is synthetic. The real ETH-USD tape is used only to anchor the calibration and is
cache-first (empty offline); ``R`` mirrors docs/results.md so the notebook re-runs for any reader.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30). Synthetic-only study.
# Null world fp 0e87032ad333 (1461 days, 1432 usable rows); real ETH anchor fp d4b0cdbf2e5e.
R = dict(
    fp_null="0e87032ad333", fp_eth="d4b0cdbf2e5e",
    n_days=1461, n_rows=1432, n_spike=144, n_calm=1288,
    eth_n=3101, eth_vol=0.0444,
    spike_mean=0.44, calm_mean=0.37, gap=0.07, gap_t=0.17, slope_t=-0.04, placebo_p=0.856,
    gross_ann=-16.2, gross_t=-1.00, net_ann=-21.2, net_t=-1.30,
    one_way_bps=10.0, borrow_bps=1000.0,
    # horizon sweep: (horizon, gap%, gap_t, slope_t)
    hz=[(1, 0.001, 0.17, -0.04), (3, 0.003, 0.34, 0.31), (7, 0.009, 0.82, 1.05),
        (14, 0.046, 2.70, 1.56), (30, 0.042, 1.57, 1.03)],
    # synthetic positive control: (beta, mean slope-t over 25 seeds)
    ctrl=[(0.0, -0.29), (-0.001, -1.47), (-0.002, -2.65), (-0.004, -5.02),
          (-0.006, -7.40), (-0.010, -12.16)],
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

from eth_gas_fees import data, strategy as st

# The HEADLINE world is the NULL (gas spikes carry no forward info) — the honest 'no real tape,
# here is what the engine reads when there is nothing' position. Deterministic and offline.
DF_NULL, _ = data.synthetic_series(top_signal_beta=0.0, seed=582)
FR = st.build_frame(DF_NULL, horizon=1)

# There is NO real gas tape (synthetic-only study). The real ETH price tape is cache-first and
# used ONLY to anchor the calibration; HAVE_REAL_ETH just gates the anchor print.
_eth = data.fetch_eth_prices(fetch=False)
HAVE_REAL_ETH = not _eth.empty
print("null world:", len(DF_NULL), "days |", len(FR), "usable rows | real ETH anchor present:",
      HAVE_REAL_ETH)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# ETH Gas Fees — does a spike mean the top is in? ⛽\n"
            "### When everyone's paying through the nose to transact, is that euphoria… or real demand?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Top_signal_on-chain%3F: Unproven](https://img.shields.io/badge/Top_signal_on--chain%3F-Unproven-8b949e?style=flat-square)\n\n"
            "Every time you do *anything* on Ethereum — send a token, swap on a DEX, mint an NFT — you "
            "pay a **gas fee**. When the network is busy, gas spikes: people bid each other up to get "
            "their transaction in first. Crypto lore says a **gas-fee spike is a top signal**: it means "
            "the crowd is frantic — aping into memecoins, chasing mints, over-leveraged — and that kind "
            "of euphoria usually marks a local peak. The opposite reading is just as plausible: high gas "
            "means people genuinely *want* the chain, which is bullish.\n\n"
            "So which is it? We'd love to check on real data — but here's the honest catch we hit "
            "immediately.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the cost "
            "maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can we test 'gas spike = top' on real data? | **No — and that's the headline.** A clean "
            "daily history of Ethereum gas fees needs an archive node or a paid API. A free retail stack "
            "reaches ETH *prices* but not *gas*. So there's no real tape to judge the claim. |\n"
            "| So what *can* we do? | Build a fair **synthetic** world where congestion follows euphoria, "
            "and prove our test would catch a real top signal if one existed — and check what it reads "
            "when there's *nothing* there. |\n"
            "| Does the test catch a planted top signal? | **Yes** — plant one and the detector lights up "
            "past the significance bar, and it's steady across 25 random seeds. |\n"
            "| And when there's no signal (the null)? | **It correctly reads flat**, and a contrarian "
            "'short ETH on the spike' trade *loses money even before fees*. |\n\n"
            "> Verdict: **Signal None** (no real tape to certify), **Tradability Mirage** (the short "
            "bleeds), **Top signal on-chain? Unproven** — a good story we simply can't test on real data "
            "yet."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When gas fees spike, the top is in.\"* — crypto on-chain folklore.\n\n"
            "The intuition: gas is the *price of using the chain*. It only spikes when demand for "
            "blockspace overwhelms supply — a memecoin mania, an airdrop farm, a leverage frenzy. That's "
            "peak-euphoria behaviour, and euphoria marks tops. So a spike should be **bearish** for "
            "forward ETH returns (a contrarian signal). The bulls answer: high gas is just proof the "
            "chain is *wanted* — that's demand, and demand is bullish. The two readings point opposite "
            "ways, which is exactly why it's worth a test."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If gas spikes really did mark tops, you'd have a free, public, real-time sell signal — no "
            "subscription, just look at the chain. That's a big 'if'. The desk has looked at other "
            "on-chain and sentiment gauges ([325 Crypto-Fear-Greed](../../325-crypto-fear-greed/), "
            "[292 Bitcoin-Hashrate](../../292-bitcoin-hashrate/)); this one goes at **blockspace "
            "congestion** — arguably the purest 'how frantic is everyone right now' meter crypto has."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Define a spike.** Compare today's gas to its own recent baseline (a rolling median). "
            "Well above baseline = a spike.\n"
            "2. **Look forward.** After a spike, what did ETH do *next*? The top-signal claim says: go "
            "*down* (or at least underperform the calm days).\n"
            "3. **Be fair about timing.** The spike is measured from *past* data only, so it's known "
            "today; we then measure tomorrow's (and next week's) return. No peeking.\n\n"
            "**The wall we hit:** step 1 needs a daily gas-fee series, and we can't get a clean, long one "
            "for free. So we do the honest next-best thing — build a *synthetic* world where gas and "
            "price move together the way the story says, and use it to (a) prove our test works and (b) "
            "see what 'no signal' looks like."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — a fair synthetic world\n\n"
            "First: **when there's genuinely no forward signal (the null), does our test correctly say "
            "so?** Here are the forward ETH returns on gas-*spike* days vs *calm* days."
        ),
        code(
            "sc = st.spike_vs_calm(FR)\n"
            "spike_m, calm_m, gap, t = sc['spike_mean']*100, sc['calm_mean']*100, sc['gap']*100, sc['t']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.3))\n"
            "ax.bar(['gas-SPIKE days','CALM days'], [spike_m, calm_m], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg next-day ETH return %')\n"
            "ax.set_title(f'Null world: spike vs calm forward return (gap {gap:+.2f}%, t {t:+.2f})')\n"
            "fig.suptitle('SYNTHETIC null world — no planted signal', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'spike days +{spike_m:.2f}% | calm days +{calm_m:.2f}% | gap {gap:+.2f}% (t {t:+.2f})')"
        ),
        md(
            f"The two bars are basically the same height — gap **+{R['gap']}%**, *t* **+{R['gap_t']}**. "
            "The contrarian claim needs spike days to be *lower* (a negative gap). Here there's nothing, "
            "which is exactly right: we planted no signal, and the test found none. Good — the detector "
            "isn't hallucinating."
        ),
        md(
            "Now the other half: **if we DO plant a real top signal, does the test catch it?** We plant "
            "signals of increasing strength and watch the detector (a slope that goes *negative* when "
            "gas spikes precede weak returns), averaged over 25 random seeds so no lucky seed can fake "
            "it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "betas = [c[0] for c in ctrl]\n"
            "ts = [st.synthetic_mean_t(data, top_signal_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (real top signal)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted top-signal strength (more negative = stronger)')\n"
            "ax.set_ylabel('detector: mean slope-t (25 seeds)')\n"
            "ax.set_title('Flat at the null, dives past -2 as we plant a real signal')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'planted {b:+.3f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "At zero (the null) the detector sits at ≈ 0; as we plant a stronger and stronger top signal "
            "it dives past −2. So the machine is **honest and faithful**: flat when there's nothing, loud "
            "when there's something. The only thing missing is a real gas tape to point it at."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Not because the idea is silly, but because there's **no real gas-fee "
            "tape** on a free stack, so we can't clear the bar for a real signal. On the synthetic null "
            "the test correctly reads flat.\n"
            "- **Tradability — Mirage.** The contrarian 'short ETH on the spike' trade *loses even "
            "gross* in the null world — and shorting crypto pays a heavy borrow on top.\n"
            "- **Top signal on-chain? — Unproven.** Coherent story, faithful detector, but untested on "
            "real data. Not busted, not confirmed."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even in the null world, the 'short ETH when gas spikes' overlay bleeds — you're fighting "
            "the coin's long-run drift and paying a stiff borrow to short it. And there's no real signal "
            "to offset that cost. On top of it all, **EIP-1559** (2021) changed how gas fees even work, "
            "and the **L2 migration** (2023-25) moved most activity off the main chain — so 'gas' in 2025 "
            "measures something different than in 2021. A real test has to untangle all that first."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The cousins.** [325 Crypto-Fear-Greed](../../325-crypto-fear-greed/) tests a *sentiment* "
            "gauge the same contrarian way; [292 Bitcoin-Hashrate](../../292-bitcoin-hashrate/) and "
            "[295 Stablecoin-Supply](../../295-stablecoin-supply/) test other on-chain meters.\n"
            "- **The real test needs the real tape.** Pull a daily gas series from an archive node / "
            "Dune / Etherscan, handle the EIP-1559 and L2 breaks, and re-run this exact engine.\n\n"
            "*Think gas spikes really do mark tops? Fork this, drop in a real archive-node gas series, "
            "and show the spike-minus-calm gap clearing t = −2 the contrarian way.*"
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
            "# ETH Gas Fees as a top signal — a quantitative teardown 🔬\n"
            "### trailing gas spike · spike−calm two-sample *t* · HAC forward slope · label-shuffle placebo · horizon sweep · costs & borrow · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Top_signal_on-chain%3F: Unproven](https://img.shields.io/badge/Top_signal_on--chain%3F-Unproven-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether an Ethereum "
            "**gas-fee spike** predicts LOW forward ETH returns (the contrarian top signal) — on a "
            "**synthetic** tape, because the real daily gas series is unreachable on a no-key stack.\n\n"
            "> ⚠️ **Not investment advice.** Synthetic-only on the signal side (no real gas tape); the "
            f"real ETH-USD price tape ({R['eth_n']} daily returns, fp `{R['fp_eth']}`) only anchors the "
            "calibration. Null world fp `" + R["fp_null"] + "`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Synthetic-only (no real gas tape → no *t* ≥ 2 possible). Null "
            f"world: spike−calm gap **+{R['gap']}%** (two-sample *t* **+{R['gap_t']}**), HAC slope-*t* "
            f"**{R['slope_t']}**, placebo *p* **{R['placebo_p']}** — flat. |\n"
            f"| **Tradability** | `MIRAGE` | Contrarian short-ETH overlay: gross **{R['gross_ann']}%**/yr "
            f"(HAC *t* {R['gross_t']}) → net **{R['net_ann']}%**/yr ({R['one_way_bps']:.0f} bps + "
            f"{R['borrow_bps']:.0f} bps/yr borrow). Loses even gross. |\n"
            "| **Top signal on-chain?** | `UNPROVEN` | Engine faithful (control below), but no real tape "
            "to decide; EIP-1559 + L2 migration muddy 'gas'. |\n\n"
            "> 💡 In plain words: the detector works (the synthetic control proves it), but the signal "
            "side has **no real data** to test — so we honestly stamp it unproven, not real."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t$ be the standardised trailing **gas spike** at day $t$ and $r_{t\\to t+H}$ the "
            "forward ETH return over the next $H$ days.\n\n"
            "- **H₁ (contrarian top signal).** $\\mathbb{E}[r \\mid \\text{spike}] < "
            "\\mathbb{E}[r \\mid \\text{calm}]$ — a *negative* spike−calm gap, at *t* ≤ −2.\n"
            "- **H₂ (forward slope).** slope of $r$ on $s < 0$ (bigger spike → lower forward return).\n"
            "- **H₃ (robustness).** the sign holds across forward horizons $H$.\n"
            "- **H₄ (net).** H₁ survives costs and the short borrow.\n\n"
            "**The binding constraint is data**: $s_t$ needs a real daily gas series we cannot get on a "
            "no-key stack. So we test the *machinery* on a synthetic world and report what it reads at "
            "the null — where H₁–H₄ are all correctly **not** rejected (there is nothing to find)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If a real gas tape carried H₁, gas would be a free public top signal — a rare thing. But "
            "two forces make even a *real* test hard: **(a)** the data barrier (archive node / keyed "
            "API), and **(b)** regime breaks — **EIP-1559** (Aug 2021) turned gas into a burned base "
            "fee + tip, and the **2023-25 L2 migration** moved most activity off mainnet, so post-2023 "
            "gas measures a different activity base. This maps to the desk's other data-starved claims "
            "([273 Lego](../../273-lego-returns/), [275 Whisky](../../275-whisky-cask/), "
            "[276 Sneaker](../../276-sneaker-resale/)): synthetic-only, capped below `REAL`."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Gas spike.** $s_t = (g_t - \\text{med}_{30}) / (1.4826\\,\\text{MAD}_{30})$ — a "
            "backward-looking, MAD-scaled deviation of gas from its 30-day baseline (public at day $t$).\n"
            "- **Buckets.** Top-decile spike days ('euphoric congestion') vs the rest.\n"
            "- **Inference.** Two-sample (Welch) *t* on spike vs calm forward returns; a HAC (Newey-West) "
            "slope of forward return on $s$; a **label-shuffle placebo** null (2000 perms).\n"
            "- **Robustness.** Re-run over horizons $H \\in \\{1,3,7,14,30\\}$ days.\n"
            "- **Frictions.** One-way turnover on every position change + a crypto short borrow on short "
            "days.\n"
            "- **Positive control.** A deterministic synthetic world with a planted top signal "
            "(`top_signal_beta < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the spike is trailing-only (public at day-$t$ close); the outcome is the forward "
            "return over $(t, t+H]$. One documented execution lag; no future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The spike−calm gap — H₁\n\n"
            "Forward ETH returns on top-decile gas-spike days vs calm days, with the two-sample *t* and "
            "the placebo null (null world — no planted signal)."
        ),
        code(
            "sc = st.spike_vs_calm(FR); p = st.placebo_pvalue(FR, n_perm=2000, seed=582)\n"
            "spike_m, calm_m, gap, t = sc['spike_mean']*100, sc['calm_mean']*100, sc['gap']*100, sc['t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['SPIKE (top decile)','CALM'], [spike_m, calm_m], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward ETH return %')\n"
            "ax.set_title(f'Spike - calm = {gap:+.2f}%  (two-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'spike +{spike_m:.2f}% (n {sc[\"n_spike\"]}) | calm +{calm_m:.2f}% (n {sc[\"n_calm\"]}) | gap {gap:+.2f}% | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **not rejected — because there is nothing to reject.** The gap is "
            f"**+{R['gap']}%** (*t* +{R['gap_t']}), placebo *p* = {R['placebo_p']}. A real top signal "
            "would show a *negative*, significant gap; the null shows the flat line the engine should "
            "read when no signal is planted."
        ),
        md(
            "### 4b · The forward slope — H₂\n\n"
            "Regress the forward ETH return on the gas spike (HAC *t*). A *negative* slope is the "
            "contrarian top signal."
        ),
        code(
            "sl = st.spike_slope(FR)\n"
            "x = FR['gas_spike'].to_numpy(); y = FR['fwd_ret'].to_numpy()*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=10, color=GREY, alpha=.5)\n"
            "xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)\n"
            "ax.plot(xs, (sl['slope']*xs + (y.mean()/100 - sl['slope']*np.nanmean(x)))*100, c=RED, lw=2)\n"
            "ax.set_xlabel('gas spike (higher = more congested)'); ax.set_ylabel('forward ETH return %')\n"
            "ax.set_title(f\"Slope-t (HAC) {sl['tstat']:+.2f} — flat (negative would be the top signal)\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope {sl['slope']:+.5f}/unit, slope-t (HAC) {sl['tstat']:+.2f}, n {sl['n']}\")"
        ),
        md(
            f"> 💡 In plain words: H₂ **not rejected** — slope-*t* **{R['slope_t']}**, indistinguishable "
            "from zero. In the null world gas spikes carry no forward information, and the HAC slope "
            "confirms it."
        ),
        md(
            "### 4c · Robustness — H₃ (does anything appear at other horizons?)\n\n"
            "The same test over forward horizons of 1, 3, 7, 14 and 30 days."
        ),
        code(
            "hz = " + repr([h[0] for h in R["hz"]]) + "\n"
            "sweep = st.horizon_sweep(DF_NULL, horizons=tuple(hz))\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "ax.plot(sweep.index, sweep['slope_t'], 'o-', c=RED, lw=2, label='forward slope-t (HAC)')\n"
            "ax.plot(sweep.index, sweep['gap_t'], 's--', c=GREY, lw=1.5, label='spike-calm gap-t')\n"
            "ax.axhline(-2, ls=':', c=GREEN, lw=1, label='t = -2 (top signal)'); ax.axhline(2, ls=':', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('forward horizon (days)'); ax.set_ylabel('t-stat')\n"
            "ax.set_title('No horizon produces a negative (top-signal) slope-t past -2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ — **nothing lights up the contrarian way at any horizon.** The one "
            "gap-*t* that pokes above +2 (14-day) is the *wrong sign* for the top-signal claim and "
            "doesn't survive as a HAC slope. No stable negative signal anywhere."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — synthetic-only (no real gas tape, so no *t* ≥ 2 possible); null world "
            f"flat (gap +{R['gap']}%, *t* +{R['gap_t']}, slope-*t* {R['slope_t']}, placebo *p* {R['placebo_p']}).\n"
            f"- **Tradability `MIRAGE`** — contrarian short overlay gross {R['gross_ann']}%/yr → net "
            f"{R['net_ann']}%/yr; loses before *and* after costs.\n"
            "- **Top signal on-chain? `UNPROVEN`** — faithful engine (control below), no real data to "
            "certify, regime breaks in what 'gas' means."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "The contrarian short-ETH overlay, gross vs net of one-way turnover + a crypto short borrow."
        ),
        code(
            "gross = st.summarize(st.overlay_returns(FR))\n"
            "net = st.summarize(st.net_overlay_returns(FR, one_way_bps=10.0, borrow_bps_ann=1000.0))\n"
            "g_ann, n_ann = gross['mean']*365*100, net['mean']*365*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [g_ann, n_ann], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('overlay return %/yr')\n"
            "ax.set_title(f'Loses gross ({g_ann:+.0f}%/yr), worse net ({n_ann:+.0f}%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {g_ann:+.1f}%/yr (HAC t {gross['tstat']:+.2f}) -> net {n_ann:+.1f}%/yr (HAC t {net['tstat']:+.2f})\")"
        ),
        md(
            "> 💡 In plain words: there's nothing to charge costs against — shorting ETH on gas spikes "
            "*loses* before frictions (you fight the drift), and the crypto borrow makes it worse. "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real top signal "
            "(`top_signal_beta < 0`) of increasing strength and watch the forward slope-*t* dive "
            "negative — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "betas = [c[0] for c in ctrl]\n"
            "ts = [st.synthetic_mean_t(data, top_signal_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (top signal)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted top_signal_beta (more negative = stronger)')\n"
            "ax.set_ylabel('mean forward slope-t (25 seeds)')\n"
            "ax.set_title('Flat at the null, past -2 as the planted top signal grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'beta {b:+.3f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "The slope-*t* is ≈ 0 at the null and dives past −2 as the planted top signal grows — so the "
            "engine is a faithful detector. The real-tape reading is therefore simply **unavailable**: "
            "the apparatus is sound, but no reachable gas series exists to point it at. For the "
            "sentiment cousin, see [325 Crypto-Fear-Greed](../../325-crypto-fear-greed/); for other "
            "on-chain meters, [292 Bitcoin-Hashrate](../../292-bitcoin-hashrate/) and "
            "[295 Stablecoin-Supply](../../295-stablecoin-supply/)."
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
