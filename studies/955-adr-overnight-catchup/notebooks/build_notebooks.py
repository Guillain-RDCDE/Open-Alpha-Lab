"""Generate the two narrative notebooks for Study 955 (ADR Catch-Up).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md`` line for line; the only live
cells run the fast synthetic control, and they are labelled as synthetic wherever they
appear. Nothing synthetic ever sits under a real-tape banner.
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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — mirror of docs/results.md.
# 8 ADRs x (home index + FX), daily total-return ADR closes vs price-only home
# indices, 2004-01-05 -> 2026-06-30, one execution lag, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2004-01-05", end="2026-06-30", n_rows=41272, n_names=8, fp="a564d6b737e5",

    # 1. the pooled stale-price regression
    alpha_bps=3.63, t_alpha=3.22, b0=0.5761, t_b0=25.0, b1=0.0209, t_b1=1.15, n_obs=41264,

    # 1b. index leg vs FX leg
    all_h=0.680, all_t_h=32.3, all_hlag=0.0023, all_t_hlag=0.12,
    all_f=0.235, all_t_f=6.9, all_flag=0.0683, all_t_flag=2.79,
    jp_h=0.362, jp_t_h=14.6, jp_hlag=0.0620, jp_t_hlag=2.43,
    jp_f=-0.081, jp_t_f=-1.5, jp_flag=0.0717, jp_t_flag=1.99,
    uk_hlag=-0.0421, uk_t_hlag=-1.31, eu_hlag=-0.0208, eu_t_hlag=-0.87,

    # 2. the region cut
    jp_n=9650, jp_b0=0.290, jp_b1=0.0809, jp_t_b1=3.25, jp_gamma=0.0331, jp_t_gamma=1.43,
    jp_rho=-0.165,
    eu_n=10558, eu_b0=0.471, eu_b1=0.0123, eu_t_b1=0.47, eu_gamma=-0.0051, eu_t_gamma=-0.24,
    uk_n=21056, uk_b0=0.843, uk_b1=-0.0479, uk_t_b1=-1.80, uk_gamma=-0.0357, uk_t_gamma=-1.16,
    tm_b1=0.0936, tm_t_b1=3.08, sony_b1=0.0681, sony_t_b1=2.75,
    all_gamma=-0.0077, all_t_gamma=-0.41, all_rho=-0.050,

    # Japan block by block
    jp_blocks=[("2004-2009", 2550, 0.0806, 1.74, 0.0347, 0.79),
               ("2010-2014", 2160, 0.1119, 2.35, 0.0551, 1.10),
               ("2015-2020", 2590, 0.0540, 1.15, -0.0012, -0.03),
               ("2021-2026", 2344, 0.0658, 2.10, 0.0391, 1.24)],
    jp_early_b1=0.0935, jp_early_t=2.60, jp_late_b1=0.0598, jp_late_t=2.25,

    # 2b. is that Japan loading linear, or tail-carried? (the audit's section 5)
    jp_tail=[("full sample", 9650, 0.0809, 3.25),
             ("trim >99.5th pct", 9602, 0.0169, 0.67),
             ("trim >99th pct", 9554, 0.0307, 1.52),
             ("trim >97.5th pct", 9408, 0.0358, 1.75),
             ("trim >95th pct", 9168, 0.0418, 1.82),
             ("winsorize @99th", 9650, 0.0603, 2.62),
             ("winsorize @97.5th", 9650, 0.0567, 2.58)],
    syn_tail=[("full sample", 0.2510, 47.8), ("trim >99th pct", 0.2489, 45.4),
              ("trim >95th pct", 0.2511, 40.4)],
    jp_buckets=[(1, -0.0192, 0.00), (2, -0.0053, 4.81), (3, 0.0005, 6.38),
                (4, 0.0061, 1.23), (5, 0.0191, 1.60)],
    jp_q5q1=1.60,
    jp_consec_share=0.920, jp_consec_b1=0.0749, jp_consec_t=3.29,
    jp_boot_lo=0.0338, jp_boot_hi=0.1225, jp_boot_neg=0.0,

    # 3. residual and the discriminating regression
    fm_rank_bp=4.60, fm_rank_t=1.32, fm_raw=0.0214, fm_raw_t=2.22,
    pool_e=-0.0168, pool_e_t=-1.06,
    coef_a=-0.0230, t_a=-1.57, coef_x=0.0056, t_x=0.27,

    # 4. the costed books
    cu_gross=-0.92, cu_sharpe=-0.049, cu_t=-0.23, cu_net=-14.81, cu_be=-0.34,
    re_gross=2.55, re_sharpe=0.161, re_t=0.72, re_net=-12.01, re_be=0.89,
    rv_gross=5.28, rv_sharpe=0.326, rv_t=1.66, rv_net=-8.57, rv_be=1.94,
    jp_cu_gross=2.30, jp_cu_sharpe=0.090, jp_cu_t=0.41, jp_cu_net=-11.52, jp_cu_be=0.85,
    turnover=1.083,
    ci_cu_lo=-0.469, ci_cu_hi=0.396, ci_cu_neg=57.4,
    ci_re_lo=-0.290, ci_re_hi=0.600, ci_rv_lo=-0.080, ci_rv_hi=0.706,
    cost_grid=[(0.0, -1.16, -0.062), (1.0, -3.89, -0.206), (2.0, -6.62, -0.351),
               (5.0, -14.81, -0.785), (10.0, -28.46, -1.506), (25.0, -69.42, -3.641)],
    borrow_lo=-14.57, borrow_hi=-16.01,
    cu_dn_sharpe=-0.369, cu_lin_sharpe=0.010,

    # era cut on the whole panel
    era_e_n=20148, era_e_b1=0.0285, era_e_t=1.12, era_e_gamma=-0.0140,
    era_l_n=21108, era_l_b1=0.0103, era_l_t=0.44, era_l_gamma=0.0018,

    # cross-checks
    nvo_real_b1=0.0194, nvo_real_t=0.43, nvo_proxy_b1=0.0381, nvo_proxy_t=0.78,
    ew_ann=11.19, ew_sharpe=0.515, ew_t=2.67, ew_bil_sharpe=0.450, ew_bil_t=2.15,

    # synthetic control
    syn_b1=0.2510, syn_t_b1=47.8, syn_sharpe=7.54,
    syn_null_b1=0.0002, syn_null_sd=0.0040, syn_null_fire=0,
    syn_bounce_fm=-0.1535, syn_bounce_fm_t=-20.6,
    syn_bounce_b1=0.0060, syn_bounce_t=0.96, syn_bounce_xa=0.1086, syn_bounce_xa_t=13.3,
)

SYNTH_NOTE = (
    "*The cell below is the **synthetic** control — a simulated panel with a known planted "
    "answer, run live so you can see the machinery is unbiased. It is not the real tape and "
    "carries no part of the verdict.*"
)

SYNTH_IMPORT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "from adr_catchup import data, strategy as st\n"
)


HEADER = f"""# Study 955 — ADR Catch-Up 🌏

**When Tokyo closed thirteen hours ago, does the New York listing still owe it a move?**

Toyota trades in Tokyo and, as an ADR, in New York. Tokyo shuts at 02:00 New York time;
London and Frankfurt at 11:30, *inside* the US session. So for part of the world the ADR
gets a whole American day to price news its home market already priced and went home on.
The trading-desk folklore: watch the home index and the currency overnight, and you know
which way the ADR is going.

We test that on **eight** cross-listed ADRs (TM, SONY, SAP, NVO, SHEL, BP, HSBC, RIO) against
their home indices and currencies, **{R['start']} → {R['end']}** ({R['n_rows']:,} name-days),
with one execution lag and real costs.

*Real numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`);
the live cells run the fast synthetic control and are labelled as such. As-of 2026-06-30.*

> ⚠️ **The honest limit, up front.** Real catch-up is an *intraday* effect. A daily-close
> tape sees only close-to-close, so what we can measure is the residue that survives to the
> ADR's own close — a weaker question than "what happens in the US session". Everything
> here answers the weaker question, and says so.
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea\n\n"
           "The ADR and the home share are the same company. If Tokyo rallied 2% overnight "
           "and Toyota's New York line has not moved 2%, somebody is about to be paid. The "
           "only question is *when* — and whether anything is still owed by the time New "
           "York shuts."),
        code(
            "R = dict(b0=%r, b1=%r, t_b1=%r, jp_b1=%r, jp_t_b1=%r, uk_b1=%r, uk_t_b1=%r)\n"
            "print('Across all eight ADRs, how much of YESTERDAY\\'s home move is still owed?')\n"
            "print('  loading on yesterday: %%+.4f   (t = %%+.2f)  -> nothing'\n"
            "      %% (R['b1'], R['t_b1']))\n"
            "print()\n"
            "print('Split by where the home market is:')\n"
            "print('  Japan  (Tokyo shuts 02:00 New York time) : %%+.4f  (t = %%+.2f)'\n"
            "      %% (R['jp_b1'], R['jp_t_b1']))\n"
            "print('  UK     (London shuts 11:30, mid-session) : %%+.4f  (t = %%+.2f)'\n"
            "      %% (R['uk_b1'], R['uk_t_b1']))"
            % (R["b0"], R["b1"], R["t_b1"], R["jp_b1"], R["jp_t_b1"], R["uk_b1"], R["uk_t_b1"])
        ),
        md("## 2. The clock predicts who has a lag\n\n"
           f"Pooled over all eight names, yesterday's home move is worth **{R['b1']:+.4f}** "
           f"with *t* = {R['t_b1']:+.2f} — nothing at all. But split them by *when their home "
           "market closes* and a clean pattern falls out.\n\n"
           f"- **Japan** — Tokyo shuts thirteen hours before the ADR does. Yesterday's move "
           f"still loads **{R['jp_b1']:+.4f}** (*t* = **{R['jp_t_b1']:+.2f}**). Both names on "
           f"their own: Toyota *t* = {R['tm_t_b1']:+.2f}, Sony *t* = {R['sony_t_b1']:+.2f}.\n"
           f"- **UK** — London shuts at 11:30 New York time, *inside* the US session, so most "
           f"of its move is already in the ADR's own day. Loading: {R['uk_b1']:+.4f} "
           f"(*t* = {R['uk_t_b1']:+.2f}) — nothing left over, exactly as the mechanism says.\n\n"
           "That is not a fishing expedition finding a lucky subgroup. It is the theory's own "
           "prediction — a lag exists where the markets do not overlap — coming true, and "
           "failing to appear where they do."),
        md("## 3. And Japan's lag is not a fluke of one decade\n\n"
           "| Block | loading on yesterday | *t* |\n|---|--:|--:|\n"
           + "\n".join(f"| {tag} | {b:+.4f} | {t:+.2f} |" for tag, _, b, t, _, _ in R["jp_blocks"])
           + f"\n| **2004–2014 half** | **{R['jp_early_b1']:+.4f}** | **{R['jp_early_t']:+.2f}** |"
           + f"\n| **2015–2026 half** | **{R['jp_late_b1']:+.4f}** | **{R['jp_late_t']:+.2f}** |\n\n"
           "Positive in every block, both halves, and still alive in the last five years. "
           "So far this looks like a finding. Section 3½ is where it stops looking like one."),
        md("## 3½. Who actually owns that number?\n\n"
           "A *loading* should be boring and linear: if Toyota's New York line repays a "
           "fixed slice of whatever Tokyo did last night, then throwing away the wildest "
           "nights should barely move the estimate. Here is what happens when we do.\n\n"
           "| what we removed | Japan's loading | *t* |\n|---|--:|--:|\n"
           + "\n".join(f"| {tag} | {b:+.4f} | {t:+.2f} |" for tag, _n, b, t in R["jp_tail"])
           + "\n\nDeleting **48 rows** — half of one per cent of the sample — takes the "
           "study's only significant number from "
           f"{R['jp_tail'][0][2]:+.4f} (*t* = {R['jp_tail'][0][3]:+.2f}) to "
           f"{R['jp_tail'][1][2]:+.4f} (*t* = {R['jp_tail'][1][3]:+.2f}). For comparison, the "
           "identical knife on a *simulated* market where a catch-up lag really was planted, "
           "and planted linearly, moves the estimate by about one per cent (next section's "
           "live cell prints it). So the honest reading is: on the 0.5% of nights when Tokyo "
           "moved enormously there is something; on an ordinary night there is nothing.\n\n"
           "The plainest version of the same question — sort the days by what Tokyo did last "
           "night, and see what the ADR paid next:\n\n"
           "| yesterday in Tokyo | what the ADR paid next |\n|---|--:|\n"
           + "\n".join(f"| {'Q%d' % b} ({x:+.2%}) | {a:+.2f} bp |" for b, x, a in R["jp_buckets"])
           + f"\n\n**Best minus worst: {R['jp_q5q1']:+.2f} basis points**, and not even in a "
           "straight line. There is no gradient to bet on."),
        md("## 4. So why isn't this a trade?\n\n"
           "Three reasons, and each one is fatal on its own.\n\n"
           f"**(a) That coefficient is not the one you can bet on.** It is measured while "
           f"*also* knowing tomorrow's home move — which you don't. Strip that out and the "
           f"bettable number in Japan is **{R['jp_gamma']:+.4f}** (*t* = {R['jp_t_gamma']:+.2f}), "
           f"about a third the size. The reason is dull and arithmetic: the Nikkei-in-dollars "
           f"is itself negatively autocorrelated ({R['jp_rho']:+.3f}), which inflates the "
           "first number relative to the second.\n\n"
           f"**(b) It is tiny against the noise, and it is not there most nights.** Read at "
           "face value it is an 8-basis-point payback on a 1% Tokyo move, on a stock that "
           "moves ~30% a year on its own news — and section 3½ showed most of even that "
           "comes from the wildest half-per-cent of nights. You would need thousands of bets "
           "to see it, and you get one a day.\n\n"
           f"**(c) The costs eat it before you start.** The book has to be re-set every "
           f"session — it turns over **{R['turnover']*100:.0f}% of the portfolio per day**. "
           f"It breaks even at **{R['jp_cu_be']:.2f} basis points** of one-way cost in Japan "
           f"and at **{R['cu_be']:+.2f} bps** across all eight, which is to say it never "
           "breaks even at all."),
        code(
            "R = dict(cu_gross=%r, cu_sharpe=%r, cu_net=%r, cu_be=%r,\n"
            "         jp_cu_gross=%r, jp_cu_sharpe=%r, jp_cu_be=%r,\n"
            "         rv_gross=%r, rv_sharpe=%r)\n"
            "print('Buy the ADRs whose home market rose, sell the ones that fell:')\n"
            "print('  all eight, before costs : %%+6.2f%%%% a year   Sharpe %%+.3f'\n"
            "      %% (R['cu_gross'], R['cu_sharpe']))\n"
            "print('  Japan only, before costs: %%+6.2f%%%% a year   Sharpe %%+.3f'\n"
            "      %% (R['jp_cu_gross'], R['jp_cu_sharpe']))\n"
            "print('  all eight, after 5bp    : %%+6.2f%%%% a year'  %% R['cu_net'])\n"
            "print()\n"
            "print('A control book that never looks at the home market at all,')\n"
            "print('and just fades whatever the ADR did yesterday:')\n"
            "print('  before costs            : %%+6.2f%%%% a year   Sharpe %%+.3f  <- better!'\n"
            "      %% (R['rv_gross'], R['rv_sharpe']))"
            % (R["cu_gross"], R["cu_sharpe"], R["cu_net"], R["cu_be"],
               R["jp_cu_gross"], R["jp_cu_sharpe"], R["jp_cu_be"],
               R["rv_gross"], R["rv_sharpe"])
        ),
        md("## 5. The line that settles it\n\n"
           f"A dumb control book — fade whatever the ADR did yesterday, never open the home "
           f"tape at all — earns **{R['rv_gross']:+.2f}%/yr** gross (Sharpe "
           f"{R['rv_sharpe']:+.3f}), while the home-informed catch-up book earns "
           f"**{R['cu_gross']:+.2f}%/yr** (Sharpe {R['cu_sharpe']:+.3f}). In tradable form, "
           "the foreign market's information is a *drag*. Every one of those Sharpes has a "
           "bootstrap confidence interval straddling zero anyway."),
        md("## 6. Live check — the machinery is not broken\n\n" + SYNTH_NOTE),
        code(
            SYNTH_IMPORT +
            "syn = data.synthetic_panel(signal_strength=1.0, seed=955)[0]\n"
            "planted = st.synthetic_detect(syn)\n"
            "null    = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=955)[0])\n"
            "print('SYNTHETIC world where 40 pct of the home move really does arrive a day late:')\n"
            "print('   detected lag %+.4f (t %+.1f)   book Sharpe %+.2f  -> the detector fires'\n"
            "      % (planted['beta_lag'], planted['t_lag'], planted['gross_sharpe']))\n"
            "print('SYNTHETIC world where the ADR prices everything the same day:')\n"
            "print('   detected lag %+.4f (t %+.2f)  -> and stays quiet'\n"
            "      % (null['beta_lag'], null['t_lag']))\n"
            "print()\n"
            "print('And the calibration for section 3.5 - the same knife, on the SYNTHETIC')\n"
            "print('world where the lag is real AND linear:')\n"
            "tail = st.tail_sensitivity(syn, quantiles=(0.99, 0.95))\n"
            "for _, r in tail.iterrows():\n"
            "    label = 'full sample' if r['mode'] == 'full' else '%s > %.3f' % (r['mode'], r['q'])\n"
            "    print('   %-18s lag %+.4f (t %+.1f)' % (label, r['beta_lag'], r['t_lag']))\n"
            "print('   -> a genuine linear lag does not care. Japan lost 79 pct of its size.')"
        ),
        md("> 🔬 **For the quants.** The same generator also plants pure bid-ask bounce with "
           "*zero* stale information, and that fires the residual-reversal rule hard while "
           "leaving the lagged-home-move test silent. That is why this study never lets a "
           "negative residual slope stand as evidence of catch-up — see notebook 02."),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The blanket folklore fails: pooled over eight ADRs, "
           f"yesterday's home move is worth {R['b1']:+.4f} (*t* = {R['t_b1']:+.2f}), and the "
           f"little that is there comes from the *currency*, which never closes, rather than "
           f"the index, which does. Japan — the one home market that shuts long before New "
           f"York — does carry **{R['jp_b1']:+.4f}** (*t* = **{R['jp_t_b1']:+.2f}**) where the "
           f"clock says it should, and the UK, closing mid-session, carries nothing. But that "
           f"number is owned by the tail: drop the 48 wildest nights and it falls to "
           f"{R['jp_tail'][1][2]:+.4f} (*t* = {R['jp_tail'][1][3]:+.2f}), where a simulated "
           f"*linear* lag would not have moved at all, and sorting the days by yesterday's "
           f"Tokyo move pays {R['jp_q5q1']:+.2f} bp best-minus-worst with no gradient. A "
           f"whiff on extreme nights, on two survivorship-picked names — not a result.\n"
           f"- **Tradability — Mirage.** Negative gross across all eight, break-even at "
           f"{R['jp_cu_be']:.2f} bps in Japan, 108% daily turnover, and beaten by a control "
           f"that ignores the home market entirely. Not a trade."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 955 — ADR Catch-Up — the teardown\n\n"
           "The stale-price HAC regression, the index-leg/FX-leg split, the region cut by "
           "closing time, the trim/winsorize knife that decides whether the one surviving "
           "coefficient is a loading or a tail, `b1` versus the bettable γ and the "
           "autocorrelation that separates "
           "them, the Fama-MacBeth residual test and its bounce confound, the costed books "
           "with breakeven costs, and the live synthetic control. Every real number is frozen "
           "from `docs/results.md` (Fingerprint `%s`), sample %s → %s, %d name-days, one "
           "execution lag. ADR closes total-return; home indices price-only."
           % (R["fp"], R["start"], R["end"], R["n_rows"])),
        code("R = %r" % (R,)),
        md("> 💡 **In plain words.** `b1` is the answer to: *knowing what the home market did "
           "today, does what it did yesterday still tell me anything about the ADR today?* If "
           "the ADR closed at fair value, the answer is no."),
        md("## 1. The pooled stale-price regression\n\n"
           "`a_t = α + b0·x_t + b1·x_{t-1}`, HAC (Newey-West) standard errors, rows pooled "
           "across names and sorted by date so the Bartlett kernel absorbs both serial and "
           "same-day cross-sectional correlation. `b0` is the stock's beta to its home "
           "*index*, not a catch-up ratio — it has no reason to be 1."),
        code(
            "print(f\"alpha {R['alpha_bps']:+.2f} bp/day (t={R['t_alpha']:+.2f})\")\n"
            "print(f\"b0  same-day home dollar move : {R['b0']:+.4f}  (t={R['t_b0']:+.1f})\")\n"
            "print(f\"b1  LAGGED home dollar move   : {R['b1']:+.4f}  (t={R['t_b1']:+.2f})   <- the test\")\n"
            "print(f\"n = {R['n_obs']:,}\")"
        ),
        md("## 2. Where does the pooled lag actually live — the index or the currency?\n\n"
           "`a_t = α + c0·h_t + c1·h_{t-1} + d0·f_t + d1·f_{t-1}`. A stale-*market* story has "
           "to show up in the index leg. FX trades around the clock and Yahoo stamps its close "
           "in New York hours, so a lagged FX loading is a snapshot artefact, not unpaid "
           "information."),
        code(
            "print(f\"ALL   h_lag {R['all_hlag']:+.4f} (t={R['all_t_hlag']:+.2f})   \"\n"
            "      f\"f_lag {R['all_flag']:+.4f} (t={R['all_t_flag']:+.2f})  <- the pooled lag is ALL currency\")\n"
            "print(f\"Japan h_lag {R['jp_hlag']:+.4f} (t={R['jp_t_hlag']:+.2f})   \"\n"
            "      f\"f_lag {R['jp_flag']:+.4f} (t={R['jp_t_flag']:+.2f})  <- a genuine stale INDEX loading\")\n"
            "print(f\"UK    h_lag {R['uk_hlag']:+.4f} (t={R['uk_t_hlag']:+.2f})\")\n"
            "print(f\"EU    h_lag {R['eu_hlag']:+.4f} (t={R['eu_t_hlag']:+.2f})\")"
        ),
        md("## 3. The region cut — the discriminating test\n\n"
           "Tokyo closes 02:00 ET, thirteen hours before the ADR. London and Frankfurt close "
           "11:30 ET, *inside* the US session. This is not a subgroup hunt: the closing clock "
           "is the mechanism, so the cut is the hypothesis, and the UK null is a confirmation "
           "rather than a failure."),
        code(
            "for tag, n, b0, b1, tb1, g, tg in [\n"
            "    ('Japan ', R['jp_n'], R['jp_b0'], R['jp_b1'], R['jp_t_b1'], R['jp_gamma'], R['jp_t_gamma']),\n"
            "    ('Europe', R['eu_n'], R['eu_b0'], R['eu_b1'], R['eu_t_b1'], R['eu_gamma'], R['eu_t_gamma']),\n"
            "    ('UK    ', R['uk_n'], R['uk_b0'], R['uk_b1'], R['uk_t_b1'], R['uk_gamma'], R['uk_t_gamma'])]:\n"
            "    print(f'{tag} n={n:6,}  b0={b0:+.3f}  b1={b1:+.4f} (t={tb1:+.2f})  gamma={g:+.4f} (t={tg:+.2f})')\n"
            "print()\n"
            "print(f\"per name: TM b1={R['tm_b1']:+.4f} (t={R['tm_t_b1']:+.2f}), \"\n"
            "      f\"SONY b1={R['sony_b1']:+.4f} (t={R['sony_t_b1']:+.2f})\")"
        ),
        md("### Japan block by block"),
        code(
            "for tag, n, b1, tb1, g, tg in R['jp_blocks']:\n"
            "    print(f'{tag} n={n:5d}: b1={b1:+.4f} (t={tb1:+.2f})   gamma={g:+.4f} (t={tg:+.2f})')\n"
            "print(f\"2004-2014 half: b1={R['jp_early_b1']:+.4f} (t={R['jp_early_t']:+.2f})\")\n"
            "print(f\"2015-2026 half: b1={R['jp_late_b1']:+.4f} (t={R['jp_late_t']:+.2f})\")"
        ),
        md("## 3½. Is that Japan loading linear, or does the tail own it?\n\n"
           "The question the HAC *t* cannot answer. A HAC standard error corrects for "
           "dependence, not for *leverage*: a slope is an average, and a handful of enormous "
           "regressor values can be its sole author. Under linearity, selecting on a "
           "regressor does not bias OLS — so if the loading is real and uniform, trimming the "
           "biggest `|x_lag|` rows should barely move it, and winsorizing (capping, keeping "
           "every row) should move it not at all. Both knives, plus the calibration on a "
           "synthetic panel where the planted lag *is* linear:"),
        code(
            "print('Japan:')\n"
            "for tag, n, b1, t in R['jp_tail']:\n"
            "    print(f'  {tag:20s} n={n:6,}  b1={b1:+.4f} (t={t:+.2f})')\n"
            "print()\n"
            "print('SYNTHETIC calibration - a genuinely LINEAR planted lag under the same knife:')\n"
            "for tag, b1, t in R['syn_tail']:\n"
            "    print(f'  {tag:20s}          b1={b1:+.4f} (t={t:+.1f})')\n"
            "drop = 1 - R['jp_tail'][1][2] / R['jp_tail'][0][2]\n"
            "syn_drop = 1 - R['syn_tail'][1][1] / R['syn_tail'][0][1]\n"
            "print()\n"
            "print(f'0.5% of rows removed: Japan loses {drop:.0%} of its coefficient, '\n"
            "      f'the linear plant loses {syn_drop:.0%} at a 1% trim.')"
        ),
        md("### The non-parametric twin: sort by yesterday's home move\n\n"
           "No regression, no leverage: bucket every Japanese name-day by `x_lag` and read "
           "what the ADR paid next. (Rows inside a bucket share dates and home markets, so a "
           "*t* here would overstate the evidence — read the basis points.)"),
        code(
            "for b, x, a in R['jp_buckets']:\n"
            "    print(f'  Q{b}  x_lag {x:+.4f}  ->  ADR next {a:+6.2f} bp')\n"
            "print(f\"\\n  Q5 - Q1 = {R['jp_q5q1']:+.2f} bp, and not monotone.\")\n"
            "print()\n"
            "print(f\"lag is strictly yesterday ({R['jp_consec_share']:.1%} of rows): \"\n"
            "      f\"b1={R['jp_consec_b1']:+.4f} (t={R['jp_consec_t']:+.2f}) - unchanged\")\n"
            "print(f\"block bootstrap over dates: b1 95% CI \"\n"
            "      f\"[{R['jp_boot_lo']:+.4f}, {R['jp_boot_hi']:+.4f}], \"\n"
            "      f\"{R['jp_boot_neg']:.1f}% of draws <= 0\")\n"
            "print()\n"
            "print('The bootstrap says it is not a sampling fluke; the trim says it is a')\n"
            "print('body-vs-tail fluke. Those are different objections and both are reported.')"
        ),
        md("> 💡 **In plain words.** Forty-eight nights out of nine thousand six hundred and "
           "fifty carry the whole result. That is not a reason to call it fake — something "
           "does happen after a huge Tokyo session — but it is a decisive reason not to call "
           "it an established loading, and it is why the Signal stamp on this study is Weak "
           "and not Mixed."),
        md("## 4. `b1` is not the tradable coefficient — and the gap is arithmetic\n\n"
           "`b1` conditions on `x_{t+1}`, which is unknown at the trade. The bettable "
           "coefficient is the univariate γ in `a_{t+1} = α + γ·x_t`. Because the home dollar "
           "move is itself negatively autocorrelated, the two-regressor estimator inflates "
           "`b1` — part of the lagged loading is the estimator undoing the same-day loading "
           "applied to a mean-reverting regressor."),
        code(
            "print(f\"ALL   b1={R['b1']:+.4f} (t={R['t_b1']:+.2f})  ->  gamma={R['all_gamma']:+.4f} \"\n"
            "      f\"(t={R['all_t_gamma']:+.2f})   rho1(x)={R['all_rho']:+.3f}\")\n"
            "print(f\"Japan b1={R['jp_b1']:+.4f} (t={R['jp_t_b1']:+.2f})  ->  gamma={R['jp_gamma']:+.4f} \"\n"
            "      f\"(t={R['jp_t_gamma']:+.2f})   rho1(x)={R['jp_rho']:+.3f}\")\n"
            "print(f\"\\ninflation factor in Japan: {R['jp_b1']/R['jp_gamma']:.2f}x\")\n"
            "print('gamma clears |t| = 2 nowhere.')"
        ),
        md("> 💡 **In plain words.** Quoting the big coefficient would have made the prize look "
           "about two and a half times larger than anything you could actually bet on. That "
           "single substitution is the difference between a headline and an honest result."),
        md("## 5. The residual test, and why it carries no weight\n\n"
           "`e_t = a_t − β_t·x_t`, with `β_t` a 252-day rolling beta on data through `t-1` "
           "only. Three ways of testing whether `e_t` predicts `a_{t+1}` disagree *in sign* — "
           "which is itself the finding. The raw cross-sectional slope divides by the day's "
           "dispersion, near zero on a quiet eight-name cross-section; the pooled slope "
           "ignores cross-correlation and overstates its own *t*. The rank-standardised "
           "Fama-MacBeth is the one to read."),
        code(
            "print(f\"Fama-MacBeth, rank-standardised : {R['fm_rank_bp']:+.2f} bp  (t={R['fm_rank_t']:+.2f})  <- read this one\")\n"
            "print(f\"Fama-MacBeth, raw slope         : {R['fm_raw']:+.4f}      (t={R['fm_raw_t']:+.2f})  <- unstable\")\n"
            "print(f\"pooled slope                    : {R['pool_e']:+.4f}     (t={R['pool_e_t']:+.2f})  <- over-optimistic\")"
        ),
        md("### And the discriminating regression: `a_{t+1} = α + φ·a_t + γ·x_t`\n\n"
           "Any residual contains `a_t`, so a negative slope on it may just be Roll bounce and "
           "Nagel liquidity-provision reversal. Put the ADR's own move in the regression and "
           "ask what the home tape adds."),
        code(
            "print(f\"own move a_t : {R['coef_a']:+.4f}  (t={R['t_a']:+.2f})   <- plain one-day reversal\")\n"
            "print(f\"home    x_t  : {R['coef_x']:+.4f}  (t={R['t_x']:+.2f})   <- the home tape adds nothing\")\n"
            "print()\n"
            "print('And this test is BIASED IN FAVOUR of catch-up: on a synthetic panel with')\n"
            "print(f\"pure bounce and ZERO stale information it returns {R['syn_bounce_xa']:+.4f} \"\n"
            "      f\"(t={R['syn_bounce_xa_t']:+.1f}).\")"
        ),
        md("## 6. The costed books\n\n"
           "Gross exposure exactly 1, weights formed at the close of `t` and held over `t+1` "
           "(**one** execution lag), one-way cost × NAV on turnover, borrow accrued daily on "
           "the short leg. Being self-financing at gross 1, the book's return *is* its "
           "excess-of-cash return, so the Sharpe race is already excess-vs-excess. The third "
           "book is the control: it fades the ADR's own move and never opens the home tape."),
        code(
            "rows = [('catch-up  +sign(x)', R['cu_gross'], R['cu_sharpe'], R['cu_t'], R['cu_net'], R['cu_be']),\n"
            "        ('residual  -sign(e)', R['re_gross'], R['re_sharpe'], R['re_t'], R['re_net'], R['re_be']),\n"
            "        ('CONTROL   -sign(a)', R['rv_gross'], R['rv_sharpe'], R['rv_t'], R['rv_net'], R['rv_be']),\n"
            "        ('catch-up, Japan   ', R['jp_cu_gross'], R['jp_cu_sharpe'], R['jp_cu_t'], R['jp_cu_net'], R['jp_cu_be'])]\n"
            "for tag, g, s, t, n, be in rows:\n"
            "    print(f'{tag}: gross {g:+6.2f}%/yr  Sharpe {s:+.3f} (t={t:+.2f})  '\n"
            "          f'net(5bp/50bp) {n:+7.2f}%/yr  breakeven {be:+.2f} bps')\n"
            "print(f\"\\nturnover {R['turnover']*100:.0f}% of NAV per day\")\n"
            "print('the CONTROL, which uses no home data at all, has the best gross Sharpe of the four.')"
        ),
        code(
            "print('bootstrap Sharpe CIs (2000 draws, 21-day blocks) — all span zero:')\n"
            "print(f\"  catch-up: [{R['ci_cu_lo']:+.3f}, {R['ci_cu_hi']:+.3f}]  ({R['ci_cu_neg']:.1f}% of draws < 0)\")\n"
            "print(f\"  residual: [{R['ci_re_lo']:+.3f}, {R['ci_re_hi']:+.3f}]\")\n"
            "print(f\"  control : [{R['ci_rv_lo']:+.3f}, {R['ci_rv_hi']:+.3f}]\")\n"
            "print()\n"
            "print('cost sweep, catch-up book (borrow held at 50 bps):')\n"
            "for c, ann, sh in R['cost_grid']:\n"
            "    print(f'  {c:5.1f} bps: {ann:+7.2f}%/yr  Sharpe {sh:+.3f}')\n"
            "print(f\"\\nborrow 0 -> 300 bps moves the net from {R['borrow_lo']:+.2f}%/yr to \"\n"
            "      f\"{R['borrow_hi']:+.2f}%/yr — the ASSUMPTION does not drive the verdict.\")\n"
            "print(f\"variants: dollar-neutral Sharpe {R['cu_dn_sharpe']:+.3f}, \"\n"
            "      f\"signal-weighted {R['cu_lin_sharpe']:+.3f} — no rescue.\")"
        ),
        md("## 7. Cross-checks and the era cut"),
        code(
            "print(f\"era cut, whole panel: 2004-2014 b1={R['era_e_b1']:+.4f} (t={R['era_e_t']:+.2f}), \"\n"
            "      f\"2015-2026 b1={R['era_l_b1']:+.4f} (t={R['era_l_t']:+.2f})\")\n"
            "print(f\"NVO home-index PROXY check (2016-12 on): real OMXC25/DKK b1={R['nvo_real_b1']:+.4f} \"\n"
            "      f\"(t={R['nvo_real_t']:+.2f}) vs GDAXI/EUR proxy {R['nvo_proxy_b1']:+.4f} \"\n"
            "      f\"(t={R['nvo_proxy_t']:+.2f}) — same null\")\n"
            "print(f\"context: EW ADR basket excess-of-^IRX {R['ew_ann']:+.2f}%/yr Sharpe \"\n"
            "      f\"{R['ew_sharpe']:+.3f} (t={R['ew_t']:+.2f}); on BIL cash, Sharpe \"\n"
            "      f\"{R['ew_bil_sharpe']:+.3f} (t={R['ew_bil_t']:+.2f})\")"
        ),
        md("## 8. Live synthetic control — the machinery, and the confound\n\n" + SYNTH_NOTE +
           "\n\nTwo things are planted independently: a genuine catch-up lag (a share of the "
           "home loading arrives a day late) and pure bid-ask bounce (a transient pricing "
           "error that unwinds tomorrow). The point is that they are *not* interchangeable."),
        code(
            SYNTH_IMPORT +
            "import numpy as np\n"
            "pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=955)[0])\n"
            "print(f\"planted lag : b1 {pl['beta_lag']:+.4f} (t {pl['t_lag']:+.1f})  \"\n"
            "      f\"x|a {pl['coef_x_ctrl']:+.4f} (t {pl['t_x_ctrl']:+.1f})  book Sharpe {pl['gross_sharpe']:+.2f}\")\n"
            "nl = [st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=955+s)[0])\n"
            "      for s in range(5)]\n"
            "b = np.array([d['beta_lag'] for d in nl]); tb = np.array([d['t_lag'] for d in nl])\n"
            "print(f\"null x5     : b1 mean {b.mean():+.4f} (sd {b.std(ddof=1):.4f}), |t|>=2 on {(abs(tb)>=2).sum()}/5\")\n"
            "bo = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, bounce_vol=0.006,\n"
            "                                             seed=955)[0])\n"
            "print(f\"pure BOUNCE, zero catch-up: residual rule fires (FM slope {bo['fm_slope_e']:+.4f}, \"\n"
            "      f\"t {bo['fm_t_e']:+.1f}) but b1 stays silent ({bo['beta_lag']:+.4f}, t {bo['t_lag']:+.2f})\")"
        ),
        md("> 💡 **In plain words.** The last line is why the residual test never carries the "
           "verdict: a market with *no* stale information at all, only a jumpy bid-ask spread, "
           "makes it look like ADRs are catching up. The lagged-home-move test is immune to "
           "that, because yesterday's foreign index is independent of today's pricing error."),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Pooled over eight ADRs the lagged home loading is "
           f"{R['b1']:+.4f} (HAC *t* = {R['t_b1']:+.2f}), and decomposed it is entirely the FX "
           f"leg ({R['all_flag']:+.4f}, *t* = {R['all_t_flag']:+.2f}) against a dead index leg "
           f"({R['all_hlag']:+.4f}, *t* = {R['all_t_hlag']:+.2f}) — a snapshot artefact in a "
           f"market that never closes. Japan is the exception the clock predicts: "
           f"b1 = **{R['jp_b1']:+.4f}** (*t* = **{R['jp_t_b1']:+.2f}**), index leg "
           f"{R['jp_hlag']:+.4f} (*t* = {R['jp_t_hlag']:+.2f}), both names alone "
           f"(*t* = {R['tm_t_b1']:+.2f} / {R['sony_t_b1']:+.2f}), positive in all four blocks "
           f"and both halves ({R['jp_early_t']:+.2f} / {R['jp_late_t']:+.2f}) — and the UK, "
           f"whose home markets close mid-session, is {R['uk_b1']:+.4f} "
           f"(*t* = {R['uk_t_b1']:+.2f}), the mechanism's own prediction confirmed. What stops "
           f"that being a result (§3½): deleting the {R['jp_tail'][0][1] - R['jp_tail'][1][1]} "
           f"largest lagged home moves — 0.5% of the rows — takes it to "
           f"{R['jp_tail'][1][2]:+.4f} (*t* = {R['jp_tail'][1][3]:+.2f}), where the identical "
           f"trim moves a synthetic *linear* planted lag by 1%; winsorized it is "
           f"{R['jp_tail'][5][2]:+.4f} (*t* = {R['jp_tail'][5][3]:+.2f}), and the bucket sort "
           f"pays {R['jp_q5q1']:+.2f} bp Q5 − Q1 with no gradient. On top of that the bettable "
           f"γ is {R['jp_gamma']:+.4f} (*t* = {R['jp_t_gamma']:+.2f}) in Japan and "
           f"{R['all_gamma']:+.4f} overall, clearing |*t*| = 2 nowhere; the home tape "
           f"adds {R['coef_x']:+.4f} (*t* = {R['t_x']:+.2f}) once the ADR's own move is "
           f"controlled for, against a test that bounce biases *in its favour*; and eight "
           f"surviving mega-caps are a survivor-picked universe.\n"
           f"- **Tradability — Mirage.** Catch-up book {R['cu_gross']:+.2f}%/yr gross "
           f"(Sharpe {R['cu_sharpe']:+.3f}, *t* = {R['cu_t']:+.2f}) with a **negative** "
           f"breakeven cost; Japan-only breaks even at {R['jp_cu_be']:.2f} bps one-way; "
           f"{R['turnover']*100:.0f}% daily turnover turns 5 bps into "
           f"{R['cu_net']:+.2f}%/yr. Every bootstrap CI spans zero, no weighting variant "
           f"rescues it, and the no-home-data control beats it gross "
           f"({R['rv_sharpe']:+.3f} vs {R['cu_sharpe']:+.3f}). Statistically visible, "
           f"economically buried."),
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
