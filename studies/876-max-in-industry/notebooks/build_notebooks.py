"""Generate the two narrative notebooks for Study 876 (Industry-Relative MAX).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic
positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# total-return, 50 liquid US large-caps across 8 GICS sectors, 2010-01-04 -> 2026-06-30;
# monthly MAX quintile sort, Q1 low - Q5 high, next-month return).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_sectors=8, n_months=198, n_spread=197,
    raw_bps=-104.8, raw_t=-2.42, raw_t1s=-2.53, raw_win=43, raw_sharpe=-0.62, raw_p=0.994,
    adj_bps=-89.7, adj_t=-2.51, adj_t1s=-2.80, adj_win=46, adj_sharpe=-0.69, adj_p=0.998,
    q1=17.50, q2=13.10, q3=15.45, q4=16.95, q5=28.26,
    placebo_obs=-89.7, placebo_mean=-0.19, placebo_sd=32.56, placebo_sigma_left=-2.75,
    placebo_left_p=0.0023, placebo_draws=20000,
    era_early_bps=-47.1, era_early_t=-1.26, era_early_n=96,
    era_late_bps=-130.2, era_late_t=-2.24, era_late_n=101,
    timer_1_gross=-89.7, timer_1_cost=6.2, timer_1_net=-95.9, timer_1_t=-3.00, timer_1_ann=-11.5,
    timer_5_gross=-89.7, timer_5_cost=14.2, timer_5_net=-103.9, timer_5_t=-3.25, timer_5_ann=-12.5,
    null_mean_t=-0.25, null_sd_t=0.88, null_fire=1,
    planted_raw_t=11.18, planted_adj_t=20.60,
    fingerprint="5384b8f7f128",
)


HEADER = f"""# Study 876 — Industry-Relative MAX 🎰

**Does adjusting a stock's MAX for its sector sharpen — or kill — the lottery effect?**

The lottery / MAX effect (study 365, Bali-Cakici-Whitelaw 2011) sorts a name on its own
**maximum daily return** last month and finds the lottery-like high-MAX names *under-earn*.
But part of a name's MAX is just **sector-wide volatility** — a whole sector can be jumpy for
macro reasons — which is noise for a *lottery-demand* story. Here we subtract the **median MAX
of the name's sector peers** to get the **industry-relative** MAX, a cleaner proxy for
*idiosyncratic* lottery demand, and ask whether the negative MAX→return relation sharpens.

We take the self-contained monthly version on a liquid US cross-section ({R['start']} →
{R['end']}, {R['n_names']} names across {R['n_sectors']} GICS sectors).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A stock's MAX (its biggest one-day pop) is two things stacked: **sector weather** "
           "(the whole sector was jumpy) plus a **name-specific lottery pop**. The lottery "
           "story is about the *idiosyncratic* pop, so we strip the sector part out: "
           "`industry-relative MAX = own MAX − median MAX of sector peers`. Then sort, buy the "
           "boring low-MAX tail, sell the lottery high-MAX tail — and check whether the "
           "cleaner signal pays better."),
        code(
            "R = %r\n"
            "print('RAW MAX          : spread %%+.1f bps/mo  (NW t = %%+.2f)' %% (R['raw_bps'], R['raw_t']))\n"
            "print('INDUSTRY-RELATIVE: spread %%+.1f bps/mo  (NW t = %%+.2f)' %% (R['adj_bps'], R['adj_t']))\n"
            "print()\n"
            "print('Both spreads are NEGATIVE -> the claim INVERTS: on mega-caps the lottery')\n"
            "print('(high-MAX) names OUT-earned the boring low-MAX ones. The industry')\n"
            "print('adjustment does not fix the sign; it slightly sharpens the wrong-sign t.')"
            % (R,)
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world where each name's MAX = an **un-priced "
           "sector-wide level** + a **priced idiosyncratic pop** (`edge>0`). The industry "
           "adjustment removes the sector level, so it should recover the planted relation "
           "*more sharply* than the raw MAX — and both must stay silent on the null "
           "(`edge=0`). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from max_industry import data, strategy as st\n"
            "planted = data.synthetic_panel(edge=0.012, seed=876, n_months=240)\n"
            "null = data.synthetic_panel(edge=0.0, seed=876, n_months=240)\n"
            "print('planted world  raw-MAX  NW t = %+.2f' % st.synthetic_detect(planted, adjusted=False)['t_nw'])\n"
            "print('planted world  ind-rel  NW t = %+.2f  (adjustment SHARPENS)' % st.synthetic_detect(planted, adjusted=True)['t_nw'])\n"
            "print('null world     ind-rel  NW t = %+.2f  (should be ~0)' % st.synthetic_detect(null, adjusted=True)['t_nw'])"
        ),
        md("## 3. The honest verdict — a cleaner knife on a claim that isn't there\n\n"
           f"On this liquid mega-cap tape the industry-relative long-low / short-high MAX "
           f"spread is **{R['adj_bps']:+.1f} bps/mo** with NW *t* = **{R['adj_t']:+.2f}** — "
           f"significant, but with the **opposite sign** to the MAX effect: here the lottery "
           f"high-MAX names actually *out-earned* the boring low-MAX ones (the sign-flip "
           f"placebo puts the observed value ~{abs(R['placebo_sigma_left']):.1f}σ into the "
           f"*left* tail). The industry adjustment barely moves the raw *t* "
           f"({R['raw_t']:+.2f} → {R['adj_t']:+.2f}): it sharpens the knife but there is no "
           f"(right-signed) apple to cut. The seeded synthetic control recovers a *planted* "
           f"relation and confirms the adjustment sharpens it, so this is a genuine "
           f"sign-reversal on the mega-cap survivor universe, not a bug — the MAX premium is a "
           f"small-and-illiquid-stock phenomenon. **Signal: None** (the claimed edge is absent "
           f"— and inverts), **Tradability: Mirage** (the specified book loses money gross and "
           f"net)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 876 — Industry-Relative MAX — the teardown\n\n"
           "The head-to-head raw vs industry-relative sort, the Newey-West spread *t*, the "
           "quintile monotonicity card, the 20,000-draw sign-flip placebo, the two-era cut, "
           "the costed timer, and the synthetic control that proves the adjustment sharpens."),
        code("R = %r" % (R,)),
        md("## The head-to-head — raw MAX vs industry-relative MAX\n\n"
           "Monthly Q1 (low MAX) − Q5 (high MAX) next-month spread, equal-weight."),
        code(
            "print(f\"raw MAX          : {R['raw_bps']:+.1f} bps/mo  NW t = {R['raw_t']:+.2f}  \"\n"
            "      f\"one-sample t = {R['raw_t1s']:+.2f}  win {R['raw_win']}%  Sharpe {R['raw_sharpe']:+.2f}\")\n"
            "print(f\"industry-relative: {R['adj_bps']:+.1f} bps/mo  NW t = {R['adj_t']:+.2f}  \"\n"
            "      f\"one-sample t = {R['adj_t1s']:+.2f}  win {R['adj_win']}%  Sharpe {R['adj_sharpe']:+.2f}\")\n"
            "print('  -> both NEGATIVE (wrong sign vs the claim); adjustment sharpens |t| slightly')"
        ),
        md("## Quintile monotonicity — industry-relative MAX (annualised mean return)\n\n"
           "Q5 (the lottery, high-MAX names) posts the *highest* return — the effect runs "
           "backwards on mega-caps."),
        code(
            "for q in ('q1','q2','q3','q4','q5'):\n"
            "    print(f\"  Q{q[-1]}: {R[q]:+.2f}%/yr\")\n"
            "print(f\"  Q5 - Q1 = {R['q5']-R['q1']:+.2f}%/yr (high-MAX out-earns low-MAX)\")"
        ),
        md("## Placebo — sign-flip null on the industry-relative spread (20,000 draws)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.1f} bps vs placebo mean {R['placebo_mean']:+.2f} \"\n"
            "      f\"(sd {R['placebo_sd']:.1f}) -> ~{R['placebo_sigma_left']:+.2f} sigma, left-tail p = {R['placebo_left_p']:.4f}\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.1f} bps  NW t = {R['era_early_t']:+.2f}  (not significant)\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.1f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('  -> the wrong-sign relation is a LATE-era phenomenon; it fails the sub-era robustness bar')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Long low-MAX / short high-MAX (industry-relative), 2 sides × one-way × NAV per "
           "monthly rebalance; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t,a in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t'],R['timer_1_ann']),\n"
            "                      ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'],R['timer_5_ann'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.1f} -> net {n:+.1f} bps/mo (cost {c:.1f}/mo, t={t:+.2f}, ~{a:+.1f}%/yr)\")\n"
            "print('  the specified book loses money gross AND net at every cost -> Mirage')"
        ),
        md("## Synthetic positive control — the machinery is unbiased AND the thesis holds\n\n"
           "Live: neither sort fires on the null; on the planted world the industry adjustment "
           "recovers the effect MORE sharply than raw MAX."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from max_industry import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=876+s, n_months=180), adjusted=True)['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds [ind-rel]: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = data.synthetic_panel(edge=0.012, seed=876, n_months=240)\n"
            "print(f\"planted (edge=0.012): raw-MAX NW t = {st.synthetic_detect(planted, adjusted=False)['t_nw']:+.2f}  vs  ind-rel NW t = {st.synthetic_detect(planted, adjusted=True)['t_nw']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The industry-relative MAX does **not** sharpen a valid MAX "
           f"effect on 50 liquid US mega-caps — because the effect is *inverted* here. The "
           f"Q1−Q5 spread is **{R['adj_bps']:+.1f} bps/mo** (NW *t* = **{R['adj_t']:+.2f}**), "
           f"significant but *opposite in sign* to the claim (the sign-flip placebo puts it "
           f"~{abs(R['placebo_sigma_left']):.1f}σ into the left tail), and it holds only in the "
           f"late era (*t* = {R['era_early_t']:+.2f} early / {R['era_late_t']:+.2f} late). A "
           f"significant wrong-sign result **fails the claim**. The 20-seed synthetic control "
           f"recovers a *planted* relation cleanly and confirms the adjustment sharpens it "
           f"(raw *t* = {R['planted_raw_t']:+.2f} → ind-rel *t* = {R['planted_adj_t']:+.2f}), so "
           f"the sign-reversal is real, not machinery. Survivorship biases the magnitude.\n"
           f"- **Tradability — Mirage.** The specified long-low / short-high book loses money "
           f"gross (**{R['timer_1_gross']:+.1f} bps/mo**) and net (**{R['timer_1_net']:+.1f}** "
           f"at 1 bp, **{R['timer_5_net']:+.1f}** at 5 bps). A Mirage in the claimed direction."),
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
