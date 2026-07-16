"""Generate the two narrative notebooks for Study 777 (Black-Friday-Retail).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached XRT/SPY
tapes under ../_cache/ (fetching once on a cache miss) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (XRT + SPY, yfinance,
# 2006-06-22 -> 2026-06-30; 20 of 20 Black Fridays resolved). Fingerprint 75498c2542dd.
R = dict(
    n_events=20, n_included=20, fp="75498c2542dd", rows=5036,
    pre_s_mean=+1.333, pre_s_t=+1.585, pre_s_hit=13, pre_s_n=20,
    pre_l_mean=+1.138, pre_l_t=+1.086, pre_l_hit=12,
    post_s_mean=+0.059, post_s_t=+0.081, post_s_hit=9,
    post_l_mean=-0.497, post_l_t=-0.435, post_l_hit=9,
    pl_pre_p=0.0262, pl_mean=-0.012, pl_sd=0.667, pl_post_p=0.5457,
    jk_lo=+1.167, jk_hi=+2.145,
    pre_s_net5=+1.233, pre_s_t5=+1.47, pre_s_net10=+1.133, pre_s_t10=+1.35,
    null_mean_t=-0.59, null_sd_t=1.49, null_hits=3,
    planted1_mean=+2.474, planted1_t=+3.72, planted2_mean=+3.474, planted2_t=+5.23,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from black_friday_retail import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching XRT + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} Black Fridays resolved")
"""


def build_curious():
    cells = [
        md("# Study 777 — Black-Friday-Retail 🛍️\n\n"
           "*For the curious.* Everyone *knows* you buy retail into Black Friday — the "
           "sector runs up on holiday-sales hopes, then sells the news. We put 20 Black "
           "Fridays (2006→2025) on the stand and asked the tape two plain questions: **does "
           "the retail ETF (XRT) rally into Black Friday?** and **does it fade after?** The "
           "answer is more interesting than the usual folklore bust."),
        md("## The claim, and why it's a clean test\n\n"
           "Black Friday (the Friday after US Thanksgiving) is the year's biggest shopping "
           "catalyst and its date is **fixed by statute** — the day after the fourth "
           "Thursday of November, known years ahead. So *buy K sessions before, sell on the "
           "day* is calendar-known and zero-look-ahead. We measure XRT's **abnormal** return "
           "(XRT − SPY, total-return) so we net out the market's own November drift and see "
           "only retail's *relative* move."),
        code(PRELUDE),
        md("## The Black-Friday calendar we test (day after the 4th Thursday of November)"),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'black_friday'])"),
        md("## The picture: mean cumulative abnormal return around Black Friday\n\n"
           "Offset 0 is Black Friday. Left of zero is the *run-up* (the supposed rally in); "
           "right of zero is the *fade* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#8e44ad', lw=2)\n"
            "ax.set_xlabel('trading days from Black Friday (0 = event)')\n"
            "ax.set_ylabel('mean cumulative AR, XRT − SPY (%)')\n"
            "ax.set_title('Retail DOES drift up into Black Friday — mildly')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | **{R['pre_s_mean']:+.2f}%** | **{R['pre_s_t']:+.2f}** | {R['pre_s_hit']}/20 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/20 |\n"
           f"| 2-week fade | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/20 |\n"
           f"| 1-month fade | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/20 |\n\n"
           f"The run-up points the **right way** for once: in the last two weeks *into* Black "
           f"Friday XRT beats SPY by {R['pre_s_mean']:+.2f}% on average, hitting "
           f"{R['pre_s_hit']} of 20 years. But it's mild — *t* is only {R['pre_s_t']:+.2f} "
           f"(short of significance) — and it *weakens* if you stretch to a month. And the "
           f"'sell the news' fade? A clean zero (both cuts |*t*| < 0.5)."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "Retail really *does* drift up into Black-Friday week — the right direction, and "
           "unusual versus random windows (the quants' notebook shows only ~2.6% of random "
           "fortnights were this positive). But the effect is small, only *marginally* "
           "significant (*t* ≈ 1.6), fades at a longer lookback, and comes with **no** "
           "'sell the news' reversal to pair it with. Verdict: **Weak signal, Mirage "
           "tradability** — a suggestive curiosity, not a bankable trade. The quants' "
           "notebook has the placebo, the jackknife, the costed leg and the synthetic "
           "control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 777 — Black-Friday-Retail — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent Black-Friday years\n\n"
           "Each Black Friday is one independent event, so the unit is a one-sample *t* of "
           "the per-year abnormal return — **not** a daily panel (which would fake "
           "precision)."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the run-up inside the luck cloud?\n\n"
           "For each event we redraw a random, non-Black-Friday 2-week window on XRT vs SPY "
           "and recompute the abnormal return; 20 seeds × 200 draws. If the observed mean "
           "sits in the tail of that null, it isn't ordinary tracking noise. Here it does — "
           "which is why the effect is *real-ish* even though the cross-year *t* (which pays "
           "the full year-to-year variance) is only ~1.6."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'pre_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "xrt, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = xrt.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(xrt.loc[common[p+10]]/xrt.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8e44ad', alpha=0.65)\n"
            "ax.axvline(pl['obs']*100, color='#c0392b', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"pre-Black-Friday pop vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the run-up one great year, or broad?"),
        code(
            "x = inc['pre_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). The run-up is a *long* in a liquid ETF, so it survives costs "
           "almost intact — but a +1.3%/yr window with *t* ≈ 1.5 is one positive cut inside "
           "a four-window search, and its 'sell the news' partner doesn't exist."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('post_s','2wk fade')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is noisy\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover "
           "a planted pre-Black-Friday bump. Note the honest small-sample false-positive "
           "rate at n = 20 with a 10-day AR: |*t*| ≥ 2 fires on ~3/20 null seeds — a reason "
           "not to over-read the observed +1.59."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=783+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=783, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=783, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The run-up points the right way (2-week run-up +1.33%, "
           "*t* = +1.59, placebo right-tail ≈ 0.026, jackknife stays positive) — a genuine, "
           "directionally-correct whiff, more than most calendar folklore delivers — but the "
           "cross-year *t* is short of significance, it weakens at a 1-month lookback, the "
           "fade is a clean zero, and the null fires |*t*| ≥ 2 on 3/20 seeds at this n. "
           "**Tradability: Mirage.** Net +1.2%/yr with *t* ≈ 1.5 is one positive cut inside "
           "a four-window search with no partner reversal — liquid and long, but no reliable "
           "edge to size."),
    ]
    return new_notebook(cells=cells)


def main():
    for name, nb in (("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())):
        path = os.path.join(HERE, f"{name}.ipynb")
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)


if __name__ == "__main__":
    main()
