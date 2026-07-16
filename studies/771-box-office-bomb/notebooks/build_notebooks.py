"""Generate the two narrative notebooks for Study 771 (Box-Office-Bomb).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached DIS/SPY tapes
under ../_cache/ (fetching once on a cache miss) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (DIS + SPY, yfinance,
# 2010-01-04 -> 2026-06-30; 16 of 16 flops resolved). Fingerprint f3ff7b3017ec.
R = dict(
    n_events=16, n_included=16, fp="f3ff7b3017ec", rows=4147,
    post_s_mean=-1.034, post_s_t=-1.533, post_s_up=9, post_s_n=16,
    post_l_mean=-1.767, post_l_t=-1.622, post_l_up=7,
    pre_s_mean=+0.001, pre_s_t=+0.001, pre_s_up=10,
    pre_l_mean=-1.109, pre_l_t=-0.809, pre_l_up=8,
    pl_post_s_p=0.169, pl_post_s_mean=-0.117, pl_post_s_sd=0.949,
    pl_post_l_p=0.139, pl_post_l_mean=-0.245, pl_post_l_sd=1.422,
    jk_lo=-1.801, jk_hi=-1.088,
    short_s_net5=+0.934, short_s_t5=+1.38, short_s_net10=+0.834, short_s_t10=+1.24,
    short_l_net5=+1.667, short_l_t5=+1.53, short_l_net10=+1.567, short_l_t10=+1.44,
    null_mean_t=+0.07, null_sd_t=1.03, null_hits=0,
    planted1_mean=-0.119, planted1_t=-0.14, planted2_mean=-1.119, planted2_t=-1.35,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from box_office_bomb import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching DIS + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} flops resolved")
"""


def build_curious():
    cells = [
        md("# Study 771 — Box-Office-Bomb 🎬💣\n\n"
           "*For the curious.* When a Disney tentpole belly-flops — *John Carter*, *The Lone "
           "Ranger*, *The Marvels*, *Snow White* — the reflex trade is obvious: **sell the "
           "studio.** A nine-figure write-down is a public humiliation, so the stock should "
           "sag once the disappointing weekend is known. We put 16 genuinely notorious Disney "
           "bombs (2012→2025) on the stand and asked the tape: **does DIS actually fall after "
           "a flop?** The answer is a shrug — the right direction, far too faint to trade."),
        md("## The claim, and why it's a clean test\n\n"
           "A wide release opens on a known Friday; the *weekend* box-office number is public "
           "by Sunday/Monday. So the first session at which 'it flopped' is common knowledge "
           "is the **Monday after opening** — a *short DIS that Monday, cover K sessions later* "
           "rule is calendar-known and zero-look-ahead. We measure DIS's **abnormal** return "
           "(DIS − SPY, total-return) so we're not just measuring the market."),
        code(PRELUDE),
        md("## The flop calendar we test (hardcoded from Box Office Mojo / studio releases)"),
        code("pd.DataFrame(dt.EVENTS, columns=['title', 'opening_date'])"),
        md("## The picture: mean cumulative abnormal return around the flop\n\n"
           "Offset 0 is the anchor (the Monday after the opening weekend, when the flop is "
           "public). Left of zero is the pre-release drift; right of zero is the post-flop "
           "window — the 'sell the news'."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#c0392b', lw=2)\n"
            "ax.set_xlabel('trading days from the flop-known anchor (0 = Monday after opening)')\n"
            "ax.set_ylabel('mean cumulative AR, DIS − SPY (%)')\n"
            "ax.set_title('Disney drifts DOWN after a flop — but only a little')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | up rate |\n|---|--:|--:|--:|\n"
           f"| 2-week post | **{R['post_s_mean']:+.2f}%** | **{R['post_s_t']:+.2f}** | {R['post_s_up']}/16 |\n"
           f"| 1-month post | **{R['post_l_mean']:+.2f}%** | **{R['post_l_t']:+.2f}** | {R['post_l_up']}/16 |\n"
           f"| 2-week pre | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_up']}/16 |\n"
           f"| 1-month pre | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_up']}/16 |\n\n"
           f"The folklore points the *right way*: after a flop DIS under-performs SPY by "
           f"{abs(R['post_s_mean']):.2f}% over two weeks and {abs(R['post_l_mean']):.2f}% over "
           f"a month, and both cuts carry a negative *t*. But neither clears significance "
           f"(|*t*| ≈ 1.5–1.6), and the up-rate is a coin-flip. A whiff, not a signal."),
        code(
            "rows = [('2wk post','post_s'),('1mo post','post_l'),('2wk pre','pre_s'),('1mo pre','pre_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<9s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  up {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "Selling Disney after a notorious flop is *directionally* right — the stock does "
           "mildly lag the market afterwards — but the effect is too small and too noisy "
           "(|*t*| ≈ 1.5, placebo p ≈ 0.15) to distinguish from Disney's ordinary "
           "underperformance vs SPY, and a 16-event, once-a-year short is a curiosity, not a "
           "strategy. Verdict: **Weak signal, Mirage tradability.** The quants' notebook has "
           "the placebo, the jackknife, the costed short leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 771 — Box-Office-Bomb — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed short leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent flop events\n\n"
           "Each flop is one independent event, so the unit is a one-sample *t* of the "
           "per-event abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk post','post_s'),('1mo post','post_l'),('2wk pre','pre_s'),('1mo pre','pre_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<9s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the post-flop sag inside the luck cloud?\n\n"
           "For each event we redraw a random, non-flop 2-week window on DIS vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits in "
           "the *left tail* of that null, it isn't ordinary tracking noise. (Note the placebo "
           "mean is itself slightly negative — DIS lagged SPY over 2010–2026.)"),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_s', k=10, tail='left')\n"
            "import numpy as np\n"
            "dis, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = dis.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(dis.loc[common[p+10]]/dis.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#c0392b', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"post-flop sag vs luck cloud (left-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the sag one bad film, or broad?"),
        code(
            "x = inc['post_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — the SHORT, net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same. "
           "You *short* DIS, so a negative post-window is a positive short P&L — but with "
           "*t* ≈ 1.4–1.5 it's not distinguishable from zero, and shorting a mega-cap 16 times "
           "in 13 years is capacity- and borrow-trivial."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, short_col, label in [('post_s','short_s_net','2wk post'),('post_l','short_l_net','1mo post')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[short_col].values); n10 = st.one_sample_t(inc10[short_col].values)\n"
            "    print(f'{label:<9s} gross DIS-SPY {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  SHORT net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  SHORT net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is quiet\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted post-flop sell-off monotonically. At n = 16 with a 10-day AR the machinery "
           "needs a *large* real drop to reach significance — which frames why the observed "
           "−1% shows up only as a whiff."),
        code(
            "null_ts = np.array([st.synthetic_detect(drop=0.0, seed=773+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for d in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(drop=d, seed=773, k=10)\n"
            "    print(f'planted -{d*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "drops = np.linspace(0, 0.04, 13)\n"
            "ts = [st.synthetic_detect(drop=d, seed=773, k=10)['t'] for d in drops]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(drops*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(-2, color='0.6', ls='--'); ax.set_xlabel('planted post-flop drop (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted sell-off is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The sign is the folklore's — DIS under-performs SPY after a flop "
           "at both horizons (−1.03% / −1.77%) and the jackknife keeps *t* negative on every "
           "leave-one-out — but |*t*| ≈ 1.5–1.6 is short of significance, the placebo p is "
           "0.14–0.17, the up-rate is a coin-flip, and part of the raw sag is just Disney's "
           "decade-long drag vs SPY. **Tradability: Mirage.** The short earns ~+0.9–1.7% per "
           "event with *t* ≈ 1.4–1.5 (insignificant), fires 16 times in 13 years, and sits "
           "inside a multi-window search."),
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
