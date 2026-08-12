"""Generate the two narrative notebooks for Study 844 (Madden-Cover-Curse).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
EA/TTWO/SPY tapes under ../_cache/ (fetching once on a cache miss) and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (EA + TTWO + SPY,
# yfinance, 2014-06-02 -> 2026-06-30; 20 of 20 launches resolved). Fingerprint a8dd4e62fc5d.
R = dict(
    n_events=20, n_included=20, fp="a8dd4e62fc5d", rows=3038,
    pre_s_mean=+0.048, pre_s_t=+0.072, pre_s_hit=10,
    pre_l_mean=-0.168, pre_l_t=-0.197, pre_l_nw=-0.227, pre_l_hit=10,
    post_s_mean=-1.213, post_s_t=-1.889, post_s_nw=-1.556, post_s_hit=8,
    post_l_mean=-0.781, post_l_t=-0.906, post_l_nw=-1.084, post_l_hit=9,
    ea_mean=-0.247, ea_t=-0.242, ttwo_mean=-1.315, ttwo_t=-0.923,
    early_mean=-0.029, early_t=-0.042, late_mean=-1.532, late_t=-0.966,
    pl_pre_p=0.645, pl_post_p=0.823, pl_mean=+0.285, pl_sd=1.191,
    jk_lo=-1.355, jk_hi=-0.369,
    post_l_net5=-0.881, post_l_t5=-1.02, post_l_net10=-0.981, post_l_t10=-1.14,
    null_mean_t=+0.34, null_sd_t=0.99, null_hits=1,
    planted2_mean=+0.621, planted2_t=+0.49, planted4_mean=+2.621, planted4_t=+2.06,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from madden_curse import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching EA + TTWO + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} launches resolved")
"""


def build_curious():
    cells = [
        md("# Study 844 — Madden-Cover-Curse 🎮\n\n"
           "*For the curious.* The gaming **\"Madden curse\"** says the athlete on the box "
           "gets hurt the next season. We're not a sports desk — we test the *finance* "
           "version: when the new **Madden** or **NBA 2K** ships, does the **publisher's "
           "stock** move? EA makes *Madden*; Take-Two (TTWO) makes *NBA 2K*. We put 20 annual "
           "launches (2015→2024) on the stand and asked two plain questions: **does the "
           "publisher rally into the launch?** and **does it move after?** The answer is a "
           "quiet double bust."),
        md("## The claim\n\n"
           "A new *Madden* or *NBA 2K* is a scheduled, heavily-marketed, once-a-year ship — "
           "the retail 'buy the hype' reflex says load up on the publisher into launch day. "
           "The catch that kills it before we start: it's a *known* date, marketed months "
           "ahead, and one sports title is a small slice of a giant multi-franchise "
           "publisher (EA also ships FC and Apex; Take-Two also ships GTA). We measure the "
           "publisher's **abnormal** return (publisher − SPY, total-return) so we're not "
           "just measuring the market — each launch anchored to *its own* publisher."),
        code(PRELUDE),
        md("## The launch calendar we test (hardcoded from Wikipedia / press releases)"),
        code("pd.DataFrame(dt.EVENTS, columns=['launch', 'publisher', 'title', 'cover_athlete'])"),
        md("## The picture: mean cumulative abnormal return around the launch\n\n"
           "Offset 0 is the launch. Left of zero is the *run-up* (the supposed hype rally); "
           "right of zero is the *drift* window (ride it or sell the news)."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#0b6e4f', lw=2)\n"
            "ax.set_xlabel('trading days from the launch (0 = ship date)')\n"
            "ax.set_ylabel('mean cumulative AR, publisher − SPY (%)')\n"
            "ax.set_title('EA / TTWO do NOT rally into (or pop after) a game launch')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 1-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/20 |\n"
           f"| 2-week run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/20 |\n"
           f"| 1-week drift | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/20 |\n"
           f"| 2-week drift | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/20 |\n\n"
           f"The bullish folklore just **isn't there**: the run-up is a clean zero "
           f"({R['pre_s_mean']:+.2f}% / {R['pre_l_mean']:+.2f}%, |*t*| < 0.25, hit 10/20 — an "
           f"exact coin flip). And after the launch the publisher, if anything, drifts "
           f"*down* ({R['post_s_mean']:+.2f}% in the first week) — the **wrong way** for a "
           f"'buy the hype' trade, and not significant anyway (*t* = {R['post_s_t']:+.2f}, "
           f"|*t*| < 2)."),
        code(
            "rows = [('1wk run-up','pre_s'),('2wk run-up','pre_l'),('1wk drift','post_s'),('2wk drift','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<11s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The finance version of the *Madden curse* is folklore: EA and TTWO don't rally "
           "into a launch (a clean zero) and don't reliably pop after (a faint, insignificant "
           "*dip* if anything — the wrong way for 'buy the hype'). On top of the missing "
           "signal, the date is public months ahead and a single sports title is a small "
           "slice of a huge publisher's revenue. Verdict: **None signal, Mirage "
           "tradability.** The quants' notebook has the placebo, the splits, the jackknife, "
           "the costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 844 — Madden-Cover-Curse — for the quants 🔬\n\n"
           "The full battery: one-sample *t* and a Newey-West (HAC) *t* per window, a "
           "random-window placebo, per-publisher and sub-era splits, a leave-one-out "
           "jackknife, the costed net leg, and a seeded synthetic positive control. "
           f"Everything offline once cached; fingerprint `{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* (and HAC *t*) across independent launches\n\n"
           "Each launch is one independent event pooled across the two publishers, so the "
           "unit is a one-sample *t* of the per-event abnormal return — **not** a daily panel "
           "(which would fake precision). A Newey-West *t* on the date-ordered series is a "
           "robustness cross-check; with well-spaced events it lands close by."),
        code(
            "for label, col in [('1wk run-up','pre_s'),('2wk run-up','pre_l'),('1wk drift','post_s'),('2wk drift','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    nw = st.newey_west_t(inc.sort_values('anchor_date')[col].values)\n"
            "    print(f'{label:<11s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}  t_NW={nw:+.3f}')"
        ),
        md(f"## 2. Per-publisher & sub-era splits — does the faint dip hold up?\n\n"
           f"A real effect should show in *both* publishers and *both* halves of the sample. "
           f"It doesn't: EA's 2-week drift is ~zero (*t* = {R['ea_t']:+.2f}), only TTWO leans "
           f"(insignificantly), the early era is flat (*t* = {R['early_t']:+.2f}) and the late "
           f"era carries what little lean there is — the opposite of a stable effect."),
        code(
            "print('by publisher (2-week drift):')\n"
            "for pub in dt.PUBLISHERS:\n"
            "    sub = inc[inc['publisher']==pub]; s = st.one_sample_t(sub['post_l'].values)\n"
            "    print(f'  {pub:<5s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')\n"
            "srt = inc.sort_values('anchor_date'); h = len(srt)//2\n"
            "print('by era (2-week drift):')\n"
            "for lab, part in [('2015-2019', srt.iloc[:h]), ('2020-2024', srt.iloc[h:])]:\n"
            "    s = st.one_sample_t(part['post_l'].values)\n"
            "    print(f'  {lab} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 3. Random-window placebo — is the move inside the luck cloud?\n\n"
           "For each event we redraw a random, non-launch 2-week window on the event's OWN "
           "publisher vs SPY and recompute the abnormal return; 20 seeds × 200 draws. If the "
           "observed mean sits in the tail of that null, it isn't ordinary tracking noise. "
           "Here it sits deep inside."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_l', k=10, tail='right')\n"
            "spy = prices[dt.BENCHMARK]\n"
            "pub_arr = {}\n"
            "for p in dt.PUBLISHERS:\n"
            "    common = prices[p].index.intersection(spy.index).sort_values()\n"
            "    pub_arr[p] = (prices[p].reindex(common).to_numpy(float), spy.reindex(common).to_numpy(float), len(common))\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "epubs = inc['publisher'].to_numpy()\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for p in epubs:\n"
            "        pa, sa, m = pub_arr[p]; q = int(rng.integers(0, m-11))\n"
            "        vals.append(float(pa[q+10]/pa[q] - sa[q+10]/sa[q]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#0b6e4f', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"post-launch drift vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 4. Jackknife — is the dip one bad launch, or broad?"),
        code(
            "x = inc['post_l'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 5. Tradability — net of costs\n\n"
           "Even gross there is no significant edge; costs only deepen the (insignificant) "
           "negative windows. 'Buy the hype' is a *losing* long here, and the mirror "
           "'sell the news' short is not significant either — nor a business on a large-cap "
           "publisher around a telegraphed annual ship. Execution lag = 0 (the date is "
           "calendar-known years ahead)."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_l','2wk run-up'),('post_l','2wk drift')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<11s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 6. Synthetic positive control — the detector works; the tape has no bump\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted launch drift with unit slope. At n = 20 with a 10-day AR the null fires "
           "|*t*| ≥ 2 on only ~1/20 seeds — an honest false-positive rate — and a real planted "
           "drift is recovered (+1% planted → +1% measured). The machinery is fine; the real "
           "tape simply has nothing to find (and what lean it has points the wrong way for "
           "the claim)."),
        code(
            "null_ts = np.array([st.synthetic_detect(drift=0.0, seed=844+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for d in (0.02, 0.04):\n"
            "    r = st.synthetic_detect(drift=d, seed=844, k=10)\n"
            "    print(f'planted +{d*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "drifts = np.linspace(0, 0.06, 13)\n"
            "ts = [st.synthetic_detect(drift=d, seed=844, k=10)['t'] for d in drifts]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(drifts*100, ts, 'o-', color='#0b6e4f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted launch drift (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted drift is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Neither half of the folklore survives — the run-up is a clean "
           "zero (|*t*| < 0.25, hit 10/20) and the post-launch move is a faint, insignificant "
           "*dip* (*t* = −1.89 at 1 week, HAC *t* = −1.56, |*t*| < 2), the wrong sign for "
           "'buy the hype', deep inside the placebo cloud (right-tail p = 0.82), and gone "
           "under per-publisher and sub-era splits. The synthetic control shows a real drift "
           "*would* have been caught. **Tradability: Mirage.** No edge gross or net, a losing "
           "long, an insignificant short, and no business around a telegraphed annual ship."),
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
