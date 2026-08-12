"""Generate the two narrative notebooks for Study 846 (Blockbuster Game-Launch Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
TTWO/EA/NTDOY/UBSFY/SPY tapes under ../_cache/ (fetching once on a cache miss) and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The
synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (TTWO + EA + NTDOY +
# UBSFY + SPY, yfinance, 2013-01-02 -> 2026-06-30; 33 of 37 launches resolved, 4 ATVI
# excluded — delisted). Fingerprint ba0709a58928.
R = dict(
    n_events=37, n_included=33, fp="ba0709a58928", rows=3393,
    pre_s_mean=-1.381, pre_s_t=-1.316, pre_s_nw=-1.844, pre_s_hit=8,
    pre_l_mean=-1.581, pre_l_t=-1.259, pre_l_nw=-1.626, pre_l_hit=13,
    post_s_mean=+0.339, post_s_t=+0.383, post_s_nw=+0.382, post_s_hit=18,
    post_l_mean=-0.165, post_l_t=-0.098, post_l_nw=-0.099, post_l_hit=16,
    ttwo_mean=+1.228, ttwo_t=+0.535, ea_mean=+0.648, ea_t=+0.401,
    ntdoy_mean=+2.766, ntdoy_t=+1.164, ubsfy_mean=-5.597, ubsfy_t=-0.971,
    early_mean=+1.164, early_t=+0.697, late_mean=-1.416, late_t=-0.489,
    pl_pre_p=0.938, pl_post_p=0.602, pl_pre_mean=+0.153, pl_pre_sd=1.188,
    pl_post_mean=+0.263, pl_post_sd=1.647,
    jk_lo=-0.324, jk_hi=+1.111,
    pre_l_net5=-1.681, pre_l_t5=-1.34, pre_l_net10=-1.781, pre_l_t10=-1.42,
    post_l_net5=-0.265, post_l_t5=-0.16, post_l_net10=-0.365, post_l_t10=-0.22,
    null_mean_t=-0.24, null_sd_t=1.21, null_hits=1,
    planted2_mean=+1.696, planted2_t=+1.38, planted4_mean=+3.696, planted4_t=+3.02,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from game_launch import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching TTWO + EA + NTDOY + UBSFY + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} launches resolved "
      f"(4 ATVI launches carry no tape — Activision delisted 2023-10-13)")
"""


def build_curious():
    cells = [
        md("# Study 846 — Blockbuster Game-Launch Drift 🕹️\n\n"
           "*For the curious.* When a marquee AAA game ships — **GTA V**, **Red Dead "
           "Redemption 2**, **Zelda: Tears of the Kingdom**, **Battlefield**, **Assassin's "
           "Creed** — does the **publisher's stock** move? Folklore says *buy the hype into "
           "launch day* and then either ride the momentum or *sell the news*. We put **~33 "
           "blockbuster launches (2013→2024)** on the stand — mapped to **TTWO** (Rockstar/2K), "
           "**EA**, **NTDOY** (Nintendo) and **UBSFY** (Ubisoft) — and asked two plain "
           "questions: **does the publisher rally into the launch?** and **does it drift after "
           "it?** The answer is a quiet double bust."),
        md("## The claim\n\n"
           "A blockbuster launch is a scheduled, heavily-marketed, months-in-advance ship — the "
           "retail 'buy the hype' reflex says load up on the publisher into launch day and ride "
           "the drift. The catch that kills it before we start: the date is *known* far ahead, "
           "and one title is a small slice of a giant multi-franchise publisher (TTWO also ships "
           "NBA 2K; EA also ships FC and Apex; Nintendo ships a whole platform). We measure the "
           "publisher's **abnormal** return (publisher − SPY, total-return) so we're not just "
           "measuring the market — each launch anchored to *its own* publisher. Note: **ATVI** "
           "(Call of Duty, Diablo IV) is in the calendar for the record but has **no tape** — "
           "Microsoft's buyout delisted it on 2023-10-13 — so its five launches are honestly "
           "excluded, not silently dropped."),
        code(PRELUDE),
        md("## The launch calendar we test (hardcoded from Wikipedia / press releases)"),
        code("pd.DataFrame(dt.EVENTS, columns=['launch', 'publisher', 'title'])"),
        md("## The picture: mean cumulative abnormal return around the launch\n\n"
           "Offset 0 is the launch. Left of zero is the *run-up* (the supposed hype rally); "
           "right of zero is the *drift* window out to ~1 month (ride it or sell the news)."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#0b6e4f', lw=2)\n"
            "ax.set_xlabel('trading days from the launch (0 = ship date)')\n"
            "ax.set_ylabel('mean cumulative AR, publisher − SPY (%)')\n"
            "ax.set_title('Publishers do NOT rally into (or drift after) a blockbuster launch')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 1-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/33 |\n"
           f"| 2-week run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/33 |\n"
           f"| 1-week drift | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/33 |\n"
           f"| ~1-month drift (20d) | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/33 |\n\n"
           f"The bullish folklore just **isn't there**. If anything the publisher *drifts down* "
           f"into the launch ({R['pre_l_mean']:+.2f}% over two weeks) — the **wrong way** for "
           f"'buy the hype' — but it's **not significant** (*t* = {R['pre_l_t']:+.2f}, |*t*| < 2). "
           f"And the headline **20-day post-launch drift is a clean zero** "
           f"({R['post_l_mean']:+.2f}%, *t* = {R['post_l_t']:+.2f}, hit {R['post_l_hit']}/33 — an "
           f"essentially even coin flip). No hype premium, no momentum, no sell-the-news edge."),
        code(
            "rows = [('1wk run-up','pre_s'),('2wk run-up','pre_l'),('1wk drift','post_s'),('20d drift','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<11s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The finance version of 'buy the hype into a game launch' is folklore: publishers "
           "don't rally into a launch (a faint, insignificant *dip* if anything — the wrong way) "
           "and don't drift after it (a clean 20-day zero). On top of the missing signal, the "
           "date is public months ahead and a single title is a small slice of a huge "
           "publisher's revenue. Verdict: **None signal, Mirage tradability.** The quants' "
           "notebook has the placebo, the per-publisher and sub-era splits, the jackknife, the "
           "costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 846 — Blockbuster Game-Launch Drift — for the quants 🔬\n\n"
           "The full battery: one-sample *t* and a Newey-West (HAC) *t* per window, a "
           "random-window placebo, per-publisher and sub-era splits, a leave-one-out "
           "jackknife, the costed net leg, and a seeded synthetic positive control. "
           f"Everything offline once cached; fingerprint `{R['fp']}`. **ATVI** (delisted "
           "2023-10-13) carries no tape, so its four launches sit in the excluded funnel."),
        code(PRELUDE),
        md("## 1. One-sample *t* (and HAC *t*) across independent launches\n\n"
           "Each launch is one independent event pooled across the four publishers, so the unit "
           "is a one-sample *t* of the per-event abnormal return — **not** a daily panel (which "
           "would fake precision). A Newey-West *t* on the date-ordered series is a robustness "
           "cross-check; with well-spaced events it lands close by."),
        code(
            "for label, col in [('1wk run-up','pre_s'),('2wk run-up','pre_l'),('1wk drift','post_s'),('20d drift','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    nw = st.newey_west_t(inc.sort_values('anchor_date')[col].values)\n"
            "    print(f'{label:<11s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}  t_NW={nw:+.3f}')"
        ),
        md(f"## 2. Per-publisher & sub-era splits — does anything hold up?\n\n"
           f"A real effect should show in *most* publishers and *both* halves of the sample. The "
           f"20-day drift does the opposite: **TTWO** {R['ttwo_mean']:+.2f}% (*t* {R['ttwo_t']:+.2f}), "
           f"**EA** {R['ea_mean']:+.2f}% (*t* {R['ea_t']:+.2f}), **NTDOY** {R['ntdoy_mean']:+.2f}% "
           f"(*t* {R['ntdoy_t']:+.2f}) but **UBSFY** {R['ubsfy_mean']:+.2f}% (*t* {R['ubsfy_t']:+.2f}) — "
           f"the signs scatter and nothing clears |*t*| ≥ 2. And the eras flip: early "
           f"{R['early_mean']:+.2f}% (*t* {R['early_t']:+.2f}) vs late {R['late_mean']:+.2f}% "
           f"(*t* {R['late_t']:+.2f}) — the opposite of a stable, era-spanning effect."),
        code(
            "print('by publisher (20-day drift):')\n"
            "for pub in dt.PUBLISHERS:\n"
            "    sub = inc[inc['publisher']==pub]; s = st.one_sample_t(sub['post_l'].values)\n"
            "    print(f'  {pub:<6s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')\n"
            "srt = inc.sort_values('anchor_date'); h = len(srt)//2\n"
            "print('by era (20-day drift):')\n"
            "for lab, part in [('early', srt.iloc[:h]), ('late', srt.iloc[h:])]:\n"
            "    s = st.one_sample_t(part['post_l'].values)\n"
            "    print(f'  {lab:<5s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 3. Random-window placebo — is the move inside the luck cloud?\n\n"
           "For each event we redraw a random, non-launch window on the event's OWN publisher "
           "vs SPY and recompute the abnormal return; 20 seeds × 200 draws. If the observed mean "
           "sits in the tail of that null, it isn't ordinary tracking noise. Here both the "
           "run-up and the 20-day drift sit deep inside."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_l', k=20, tail='right')\n"
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
            "        pa, sa, m = pub_arr[p]; q = int(rng.integers(0, m-21))\n"
            "        vals.append(float(pa[q+20]/pa[q] - sa[q+20]/sa[q]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#0b6e4f', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 20-day AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"post-launch drift vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 4. Jackknife — is the (already-zero) drift one launch, or broad?"),
        code(
            "x = inc['post_l'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 5. Tradability — net of costs\n\n"
           "Even gross there is no significant edge; costs only deepen the (insignificant) "
           "windows. 'Buy the hype' is a *losing* long here, and there is no momentum to ride "
           "after the ship. Execution lag = 0 (the date is calendar-known months ahead — the "
           "one thing that *is* real is that you could place the trade on time; there's just "
           "nothing to place it on)."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_l','2wk run-up'),('post_l','20d drift')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<11s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 6. Synthetic positive control — the detector works; the tape has no bump\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted launch drift with unit slope. At n = 32 with a 20-day AR the null fires "
           "|*t*| ≥ 2 on only ~1/20 seeds — an honest false-positive rate — and a real planted "
           "drift is recovered (+1% planted → +1% measured). The machinery is fine; the real "
           "tape simply has nothing to find."),
        code(
            "null_ts = np.array([st.synthetic_detect(drift=0.0, seed=846+s, k=20)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for d in (0.02, 0.04):\n"
            "    r = st.synthetic_detect(drift=d, seed=846, k=20)\n"
            "    print(f'planted +{d*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "drifts = np.linspace(0, 0.06, 13)\n"
            "ts = [st.synthetic_detect(drift=d, seed=846, k=20)['t'] for d in drifts]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(drifts*100, ts, 'o-', color='#0b6e4f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted launch drift (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted drift is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Neither half of the folklore survives — the run-up is a faint, "
           "insignificant *dip* (2-week −1.58%, *t* = −1.26, HAC *t* = −1.63, the wrong sign for "
           "'buy the hype', placebo right-tail p = 0.94) and the 20-day post-launch drift is a "
           "clean zero (−0.17%, *t* = −0.10, hit 16/33, placebo p = 0.60), gone under "
           "per-publisher (signs scatter, UBSFY −5.6% vs NTDOY +2.8%) and sub-era (early +1.16% "
           "vs late −1.42%) splits. The synthetic control shows a real drift *would* have been "
           "caught. **Tradability: Mirage.** No edge gross or net, a losing long into the "
           "launch, and no momentum to ride out of it."),
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
