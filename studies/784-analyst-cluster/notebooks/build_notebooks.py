"""Generate the two narrative notebooks for Study 784 (Analyst-Cluster).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached NVDA/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (NVDA + SPY, yfinance,
# 2015-01-02 -> 2026-06-30; 39 of 39 clusters resolved). Fingerprint 31db942be633.
R = dict(
    n_events=39, n_included=39, fp="31db942be633", rows=2889,
    pre_s_mean=+2.292, pre_s_t=+2.339, pre_s_hit=26, pre_s_n=39,
    pre_l_mean=+5.274, pre_l_t=+3.562, pre_l_hit=30,
    post_s_mean=+4.756, post_s_t=+2.126, post_s_hit=23,
    post_l_mean=+5.060, post_l_t=+1.964, post_l_hit=23,
    pl_runup_p=0.408, pl_mean=+2.003, pl_sd=1.243, pl_fade_p=0.985,
    jk_lo=+1.838, jk_hi=+2.464,
    post_s_net5=+4.656, post_s_t5=+2.08, post_s_net10=+4.556, post_s_t10=+2.04,
    null_mean_t=-0.02, null_sd_t=1.30, null_hits=3,
    planted1_mean=+0.777, planted1_t=+1.00, planted2_mean=+1.777, planted2_t=+2.30,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from analyst_cluster import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching NVDA + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} clusters resolved")
"""


def build_curious():
    cells = [
        md("# Study 784 — Analyst-Cluster 📈\n\n"
           "*For the curious.* The contrarian's favourite: when *everybody* upgrades a "
           "stock in the same week, fade it — the sell-side pile-in is the last marginal "
           "buyer. We put that trade on NVDA's tape across 39 analyst-upgrade clusters "
           "(2016→2025) and asked two plain questions: **does NVDA run up into the cluster?** "
           "and **does it fade after?** The answer to the fade is a loud, wrong-signed *no*."),
        md("## The claim, and the honest proxy\n\n"
           "> **LABELLED PROXY.** There's no free point-in-time analyst-upgrade feed here, so "
           "the *cluster week* is proxied by **NVDA's real quarterly earnings week** — the "
           "canonical trigger of a same-week price-target-hike wave. The earnings dates are "
           "real and known ~3-4 weeks ahead (zero look-ahead). The catch, stated up front: "
           "this proxy inherits **post-earnings drift** as a confound.\n\n"
           "We measure NVDA's **abnormal** return (NVDA − SPY, total-return) so we're not just "
           "measuring the fact that NVDA went up ~300× over the decade."),
        code(PRELUDE),
        md("## The clusters we test (LABELLED PROXY = NVDA earnings weeks)"),
        code("pd.DataFrame(dt.EVENTS, columns=['cluster_tag', 'anchor_date (PROXY)'])"),
        md("## The picture: mean cumulative abnormal return around the cluster\n\n"
           "Offset 0 is the cluster/earnings day. Left of zero is the *run-up*; right of zero "
           "is the *fade* window the folklore says to short. If the fade were real the line "
           "would turn **down** after 0. It doesn't."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#2ea44f', lw=2)\n"
            "ax.set_xlabel('trading days from the cluster (0 = earnings/upgrade week)')\n"
            "ax.set_ylabel('mean cumulative AR, NVDA − SPY (%)')\n"
            "ax.set_title('NVDA does NOT fade after the upgrade cluster — it drifts UP')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/39 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/39 |\n"
           f"| **2-week fade** | **{R['post_s_mean']:+.2f}%** | **{R['post_s_t']:+.2f}** | {R['post_s_hit']}/39 |\n"
           f"| 1-month fade | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/39 |\n\n"
           f"The 'fade' is **wrong-signed**: after the cluster NVDA *gains* "
           f"{R['post_s_mean']:+.2f}% vs SPY over two weeks (*t* = {R['post_s_t']:+.2f}). A "
           f"contrarian who shorted it would have lost that, 39 times over. Note the trap, "
           f"though: NVDA is one hand-picked mega-winner, so *random* 2-week windows already "
           f"beat SPY by ~+{R['pl_mean']:.1f}% — read every number against that, not zero."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The *fade the analyst cluster* trade is busted on NVDA — the stock **drifts up** "
           "after the cluster, so fading it loses money. The only real effect in the data is a "
           "single-name **post-earnings drift** sitting on top of NVDA's +2%/fortnight "
           "baseline — neither the claim, nor a generalizable edge. Verdict: **Weak signal "
           "(wrong-signed), Mirage tradability.** The quants' notebook has the placebo, the "
           "jackknife, the costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 784 — Analyst-Cluster — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`.\n\n"
           "**LABELLED PROXY:** cluster week = NVDA's real earnings week (post-print PT-hike "
           "wave). Confound to keep in mind throughout: this window also carries PEAD."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent clusters\n\n"
           "Each cluster is one independent event, so the unit is a one-sample *t* of the "
           "per-cluster abnormal return — **not** a daily panel (which would fake precision). "
           "The fade (`post_*`) is the tested short; a negative mean would support it."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the post-cluster move inside NVDA's luck cloud?\n\n"
           "For each event we redraw a random, non-cluster 2-week window on NVDA vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. The **placebo mean is "
           "already ~+2%** — that's the single-name selection baseline. The fade window sits "
           "in the far *right* tail of that null (left-tail p ≈ 0.985): abnormally *positive*, "
           "the opposite of a fade."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_s', k=10, tail='left')\n"
            "aapl, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = aapl.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(aapl.loc[common[p+10]]/aapl.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#2ea44f', lw=2, label=f\"observed fade {pl['obs']*100:+.2f}%\")\n"
            "ax.axvline(pl['placebo_mean']*100, color='#c0392b', lw=1.5, ls='--', label=f\"placebo mean {pl['placebo_mean']*100:+.2f}% (selection baseline)\")\n"
            "ax.set_xlabel('mean 2-week AR of random NVDA windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"post-cluster window vs NVDA luck cloud (left-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the upward drift one AI-boom quarter, or broad?"),
        code(
            "x = inc['post_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample fade t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window coincide. The "
           "*fade* (the tested trade) is a **loser** at every cost level — you'd be shorting a "
           "+4.7%/2wk window on one of the priciest names to borrow. The mirror long-the-drift "
           "survives costs but is plain PEAD on a single hand-picked winner."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('post_s','2wk fade'),('pre_s','2wk run-up')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is honest\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-cluster bump. At n = 39 the small-sample false-positive rate is "
           "near-nominal (|*t*| ≥ 2 on ~3/20 null seeds), and a planted bump is recovered "
           "monotonically."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=803+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=803, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=803, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The tested fade is not real — it is confidently *wrong-signed*: "
           "NVDA drifts **up** +4.76%/2wk (*t* = +2.13, placebo far-right p ≈ 0.985, jackknife "
           "[+1.84, +2.46]) after the cluster. What's present is an *upward* post-earnings "
           "drift, but on a single hand-picked mega-winner whose random windows already beat "
           "SPY by +2% — neither the claim nor generalizable. **Tradability: Mirage.** The "
           "fade loses ~+4.7%/event before borrow; the mirror long is one-name PEAD on a +2% "
           "selection baseline, no capacity or cross-sectional edge."),
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
