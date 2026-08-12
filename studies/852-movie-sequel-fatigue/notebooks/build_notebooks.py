"""Generate the two narrative notebooks for Study 852 (Movie-Sequel Fatigue).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
DIS/CMCSA/PARA/SPY tapes under ../_cache/ (fetching once on a cache miss) and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The heavy placebos
are shown live at reduced draw counts (identical shape, faster); the synthetic positive
control runs anywhere with no network. Fingerprint 5a2c01b12a9b.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (DIS+CMCSA+PARA+SPY,
# yfinance, 2003-01-02 -> 2026-06-30; 43 of 46 entries resolved). Fingerprint 5a2c01b12a9b.
R = dict(
    n_events=46, n_included=43, fp="5a2c01b12a9b", dis_rows=5910,
    car_mean=-0.734, car_t=-1.831, car_nwt=-2.056, up_k=17, up_n=43, up_lo=26.4, up_hi=54.4,
    raw_slope=-0.216, raw_t=-1.196, raw_r=-0.184,
    dem_slope=-0.902, dem_t=-3.054, dem_r=-0.430,
    era_early_slope=-0.405, era_early_t=-0.952, n_early=25,
    era_late_slope=-3.504, era_late_t=-8.853, n_late=18,
    pairs=29, ar1_slope=0.146, ar1_t=0.810,
    down_mean=-1.927, up_mean=-0.730, welch_t=-1.364, n_down=14, n_up=15,
    perm_dem_pleft=0.0000, perm_dem_ptwo=0.0000, perm_dem_nullsd=0.226,
    perm_raw_pleft=0.116, perm_raw_ptwo=0.237,
    rd_obs=-0.734, rd_pmean=-0.083, rd_psd=0.480, rd_ptwo=0.114,
    tm_n=14, tm_gross=1.927, tm_tg=2.95, tm_net5=1.821, tm_t5=2.79, tm_net10=1.721, tm_t10=2.63,
    null_mean_t=-0.01, null_sd_t=1.25, null_hits=2,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sequel_fatigue import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching DIS + CMCSA + PARA + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
cars = st.build_event_cars(prices)
inc = cars[cars["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} franchise entries resolved")
"""


def build_curious():
    cells = [
        md("# Study 852 — Movie-Sequel Fatigue 🎬\n\n"
           "*For the curious.* Everyone *knows* franchises get tired: *Ant-Man: Quantumania*, "
           "*The Marvels*, *Indiana Jones 5*, *Fast X* — later sequels open softer, reviews "
           "sour, the magic fades. If that's real, the reflex trade is obvious: the **studio** "
           "should react worse to sequel N than to sequel N-1. We put **46 franchise entries** "
           "(Marvel, Star Wars, Fast & Furious, Jurassic World, Pirates, Transformers) on the "
           "stand and asked the tape: **does the studio's stock actually flinch harder as the "
           "franchise ages?** The answer is a shrug with one asterisk."),
        md("## The claim, and why it's a clean test\n\n"
           "A wide release opens on a known Friday; the *weekend* box-office is public by "
           "Sunday/Monday. So the first session at which 'how it opened' is common knowledge is "
           "the **Monday after opening** — we anchor the studio-reaction window there (base = "
           "the opening-Friday close), measure the studio's **abnormal** return (studio − SPY, "
           "total-return), and index each film by its **sequel number**. Zero look-ahead."),
        code(PRELUDE),
        md("## The franchise calendar we test (hardcoded from Box Office Mojo / studio releases)"),
        code("dt.events_frame()[['franchise','title','seq','opening','ticker']]"),
        md("## The picture: mean studio reaction by sequel number\n\n"
           "If 'fatigue' is real, this should slope **down** — later sequels drawing a worse "
           "opening reaction. It does tilt down, but shallowly, and the scatter is a cloud."),
        code(
            "g = inc.groupby('seq')['car'].mean() * 100\n"
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))\n"
            "ax[0].bar(g.index, g.values, color='#c0392b', alpha=0.85)\n"
            "ax[0].axhline(0, color='0.6', lw=0.8)\n"
            "ax[0].set_xlabel('sequel number'); ax[0].set_ylabel('mean opening-reaction CAR (%)')\n"
            "ax[0].set_title('later sequels react a touch worse — on average')\n"
            "sl = st.fatigue_slope(inc, demean=False)\n"
            "ax[1].scatter(inc['seq'], inc['car']*100, color='#8b949e', alpha=0.7)\n"
            "xs = np.array([inc['seq'].min(), inc['seq'].max()])\n"
            "ax[1].plot(xs, (sl['intercept'] + sl['slope']*xs)*100, color='#c0392b', lw=2,\n"
            "           label=f\"raw slope {sl['slope']*100:+.2f}%/seq (t={sl['t']:+.2f})\")\n"
            "ax[1].axhline(0, color='0.6', lw=0.8); ax[1].legend()\n"
            "ax[1].set_xlabel('sequel number'); ax[1].set_ylabel('opening-reaction CAR (%)')\n"
            "ax[1].set_title('...but the raw fit is not significant')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| what | value |\n|---|--:|\n"
           f"| mean opening reaction (all {R['n_included']}) | **{R['car_mean']:+.2f}%** (NW *t* = {R['car_nwt']:+.2f}) |\n"
           f"| raw fatigue slope (CAR vs sequel #) | **{R['raw_slope']:+.2f}%/seq** (*t* = {R['raw_t']:+.2f}) |\n"
           f"| within-franchise slope (fixed effect) | **{R['dem_slope']:+.2f}%/seq** (*t* = {R['dem_t']:+.2f}) |\n\n"
           f"Studios do dip a mild **{abs(R['car_mean']):.2f}%** in the week a sequel opens, and "
           f"the fatigue slope points the *right way* (negative). But the *natural* reading — "
           f"'CAR declines with sequel number' — is **insignificant** (*t* = {R['raw_t']:+.2f}). "
           f"It only becomes significant once you add a franchise fixed effect (*t* = "
           f"{R['dem_t']:+.2f})... and that's where the story gets slippery."),
        code(
            "for label, dem in [('raw', False), ('franchise fixed effect', True)]:\n"
            "    s = st.fatigue_slope(inc, demean=dem)\n"
            "    print(f'{label:<24s} slope={s[\"slope\"]*100:+.3f}%/seq  t={s[\"t\"]:+.3f}  r={s[\"r\"]:+.3f}')"
        ),
        md(f"## The asterisk: it's a recent-era artifact\n\n"
           f"Split the within-franchise slope by era and the 'signal' evaporates before 2018:\n\n"
           f"| era | n | slope | *t* |\n|---|--:|--:|--:|\n"
           f"| early (< 2018) | {R['n_early']} | {R['era_early_slope']:+.2f}%/seq | {R['era_early_t']:+.2f} |\n"
           f"| late (≥ 2018) | {R['n_late']} | {R['era_late_slope']:+.2f}%/seq | {R['era_late_t']:+.2f} |\n\n"
           f"All of the 'fatigue' lives in the post-2018 window (a handful of 2-entry "
           f"franchises), and even the mean reaction sits **inside** a random-date placebo cloud "
           f"(p = {R['rd_ptwo']:.2f}). So: the right sign, one significant-but-fragile "
           f"specification, nothing you'd bet on."),
        code(
            "era = st.era_slopes(inc, demean=True)\n"
            "for k in ('early','late'):\n"
            "    e = era[k]; print(f'{k:<6s} n={e[\"n\"]:2d}  slope={e[\"slope\"]*100:+.3f}%/seq  t={e[\"t\"]:+.3f}')"
        ),
        md("## So what?\n\n"
           "'Franchise fatigue' is *directionally* real in studio reactions — later sequels do "
           "draw a slightly worse opening move — but it's insignificant in its natural form, "
           "reaches significance only in a fixed-effect specification that is entirely a "
           "post-2018 artifact, and the average reaction is indistinguishable from random dates. "
           "**Verdict: Weak signal, Fragile tradability.** The quants' notebook has the two "
           "placebos, the persistence test, the costed short, and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 852 — Movie-Sequel Fatigue — for the quants 🔬\n\n"
           "The full battery: the raw vs franchise-FE fatigue slope, the two-era robustness cut, "
           "a sequel-number label-permutation placebo, a random-date placebo, the within-franchise "
           "AR(1) persistence, a costed short-the-tired-sequel leg, and a 20-seed synthetic "
           f"positive control. Everything offline once cached; fingerprint `{R['fp']}`."),
        code(PRELUDE),
        md("## 1. The mean reaction + the fatigue slope (H1)\n\n"
           "Each film reduces to one number: the studio's abnormal return (studio − SPY) over "
           "`[anchor−1 .. anchor+3]`. The unit is a cross-event *t* (independent releases), and "
           "the fatigue slope is an OLS of CAR on sequel number — raw, then with a franchise "
           "fixed effect (the *within*-franchise tilt)."),
        code(
            "d0 = st.day0_stats(inc, 'car')\n"
            "print(f'mean CAR {d0[\"mean\"]*100:+.3f}%  t={d0[\"t\"]:+.3f}  NW-t={d0[\"t_nw\"]:+.3f}  up {d0[\"up_k\"]}/{d0[\"up_n\"]}')\n"
            "for label, dem in [('raw', False), ('franchise-FE', True)]:\n"
            "    s = st.fatigue_slope(inc, demean=dem)\n"
            "    print(f'{label:<12s} slope={s[\"slope\"]*100:+.4f}%/seq  t={s[\"t\"]:+.3f}  r={s[\"r\"]:+.3f}')"
        ),
        md("## 2. Two-era robustness — the fixed-effect slope is a post-2018 artifact\n\n"
           "A 'Real' stamp needs the sign to hold across sub-eras. It does not: the "
           "within-franchise slope is a dead ~−0.95 *t* before 2018 and an implausibly tight "
           "−8.9 *t* after (a few 2-entry franchises)."),
        code(
            "era = st.era_slopes(inc, demean=True)\n"
            "for k in ('early','late'):\n"
            "    e = era[k]; print(f'{k:<6s} n={e[\"n\"]:2d}  slope={e[\"slope\"]*100:+.4f}%/seq  t={e[\"t\"]:+.3f}  r={e[\"r\"]:+.3f}')"
        ),
        md("## 3. Placebo A — permute the sequel-number labels (H1 falsification)\n\n"
           "Shuffle the sequel numbers across events (breaking any seq→CAR link) and recompute "
           "the slope. The franchise-FE slope is outside the shuffle cloud (a real *within*-"
           "franchise ordering); the **raw** slope sits comfortably inside it — the significance "
           "lives only in the fixed-effect specification. (Shown live at 1,500 draws; "
           f"docs/results.md uses 5,000 → raw p_two = {R['perm_raw_ptwo']:.2f}.)"),
        code(
            "pl = st.permute_slope_pvalue(inc, demean=True, n_perm=1500)\n"
            "plr = st.permute_slope_pvalue(inc, demean=False, n_perm=1500)\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(plr['draws']*100, bins=50, color='#8b949e', alpha=0.85, label='raw null')\n"
            "ax.axvline(plr['obs_slope']*100, color='#c0392b', lw=2, label=f\"raw obs {plr['obs_slope']*100:+.2f}% (p={plr['p_two']:.2f})\")\n"
            "ax.axvline(pl['obs_slope']*100, color='#2ea44f', lw=2, ls='--', label=f\"FE obs {pl['obs_slope']*100:+.2f}% (p={pl['p_two']:.3f})\")\n"
            "ax.set_xlabel('permuted fatigue slope (%/sequel)'); ax.set_ylabel('count')\n"
            "ax.set_title('raw slope is inside the label-shuffle cloud; only the FE version escapes'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('raw:', {k: round(v,4) for k,v in plr.items() if isinstance(v,float)})"
        ),
        md("## 4. Placebo B — random pseudo-event dates\n\n"
           "For each event draw a random, non-event same-length window on the SAME studio ticker "
           "vs SPY. If the observed mean CAR is inside this cloud, the sequel reactions are "
           "indistinguishable from random dates."),
        code(
            "rd = st.random_date_placebo(prices, inc, n_seeds=10, n_draws_per_seed=200)\n"
            "print(f\"observed mean CAR {rd['obs']*100:+.3f}%  vs random-window mean {rd['placebo_mean']*100:+.3f}% \"\n"
            "      f\"(sd {rd['placebo_sd']*100:.3f}%)  p_two = {rd['p_two']:.3f}  over {rd['n_draws']:,} draws\")"
        ),
        md(f"## 5. Persistence (H2) — does a down sequence drag the next?\n\n"
           f"Within each franchise, pair consecutive entries and (a) regress N's CAR on N-1's "
           f"CAR, (b) split N's CAR by whether N-1 was down. Both point the right way; both are "
           f"insignificant (AR(1) *t* = {R['ar1_t']:+.2f}, down-vs-up Welch *t* = "
           f"{R['welch_t']:+.2f})."),
        code(
            "per = st.fatigue_persistence(inc)\n"
            "print(f'pairs={per[\"n_pairs\"]}  AR(1) slope={per[\"ar1_slope\"]:+.4f}  t={per[\"ar1_t\"]:+.3f}')\n"
            "print(f'next-CAR | prev-down {per[\"down_mean\"]*100:+.3f}% (n={per[\"n_down\"]})  vs  prev-up {per[\"up_mean\"]*100:+.3f}% (n={per[\"n_up\"]})  Welch t={per[\"welch_t\"]:+.3f}')"
        ),
        md("## 6. Tradability — the short-the-fatigued-sequel leg, net of costs\n\n"
           "Short the studio at the anchor of an entry whose predecessor reacted negatively; hold "
           "the reaction window. It *nominally* clears costs — but it fires **14 times in 20 "
           "years**, is conditioned in-sample on the (insignificant) persistence above, and lives "
           "in the same post-2018 window. What the *t* really tests is 'these 14 entries dipped', "
           "which Placebo B already puts inside the noise. **Fragile, not deployable.**"),
        code(
            "for c in (5.0, 10.0):\n"
            "    tm = st.fatigue_timer(inc, cost_bps=c)\n"
            "    print(f'@{c:>4.0f}bps  n={tm[\"n\"]}  gross {tm[\"gross_mean\"]*100:+.3f}% (t={tm[\"t_gross\"]:+.2f})  net {tm[\"net_mean\"]*100:+.3f}% (t={tm[\"t_net\"]:+.2f})  [cost {tm[\"cost_bps\"]:.1f} bps/rt]')"
        ),
        md("## 7. Synthetic positive control — the detector works, the null is quiet\n\n"
           "The fatigue-slope detector must stay quiet on a planted-null world and recover a "
           "planted fatigue slope monotonically. At ~41 events with a franchise fixed effect the "
           "slope *t* is noisy — which frames why the real, shallow tilt shows up only as a whiff."),
        code(
            "null_t = np.array([st.synthetic_detect(edge=0.0, seed=852+s)['t'] for s in range(20)])\n"
            "print(f'null: mean slope-t={null_t.mean():+.2f} sd={null_t.std(ddof=1):.2f}  |t|>=2 in {(np.abs(null_t)>=2).sum()}/20')\n"
            "edges = np.linspace(0, 0.014, 8)\n"
            "ts = [st.synthetic_detect(edge=e, seed=852)['t'] for e in edges]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(edges, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(-2, color='0.6', ls='--'); ax.set_xlabel('planted fatigue edge'); ax.set_ylabel('detector slope-t')\n"
            "ax.set_title('planted fatigue is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The fatigue slope points the folklore's way but is insignificant "
           "raw (*t* = −1.20), significant only with a franchise fixed effect (*t* = −3.05) that "
           "lives entirely post-2018 (early *t* = −0.95 vs late *t* = −8.85, failing the cross-era "
           "bar), the mean reaction is inside the random-date cloud (p = 0.11), and persistence is "
           "insignificant. **Tradability: Fragile.** The short nominally clears costs (14 fires, "
           "+1.82% net @ 5 bps, *t* = 2.79) but rests on in-sample conditional selection of an "
           "insignificant persistence signal in a recent-era window — it will not survive out of "
           "sample."),
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
