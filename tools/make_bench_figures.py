#!/usr/bin/env python3
"""Build the bench map — docs/bench_map.png — from the root README studies table.

The README table is the single source of truth for verdicts; this script parses
it (it never re-judges anything) and draws the whole bench as one 3x3 matrix:

    Signal (rows)        x  Tradability (columns)
    REAL / WEAK / NONE      INVESTABLE / FRAGILE / MIRAGE

Every published study is a numbered chip in its cell. Special stamps
(e.g. the pre-registered Gamma-Gospel study) are listed in a footnote rather
than forced into a cell. ``Mixed`` counts with ``Weak`` (same amber bucket,
see METHODOLOGY.md).

Deterministic — no data fetch, no randomness. Re-run whenever the table
changes; it works for any number of studies.

Usage:  python tools/make_bench_figures.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BENCH_MD = ROOT / "docs" / "bench.md"  # canonical family taxonomy ("mortality by family")
OUT = ROOT / "docs" / "bench_map.png"
OUT_JSON = ROOT / "docs" / "bench.json"  # feeds the interactive page (docs/index.html)

# ---------------------------------------------------------------- palette ---
GREEN = "#2ea44f"
AMBER = "#dab617"
RED = "#c0392b"
GREY = "#8b949e"
INK = "#24292f"
PAPER = "#fcfcfa"

SIGNALS = ["Real", "Weak", "None"]            # row order, top to bottom
TRADABILITIES = ["Investable", "Fragile", "Mirage"]  # column order, left to right
AXIS_COLOR = {"Real": GREEN, "Weak": AMBER, "None": RED,
              "Investable": GREEN, "Fragile": AMBER, "Mirage": RED}
# severity per stamp: 0 = green, 1 = amber, 2 = red (used to tint each cell)
SEVERITY = {"Real": 0, "Weak": 1, "None": 2,
            "Investable": 0, "Fragile": 1, "Mirage": 2}

# --------------------------------------------------------------- parsing ----
# A study row looks like:
#   | **[16](studies/16-storm-shy/)** | **Storm-Shy** | claim... | ![Real](...) | ![Investable](...) |
ROW_RE = re.compile(
    r"^\|\s*\*\*\[(?P<num>\d+)\]\((?P<href>studies/[^)]*)\)\*\*"  # | **[NN](studies/...)**
    r"\s*\|\s*\*\*(?P<name>.+?)\*\*"                      # | **Name**
    r"\s*\|\s*(?P<claim>.*?)"                             # | claim
    r"\s*\|\s*(?P<signal>!\[[^\]]+\][^|]*)"               # | ![Signal badge]
    r"\s*\|\s*(?P<trad>!\[[^\]]+\][^|]*)"                 # | ![Tradability badge]
    r"\s*\|\s*$"
)
BADGE_ALT_RE = re.compile(r"!\[([^\]]+)\]")


def _stamp(cell: str) -> str:
    """Alt text of the (first) shields badge in a table cell, e.g. 'Real'."""
    m = BADGE_ALT_RE.search(cell)
    return m.group(1).strip() if m else cell.strip()


def parse_readme(path: Path = README) -> list[dict]:
    """Every published study row in the README table, in table order."""
    studies = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        signal = _stamp(m["signal"])
        trad = _stamp(m["trad"])
        studies.append({
            "num": int(m["num"]),
            "name": m["name"].strip(),
            "claim": m["claim"].strip(),
            "signal": signal,
            "tradability": trad,
            "href": m["href"].strip(),
        })
    if not studies:
        sys.exit(f"No study rows matched in {path} — has the table format changed?")
    return studies


# A family row in docs/bench.md looks like:
#   | Calendar & seasonal — [01](../studies/01-...) [41](...) ... | 34 | 4 | 3 | 0 |
# The first cell holds the family name, an em-dash, then the member links.
FAMILY_ROW_RE = re.compile(r"^\|\s*(?P<fam>.+?)\s+—\s+(?P<rest>\[\d+\].+?)\s*\|\s*\d+\s*\|")


def parse_families(path: Path = BENCH_MD) -> dict[int, str]:
    """Map study number -> family, parsed from the bench.md taxonomy table.

    bench.md is the canonical (desk-maintained) taxonomy; this never invents
    families. Studies absent from the table fall back to 'Unclassified'.
    """
    fam_of: dict[int, str] = {}
    if not path.exists():
        return fam_of
    for line in path.read_text(encoding="utf-8").splitlines():
        m = FAMILY_ROW_RE.match(line)
        if not m:
            continue
        fam = m["fam"].strip()
        for num in re.findall(r"\[(\d+)\]", m["rest"]):
            fam_of[int(num)] = fam
    return fam_of


def bucket(studies: list[dict]):
    """Split studies into the 3x3 grid + the special-stamp leftovers."""
    grid = {(s, t): [] for s in SIGNALS for t in TRADABILITIES}
    special = []
    for st in studies:
        sig = "Weak" if st["signal"] == "Mixed" else st["signal"]  # same amber bucket
        if (sig, st["tradability"]) in grid:
            grid[(sig, st["tradability"])].append(st)
        else:
            special.append(st)  # e.g. Pre-reg / Pre-reg
    return grid, special


# --------------------------------------------------------------- drawing ----
def _hex2rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _lerp(c1, c2, t):
    a, b = _hex2rgb(c1), _hex2rgb(c2)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def cell_color(sig: str, trad: str):
    """Green -> amber -> red along the mean severity of the two stamps."""
    s = (SEVERITY[sig] + SEVERITY[trad]) / 2  # 0..2
    return _lerp(GREEN, AMBER, s) if s <= 1 else _lerp(AMBER, RED, s - 1)


def _chip_layout(n: int):
    """(n_cols, n_rows) for n chips in one cell — at most 6 per row."""
    cols = min(6, max(1, math.ceil(math.sqrt(1.8 * n))))
    return cols, math.ceil(n / cols)


def draw(grid, special, total: int, out: Path = OUT) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10.2), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(-0.92, 3.06)
    ax.set_ylim(-0.62, 4.10)
    ax.set_aspect("equal")
    ax.axis("off")

    pad = 0.035  # gap between cells

    for i, sig in enumerate(SIGNALS):            # rows, top to bottom
        for j, trad in enumerate(TRADABILITIES):  # columns, left to right
            x0, y0 = j + pad, (2 - i) + pad       # cell lower-left
            w = h = 1 - 2 * pad
            members = grid[(sig, trad)]
            col = cell_color(sig, trad)
            empty = not members

            box = FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle="round,pad=0,rounding_size=0.045",
                facecolor=(*col, 0.10 if not empty else 0.04),
                edgecolor=(*col, 0.85 if not empty else 0.30),
                linewidth=1.6, linestyle="-" if not empty else (0, (4, 3)),
            )
            ax.add_patch(box)

            # count, top-right corner of the cell
            ax.text(x0 + w - 0.055, y0 + h - 0.058, str(len(members)),
                    ha="right", va="top", fontsize=21, fontweight="bold",
                    color=col if not empty else GREY, alpha=0.92)

            if empty:
                ax.text(x0 + w / 2, y0 + h / 2, "—", ha="center", va="center",
                        fontsize=16, color=GREY, alpha=0.55)
                continue

            # numbered chips, centred in the cell (leaving the count corner room)
            ncols, nrows = _chip_layout(len(members))
            r = min(0.072, 0.40 / max(ncols, nrows))   # chip radius
            step_x = min(2.55 * r, (w - 0.16) / max(ncols - 1, 1)) if ncols > 1 else 0
            step_y = min(2.55 * r, (h - 0.30) / max(nrows - 1, 1)) if nrows > 1 else 0
            cx0 = x0 + w / 2 - step_x * (ncols - 1) / 2
            cy0 = y0 + (h - 0.10) / 2 + step_y * (nrows - 1) / 2
            for k, st in enumerate(sorted(members, key=lambda s: s["num"])):
                cx = cx0 + (k % ncols) * step_x
                cy = cy0 - (k // ncols) * step_y
                ax.add_patch(Circle((cx, cy), r, facecolor=col,
                                    edgecolor="white", linewidth=1.1, zorder=3))
                ax.text(cx, cy, f"{st['num']:02d}", ha="center", va="center",
                        fontsize=max(7.5, 118 * r), fontweight="bold",
                        color="white", zorder=4)

    # ----- axis headers -------------------------------------------------
    for j, trad in enumerate(TRADABILITIES):
        ax.text(j + 0.5, 3.20, trad.upper(), ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=AXIS_COLOR[trad])
    ax.text(1.5, 3.44, "Tradability — does it survive costs, capacity & scale?",
            ha="center", va="bottom", fontsize=11.5, color=INK, style="italic")

    for i, sig in enumerate(SIGNALS):
        ax.text(-0.16, (2 - i) + 0.5, sig.upper(), ha="center", va="center",
                fontsize=15, fontweight="bold", color=AXIS_COLOR[sig],
                rotation=90)
    ax.text(-0.62, 1.5, "Signal — is the effect statistically real?",
            ha="center", va="center", fontsize=11.5, color=INK,
            style="italic", rotation=90)

    # ----- title & footnotes --------------------------------------------
    ax.text(1.5, 4.07, f"{total} famous trading ideas, one protocol",
            ha="center", va="top", fontsize=20, fontweight="bold", color=INK)
    ax.text(1.5, 3.88, "Each chip is a study — its number in the README table. "
                       "Same test bench, two stamps each.",
            ha="center", va="top", fontsize=11.5, color="#57606a")

    notes = []
    if special:
        det = ", ".join(f"{s['num']:02d} {s['name']} ({_pretty(s)})" for s in special)
        notes.append(f"Not on the grid: {det}.")
    notes.append("Study 03's “Mixed” signal counts with Weak (same amber bucket).")
    for k, note in enumerate(notes):
        ax.text(-0.80, -0.26 - 0.155 * k, note, ha="left", va="top",
                fontsize=9.5, color=GREY)
    ax.text(3.04, -0.26, "github.com/Guillain-RDCDE/Open-Alpha-Lab",
            ha="right", va="top", fontsize=9.5, color=GREY)

    fig.savefig(out, dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.22)
    plt.close(fig)


def _pretty(st: dict) -> str:
    if st["signal"] == st["tradability"]:
        return st["signal"].lower().replace("--", "-")
    return f"{st['signal']}/{st['tradability']}"


# ----------------------------------------------------------------- main -----
def main() -> None:
    studies = parse_readme()
    grid, special = bucket(studies)
    total = len(studies)

    # counts, printed so docs/bench.md can quote exact numbers
    print(f"Parsed {total} studies from {README.name}\n")
    print(f"{'':>6}  " + "".join(f"{t:>12}" for t in TRADABILITIES) + f"{'row':>8}")
    for sig in SIGNALS:
        row = [len(grid[(sig, t)]) for t in TRADABILITIES]
        print(f"{sig:>6}  " + "".join(f"{n:>12}" for n in row) + f"{sum(row):>8}")
    col_tot = [sum(len(grid[(s, t)]) for s in SIGNALS) for t in TRADABILITIES]
    print(f"{'col':>6}  " + "".join(f"{n:>12}" for n in col_tot)
          + f"{sum(col_tot):>8}")
    for s in special:
        print(f"\nSpecial stamp: {s['num']:02d} {s['name']} - "
              f"{s['signal']} / {s['tradability']}")
    print("\nPer cell:")
    for sig in SIGNALS:
        for t in TRADABILITIES:
            nums = ", ".join(f"{m['num']:02d}" for m in
                             sorted(grid[(sig, t)], key=lambda m: m["num"]))
            print(f"  {sig:>4} x {t:<10} ({len(grid[(sig, t)]):>2}): {nums or '-'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    draw(grid, special, total)
    print(f"\nWrote {OUT.relative_to(ROOT)}")

    write_json(studies, grid, special, total)
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")


def write_json(studies, grid, special, total: int, out: Path = OUT_JSON) -> None:
    """Dump the parsed bench as JSON for the interactive page (docs/index.html).

    Same single source of truth as the PNG: it never re-judges, it just serialises
    what the README table already says. Cell counts are precomputed for convenience;
    the page can also recompute them from ``studies``.
    """
    counts = {sig: {trad: len(grid[(sig, trad)]) for trad in TRADABILITIES}
              for sig in SIGNALS}
    fam_of = parse_families()
    enriched = [{**st, "family": fam_of.get(st["num"], "Unclassified")}
                for st in studies]
    payload = {
        "total": total,
        "repo": "https://github.com/Guillain-RDCDE/Open-Alpha-Lab",
        "signals": SIGNALS,
        "tradabilities": TRADABILITIES,
        "counts": counts,
        "studies": enriched,
        "special": [{**st, "family": fam_of.get(st["num"], "Unclassified")}
                    for st in special],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
