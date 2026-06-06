"""One-command fetch of the source + supporting articles — from official sources.

We do **not** redistribute the PDFs (copyright stays with the authors); this
script downloads the openly available ones *for you* straight from arXiv, the
authors' university pages, the NY Fed and the publishers' open repositories.

Usage:
    python papers/download_papers.py

Two papers are behind a login wall (SSRN) and cannot be fetched automatically;
their links are printed at the end for a manual download.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (output filename, official direct-PDF URL, one-line description)
OPEN_PAPERS = [
    (
        "Knuteson_2019_Celebrating-Three-Decades-of-Worldwide-Stock-Market-Manipulation.pdf",
        "https://arxiv.org/pdf/1912.01708",
        "Knuteson (2019) — the figures we reproduce [arXiv 1912.01708]",
    ),
    (
        "Knuteson_2020_Strikingly-Suspicious-Overnight-and-Intraday-Returns.pdf",
        "https://arxiv.org/pdf/2010.01727",
        "Knuteson (2020) — the substantive overnight/intraday data paper [arXiv 2010.01727]",
    ),
    (
        "Knuteson_2022_They-Still-Havent-Told-You.pdf",
        "https://arxiv.org/pdf/2201.00223",
        "Knuteson (2022) — the attribution follow-up [arXiv 2201.00223]",
    ),
    (
        "Lou-Polk-Skouras_2019_A-Tug-of-War-Overnight-vs-Intraday-Expected-Returns.pdf",
        "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf",
        "Lou, Polk, Skouras (2019), J. Financial Economics [LSE copy]",
    ),
    (
        "Haghani-Ragulin-Dewey_2022_Night-Moves-Overnight-Drift.pdf",
        "https://elmwealth.com/wp-content/uploads/2026/04/Elm-Night-Moves-Overnight-Drift.pdf",
        "Haghani, Ragulin, Dewey (2022), 'Night Moves' [Elm Wealth]",
    ),
    (
        "Qiao-Dam_2020_Overnight-Return-Puzzle-and-T+1-Trading-Rule-China.pdf",
        "https://pure.rug.nl/ws/files/132141807/1_s2.0_S1386418120300033_main.pdf",
        "Qiao & Dam (2020), the Chinese T+1 case [U. Groningen open repo]",
    ),
    (
        "Boyarchenko-Larsen-Whelan_2020_The-Overnight-Drift_FedNY-SR917.pdf",
        "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf",
        "Boyarchenko, Larsen, Whelan (2020), 'The Overnight Drift' [NY Fed SR 917]",
    ),
    (
        "Cooper-Cliff-Gulen_2008_Return-Differences-Trading-vs-Non-Trading-Hours.pdf",
        "https://web.archive.org/web/20210530111718if_/https://www.krannert.purdue.edu/faculty/hgulen/Day_and_Night.pdf",
        "Cooper, Cliff, Gulen (2008), 'Like Night and Day' [Purdue copy via Wayback Machine]",
    ),
]

# Login-walled (SSRN bot wall); cannot be automated. No free copy exists
# anywhere — not on arXiv, the author's site, or any repository. Its data and
# figures are, however, fully covered by Knuteson's 2019 + 2020 papers above.
GATED = [
    (
        "Knuteson (2023), 'Nothing to See Here' — the central pamphlet",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084",
    ),
]


def _download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (overnight-alpha)"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 (trusted URLs)
            data = r.read()
    except Exception as exc:
        print(f"  ! failed: {exc}")
        return False
    if data[:4] != b"%PDF":
        print("  ! response was not a PDF (link moved / gated); skipped.")
        return False
    dest.write_bytes(data)
    print(f"  -> saved {dest.name}  ({len(data)//1024} KB)")
    return True


def main() -> int:
    try:  # Windows consoles default to cp1252 and mangle the em-dashes below.
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print("Fetching openly available articles from their official sources...\n")
    ok = 0
    for fname, url, desc in OPEN_PAPERS:
        print(f"- {desc}")
        if _download(url, HERE / fname):
            ok += 1

    print(f"\n{ok}/{len(OPEN_PAPERS)} open papers downloaded into {HERE}\n")
    print("Behind a login wall (SSRN) — no free copy exists; download manually:")
    for desc, url in GATED:
        print(f"  - {desc}\n      {url}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
