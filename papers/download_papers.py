"""One-command fetch of the source articles — from their OFFICIAL sources.

We do **not** redistribute the PDFs: the arXiv papers are under arXiv's
``nonexclusive-distrib/1.0`` licence (the author keeps copyright and grants only
arXiv the right to distribute), and the SSRN paper is access-gated. So this
script downloads them *for you* straight from arXiv into this folder, leaving
copyright with the author and pointing everyone at the primary source.

Usage:
    python papers/download_papers.py

The SSRN paper ("Nothing to See Here", 4619084) requires a manual download
(login wall); the link is printed at the end.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (arXiv id, official abstract page, output filename)
ARXIV_PAPERS = [
    (
        "1912.01708",
        "https://arxiv.org/abs/1912.01708",
        "Knuteson_2019_Celebrating-Three-Decades-of-Worldwide-Stock-Market-Manipulation.pdf",
    ),
    (
        "2201.00223",
        "https://arxiv.org/abs/2201.00223",
        "Knuteson_2022_They-Still-Havent-Told-You.pdf",
    ),
]

SSRN = ("4619084", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084")


def _download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (overnight-alpha)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (trusted arXiv URL)
            data = r.read()
    except Exception as exc:  # network issues shouldn't crash the whole run
        print(f"  ! failed: {exc}")
        return False
    if not data[:4] == b"%PDF":
        print("  ! response was not a PDF (network / mirror issue); skipped.")
        return False
    dest.write_bytes(data)
    print(f"  -> saved {dest.name}  ({len(data)//1024} KB)")
    return True


def main() -> int:
    print("Fetching source articles from arXiv (official source)...\n")
    ok = 0
    for arxiv_id, abs_url, fname in ARXIV_PAPERS:
        print(f"[{arxiv_id}] {abs_url}")
        if _download(f"https://arxiv.org/pdf/{arxiv_id}", HERE / fname):
            ok += 1

    print(f"\n{ok}/{len(ARXIV_PAPERS)} arXiv papers downloaded into {HERE}")
    print(
        "\nThe central pamphlet, *Nothing to See Here: How to Say It When You "
        f"Need to* (SSRN {SSRN[0]}), is behind a login wall — download it manually:\n"
        f"    {SSRN[1]}\n"
        "Its figures are reproduced in the 1912.01708 paper above."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
