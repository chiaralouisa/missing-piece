#!/usr/bin/env python3
"""Render the ESMO poster to a print-ready PDF at true size.

Browsers are inconsistent about honouring a custom @page size from the print
dialog, and "Fit to page" silently rescales a 1.45 m board down to A4. Driving
the export headlessly removes both failure modes: the page size is passed
explicitly and the output is verified before it is handed over.

Usage:
    python scripts/poster_to_pdf.py                      # docs/poster/poster.html
    python scripts/poster_to_pdf.py --check              # verify an existing PDF
    python scripts/poster_to_pdf.py -i my.html -o my.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WIDTH_MM = 1450
HEIGHT_MM = 1150
MM_PER_PT = 25.4 / 72

#: Chromium ships with the sandbox image; Playwright's own download may be a
#: different build, so prefer whatever is actually installed.
_CHROMIUM_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]


def _executable() -> str | None:
    for path in _CHROMIUM_CANDIDATES:
        if Path(path).exists():
            return path
    return None  # let Playwright resolve its own


def render(src: Path, out: Path, width_mm: int, height_mm: int) -> Path:
    from playwright.sync_api import sync_playwright

    out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=_executable())
        page = browser.new_page()
        page.goto(src.resolve().as_uri())
        page.wait_for_timeout(4000)          # let webfonts settle
        page.emulate_media(media="print")
        page.wait_for_timeout(800)
        page.pdf(
            path=str(out),
            width=f"{width_mm}mm",
            height=f"{height_mm}mm",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        browser.close()
    return out


def check(pdf: Path, width_mm: int, height_mm: int) -> bool:
    """Confirm the PDF is exactly one page at the requested size."""
    data = pdf.read_bytes()
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", data)
    if not boxes:
        print(f"FAIL  {pdf}: no MediaBox found", file=sys.stderr)
        return False

    ok = True
    if len(boxes) != 1:
        # A stray second page is almost always leftover screen-preview layout
        # escaping into print; it wastes a whole sheet at this size.
        print(f"FAIL  {len(boxes)} pages, expected 1", file=sys.stderr)
        ok = False

    nums = [float(x) for x in boxes[0].split()]
    w_mm, h_mm = nums[2] * MM_PER_PT, nums[3] * MM_PER_PT
    if abs(w_mm - width_mm) > 1 or abs(h_mm - height_mm) > 1:
        print(f"FAIL  {w_mm:.1f} x {h_mm:.1f} mm, expected {width_mm} x {height_mm}", file=sys.stderr)
        ok = False

    print(f"{'OK   ' if ok else 'FAIL '} {pdf.name}: "
          f"{len(boxes)} page, {w_mm:.1f} x {h_mm:.1f} mm, "
          f"{pdf.stat().st_size / 1e6:.1f} MB")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=Path("docs/poster/poster.html"))
    ap.add_argument("-o", "--output", type=Path, default=Path("docs/poster/poster.pdf"))
    ap.add_argument("--width", type=int, default=WIDTH_MM, help="board width in mm")
    ap.add_argument("--height", type=int, default=HEIGHT_MM, help="board height in mm")
    ap.add_argument("--check", action="store_true", help="only verify the existing PDF")
    args = ap.parse_args(argv)

    if args.check:
        return 0 if check(args.output, args.width, args.height) else 1

    if not args.input.exists():
        print(f"no such file: {args.input}", file=sys.stderr)
        return 1

    render(args.input, args.output, args.width, args.height)
    return 0 if check(args.output, args.width, args.height) else 1


if __name__ == "__main__":
    raise SystemExit(main())
