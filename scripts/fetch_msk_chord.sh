#!/usr/bin/env bash
# Fetch MSK-CHORD (msk_chord_2024) and the MSK-IMPACT gene panel definitions.
#
# Run this on a machine with outbound access to cbioportal.org. Some managed
# environments (including Claude Code web sessions behind a restrictive egress
# policy) cannot reach the cBioPortal hosts at all; in that case run this
# locally and copy the resulting directory across.
#
# Usage: scripts/fetch_msk_chord.sh [DEST_DIR]
set -euo pipefail

DEST="${1:-data/msk_chord_2024}"
STUDY_URL="https://cbioportal-datahub.s3.amazonaws.com/msk_chord_2024.tar.gz"
PANEL_REPO="https://github.com/cBioPortal/datahub.git"

mkdir -p "$(dirname "$DEST")"

echo "==> downloading msk_chord_2024"
if [ ! -f "${DEST}.tar.gz" ]; then
  curl -fL --retry 4 --retry-delay 2 -o "${DEST}.tar.gz" "$STUDY_URL"
fi
mkdir -p "$DEST"
tar -xzf "${DEST}.tar.gz" -C "$(dirname "$DEST")"

echo "==> checking for Git LFS pointer files"
if head -c 40 "${DEST}/data_mutations.txt" | grep -q "git-lfs"; then
  echo "ERROR: ${DEST}/data_mutations.txt is a Git LFS pointer, not data." >&2
  echo "       Install git-lfs and re-download, or fetch the study tarball" >&2
  echo "       directly from https://www.cbioportal.org/datasets" >&2
  exit 1
fi

echo "==> fetching MSK-IMPACT gene panel definitions"
# The study tarball ships the panels it uses; if IMPACT341/IMPACT505 are not
# both present, pull them from the datahub reference data (needs git-lfs).
NEED_PANELS=0
for panel in impact341 impact505; do
  if ! ls "${DEST}"/data_gene_panel_*"${panel}"* >/dev/null 2>&1 \
     && ! ls "${DEST}"/data_gene_panel_*"${panel^^}"* >/dev/null 2>&1; then
    NEED_PANELS=1
  fi
done

if [ "$NEED_PANELS" -eq 1 ]; then
  TMP="$(mktemp -d)"
  git clone --depth 1 --filter=blob:none --sparse "$PANEL_REPO" "$TMP/datahub"
  git -C "$TMP/datahub" sparse-checkout set reference_data/gene_panels
  git -C "$TMP/datahub" lfs pull --include "reference_data/gene_panels/*" || {
    echo "ERROR: git lfs pull failed; install git-lfs (https://git-lfs.com)." >&2
    exit 1
  }
  mkdir -p "${DEST}/gene_panels"
  cp "$TMP/datahub"/reference_data/gene_panels/data_gene_panel_impact*.txt \
     "${DEST}/gene_panels/"
  rm -rf "$TMP"
fi

echo "==> verifying panel sizes"
python3 - "$DEST" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from missing_piece.panels import PanelPair, load_panels  # noqa: E402

dest = Path(sys.argv[1])
for directory in (dest / "gene_panels", dest):
    try:
        panels = load_panels(directory)
    except FileNotFoundError:
        continue
    ids = {k.upper() for k in panels}
    if {"IMPACT341", "IMPACT505"} <= ids:
        pair = PanelPair(panels["IMPACT341"], panels["IMPACT505"])
        print(pair.describe())
        expected = {"IMPACT341": 341, "IMPACT505": 505}
        for name, n in expected.items():
            got = len(panels[name])
            flag = "OK" if got == n else f"WARNING expected {n}"
            print(f"  {name}: {got} genes [{flag}]")
        sys.exit(0)
print("WARNING: IMPACT341/IMPACT505 panel files not found", file=sys.stderr)
sys.exit(1)
PY

echo
echo "Done. Run the real-data experiment with:"
echo "  python -m missing_piece run -c configs/msk_chord_nsclc.yaml"
