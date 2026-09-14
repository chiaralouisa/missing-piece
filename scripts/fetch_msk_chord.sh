#!/usr/bin/env bash
# Fetch MSK-CHORD (msk_chord_2024) and the MSK-IMPACT gene panel definitions.
#
# Run this on a machine with outbound access to cbioportal.org. Managed
# environments behind a restrictive egress policy (including Claude Code web
# sessions) cannot reach the cBioPortal hosts at all; there, run this locally
# and copy the resulting directory across.
#
# Usage:
#   scripts/fetch_msk_chord.sh [DEST_DIR]        # default: data/msk_chord_2024
#   scripts/fetch_msk_chord.sh --check [DEST]    # verify an existing download
#
# Portability: kept to POSIX-ish bash so it runs on macOS's bash 3.2 as well as
# Linux. No ${var^^}, no mapfile, no associative arrays.
set -euo pipefail

STUDY_NAME="msk_chord_2024"
STUDY_URL="https://cbioportal-datahub.s3.amazonaws.com/${STUDY_NAME}.tar.gz"
PANEL_REPO="https://github.com/cBioPortal/datahub.git"

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
  CHECK_ONLY=1
  shift
fi
DEST="${1:-data/${STUDY_NAME}}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "==> $*"; }

upper() { printf '%s' "$1" | tr '[:lower:]' '[:upper:]'; }

# --- prerequisites ----------------------------------------------------------
# Checked up front with actionable messages: a failure 20 minutes into a
# multi-gigabyte download because git-lfs is missing is a bad way to find out.
check_prereqs() {
  command -v curl >/dev/null 2>&1 || die "curl not found; install it and retry."
  command -v tar  >/dev/null 2>&1 || die "tar not found; install it and retry."
  command -v python3 >/dev/null 2>&1 || die "python3 not found; install Python 3.10+."
  if ! command -v git >/dev/null 2>&1; then
    echo "WARNING: git not found. If the study tarball does not ship the" >&2
    echo "         IMPACT341/IMPACT505 panels, this script cannot fetch them." >&2
  elif ! git lfs version >/dev/null 2>&1; then
    echo "WARNING: git-lfs not installed (https://git-lfs.com)." >&2
    echo "         The gene-panel files are LFS-tracked; without it they arrive" >&2
    echo "         as ~130-byte pointer stubs and the loader will refuse them." >&2
  fi
}

# --- verification -----------------------------------------------------------
is_lfs_pointer() {
  [ -f "$1" ] && head -c 40 "$1" 2>/dev/null | grep -q "git-lfs"
}

verify() {
  local dest="$1" problems=0

  [ -d "$dest" ] || die "$dest does not exist. Run without --check to download."

  note "verifying $dest"
  for f in data_clinical_sample.txt data_mutations.txt data_cna.txt \
           data_gene_panel_matrix.txt; do
    if [ ! -f "${dest}/${f}" ]; then
      echo "  MISSING  ${f}" >&2
      problems=$((problems + 1))
    elif is_lfs_pointer "${dest}/${f}"; then
      echo "  LFS STUB ${f}  <- not real data; install git-lfs and re-download" >&2
      problems=$((problems + 1))
    else
      # du -h is portable; ls -lh formatting is not.
      echo "  ok       ${f}  ($(du -h "${dest}/${f}" | cut -f1))"
    fi
  done

  local found_panels=0
  for dir in "${dest}/gene_panels" "${dest}"; do
    [ -d "$dir" ] || continue
    for panel in impact341 impact505; do
      local up
      up="$(upper "$panel")"
      if ls "${dir}"/data_gene_panel_*"${panel}"*.txt >/dev/null 2>&1 \
         || ls "${dir}"/data_gene_panel_*"${up}"*.txt  >/dev/null 2>&1; then
        found_panels=$((found_panels + 1))
      fi
    done
    [ "$found_panels" -ge 2 ] && break
    found_panels=0
  done
  if [ "$found_panels" -lt 2 ]; then
    echo "  MISSING  IMPACT341 and/or IMPACT505 gene-panel files" >&2
    problems=$((problems + 1))
  else
    echo "  ok       IMPACT341 + IMPACT505 gene panels"
  fi

  [ "$problems" -eq 0 ] || die "$problems problem(s) above; see messages."
  note "download looks complete"
}

if [ "$CHECK_ONLY" -eq 1 ]; then
  verify "$DEST"
  exit 0
fi

check_prereqs

# --- download ---------------------------------------------------------------
PARENT="$(dirname "$DEST")"
mkdir -p "$PARENT"
TARBALL="${PARENT}/${STUDY_NAME}.tar.gz"

if [ -f "$TARBALL" ]; then
  note "reusing existing $TARBALL ($(du -h "$TARBALL" | cut -f1))"
else
  note "downloading ${STUDY_NAME} (several GB, this takes a while)"
  # -C - resumes a partial download; write to .part so an interrupted run never
  # leaves a truncated file that looks complete to the next invocation.
  curl -fL --retry 4 --retry-delay 2 -C - --progress-bar \
       -o "${TARBALL}.part" "$STUDY_URL" \
    || die "download failed. If this is a 403, your network blocks the cBioPortal
       datahub; download the study manually from https://www.cbioportal.org/datasets"
  mv "${TARBALL}.part" "$TARBALL"
fi

note "extracting"
tar -xzf "$TARBALL" -C "$PARENT"

# The tarball extracts to its own study-named directory. Honour a DEST with a
# different basename rather than silently leaving the files somewhere else.
EXTRACTED="${PARENT}/${STUDY_NAME}"
if [ "$EXTRACTED" != "$DEST" ] && [ -d "$EXTRACTED" ]; then
  note "moving ${EXTRACTED} -> ${DEST}"
  rm -rf "$DEST"
  mv "$EXTRACTED" "$DEST"
fi
[ -d "$DEST" ] || die "extraction did not produce $DEST; inspect $TARBALL"

# --- gene panels ------------------------------------------------------------
note "checking for IMPACT341 / IMPACT505 gene panels"
NEED_PANELS=0
for panel in impact341 impact505; do
  UP="$(upper "$panel")"
  if ! ls "${DEST}"/data_gene_panel_*"${panel}"*.txt  >/dev/null 2>&1 \
     && ! ls "${DEST}"/data_gene_panel_*"${UP}"*.txt   >/dev/null 2>&1 \
     && ! ls "${DEST}"/gene_panels/data_gene_panel_*"${panel}"*.txt >/dev/null 2>&1 \
     && ! ls "${DEST}"/gene_panels/data_gene_panel_*"${UP}"*.txt    >/dev/null 2>&1; then
    NEED_PANELS=1
  fi
done

if [ "$NEED_PANELS" -eq 1 ]; then
  note "panels not in the study download; fetching from the datahub reference data"
  command -v git >/dev/null 2>&1 || die "git is required to fetch the gene panels."
  git lfs version >/dev/null 2>&1 \
    || die "git-lfs is required for the gene panels. Install from https://git-lfs.com"

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --depth 1 --filter=blob:none --sparse "$PANEL_REPO" "$TMP/datahub" \
    || die "could not clone $PANEL_REPO"
  git -C "$TMP/datahub" sparse-checkout set reference_data/gene_panels
  git -C "$TMP/datahub" lfs pull --include "reference_data/gene_panels/*" \
    || die "git lfs pull failed; check that git-lfs is initialised (git lfs install)."
  mkdir -p "${DEST}/gene_panels"
  cp "$TMP/datahub"/reference_data/gene_panels/data_gene_panel_impact*.txt \
     "${DEST}/gene_panels/" \
    || die "expected panel files were not present in the datahub checkout."
fi

verify "$DEST"

# --- report what the task will actually be ----------------------------------
note "panel summary"
python3 - "$DEST" <<'PY'
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(Path.cwd() / "src"))
try:
    from missing_piece.panels import PanelPair, load_panels
except ImportError:
    print("  (run from the repo root, or `pip install -e .`, to see the panel summary)")
    sys.exit(0)

dest = Path(sys.argv[1])
for directory in (dest / "gene_panels", dest):
    try:
        panels = load_panels(directory)
    except (FileNotFoundError, NotADirectoryError):
        continue
    ids = {k.upper(): v for k, v in panels.items()}
    if {"IMPACT341", "IMPACT505"} <= set(ids):
        pair = PanelPair(ids["IMPACT341"], ids["IMPACT505"])
        for line in pair.describe().splitlines():
            print("  " + line)
        for name, expected in (("IMPACT341", 341), ("IMPACT505", 505)):
            got = len(ids[name])
            flag = "ok" if got == expected else f"WARNING expected {expected}"
            print(f"  {name}: {got} genes [{flag}]")
        sys.exit(0)
print("  WARNING: IMPACT341/IMPACT505 not both readable", file=sys.stderr)
PY

cat <<'MSG'

Done. Next:

  python -m missing_piece inspect data/msk_chord_2024
  python -m missing_piece run -c configs/msk_chord_nsclc.yaml

Run `inspect` first and read the cohort cascade before running anything else.
MSG
