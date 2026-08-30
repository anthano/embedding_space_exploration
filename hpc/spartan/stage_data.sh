#!/usr/bin/env bash
# Copy the parts of EHRSHOT the extraction actually reads onto Spartan.
#
# Run this from the LAPTOP, not from Spartan. It pushes ~280 MB, not the 17 GB
# the extract weighs on disk: the forward pass needs the `meds_reader` database
# (the timelines), `benchmark/` (the label times that define the perlabel-scout
# anchor) and `splits/`. It does not need `features/` (5.7 GB of CLMBR vectors),
# `femr/` (7.4 GB), `data/` (3.2 GB) or `models/` (1.1 GB) -- those feed the
# protocol oracle and the published-AUROC comparison, both of which run on the
# laptop against matrices the cluster sends back. Copying them would be 60x the
# transfer for nothing, on licensed data, which is also the wrong direction to
# be careless in.
#
# `results/` (1.3 MB) is included: it is the published-AUROC table, it is tiny,
# and having it there means the per-cell oracle can be run on the cluster too if
# that ever becomes convenient.
#
# Usage:
#   ./stage_data.sh <user>@spartan.hpc.unimelb.edu.au:/data/scratch/projects/punimXXXX/ehrshot
set -euo pipefail

LOCAL_ROOT="${EHRSHOT_ROOT:-$HOME/Documents/Datasets/EHRSHOT_files}"
REMOTE="${1:-}"

if [[ -z "$REMOTE" ]]; then
  echo "usage: $0 <user>@<host>:<remote-ehrshot-root>" >&2
  exit 2
fi
if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "no EHRSHOT extract at $LOCAL_ROOT (set EHRSHOT_ROOT)" >&2
  exit 2
fi

echo "staging from $LOCAL_ROOT"
echo "          to $REMOTE"
echo

# --relative keeps the directory shape under the remote root, so EHRSHOT_ROOT on
# the cluster points at one directory and `config.py` needs no cluster-specific
# branch. -P resumes a transfer that dropped, which over a campus VPN it will.
rsync -avhP --relative \
  "$LOCAL_ROOT/./meds_reader_omop_ehrshot" \
  "$LOCAL_ROOT/./EHRSHOT_ASSETS/benchmark" \
  "$LOCAL_ROOT/./EHRSHOT_ASSETS/splits" \
  "$LOCAL_ROOT/./EHRSHOT_ASSETS/results" \
  "$REMOTE/"

echo
echo "staged. On Spartan, point the jobs at it with:"
echo "  export EHRSHOT_ROOT=<remote-ehrshot-root>"
