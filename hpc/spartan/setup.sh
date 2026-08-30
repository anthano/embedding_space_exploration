#!/usr/bin/env bash
# Everything that must happen on a LOGIN node before the array can be submitted.
#
#   ./hpc/spartan/setup.sh
#
# Run it as many times as you like: every step is idempotent, so a re-run after
# a failure resumes rather than redoing.
#
# ----------------------------------------------------------------------------
# Why this is not part of the job
# ----------------------------------------------------------------------------
# The obvious simplification is to build the environment inside `extract.slurm`
# and submit one thing instead of two. It does not work, for three reasons, two
# of which hold even where compute nodes can reach the internet.
#
# 1. The array is sixteen elements that start at once, all pointing at the same
#    `.pixi/` directory. Sixteen concurrent installs racing to write one
#    environment corrupt it; they do not produce sixteen good ones. Avoiding
#    that needs either a lock or a per-element environment, at sixteen times the
#    disk and the download.
# 2. It is the most expensive possible place to do cheap work. gpu-a100 bills
#    GRES/gpu at 100x a CPU core, so a solve-and-download is ten minutes of
#    network and disk with a GPU held idle -- times sixteen. And a solve that
#    fails, fails sixteen times, each having queued for and taken a GPU, rather
#    than once in thirty seconds somewhere you can read the error.
# 3. Compute nodes are not expected to have outbound network, which is also why
#    the jobs run with HF_HUB_OFFLINE=1 and why `prefetch` exists at all.
#
# So: the slow, networked, once-per-cluster work happens here, and the job does
# nothing but forward passes.

# Not exercised end to end: the 2026-08-30 cluster setup was done by hand from
# the README before this existed, so treat a failure here as a bug in this
# script rather than in the steps, which are known to work.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# A login node has no GPU, so pixi cannot satisfy the manifest's `__cuda >= 12`
# and refuses to do anything at all. This asserts the driver the *compute* nodes
# have; it is a prerequisite of building here, not a workaround.
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12}"

PROJECT="${ESX_PROJECT:-/data/gpfs/projects/punim1993/students/Anoja}"
export EHRSHOT_ROOT="${EHRSHOT_ROOT:-$PROJECT/ehrshot}"
export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

printf 'repo         %s\n' "$REPO"
printf 'ehrshot      %s\n' "$EHRSHOT_ROOT"
printf 'hf_home      %s\n' "$HF_HOME"
printf 'cuda override %s\n' "$CONDA_OVERRIDE_CUDA"

# ----------------------------------------------------------------------------
step "1/6  checking the staged extract"
# ----------------------------------------------------------------------------
# Checked before anything slow runs. A wrong EHRSHOT_ROOT is the likeliest
# misconfiguration, and the classic cause is an rsync trailing slash putting the
# contents one level deeper than intended.
[[ -d "$EHRSHOT_ROOT" ]] || die "no extract at $EHRSHOT_ROOT (set ESX_PROJECT or EHRSHOT_ROOT)"
for required in meds_reader_omop_ehrshot EHRSHOT_ASSETS/benchmark EHRSHOT_ASSETS/splits; do
  [[ -e "$EHRSHOT_ROOT/$required" ]] || die "missing $EHRSHOT_ROOT/$required
  If the staging landed one level deeper, point EHRSHOT_ROOT at the inner
  directory rather than copying again."
done
echo "  ok"

# ----------------------------------------------------------------------------
step "2/6  solving the environment (slow the first time)"
# ----------------------------------------------------------------------------
cd "$HERE"
pixi install || die "pixi install. The conda half is known to resolve; if this is
  the PyPI half (meds-reader, transformers, the editable root package), that
  combination has not been solved on linux-64 before and the error names what
  to pin."

# ----------------------------------------------------------------------------
step "3/6  installing hf_ehr (CLMBRTokenizer)"
# ----------------------------------------------------------------------------
# --no-deps, so re-running is cheap and safe.
pixi run install-hf-ehr || die "install-hf-ehr"

# ----------------------------------------------------------------------------
step "4/6  caching the eight checkpoints"
# ----------------------------------------------------------------------------
# Must happen here: the jobs run with HF_HUB_OFFLINE=1, so a checkpoint that is
# not in HF_HOME by now is an error the array cannot recover from.
mkdir -p "$HF_HOME"
pixi run prefetch || die "prefetch. Gated repos need \`huggingface-cli login\`."

# ----------------------------------------------------------------------------
step "5/6  building the patient timeline"
# ----------------------------------------------------------------------------
# Deliberately not left to the array: all sixteen elements read this file, and
# letting them race to write it is how a matrix ends up with the wrong number of
# rows and nothing to show for it.
pixi run timeline || die "timeline"

# ----------------------------------------------------------------------------
step "6/6  dry-running both anchor levels"
# ----------------------------------------------------------------------------
# Loads no model and touches no GPU. The two counts confirm that the timeline
# summary and the label files are both readable, which is everything the array
# needs from disk.
for spec in "lastevent 6,731" "perlabel-scout 14,204"; do
  set -- $spec
  echo "  --anchor $1 (expect $2 anchors)"
  pixi run python -m embedding_space_exploration.data_management.run_extraction \
    --anchor "$1" --key-index 0 --plan | sed 's/^/    /' || die "--plan for $1"
done

cat <<EOF

$(printf '\033[32mSetup complete.\033[0m') Submit the array from the repo root:

    cd $REPO
    mkdir -p logs
    sbatch --test-only --account=punim1993 hpc/spartan/extract.slurm   # validate
    sbatch --account=punim1993 hpc/spartan/extract.slurm               # go

Then watch the first element for the anchors/s figure:

    tail -f logs/extract-*_0.out
EOF
