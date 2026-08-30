#!/usr/bin/env bash
# Tier 1.2 extraction, stage one: GPT and Llama at four contexts each, at two
# anchor levels. Sixteen array elements, one GPU each.
#
#   sbatch --account=punim1993 hpc/spartan/extract.slurm
#
# ----------------------------------------------------------------------------
# What the sixteen elements are
# ----------------------------------------------------------------------------
# 8 extraction keys x 2 anchor levels. A *key* is (family, size, context), not a
# cell: last-token and mean-pool are two reductions of one forward pass, so each
# element writes two matrices and the array is 8 wide per anchor rather than 16.
# That is `registry.extraction_key`, and it is why the mean-pool arm (P4) is free
# rather than a doubling.
#
#   element  0- 7  anchor=lastevent        6,731 anchors   one vector per patient
#   element  8-15  anchor=perlabel-scout  14,204 anchors   9 of the 14 tasks
#
# Key order comes from `run_extraction.keys_for`, longest context first, so a
# throttled array starts the expensive cells rather than finishing with them.
# Python owns that order; this script only passes an index, so the two cannot
# drift apart.
#
# ----------------------------------------------------------------------------
# Why not the full perlabel anchor
# ----------------------------------------------------------------------------
# It is 381,522 anchors against the scout's 14,204, and the 27x is bought by
# five lab tasks that label the same ~6k patients at every qualifying draw.
# Those five carry 96.3% of the extraction cost for 5/14 of the oracle's
# evidence. The scout runs nine tasks *complete* -- every label each one has --
# so each compares to its published AUROC with no subsampling caveat. Adding the
# lab tasks is a later, separate submission against the same code, and the
# throughput this array reports is what should size it.
#
# ----------------------------------------------------------------------------
# Resume
# ----------------------------------------------------------------------------
# `extract_resumable` journals whole batches and restarts where the journal
# ends, and a resumed run is bitwise identical to an uninterrupted one. So a
# element killed at the walltime is *re-submitted*, not restarted: run the same
# sbatch again and finished elements exit in seconds while unfinished ones pick
# up mid-cell. Nothing here needs `--requeue` to be safe.

#SBATCH --job-name=esx-extract
#SBATCH --array=0-15
# gpu-a100: 25 nodes x 4 GPUs (spartan-gpgpu101-125), MaxTime 7 days. Chosen over
# gpu-h100 deliberately -- a 116M-parameter model at batch 16 does not saturate
# an A100, so the newer card would buy perhaps 1.5x on the 4096-context cells and
# almost nothing on the 512s, against a narrower QoS list and a longer queue.
# Revisit for the full `perlabel` anchor, where the arithmetic changes. Switching
# needs no edit here: `sbatch --partition=gpu-h100 ...` overrides it.
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
# The partition defaults to DefCpuPerGPU=1 and DefMemPerGPU=4000, both too small:
# the timeline prep is CPU-bound Python, and one patient in this extract carries
# 237,807 events.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Well inside the partition's 7-day cap, and far above the estimate -- slack for
# queue variance and a cold model load. The partition's own default is 10
# minutes, which would kill every element, so this is not optional.
#
# Note the billing: TRESBillingWeights charges GRES/gpu at 100x a CPU core, so a
# GPU-hour here is expensive. Overrunning the walltime costs nothing (the job is
# killed and resumes on re-submission); over-requesting *nodes* would.
#SBATCH --time=04:00:00
# Relative to the SUBMIT directory, which the README makes the repo root, and
# Slurm will not create it -- an array whose log directory is missing fails at
# submit with a bare "Invalid directory specified". `mkdir -p logs` first.
#SBATCH --output=logs/extract-%A_%a.out
#SBATCH --error=logs/extract-%A_%a.out

# `--account` is deliberately NOT set here, so that a committed file cannot
# quietly submit work against the wrong project. Pass it, or export
# SBATCH_ACCOUNT:
#     sbatch --account=punim1993 hpc/spartan/extract.slurm
#
# On QoS. Checked 2026-08-28, gpu-a100 has AllowAccounts=ALL but
#     AllowQos=hpcadmin,normal,publicgpu,viz,debug,punim1257,punim0693
# so the partition is gated on QoS rather than on the account, and punim1993 has
# no dedicated entry (punim1257 and punim0693 are two projects with purchased
# allocations). `normal` is on the list, so the default QoS most users hold is
# already sufficient and no --qos flag should be needed. If a submit is rejected
# with a QoS error, `publicgpu` is the general-access GPU QoS to request from
# Spartan support, and it is then passed the same way:
#     sbatch --account=punim1993 --qos=publicgpu hpc/spartan/extract.slurm

set -euo pipefail

# ----------------------------------------------------------------------------
# Paths -- override by exporting before sbatch, or edit here
# ----------------------------------------------------------------------------
# Everything hangs off one project directory, so relocating the work is one
# export rather than four. Project storage, never $HOME: this run writes ~1 GB
# of matrices under $REPO/bld, and home directories on Spartan are quota'd.
PROJECT="${ESX_PROJECT:-/data/gpfs/projects/punim1993/students/Anoja}"
REPO="${ESX_REPO:-$PROJECT/embedding_space_exploration}"
# Must contain `meds_reader_omop_ehrshot/` and `EHRSHOT_ASSETS/` -- that is the
# shape `stage_data.sh` produces and what `config.EHRSHOT_ROOT` expects.
export EHRSHOT_ROOT="${EHRSHOT_ROOT:-$PROJECT/ehrshot}"
# Must be the same HF_HOME `pixi run prefetch` populated, or every element dies
# on a cache miss it cannot resolve offline.
export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"

# Cache misses become an immediate error rather than a stalled connection that
# can never succeed -- compute nodes have no outbound network. If an element
# dies here, `pixi run prefetch` was not run, or HF_HOME differs from the one it
# populated.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# The manifest's platform requires `__cuda >= 12`, which pixi detects from the
# driver. Asserting it here costs nothing on a node that has one and avoids a
# confusing pixi-level failure on one where detection is flaky -- torch remains
# the real check, since `--device cuda` below raises immediately without a GPU.
export CONDA_OVERRIDE_CUDA=12
# Match the CPU allocation. Left unset, torch spawns one thread per core on the
# whole node and the threads contend rather than help.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ANCHORS=(lastevent perlabel-scout)
N_KEYS=8

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ANCHOR="${ANCHORS[$((TASK_ID / N_KEYS))]}"
KEY_INDEX=$((TASK_ID % N_KEYS))

cd "$REPO/hpc/spartan"

# Fail here, loudly, rather than 30s into a model load: a wrong EHRSHOT_ROOT is
# the single most likely misconfiguration and its natural error arrives much
# later and much less legibly.
for required in "$REPO/hpc/spartan" "$EHRSHOT_ROOT/meds_reader_omop_ehrshot" "$HF_HOME"; do
  if [[ ! -d "$required" ]]; then
    echo "missing: $required" >&2
    echo "check ESX_PROJECT / ESX_REPO / EHRSHOT_ROOT / HF_HOME" >&2
    exit 2
  fi
done

echo "host        $(hostname)"
echo "array task  ${TASK_ID}  ->  anchor=${ANCHOR}  key-index=${KEY_INDEX}"
echo "repo        ${REPO}"
echo "ehrshot     ${EHRSHOT_ROOT}"
echo "hf_home     ${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

srun pixi run python -m embedding_space_exploration.data_management.run_extraction \
  --anchor "$ANCHOR" \
  --key-index "$KEY_INDEX" \
  --device cuda \
  --batch-size "${ESX_BATCH_SIZE:-16}"
