# Running Tier 1.2 extraction on Spartan

The forward passes that produce `bld/tier1/**/embeddings.parquet`. Everything downstream
— the label-free battery, CKA, P1–P4, both dependent variables — re-reads those
matrices, so this is the one part of the project that needs a GPU and the one part that
does not run on the laptop.

It does not run on the laptop for a measured reason, not a cautious one: `gpt-base-4096`
on MPS falls to **733 tokens/s** against 7,100 at context 2048, because a 4096-wide
attention matrix plus 12 retained hidden-state layers does not fit in 8 GB of unified
memory and the machine starts swapping. Llama at the same context holds 9,343 tokens/s,
so it is the memory ceiling rather than the model.

______________________________________________________________________

## What this submission covers, and what it deliberately leaves out

**Stage one of the Build Plan §1.2 sequence: GPT and Llama, four contexts each.** That
is 8 extraction keys → 16 matrices, because last-token and mean-pool are two reductions
of the same forward pass (`registry.extraction_key`), so the P4 arm is free rather than
a doubling.

Two anchor levels, 16 array elements in total:

| level            | anchors | what it buys                                         |
| ---------------- | ------: | ---------------------------------------------------- |
| `lastevent`      |   6,731 | one vector per patient — CKA, the battery, P1–P4     |
| `perlabel-scout` |  14,204 | the extraction oracle on 9 of the 14 benchmark tasks |

**Not** the full `perlabel` anchor, and this is the main scoping decision. It is 381,522
anchors against the scout's 14,204, and the entire 27× is bought by five tasks:
`lab_anemia`, `lab_hyperkalemia`, `lab_hypoglycemia`, `lab_hyponatremia` and
`lab_thrombocytopenia` label the same ~6k patients at *every qualifying lab draw*, so
they carry **96.3% of the extraction cost for 5/14 of the oracle's evidence**. The other
nine tasks are one or two anchors per patient.

The scout is not a subsample. It is the *complete* anchor set for nine complete tasks,
so each of the nine compares against its published AUROC with no subsampling caveat — a
claim a reader can check, where "we used 4% of the anchors" is one they cannot. Adding
the five lab tasks later is the same code with `--anchor perlabel`, sized against the
throughput this run reports.

**Not** `shared`, the anchor the experiment actually reports: it cuts at the last event
before a declared outcome window, and both the window and the phenotype are open
(decision B1). `lastevent` is the robustness arm for it, needs no decision, and answers
the CKA question that gates whether Mamba and Hyena get built at all.

**Not** Mamba, Hyena, MOTOR or the encoder. Those are three further loader integrations,
and the staged sequence exists so CKA over these eight cells decides whether they are
worth building.

______________________________________________________________________

## Expected cost

Measured inputs: ~0.80 tokens per event (n=40, stratified by record length), median
2,503 tokens per patient at the `lastevent` anchor, and CPU-side timeline prep of ~2.0
ms per 1,000 events.

|                                  | tokens, both families |
| -------------------------------- | --------------------: |
| `lastevent`, 4 contexts          |                  67 M |
| `perlabel-scout`, 4 contexts     |                ~141 M |
| **this submission**              |            **~208 M** |
| full `perlabel` (for comparison) |                ~3.8 B |

The array's wall clock is the *longest single element*, not the sum: `gpt-base-4096` at
`perlabel-scout`, roughly 33 M tokens. At a conservative 50k tokens/s on an A100 that is
~11 minutes of GPU plus ~2 minutes of timeline prep; even at the laptop's MPS throughput
it stays inside an hour. The 4-hour walltime in the header is slack for queue variance
and a cold model load, not a forecast — **budget 1–2 hours end to end**, more if the
queue hands you the GPUs a few at a time.

The full `perlabel` anchor would be ~5–8 hours *per element* at the pessimistic end.
That is the line this scope is drawn to stay under.

______________________________________________________________________

## Setup

Once per cluster. Steps 2, 4 and 5 must run on a **login** node — compute nodes have no
outbound network.

**1. Confirm GPU access.** Account is `punim1993`, partition is `gpu-a100`.

The gate on Spartan is **QoS, not the account** — a distinction worth knowing before it
costs an afternoon. Checked 2026-08-28, `gpu-a100` reports `AllowAccounts=ALL` but
`AllowQos=hpcadmin,normal,publicgpu,viz,debug,punim1257,punim0693`, so any project may
submit provided it holds one of those. `punim1257` and `punim0693` are two projects with
purchased allocations; `punim1993` is not among them, and does not need to be, because
`normal` is on the list and is the default QoS most users hold.

So no `--qos` flag should be needed. Verify before staging 280 MB:

```bash
srun --account=punim1993 --partition=gpu-a100 --gres=gpu:1 \
     --time=00:02:00 --mem=4G nvidia-smi
```

If that is rejected with a QoS error, `publicgpu` is the general-access GPU QoS to
request from Spartan support, then passed as `--qos=publicgpu`. Note that
`sacctmgr show assoc user=$USER format=account,partition` does **not** answer this: a
blank partition field means the association is simply not partition-scoped, which is the
normal case and says nothing about GPU entitlement.

**2. Get pixi and the repo onto Spartan.** Neither is there by default, and the repo
must be pushed first — `hpc/` and the extraction modules are new, so a clone of an
unpushed branch gets you a tree without any of this.

Install pixi (login nodes have outbound network, compute nodes do not):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pixi --version
```

Then clone — **into project storage, not `$HOME`**. The 32 matrices this run produces
are ~1 GB (6,731 × 768 float32 per `lastevent` cell, 14,204 × 768 per `perlabel-scout`
cell, ×16), they land in `$REPO/bld/`, and home directories on Spartan are small and
quota'd. pixi's own package cache is worth moving off home for the same reason:

```bash
export PROJECT=/data/gpfs/projects/punim1993/students/Anoja
export PIXI_CACHE_DIR=$PROJECT/.pixi_cache

cd $PROJECT
git clone https://github.com/anthano/embedding_space_exploration.git
cd embedding_space_exploration
git checkout tier0-calibration-harness
```

`extract.slurm` derives everything from that one directory —
`$PROJECT/embedding_space_exploration`, `$PROJECT/ehrshot`, `$PROJECT/hf_cache` — so if
your layout matches the above, no path exports are needed at submit time. If it does
not, override the base rather than each path:

```bash
export ESX_PROJECT=/somewhere/else
```

Put `PROJECT` and `PIXI_CACHE_DIR` in your `~/.bashrc` once; the login-node steps below
read them. The job validates all three directories before loading a model and exits with
`missing: <path>` rather than failing later and less legibly.

**3. Stage the data** (from the laptop). ~280 MB, not the 17 GB the extract weighs: the
forward pass reads the `meds_reader` database, `benchmark/` and `splits/`, and none of
`features/`, `femr/`, `data/` or `models/`.

```bash
./hpc/spartan/stage_data.sh you@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1993/students/Anoja/ehrshot
```

**4. Build the environment** (on the login node):

```bash
export CONDA_OVERRIDE_CUDA=12   # put this in ~/.bashrc too
cd $PROJECT/embedding_space_exploration/hpc/spartan
pixi install                # solves the CUDA build of pytorch; writes pixi.lock here
pixi run install-hf-ehr     # CLMBRTokenizer, without its training-only deps
```

**`CONDA_OVERRIDE_CUDA=12` is required on a login node and is not a workaround.** The
manifest's platform is `linux-64-cuda-12`, so pixi wants the `__cuda` virtual package —
and a login node has no GPU, so it fails with `missing virtual packages: __cuda >= 12`.
Login nodes are precisely where the environment gets built, so the override asserts the
driver the *compute* nodes have. Every `pixi` command you run outside a job needs it in
scope, which is why it belongs in `~/.bashrc` rather than on one line.

This is a **separate pixi manifest**, not a `gpu` feature in the root `pyproject.toml`.
The root workspace installs itself as an editable PyPI dependency, so resolving PyPI for
`linux-64` needs a build dispatch that only runs on `linux-64` — and pixi then refuses
*every* `pixi run` in the repo until the lock covers all environments. Verified
2026-08-28: adding the feature broke the laptop's own tasks with "the environment 'gpu'
does not support 'osx-arm64' on this machine". The cluster manifest is also deliberately
the extraction subset — no jupyter-book, node, plotly, scikit-learn or scipy — since
none is imported by the forward-pass path.

Commit the generated `hpc/spartan/pixi.lock` back from Spartan: it is what makes a
re-run months later the same environment.

**5. Prefetch weights and build the timeline** (login node):

```bash
export HF_HOME=$PROJECT/hf_cache
export EHRSHOT_ROOT=$PROJECT/ehrshot

# The layout `stage_data.sh` produces; check it before anything reads it.
ls $EHRSHOT_ROOT            # -> meds_reader_omop_ehrshot  EHRSHOT_ASSETS
ls $EHRSHOT_ROOT/EHRSHOT_ASSETS   # -> benchmark  splits  results

pixi run prefetch    # the 8 checkpoints; jobs run with HF_HUB_OFFLINE=1
pixi run timeline    # bld/ is gitignored, so this does not arrive with the repo
```

`pixi run timeline` must happen before submitting: all 16 elements read that file and
letting them race to write it is how a matrix silently ends up with the wrong number of
rows.

______________________________________________________________________

## Submit

```bash
cd $REPO          # the log paths in the header are relative to where you submit
mkdir -p logs
sbatch --account=punim1993 hpc/spartan/extract.slurm
```

Check the plan without a GPU or a model load first — this is cheap and catches a wrong
`EHRSHOT_ROOT` before the queue does:

```bash
pixi run python -m embedding_space_exploration.data_management.run_extraction \
    --anchor perlabel-scout --key-index 0 --plan
```

**Resume is automatic.** `extract_resumable` journals whole batches and restarts where
the journal ends, and a resumed run is bitwise identical to an uninterrupted one. An
element killed at the walltime is *re-submitted*, not restarted: run the same `sbatch`
again and finished elements exit in seconds while unfinished ones pick up mid-cell.

Progress lines report anchors/s and an ETA. **Read them on the first run** — the
walltime and the sizing of any future `perlabel` submission both rest on a throughput
number this project has never measured on a real GPU.

## Check before you copy

The transfer is the cheap half. What is expensive is discovering three weeks into the
battery that one cell's rows are in a different order from the other fifteen, because
CKA, the paired contrasts and the battery all read row *i* of two matrices as the same
anchor — a misalignment produces numbers rather than an error.

```bash
cd $REPO/hpc/spartan
pixi run python check_run.py
```

It needs no GPU and no model weights: every check is a cross-check between files the run
already wrote. Exit status is 0 when nothing failed. What it settles, in the order the
failures matter:

- **No `_journal/` directory survives.** `extract_resumable` deletes it as its last act,
  after verifying it holds exactly as many rows as the index — so a journal still on
  disk *is* the run's own statement that the element was killed. Re-submit the same
  `sbatch` rather than copying that cell.
- **All 32 cell directories exist**, each with `embeddings.parquet`,
  `truncation.parquet` and `extraction.json`. Sixteen per anchor level, because
  last-token and mean-pool are two matrices off one forward pass.
- **Every cell at an anchor carries the same anchors in the same order**, and that order
  is `ordered_index`'s. The check that justifies the script.
- **`truncation_side` is `left` in all 32 records**, and `device` is `cuda`. The first
  is the finding the module docstring calls load-bearing; the second is why two cells
  are comparable in the last decimals.
- **`truncated_share` falls and `median_covered` rises with context**, within each
  family. If they do not move, every cell read the same opening tokens and P1's null was
  built into the design rather than measured — which is the failure Study Design Freeze
  §7 names twice.
- **No NaN, no all-zero rows, no cell that is an exact copy of another.** Reading the
  backbone through `AutoModel` silently reinitialises the embedding table;
  `n_empty_histories` in bulk means cutoffs arrived as `NaT` and patients were embedded
  as a single PAD token. Both are green runs that produce noise.
- **Throughput per key.** Printed rather than judged: this is the first measurement of
  this pipeline on a real GPU, and it is what sizes any future `perlabel` submission.

Skim the logs for the two silent ones the files cannot show, then archive them beside
the matrices — they age out of `logs/` and the throughput lines are the record:

```bash
grep -il "newly initialized\|were not used when initializing" logs/extract-*.out
grep -h "^dropped" logs/extract-*.out | sort | uniq -c   # one distinct count per anchor
```

## Bring the results home

```bash
# from the laptop, at the repo root
rsync -avhP \
  you@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1993/students/Anoja/embedding_space_exploration/bld/tier1 \
  ./bld/
```

Then re-run the same checks locally against what actually arrived, which is also the
cheapest possible verification of the transfer:

```bash
pixi run python hpc/spartan/check_run.py
```

Leave the cluster copy in place until that passes. The matrices are the deliverable;
everything downstream runs on the laptop against them.

# Dataset

Analyses use the 6,731 patients present in the EHRSHOT MEDS extract; 8 of the 6,739
patients in the official splits file have labels but no timeline in the MEDS release and
are excluded (216 of 381,522 anchors, 0.06%).
