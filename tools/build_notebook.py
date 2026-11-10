#!/usr/bin/env python3
"""Generate BlitzGen_BinderDesign.ipynb.

Written as a generator rather than by hand so the K16 folding config is copied
from config/fold_blitz_k16.yaml at build time. That file is a vendored copy of
the campaign's own config; embedding it by reference means the notebook's
sampler schedule cannot silently drift from the one that produced BinderBase.

The committed notebook must always equal this script's output -- CI enforces
that (tools/check_notebook.py). Edit the generator, never the .ipynb.
"""
import hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "fold_blitz_k16.yaml"
K16_CFG = CFG_PATH.read_text()

PROV = json.loads((ROOT / "provenance.json").read_text())
VERSION = (ROOT / "VERSION").read_text().strip()

CFG_SHA = hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()
if CFG_SHA != PROV["fold_config_sha256"]:
    sys.exit(f"config/fold_blitz_k16.yaml sha256 is {CFG_SHA}\n"
             f"provenance.json says          {PROV['fold_config_sha256']}\n"
             "If the config changed on purpose, bump VERSION and update both "
             "provenance.json and CHANGELOG.md in the same commit.")

# Tokens substituted into every cell just before writing, so that version and
# checksum facts live in exactly one place (provenance.json / VERSION).
SUBST = {
    "@@VERSION@@":        VERSION,
    "@@BLITZGEN_SRC@@":   PROV["blitzgen_src"],
    "@@CFG_SHA@@":        PROV["fold_config_sha256"],
    "@@CKPT_SHA@@":       PROV["k16_checkpoint_sha256"],
    "@@CKPT_REPO@@":      PROV["k16_checkpoint_hf_repo"],
}

def md(*lines):  return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}
def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(lines)}

cells = []

# ----------------------------------------------------------------- intro ----
cells.append(md("""# Free binder design — BoltzGen + Blitz K16

Design de novo protein binders against a target epitope, on a free Colab GPU.

**Runtime → Change runtime type → GPU.** Any GPU works; the notebook adapts.

---

## What this gives you

A handful of designed binder sequences with predicted structures and confidence
metrics, for a target and epitope you choose. Everything is MIT-licensed and the
model weights are public.

## Before you start

**You choose the epitope, and that choice is the run.** A binder designed
against a badly chosen site will still come back with good-looking scores — the
model answers the question you asked. Pick a solvent-exposed, reasonably compact
surface patch, and treat the confidence metrics as "did the model do what I
asked", not "is this a good drug target".

Designs are drawn fresh each run, so two runs give you different binders rather
than the same ones refined. That is normal.

## Credit

BoltzGen is MIT, © 2025 Hannes Stärk ([paper](https://jclinic.mit.edu/boltzgen/)).
Blitz K16 is a distilled Boltz-2 student from
[EpiGenomicsCode/Blitz](https://github.com/EpiGenomicsCode/Blitz); weights at
[@@CKPT_REPO@@](https://huggingface.co/@@CKPT_REPO@@).

## Provenance

If you saved this notebook and are returning to it later, this is exactly what
you have. Nothing below is fetched at runtime from a moving reference.

| | |
|---|---|
| notebook version | `@@VERSION@@` |
| BlitzGen source | `@@BLITZGEN_SRC@@` |
| folding config sha256 | `@@CFG_SHA@@` |
| K16 checkpoint sha256 | `@@CKPT_SHA@@` |

Newest version: <https://github.com/EpiGenomicsCode/blitzgen-colab>
"""))

# ------------------------------------------------------------ 1. probe ------
cells.append(md("""---
## 1 · Hardware probe

**Run this first.** It needs no install and finishes in seconds, so a card that
cannot run the job is discovered now rather than after a 7.5 GB download.

It picks the precision for you:

* **compute capability ≥ 8.0** (A100, L4, RTX 30xx/40xx) → `bf16-mixed`.
* **below 8.0** (T4, V100) → `fp32`. bf16 needs Ampere.

`fp16` is *not* an option on any card. It was tested and fails: the diffusion
sampler's coordinate arithmetic overflows fp16's 5-bit exponent upstream of
`weighted_rigid_align`, and the resulting non-finite coordinates surface as
`linalg.svd: failed to converge`. Forcing the alignment itself to fp32 does not
help — the coordinates are already broken on arrival.
"""))

cells.append(code("""import subprocess, shutil, sys, textwrap

def smi(q):
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            text=True, timeout=30).strip().splitlines()[0]
    except Exception as e:
        return f"ERROR: {e}"

if not shutil.which("nvidia-smi"):
    raise SystemExit("No GPU. Runtime -> Change runtime type -> GPU, then rerun.")

name   = smi("name")
vram   = smi("memory.total")
cap    = smi("compute_cap")
driver = smi("driver_version")

try:
    cap_f = float(cap)
except ValueError:
    cap_f = 0.0

PRECISION = "bf16-mixed" if cap_f >= 8.0 else "32"
vram_mib  = int("".join(c for c in vram if c.isdigit()) or 0)

print(f"GPU            {name}")
print(f"VRAM           {vram}")
print(f"capability     {cap}")
print(f"driver         {driver}")
print(f"gcc            {shutil.which('gcc') or 'MISSING -- triton kernels need a C compiler'}")
print()
print(f"=> precision   {PRECISION}" + ("   (Ampere+)" if cap_f >= 8.0
      else "   (pre-Ampere: no bf16, so full fp32)"))

# Peak VRAM measured on an A100 for a 361-token target at batch size 1:
#   bf16-mixed 8,555 MiB   fp32 8,899 MiB
# fp32 costs only ~4% more memory because bf16-mixed is MIXED -- AMP already
# keeps fp32 master weights, so going full fp32 only promotes the bf16
# activation subset.
NEED = 8900 if PRECISION == "32" else 8600
if vram_mib and vram_mib < NEED + 2000:
    print(f"\\nWARNING: {vram_mib} MiB may be tight. A 361-token target peaked at "
          f"{NEED} MiB; larger targets need more. Reduce the crop if you hit OOM.")
else:
    print(f"\\nHeadroom OK: ~{vram_mib - NEED} MiB above the {NEED} MiB reference peak.")

if cap_f < 8.0:
    print(textwrap.dedent('''
        SPEED NOTE. Measured on a Colab T4: 8 designs at this
        notebook's defaults took 40 min end to end -- design 18 min,
        folding 16 min, everything else under 5 min combined. This
        workload is bandwidth-bound, not matmul-bound: disabling TF32
        on an A100 cost only 2.4%, so a T4's 320 GB/s against the
        A100's 1555 is what sets the pace. Below capability 8.0
        BoltzGen also runs its pure-PyTorch triangle path instead of
        the cuequivariance kernels; that cost is included here.
    '''))"""))

# ---------------------------------------------------------- 2. install ------
cells.append(md("""---
## 2 · Install

Installs [BlitzGen](https://github.com/EpiGenomicsCode/blitzgen) `v0.1.0`,
pinned to a tag so this notebook behaves the same way next month as it does
today. BlitzGen provides `--folding_config` and `--seed`; upstream BoltzGen has
neither, so a stock `pip install boltzgen` will not work here.

No CUDA extra is requested. BlitzGen's `[cu12]`/`[cu13]` extras carry the
`cuequivariance_ops_*` wheels, which are large and are never actually loaded by
this pipeline — the triton kernels behind `cuequivariance_torch` are what run.
Asking for them would only add a way for the install to fail.

Takes a couple of minutes.
"""))

cells.append(code("""#@title Install BlitzGen
BLITZGEN_SRC = "@@BLITZGEN_SRC@@"  #@param {type:"string"}

# Expect 4-8 minutes on a fresh Colab VM. pip clones the repository and builds
# a wheel from source before it begins resolving dependencies, and the pinned
# numpy/numba force a re-resolve against the preinstalled stack.
#
# The output below is streamed line by line rather than captured, so this cell
# shows progress while it works. A cell that prints nothing for minutes is a
# stalled install, not a working one.
import subprocess, sys

print("installing:", BLITZGEN_SRC, flush=True)
p = subprocess.Popen([sys.executable, "-m", "pip", "install", BLITZGEN_SRC],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
for line in p.stdout:
    print(line.rstrip(), flush=True)
if p.wait() != 0:
    raise SystemExit(
        "Install failed.\\n"
        "If this is a 404/auth error the repository is private or does not exist.\\n"
        "A public source is required; see the final section of this notebook.")

# pip reports conflicts against the host's preinstalled scientific stack here:
# on Colab, jax / tifffile / dm-tree / ml-dtypes want numpy>=2.1 and stumpy
# wants numba>=0.61.2, while BlitzGen pins numpy==2.0.2 and numba==0.61.0.
# None of those packages are on BlitzGen's import path. The warnings are
# expected, they are not errors, and this notebook is unaffected.
from importlib.metadata import version, PackageNotFoundError
import boltzgen  # noqa: F401
try:
    print("blitzgen", version("blitzgen"), "(import package: boltzgen)")
except PackageNotFoundError:
    print("blitzgen installed but metadata missing -- version unverified")"""))

# ---------------------------------------------------------- 3. weights ------
cells.append(md("""---
## 3 · Weights  (~7.5 GB)

Two downloads:

* **`boltzgen/boltzgen-1`** (~5.6 GB) — the design and inverse-folding models.
* **`vinaymatt/Blitz-Boltz2 / blitz-boltz2-k16.ckpt`** (1.9 GB) — the Blitz K16
  folding student.

The K16 file is checked against the publisher's own `SHA256SUMS`. The expected
digest is `343f252b…`, the same checkpoint used to build the
[BinderBase](https://binderbase.org) catalogue — so folding here is the real
model, not a lookalike.

Mounting Drive caches both across sessions and is worth it if you will run more
than once.
"""))

cells.append(code("""#@title Download weights
USE_DRIVE_CACHE = False  #@param {type:"boolean"}

import os, hashlib, pathlib, subprocess, sys

CACHE = "/content/blitzgen_cache"
if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE = "/content/drive/MyDrive/blitzgen_cache"
HF_HOME = f"{CACHE}/hf"; CKPT_DIR = f"{CACHE}/ckpt"
os.makedirs(HF_HOME, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
from huggingface_hub import hf_hub_download, snapshot_download

K16_SHA = "@@CKPT_SHA@@"
CKPT = pathlib.Path(CKPT_DIR) / "blitz-boltz2-k16.ckpt"

if not CKPT.exists():
    print("Downloading Blitz K16 (1.9 GB) ...")
    src = hf_hub_download("vinaymatt/Blitz-Boltz2", "blitz-boltz2-k16.ckpt",
                          local_dir=CKPT_DIR)
    CKPT = pathlib.Path(src)

print("Verifying checksum ...")
h = hashlib.sha256()
with open(CKPT, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""):
        h.update(chunk)
got = h.hexdigest()
if got != K16_SHA:
    raise SystemExit(f"CHECKSUM MISMATCH\\n  expected {K16_SHA}\\n  got      {got}\\n"
                     "Delete the file and re-run; do not design against it.")
print(f"  OK  {got[:16]}...  checkpoint verified")

print("\\nDownloading BoltzGen models (~5.6 GB, slowest step) ...")
snapshot_download("boltzgen/boltzgen-1")
print("done")"""))

# ------------------------------------------------------- 4. K16 config ------
cells.append(md("""---
## 4 · The K16 folding schedule

**This cell is not optional, and it is the easiest thing to get silently wrong.**

K16 weights must run on the K16 schedule — 16 sampling steps, `sigma_min 0.01`.
Pair them with the default Boltz-2 schedule (200 steps) and *nothing errors*:
the structures look plausible and the sampler is simply wrong.

The config below is BlitzGen's own `fold_blitz_k16.yaml`, copied verbatim, so
this notebook folds on exactly the schedule K16 was tuned for.
"""))

cells.append(code('K16_YAML = r"""' + K16_CFG + '"""\n\n' + """import pathlib, re

# The shipped config hardcodes precision: bf16-mixed, because the machines K16
# was developed on were all Ampere or newer. On a pre-Ampere card the design
# step would correctly run fp32 (set on the command line) while FOLDING quietly
# tried bf16 and died. Both folding and design_folding read this one file, so
# rewriting it here covers both.
cfg_text, n = re.subn(r"^(\\s*precision:)\\s*\\S+", r"\\1 " + PRECISION,
                      K16_YAML, flags=re.M)
cfg = pathlib.Path("/content/fold_blitz_k16.yaml")
cfg.write_text(cfg_text)

steps = re.search(r"^sampling_steps:\\s*(\\d+)", cfg_text, re.M)
smin  = re.search(r"sigma_min:\\s*([0-9.]+)", cfg_text)
prec  = re.search(r"^\\s*precision:\\s*(\\S+)", cfg_text, re.M)
print(f"wrote {cfg}")
print(f"  sampling_steps {steps.group(1) if steps else '?'}   (Boltz-2 default is 200)")
print(f"  sigma_min      {smin.group(1) if smin else '?'}")
print(f"  precision      {prec.group(1) if prec else '?'}   ({n} field(s) set from the probe)")"""))

# -------------------------------------------------------- 5. target ---------
cells.append(md("""---
## 5 · Your target and epitope

Give a **UniProt accession** and the structure is pulled from AlphaFold DB.

`BINDING` is the epitope — the residues you want the binder to contact, in
UniProt numbering (`"352..363,367..372"`).

Two things worth knowing before you choose:

* A BoltzGen minibinder contacts roughly **21 residues**. Specifying far more
  does not aim it harder, it just aims it vaguely.
* `CROP` trims the target to a neighbourhood of the epitope. It is how you keep
  a 900-residue protein inside a free GPU. Residues outside the crop simply do
  not exist for the model.

Good epitopes are solvent-exposed, high-pLDDT and compact. If you have no basis
for choosing one, that is the honest limit of this notebook: choosing well is
the hard part, and it is the part left to you.
"""))

cells.append(code("""#@title Target
UNIPROT     = "P14868"              #@param {type:"string"}
CHAIN       = "A"                   #@param {type:"string"}
CROP        = "233..501"            #@param {type:"string"}
BINDING     = "352..363,367..372,374..375"  #@param {type:"string"}
BINDER_LEN  = "55..130"             #@param {type:"string"}
MASK_CUT_FACES = True               #@param {type:"boolean"}
MASK_WIDTH  = 3                     #@param {type:"integer"}

import urllib.request, json, pathlib, textwrap
work = pathlib.Path("/content/work"); (work / "targets").mkdir(parents=True, exist_ok=True)
(work / "specs").mkdir(exist_ok=True)

cif = work / "targets" / f"AF-{UNIPROT}-F1.cif"
if not cif.exists():
    # Resolve the download URL through the AlphaFold DB API rather than
    # building it from a version string. AFDB revises its models and retires
    # the old filenames: the hardcoded "_v4" URL this notebook shipped with
    # began returning 404 for every accession once the database moved to v6.
    # The API always names the file that currently exists.
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT}"
    print(f"resolving {UNIPROT} via {api}", flush=True)
    try:
        with urllib.request.urlopen(api, timeout=60) as r:
            meta = json.load(r)
        if not meta:
            raise ValueError("accession has no AlphaFold DB entry")
        url = meta[0]["cifUrl"]
        afdb_version = meta[0].get("latestVersion", "?")
    except Exception as e:
        raise SystemExit(f"Could not resolve {UNIPROT} in AlphaFold DB: {e}\\n"
                         "Check the accession at https://alphafold.ebi.ac.uk/\\n"
                         f"or upload a .cif to {cif} yourself.")
    print(f"fetching {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, cif)
    except Exception as e:
        raise SystemExit(f"Could not fetch {UNIPROT} from AlphaFold DB: {e}\\n"
                         f"Upload a .cif to {cif} yourself to continue.")
    print(f"AlphaFold DB model version: v{afdb_version}")
print(f"target {cif}  ({cif.stat().st_size/1e6:.1f} MB)")

# Cropping invents surfaces. The residues at a cut are buried in the real
# protein but fully exposed in the crop, and the model will cheerfully design a
# binder against that artefact. Mask both cut boundaries with not_binding. If a
# boundary is the true chain terminus the mask is simply redundant, which costs
# nothing.
_lo, _hi = (int(x) for x in CROP.split(".."))
_mask = []
if MASK_CUT_FACES and _hi - _lo > 2 * MASK_WIDTH:
    _mask = [f"{_lo}..{_lo + MASK_WIDTH - 1}", f"{_hi - MASK_WIDTH + 1}..{_hi}"]
MASK_LINE = (f"            not_binding: {','.join(_mask)}\\n" if _mask else "")

spec = textwrap.dedent(f'''\\
# Generated by the BlitzGen Colab notebook.
# The epitope below was chosen by hand, not by a scored triage pipeline.
entities:
  - protein:
      id: B
      sequence: {BINDER_LEN}
  - file:
      path: ../targets/AF-{UNIPROT}-F1.cif
      include:
        - chain:
            id: {CHAIN}
            res_index: {CROP}
      binding_types:
        - chain:
            id: {CHAIN}
            binding: {BINDING}
{MASK_LINE}''')
spec_path = work / "specs" / f"{UNIPROT}_notebook.yaml"
spec_path.write_text(spec)
print(f"\\nspec {spec_path}:\\n"); print(spec)

n_bind = sum(len(range(int(a), int(b) + 1)) if ".." in seg else 1
             for seg in BINDING.split(",")
             for a, b in [seg.split("..") if ".." in seg else (seg, seg)])
print(f"masked cut faces: {','.join(_mask) if _mask else 'none'}")
print(f"epitope is {n_bind} residues; a minibinder contacts ~21."
      + ("  (reasonable)" if 10 <= n_bind <= 40 else "  <-- consider narrowing"))"""))

# ----------------------------------------------------------- 6. run ---------
cells.append(md("""---
## 6 · Design

Six pipeline steps run as separate processes — design, inverse folding, folding,
design folding, analysis, filtering. Each loads its own checkpoint, which is why
they add about **2 minutes** of startup across the six.

Measured for 8 designs at this notebook's defaults: **40 min on a Colab T4** —
design 18 min, folding 16 min, the rest under 5 min combined. A smaller target
on an A100 took about 9 min.
Do not close the tab: Colab disconnects after 90 idle minutes.
"""))

cells.append(code("""#@title Run
NUM_DESIGNS = 8                     #@param {type:"integer"}
PROTOCOL    = "protein-anything"    #@param ["protein-anything", "peptide-anything", "nanobody-anything", "antibody-anything"]

# PROTOCOL selects BoltzGen's analysis and filtering presets. It does not
# change the design architecture:
#
#   protein-anything    general binder. The default, and what BinderBase uses.
#   peptide-anything    filters cysteines and drops the hydrophobic-patch
#                       checks. Pair it with a shorter BINDER_LEN in step 5.
#   nanobody-anything   filters cysteines, hydrophobic-patch checks off.
#   antibody-anything   same presets as nanobody.
#
# Two of BoltzGen's protocols are deliberately absent from this list.
# protein-small_molecule needs a ligand in the spec, and step 5 builds a
# protein-only spec from AlphaFold. protein-redesign rewrites an existing
# chain instead of designing a new binder, so it does not fit this spec shape.
#
# NUM_DESIGNS costs time, not memory: peak VRAM was measured flat at 8.5-8.7 GB
# from 8 designs to 100. Lowering it will not rescue an out-of-memory failure --
# narrow CROP in step 5 for that.
import subprocess, sys, time, os, re

cmd = [
    "boltzgen", "run", str(spec_path),
    "--output", "/content/work/run",
    "--protocol", PROTOCOL,
    "--num_designs", str(NUM_DESIGNS),
    "--cache", os.environ["HF_HOME"],
    "--folding_checkpoint", str(CKPT),
    "--folding_config", "/content/fold_blitz_k16.yaml",
    "--seed", "42",
    "--config", "design", f"trainer.precision={PRECISION}",
    "--config", "filtering", f"budget={NUM_DESIGNS}",
]
print(" ".join(cmd), "\\n" + "=" * 78)

# boltzgen writes to a pipe here, not a terminal, so Python block-buffers its
# output and the progress bar arrives in ~8 KB dumps -- a 17-minute design step
# can jump straight from 0/8 to 8/8. PYTHONUNBUFFERED makes the child flush as
# it goes, and readline() avoids the read-ahead that file iteration applies to
# pipes.
env = dict(os.environ, PYTHONUNBUFFERED="1")
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1, env=env)

# tqdm redraws the same bar many times a second. Each redraw ends in a carriage
# return, which text mode turns into its own line, so printing every one would
# bury the real output. Print a progress line only when its "n/total" changes.
_bar  = re.compile(r"\\b(\\d+)/(\\d+)\\s*\\[")
_last = None
for line in iter(p.stdout.readline, ""):
    hit = _bar.search(line)
    if hit:
        if hit.group(0) == _last:
            continue
        _last = hit.group(0)
    sys.stdout.write(line if line.endswith("\\n") else line + "\\n")
    sys.stdout.flush()
p.wait()
print("=" * 78)
print(f"exit {p.returncode}   elapsed {time.time()-t0:.0f}s")
if p.returncode != 0:
    print("\\nIf this is CUDA out-of-memory, narrow CROP and rerun cell 5 and 6.")"""))

# -------------------------------------------------------- 7. results --------
cells.append(md("""---
## 7 · Results

`final_ranked_designs/` holds the ranked designs and their metrics.

Reading them: **`design_to_target_iptm`** is the headline number — the model's
own confidence in the binder-target interface. **`min_design_to_target_pae`** is
the predicted error across that interface, where lower is better.
**`filter_rmsd`** is how far the refolded binder drifted from the designed
backbone; large values mean the design does not fold to what was drawn.
**`plip_hbonds_refolded`** and **`delta_sasa_refolded`** describe the interface
itself — hydrogen bonds, and buried surface area in square angstroms.

A confident-looking score is not a binder. These are *in-silico* designs, and the
published hit rates for this class of method are single-digit percent before
experimental screening.
"""))

cells.append(code("""import glob, csv, os, shutil

hits = glob.glob("/content/work/run/final_ranked_designs/all_designs_metrics.csv")
if not hits:
    print("No metrics file - did cell 6 finish without error?")
else:
    with open(hits[0]) as fh:
        rows = list(csv.DictReader(fh))
    print(f"{len(rows)} designs\\n")
    cols = [c for c in ("id", "final_rank", "design_to_target_iptm",
                        "min_design_to_target_pae", "design_ptm", "filter_rmsd",
                        "plip_hbonds_refolded", "delta_sasa_refolded")
            if rows and c in rows[0]]
    print("  ".join(f"{c[:22]:>22s}" for c in cols))
    print("-" * (24 * len(cols)))
    def k(r):
        try: return -float(r.get("design_to_target_iptm") or 0)
        except ValueError: return 0
    for r in sorted(rows, key=k)[:10]:
        print("  ".join(f"{str(r.get(c,''))[:22]:>22s}" for c in cols))

    shutil.make_archive("/content/blitzgen_designs", "zip", "/content/work/run")
    print("\\nStructures (.cif) are under /content/work/run/final_ranked_designs/")
    try:
        from google.colab import files
        files.download("/content/blitzgen_designs.zip")
    except Exception as e:
        print(f"(auto-download unavailable: {e}; the zip is at "
              "/content/blitzgen_designs.zip)")"""))


# ------------------------------------------------------- 8. visualise ------
cells.append(md("""---
## 8 · Look at them

A score is not a structure. This renders any design that passes a cutoff you
choose, with the target in grey and the binder coloured by its own per-residue
confidence (red low, blue high). Residues within 5 A of the target are drawn as
sticks, so you can see whether the interface is real contact or a near miss.

Loosen `CUTOFF` if the dropdown comes up empty; the cell prints the observed
range of whichever metric you pick.
"""))

cells.append(code("""#@title Visualise designs
FILTER_BY = "design_to_target_iptm"  #@param ["design_to_target_iptm", "design_ptm", "min_design_to_target_pae", "filter_rmsd", "plip_hbonds_refolded", "delta_sasa_refolded", "liability_score"]
CUTOFF    = 0.6  #@param {type:"number"}

import subprocess, sys, csv, pathlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py3Dmol==2.5.5"],
               check=False)
import py3Dmol, ipywidgets as W
from IPython.display import display, clear_output

# For these, a smaller number is the better one. Everything else is
# higher-is-better, which decides the direction of the cutoff and the sort.
LOWER_IS_BETTER = {"min_design_to_target_pae", "filter_rmsd", "liability_score",
                   "design_largest_hydrophobic_patch_refolded"}

RANKED = pathlib.Path("/content/work/run/final_ranked_designs")
csvs = (sorted(RANKED.glob("final_designs_metrics_*.csv"))
        or sorted(RANKED.glob("all_designs_metrics.csv")))
if not csvs:
    raise SystemExit("No metrics CSV in final_ranked_designs -- did step 6 finish?")
rows = list(csv.DictReader(open(csvs[0])))

def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None

vals = [v for v in (num(r, FILTER_BY) for r in rows) if v is not None]
if not vals:
    raise SystemExit(f"{FILTER_BY} is not present in {csvs[0].name}")
lower = FILTER_BY in LOWER_IS_BETTER
print(f"{FILTER_BY}: min {min(vals):.3f}   median {sorted(vals)[len(vals)//2]:.3f}"
      f"   max {max(vals):.3f}   ({'lower' if lower else 'higher'} is better)")

keep = [(v, r) for r, v in ((r, num(r, FILTER_BY)) for r in rows)
        if v is not None and (v <= CUTOFF if lower else v >= CUTOFF)]
keep.sort(key=lambda t: t[0], reverse=not lower)
print(f"{len(keep)} of {len(rows)} designs pass "
      f"{FILTER_BY} {'<=' if lower else '>='} {CUTOFF}")
if not keep:
    raise SystemExit("Nothing passes -- loosen CUTOFF and re-run this cell.")

options = []
for v, r in keep:
    hit = next(RANKED.glob(f"final_*_designs/*{r['id']}.cif"), None)
    if hit is None:
        continue
    rank = int(float(r.get("final_rank") or 0))
    options.append((f"rank {rank:>3}   {FILTER_BY} {v:.3f}   {r['id']}", str(hit)))
if not options:
    raise SystemExit("Metrics rows found but no matching .cif files.")

dd  = W.Dropdown(options=options, description="design:",
                 layout=W.Layout(width="640px"))
out = W.Output()

def render(_=None):
    with out:
        clear_output(wait=True)
        # Chain A is the designed binder, chain B the target crop. The B-factor
        # column carries per-residue confidence, which is what colours the binder.
        view = py3Dmol.view(width=760, height=520)
        view.addModel(pathlib.Path(dd.value).read_text(), "cif")
        view.setStyle({"chain": "B"}, {"cartoon": {"color": "lightgrey"}})
        view.setStyle({"chain": "A"},
                      {"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb",
                                                   "min": 0.5, "max": 1.0}}})
        view.addStyle({"chain": "A",
                       "within": {"distance": 5, "sel": {"chain": "B"}}},
                      {"stick": {"radius": 0.15}})
        view.zoomTo({"chain": "A"})
        view.show()

dd.observe(render, names="value")
display(dd, out)
render()"""))

# --------------------------------------------------------- 9. caveats ------
cells.append(md("""---
## Known limitations

**This needs BlitzGen, not BoltzGen.** `--folding_config` and `--seed` are
BlitzGen additions; upstream BoltzGen has neither, and without
`--folding_config` the K16 weights silently run on a 200-step Boltz-2 schedule —
no error, plausible-looking structures, wrong sampler. Cell 2 pins
[`@@BLITZGEN_SRC@@`](https://github.com/EpiGenomicsCode/blitzgen/tree/v0.1.0).

**Run on a T4.** 8 designs at the default settings took 40 min end to end on a
Colab free T4 in fp32, with no intervention. The `sm_75` question that used to
sit here does not arise: below compute capability 8.0 BoltzGen sets
`use_kernels=False` itself, so the `cuequivariance` triangle kernels are never
reached on a T4 and the pure-PyTorch path is what runs. Its cost is already in
that 40 min.

**Report the precision you ran at.** On a pre-Ampere card the notebook folds
in `fp32` rather than `bf16-mixed`. Same model and same weights, different
arithmetic and therefore different designs — worth stating alongside any
results you publish.
"""))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

for _cell in nb["cells"]:
    _cell["source"] = [
        _line if "@@" not in _line
        else __import__("functools").reduce(
            lambda t, kv: t.replace(*kv), SUBST.items(), _line)
        for _line in _cell["source"]]

_left = [t for c in nb["cells"] for l in c["source"] for t in SUBST if t in l]
if _left:
    sys.exit(f"unsubstituted tokens remain: {sorted(set(_left))}")

out = (pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
       if "--out" in sys.argv else ROOT / "BlitzGen_BinderDesign.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells, v{VERSION})")
