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
    "@@CAMPAIGN@@":       PROV["campaign"],
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

## What this is *not*

This is deliberately **not** identical to the [BinderBase](https://binderbase.org)
catalogue, and you should not mix the numbers without saying so:

| | BinderBase campaign | this notebook |
|---|---|---|
| precision | `bf16-mixed` | `bf16-mixed` on Ampere+, **`fp32` on older cards** |
| designs per target | 100 | ~8–20 (session limits) |
| epitope choice | HPA PrEST antibody-validated triage | **whatever you specify** |
| seed | fixed, 42 | fixed where supported |

The precision difference matters most. The diffusion sampler is stochastic, so
changing precision does not perturb designs — it produces **different** designs
drawn from the same distribution. Measured on 8 paired designs, fp32-vs-bf16
spread sat inside the bf16-vs-bf16 run-to-run floor on every comparable metric,
but that is an indication from a small sample, not a guarantee of equivalence.

**The epitope row is the big one.** BinderBase targets come from a triage pipeline
that picks antibody-validated, solvent-exposed, compact surface patches. Here you
pick the site. A binder designed against a poorly chosen epitope will still score
well — the model answers the question you asked.

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
| matches campaign | `@@CAMPAIGN@@` |

Newest version: <https://github.com/EpiGenomicsCode/blitzgen-colab>
"""))

# ------------------------------------------------------------ 1. probe ------
cells.append(md("""---
## 1 · Hardware probe

**Run this first.** It needs no install and finishes in seconds, so a card that
cannot run the job is discovered now rather than after a 7.5 GB download.

It picks the precision for you:

* **compute capability ≥ 8.0** (A100, L4, RTX 30xx/40xx) → `bf16-mixed`, same as the campaign.
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
print(f"=> precision   {PRECISION}" + ("   (Ampere+, matches the campaign)" if cap_f >= 8.0
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
        SPEED NOTE. Reference: 8 designs took 489 s (bf16) / 556 s (true fp32)
        on an A100. This workload is NOT matmul-bound -- disabling TF32 cost
        only 2.4% -- so the limit on an older card is memory bandwidth, not
        FLOPS. A T4 (320 GB/s vs the A100's 1555) should land near 30-40 min
        for 8 designs. That is an estimate from a bandwidth ratio, not a
        measurement; please report what you actually see.
    '''))"""))

# ---------------------------------------------------------- 2. install ------
cells.append(md("""---
## 2 · Install

`BLITZGEN_SRC` must point at a **public** install source providing
`--folding_config` and `--seed`. Upstream BoltzGen has neither.

> **If this cell fails**, the fork is not published yet. See the note at the
> bottom of the notebook — this is the one piece that cannot be worked around
> from inside Colab.
"""))

cells.append(code("""#@title Install BlitzGen
BLITZGEN_SRC = "@@BLITZGEN_SRC@@"  #@param {type:"string"}

import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", BLITZGEN_SRC],
                   capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-3000:])
if r.returncode != 0:
    raise SystemExit(
        "Install failed.\\n"
        "If this is a 404/auth error the repository is private or does not exist.\\n"
        "A public source is required; see the final section of this notebook.")

import importlib, boltzgen  # noqa
print("boltzgen", getattr(boltzgen, "__version__", "(unversioned)"))"""))

# ---------------------------------------------------------- 3. weights ------
cells.append(md("""---
## 3 · Weights  (~7.5 GB)

Two downloads:

* **`boltzgen/boltzgen-1`** (~5.6 GB) — the design and inverse-folding models.
* **`vinaymatt/Blitz-Boltz2 / blitz-boltz2-k16.ckpt`** (1.9 GB) — the Blitz K16
  folding student.

The K16 file is checked against the publisher's own `SHA256SUMS`. The expected
digest is `343f252b…`, which is **byte-identical to the checkpoint that produced
the BinderBase campaign** — so folding here is the same model, not a lookalike.

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
print(f"  OK  {got[:16]}...  matches the campaign checkpoint")

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

The config below is copied verbatim from the campaign's own
`fold_blitz_k16.yaml`, so this notebook folds on exactly the schedule BinderBase
folded on.
"""))

cells.append(code('K16_YAML = r"""' + K16_CFG + '"""\n\n' + """import pathlib, re

# The campaign config hardcodes precision: bf16-mixed, because every machine in
# the campaign was Ampere or newer. On a pre-Ampere card the design step would
# correctly run fp32 (set on the command line) while FOLDING quietly tried bf16
# and died. Both folding and design_folding read this one file, so rewriting it
# here covers both.
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
for choosing, that is the honest limit of this notebook — the BinderBase
catalogue exists because choosing well is the hard part.
"""))

cells.append(code("""#@title Target
UNIPROT     = "P14868"              #@param {type:"string"}
CHAIN       = "A"                   #@param {type:"string"}
CROP        = "233..501"            #@param {type:"string"}
BINDING     = "352..363,367..372,374..375"  #@param {type:"string"}
BINDER_LEN  = "55..130"             #@param {type:"string"}
NUM_DESIGNS = 8                     #@param {type:"integer"}
MASK_CUT_FACES = True               #@param {type:"boolean"}
MASK_WIDTH  = 3                     #@param {type:"integer"}

import urllib.request, pathlib, textwrap
work = pathlib.Path("/content/work"); (work / "targets").mkdir(parents=True, exist_ok=True)
(work / "specs").mkdir(exist_ok=True)

cif = work / "targets" / f"AF-{UNIPROT}-F1.cif"
if not cif.exists():
    url = f"https://alphafold.ebi.ac.uk/files/AF-{UNIPROT}-F1-model_v4.cif"
    print(f"fetching {url}")
    try:
        urllib.request.urlretrieve(url, cif)
    except Exception as e:
        raise SystemExit(f"Could not fetch {UNIPROT} from AlphaFold DB: {e}\\n"
                         "Check the accession, or upload a .cif to "
                         f"{cif} yourself.")
print(f"target {cif}  ({cif.stat().st_size/1e6:.1f} MB)")

# Cropping invents surfaces. The residues at a cut are buried in the real
# protein but fully exposed in the crop, and the model will cheerfully design a
# binder against that artefact. The campaign masks them with not_binding; so do
# we, at both cut boundaries. If a boundary is the true chain terminus the mask
# is simply redundant, which costs nothing.
_lo, _hi = (int(x) for x in CROP.split(".."))
_mask = []
if MASK_CUT_FACES and _hi - _lo > 2 * MASK_WIDTH:
    _mask = [f"{_lo}..{_lo + MASK_WIDTH - 1}", f"{_hi - MASK_WIDTH + 1}..{_hi}"]
MASK_LINE = (f"            not_binding: {','.join(_mask)}\\n" if _mask else "")

spec = textwrap.dedent(f'''\\
# Generated by the BlitzGen Colab notebook.
# NOT from the HPA PrEST triage used by the BinderBase campaign -- the epitope
# below was chosen by the notebook user, not by select_patch.py.
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
there is a fixed cost of roughly **4 minutes** before per-design work begins.

Expect **~8 min** for 8 designs on an A100, **~30–40 min** on a T4.
Do not close the tab: Colab disconnects after 90 idle minutes.
"""))

cells.append(code("""import subprocess, sys, time, os

cmd = [
    "boltzgen", "run", str(spec_path),
    "--output", "/content/work/run",
    "--protocol", "protein-anything",
    "--num_designs", str(NUM_DESIGNS),
    "--cache", os.environ["HF_HOME"],
    "--folding_checkpoint", str(CKPT),
    "--folding_config", "/content/fold_blitz_k16.yaml",
    "--seed", "42",
    "--config", "design", f"trainer.precision={PRECISION}",
    "--config", "filtering", f"budget={NUM_DESIGNS}",
]
print(" ".join(cmd), "\\n" + "=" * 78)
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
for line in p.stdout:
    sys.stdout.write(line)
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
own confidence in the binder-target interface. **`complex_plddt`** is per-atom
confidence. **`bb_rmsd_design`** is how far the refolded binder drifted from the
designed backbone; large values mean the design does not fold to what was drawn.

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
    cols = [c for c in ("id", "design_to_target_iptm", "complex_plddt",
                        "design_ptm", "bb_rmsd_design", "quality_score")
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

# --------------------------------------------------------- 8. caveats ------
cells.append(md("""---
## Known limitations, and the one open prerequisite

**The install source must be public.** `--folding_config` and `--seed` are
BlitzGen additions; upstream BoltzGen has neither, and without
`--folding_config` the K16 weights silently run on a 200-step Boltz-2 schedule.
As of 2026-11-09 the current K16 fork exists only as an unversioned local
directory — `EpiGenomicsCode/boltzgen` is private and predates K16. **Publishing
it is the one step that cannot be done from inside this notebook.**

**Untested on a T4.** Every number here was measured on an A100 and extrapolated
by a memory-bandwidth ratio. In particular, whether the `cuequivariance` triton
kernels used by triangle attention support `sm_75` is **unknown**. If cell 6
fails inside `triangular.py`, add `--config design use_kernels=false` — there is
a pure-PyTorch fallback, at an unmeasured cost in speed and memory.

**No epitope triage.** You choose the site; the campaign used a scored pipeline.
This is the largest quality gap between a notebook run and a catalogue entry.

**fp32 results are not campaign results.** Same model, same weights, different
arithmetic, different designs. Say so if you publish them.
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
