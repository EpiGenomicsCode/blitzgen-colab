#!/usr/bin/env python3
"""Validate the committed notebook. Run by CI; run it before you commit.

Checks, in order of how much damage they prevent:

  1. the committed .ipynb is exactly what tools/build_notebook.py produces
  2. the folding config embedded in the notebook is byte-identical to
     config/fold_blitz_k16.yaml
  3. config/fold_blitz_k16.yaml matches the sha256 in provenance.json
  4. the notebook states the checkpoint sha256 from provenance.json
  5. no cell outputs or execution counts are committed
  6. BLITZGEN_SRC is pinned to an immutable ref

(6) is a warning, not a failure, while the fork is unpublished -- otherwise the
repository could not hold a buildable notebook at all. It becomes an error the
moment a real source is set, because a notebook someone saved in March must
still install the same code in September.
"""
import hashlib, json, pathlib, re, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "BlitzGen_BinderDesign.ipynb"
CFG = ROOT / "config" / "fold_blitz_k16.yaml"
PROV = json.loads((ROOT / "provenance.json").read_text())

fail, warn = [], []


def check(cond, msg):
    (fail if not cond else []).append(msg)


# 1 -- notebook equals generator output -------------------------------------
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td) / "nb.ipynb"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py"),
                        "--out", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        fail.append(f"build_notebook.py exited {r.returncode}: {r.stderr.strip()}")
    else:
        check(tmp.read_bytes() == NB.read_bytes(),
              "BlitzGen_BinderDesign.ipynb differs from build_notebook.py output. "
              "The notebook is generated -- edit tools/build_notebook.py and "
              "re-run it, do not hand-edit the .ipynb.")

nb = json.loads(NB.read_text())
cells = nb["cells"]
src_all = "".join(l for c in cells for l in c["source"])

# 2 -- embedded config is verbatim ------------------------------------------
cfg_text = CFG.read_text()
check(cfg_text in src_all,
      "the folding config in the notebook is not byte-identical to "
      f"{CFG.relative_to(ROOT)} -- the notebook would fold on a different "
      "schedule than the campaign")

# 3 -- config matches provenance --------------------------------------------
cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()
check(cfg_sha == PROV["fold_config_sha256"],
      f"config sha256 {cfg_sha} != provenance.json {PROV['fold_config_sha256']}")

# 4 -- checkpoint sha is stated ---------------------------------------------
check(PROV["k16_checkpoint_sha256"] in src_all,
      "the notebook does not state the K16 checkpoint sha256 from provenance.json")

# 5 -- no outputs committed --------------------------------------------------
dirty = [i for i, c in enumerate(cells)
         if c.get("outputs") or c.get("execution_count") is not None]
check(not dirty, f"cells {dirty} carry outputs or execution counts; "
                 "committed notebooks must be clean so diffs stay readable")

# 6 -- install source is pinned ---------------------------------------------
src = PROV["blitzgen_src"]
pinned = bool(re.search(r"@(v?\d+\.\d+\.\d+|[0-9a-f]{40})$", src))
if "UNPUBLISHED" in src:
    warn.append("blitzgen_src is still the UNPUBLISHED placeholder -- the "
                "notebook cannot run anywhere until the fork is public")
elif not pinned:
    fail.append(f"blitzgen_src {src!r} is not pinned to a tag or 40-char commit "
                "sha; a branch ref lets a saved notebook change behaviour")

for w in warn:
    print(f"WARN  {w}")
for f in fail:
    print(f"FAIL  {f}")
print(f"\n{len(cells)} cells, {NB.stat().st_size/1024:.1f} KB, "
      f"v{(ROOT / 'VERSION').read_text().strip()} -- "
      f"{len(fail)} failed, {len(warn)} warnings")
sys.exit(1 if fail else 0)
