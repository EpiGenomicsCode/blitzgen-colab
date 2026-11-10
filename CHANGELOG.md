# Changelog

Versions are of the notebook, not of BoltzGen or the K16 weights — those are
pinned in `provenance.json` and move independently.

## 0.1.1 — 2026-11-10

Everything below came out of the first real Colab (T4) run of 0.1.0. Nothing
was released between 0.1.0 and this, so it all lands as one version.

- **New step 8: a 3D viewer.** Pick a metric and a cutoff; the designs that
  pass populate a dropdown, and the selected one renders interactively — target
  in grey, binder coloured by per-residue confidence, interface residues within
  5 A drawn as sticks. Uses `py3Dmol`, not PyMOL: open-source PyMOL cannot show
  an interactive view in Colab, only ray-traced stills, while `py3Dmol` is a
  small pure-Python wrapper over WebGL that renders in the output cell.

  The filter knows which metrics are lower-is-better
  (`min_design_to_target_pae`, `filter_rmsd`, `liability_score`,
  `design_largest_hydrophobic_patch_refolded`) and flips both the comparison
  and the sort accordingly. The cell prints each metric's observed min, median
  and max, so a cutoff can be chosen rather than guessed.

- **Fixed: the results table asked for columns that do not exist.** Step 7
  selected `complex_plddt`, `bb_rmsd_design` and `quality_score`, none of which
  BoltzGen emits, and the step 7 prose documented two of them as the headline
  metrics. Because the column list was filtered by presence, this failed
  silently as a narrower table. Replaced with the real columns:
  `final_rank`, `min_design_to_target_pae`, `filter_rmsd`,
  `plip_hbonds_refolded` and `delta_sasa_refolded`.

- **Timings are now measured on a T4, not extrapolated.** 0.1.0 shipped every
  performance figure as an A100 measurement scaled by a memory-bandwidth ratio,
  with a standing warning that the notebook had never been run on a T4. It has
  now been run end to end on a Colab free T4 at the default settings: **40 min**
  for 8 designs in fp32 (design 18 min, folding 16 min, ~2 min of startup across
  the six steps). The README warning block, the `30-40 min` estimates and the
  "Untested on a T4" note are replaced with the measurement.

  This also retires an open question rather than answering it: the docs warned
  that `cuequivariance`'s `sm_75` support was unknown and that cell 6 might fail
  inside `triangular.py`. It cannot. Below compute capability 8.0 BoltzGen sets
  `use_kernels=False` itself, so those kernels are never reached on a T4, and
  the pure-PyTorch path that runs instead costs 1.14x at batch size 1 and 1.35x
  at the batch size used for 100 designs — already inside the 40 min.

- **Fixed: no target could be downloaded.** Step 5 built its URL as
  `AF-{acc}-F1-model_v4.cif`, and AlphaFold DB has since moved to v6 and
  retired the v4 filenames, so that URL returned 404 for *every* accession —
  0.1.0 could not fetch any target at all. Step 5 now resolves the file through
  the AFDB API (`/api/prediction/{acc}`) and uses the `cifUrl` it returns, so
  it follows the database forward instead of pinning a filename AFDB never
  promised to keep. The model version fetched is printed.

  AFDB v6 coordinates are not identical to v4, so a target fetched now may
  differ from the same accession fetched before this change. The v4 files are
  gone, so there is no way to avoid that.

- **Fixed: long cells looked like hangs.** Two separate causes, same symptom.
  The install cell ran pip with `-q` under `capture_output=True`, printing
  nothing until pip exited — 4-8 minutes of silence. Step 6 piped `boltzgen`,
  which block-buffers when its output is not a terminal, so a 17-minute design
  step on a T4 jumped from `0/8` straight to `8/8`. The install cell now
  streams pip line by line; step 6 runs the child with `PYTHONUNBUFFERED=1` and
  reads with `readline()` rather than iterating the pipe, which applies a
  read-ahead buffer of its own.

- **Progress-bar redraws are collapsed.** Unbuffering step 6 alone would have
  replaced silence with noise, since each tqdm repaint becomes its own line
  once text mode translates its carriage return. Step 6 prints a progress line
  only when `n/total` changes. Verified by replaying a real run log through the
  filter: every batch transition survives and non-progress output is untouched.

- **`NUM_DESIGNS` moved from step 5 to step 6.** It is a run parameter, not a
  property of the target, and nothing in the generated spec referenced it — but
  sitting on the target form it meant changing the design count forced a
  re-fetch and a spec rebuild.

- **`PROTOCOL` is now a dropdown** in step 6, replacing a hardcoded
  `protein-anything`: `protein-anything` (default), `peptide-anything`,
  `nanobody-anything`, `antibody-anything`, all checked against BoltzGen's own
  `protocol_configs`. These select analysis and filtering presets, not a
  different design architecture, and the cell says what each changes.

  `protein-small_molecule` and `protein-redesign` are excluded on purpose: the
  first needs a ligand in the spec while step 5 builds a protein-only spec, and
  the second rewrites an existing chain rather than designing a binder. A
  dropdown whose values cannot all produce a working run is worse than none.

- **The install cell reports a real version.** It printed
  `boltzgen (unversioned)`, since the import package carries no `__version__`.
  It now reads `importlib.metadata.version("blitzgen")` — a notebook built
  around a pinned stack should be able to say what it installed.

- Documented pip's resolver warnings against the Colab image (`jax`,
  `tifffile`, `dm-tree`, `ml-dtypes` want `numpy>=2.1`; `stumpy` wants
  `numba>=0.61.2`, against BlitzGen's `numpy==2.0.2` / `numba==0.61.0` pins).
  Verified none of those packages are on BlitzGen's import path. They are
  harmless, but they print in red and read as failure.

- Step 6 notes that `NUM_DESIGNS` costs time rather than memory — peak VRAM
  measured flat at 8.5-8.7 GB from 8 designs to 100 — so that lowering it is
  not mistaken for an out-of-memory remedy. Narrowing `CROP` is that remedy.

## 0.1.0 — 2026-11-09

First version.

- Pinned to [BlitzGen `v0.1.0`](https://github.com/EpiGenomicsCode/blitzgen/tree/v0.1.0),
  published 2026-11-09. Installed without a CUDA extra: the
  `cuequivariance_ops_*` wheels are large and never loaded by this pipeline, so
  requesting them would only add a way for the install to fail.

- 16-cell notebook: hardware probe → install → weights → K16 config → spec →
  run → results.
- Hardware probe selects `bf16-mixed` on compute capability ≥ 8.0 and `fp32`
  below, and rewrites the folding config to match. Both arms simulated.
- `fp16` deliberately unsupported — tested twice, including with an fp32 island
  around the alignment, and the sampler produces non-finite coordinates either
  way.
- Folding config vendored verbatim from the campaign
  (`sampling_steps: 16`, `sigma_min: 0.01`, `recycling_steps: 5`) and embedded
  byte-identically; CI enforces both.
- K16 checkpoint verified by sha256 before use; the notebook refuses to run on
  a mismatch.
- Spec generation masks the two faces created by cropping (`not_binding`), so
  designs are not drawn to surfaces that are artifacts of the crop.
