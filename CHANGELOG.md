# Changelog

Versions are of the notebook, not of BoltzGen or the K16 weights — those are
pinned in `provenance.json` and move independently.

## 0.1.0 — unreleased

First version. Not yet runnable: `blitzgen_src` is a placeholder pending
publication of the K16 fork.

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
