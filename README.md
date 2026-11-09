# BlitzGen Colab — free de novo binder design

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EpiGenomicsCode/blitzgen-colab/blob/main/BlitzGen_BinderDesign.ipynb)
[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/kernels/welcome?src=https://github.com/EpiGenomicsCode/blitzgen-colab/blob/main/BlitzGen_BinderDesign.ipynb)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Design de novo protein binders against an epitope you choose, on a free cloud
GPU, in about half an hour. Runs
[BlitzGen](https://github.com/EpiGenomicsCode/blitzgen) `v0.1.0`: BoltzGen for
design, [Blitz K16](https://github.com/EpiGenomicsCode/Blitz) — a 16-step
distilled Boltz-2 student — for folding. The same models that built the
[BinderBase](https://binderbase.org) catalogue.

The GPU is **yours**, not ours. Each run happens in your own Colab or Kaggle
account on your own quota. Nothing is submitted to a server we operate, and
there is nothing to pay for or sign up to beyond the host itself.

> ### ⚠️ Not yet run on a T4
>
> Every performance number below was measured on an A100 and scaled by memory
> bandwidth. Whether the `cuequivariance` triton kernels used by triangle
> attention support `sm_75` is **unknown**; if cell 6 fails inside
> `triangular.py`, add `--config design use_kernels=false` for a pure-PyTorch
> fallback at an unmeasured cost.

## Hosts

| | GPU | session | quota | notes |
|---|---|---|---|---|
| Colab free | T4 16 GB | ~12 h cap, **90 min idle** | floats | the idle timeout is the real constraint — don't walk away mid-run |
| Kaggle free | T4 ×2 16 GB | 12 h | ~30 h/week | longer leash; the better default for a 30–40 min job |

Kaggle's weekly quota is not officially published and does drift; treat the
number as indicative.

## What to expect

Measured on an A100 and scaled to a T4 by memory bandwidth — **the notebook has
not yet been run on a T4**:

| | |
|---|---|
| peak VRAM | 8.9 GB fp32 / 8.6 GB bf16 — fits a 16 GB T4 with room to spare |
| 8 designs | ~30–40 min on a T4 (8 min measured on an A100) |
| download | ~7.5 GB of weights, once per session |

Cards older than Ampere (compute capability < 8.0, which includes the T4) have
no `bf16`, so the notebook detects this and switches to `fp32`. That costs 4%
VRAM and 11% wall-clock. `fp16` was tested and **does not work** — the sampler
produces non-finite coordinates and the downstream SVD fails.

## Layout

```
BlitzGen_BinderDesign.ipynb   generated — do not hand-edit
tools/build_notebook.py       the generator; edit this instead
tools/check_notebook.py       validation, also run by CI
config/fold_blitz_k16.yaml    vendored verbatim from the campaign
provenance.json               every version fact, in one place
```

## Development

```sh
python3 tools/build_notebook.py    # regenerate the notebook
python3 tools/check_notebook.py    # validate before committing
```

The notebook is generated so that the folding config cannot silently drift from
the campaign's. `check_notebook.py` re-runs the generator and byte-compares, so
a hand-edit to the `.ipynb` is a CI failure rather than a mystery six months on.

### Versioning

`provenance.json` is the single source of truth; `build_notebook.py` substitutes
its values into the cells, and the notebook prints them on its first page. A
user who saved a copy can tell exactly what they have.

Change a value there, bump `VERSION`, and add a `CHANGELOG.md` entry in the same
commit. `blitzgen_src` must be pinned to a tag or a 40-character commit sha —
CI rejects a branch ref, because `@main` would let a saved notebook change
behaviour underneath whoever saved it.

## Credit

BoltzGen is MIT, © 2025 Hannes Stärk. Blitz K16 weights:
[vinaymatt/Blitz-Boltz2](https://huggingface.co/vinaymatt/Blitz-Boltz2).
This repository is MIT.
