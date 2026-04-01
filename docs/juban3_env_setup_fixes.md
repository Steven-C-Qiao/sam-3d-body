# juban3 Environment Setup Fixes

Env: Python 3.12, torch 2.8.0+cu128, system CUDA 13.0.

## 1. pytorch3d — CUDA visibility flags not applied

**Problem:** pytorch3d's `setup.py` uses `torch.version.cuda` (12.8) to decide whether to add CUDA 13.0 symbol visibility flags. Since torch was compiled for 12.8, the check `major >= 13` never triggers, but the system `nvcc` is 13.0 — causing a linker error:

```
hidden symbol `_ZN6pulsar8Renderer7fill_bgILb1EEEvNS0_8RendererENS_7CamInfoEPKffj' isn't defined
final link failed: bad value
```

**Fix:** In `/scratch/cq244/pytorch3d/pytorch3d/setup.py`, replace the CUDA version detection block with one that reads the actual `nvcc --version` output:

```python
nvcc_major = 0
try:
    nvcc_out = subprocess.check_output(
        ["nvcc", "--version"], stderr=subprocess.STDOUT
    ).decode()
    m = re.search(r"release (\d+)\.", nvcc_out)
    if m:
        nvcc_major = int(m.group(1))
except Exception:
    cuda_version = torch.version.cuda
    if cuda_version is not None:
        nvcc_major = int(cuda_version.split(".")[0])
if nvcc_major >= 13:
    nvcc_args.extend([
        "--device-entity-has-hidden-visibility=false",
        "-static-global-template-stub=false",
    ])
```

Also add `import re, subprocess` at the top. Then install:

```bash
pip install -e /scratch/cq244/pytorch3d/pytorch3d --no-build-isolation
```

**Bonus:** pytorch3d's `libc10.so` won't resolve without torch's lib dir in `LD_LIBRARY_PATH`. Add a conda activation hook:

```bash
# /scratch/cq244/conda/condaenvs/<env>/etc/conda/activate.d/torch_lib_path.sh
export LD_LIBRARY_PATH="/scratch/cq244/conda/condaenvs/<env>/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
```

## 2. pymomentum-gpu — segfault in MHR.from_files

**Problem:** `pymomentum-gpu>=0.1.107` segfaults inside `character.with_blend_shape()` when paired with torch 2.8.0+cu128 on this system. Version `0.1.95.post0` is stable.

**Fix:**

```bash
pip install "pymomentum-gpu==0.1.95.post0"
```
