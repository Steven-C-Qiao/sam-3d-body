# Claude Code Context

**Host: juban.** `/scratch` below means juban's local disk. If `hostname -s` says otherwise, you are on
a different machine and these paths point at unrelated storage — reach this repo at
`/scratches/juban/cq244/sam-3d-body` and use that machine's own environment.

**This copy:** `origin` = github.com/Steven-C-Qiao/sam-3d-body · `main` @ 472af97 · 6 ahead of origin,
16 files uncommitted. The same-named copy on columbo2 is ~80 commits behind, has no NF head, and lacks
the `-C`/`--lr` flags — it is not the same repo.

## Project Overview

SAM 3D Body reconstructs full 3D human body meshes (pose, shape, camera) from a single RGB image. It uses a promptable architecture (optional 2D keypoint/mask inputs) and a custom parametric body representation called **MHR (Momentum Human Rig)**, which decouples skeletal structure from surface shape.

## Research Focus: Probabilistic Uncertainty via Normalising Flows

The core research extends SAM 3D Body with a **conditional normalising flow** that models uncertainty over MHR body parameters. This enables:

1. Per-image probability distributions over pose and shape (not just point estimates)
2. Multi-view fusion via importance sampling

### Factorised Autoregressive NF Head (`sam_3d_body/models/heads/prohmr_ar_head.py`)

The flow (`NFARHead`) operates on **residuals** (GT minus mean prediction) and is factorised into two sequential stages:

**Stage 1 — Shape & Scale:** `p(Δβ | c, μβ)`

- Models residuals over 45 shape + 10 scale parameters
- Context: `[flow_context (1024D), shape_mean (45D), scale_mean_selected (10D)]` → projected to 2048D via `beta_context_proj`
- Flow: `ConditionalGlow` with 4 layers, 1024 hidden features

**Stage 2 — Pose (autoregressive on shape):** `p(Δθ | c, μθ, Δβ)`

- Models residuals over 39 3-DOF joint angles + 34 1-DOF joint angles (hands excluded)
- Context: `[flow_context, shape_sample, scale_sample, aa_3dofs, params_1dofs]` → projected to 2048D via `pose_context_proj`
- Conditioned on the *sampled* shape from stage 1, capturing shape-pose correlation
- Flow: `ConditionalGlow` with 4 layers, 1024 hidden features

Pose is parameterised as axis-angles (3-DOF joints) and scalar angles (1-DOF joints), converted from the continuous 6D representation used by the mean prediction head.

### Training Loss (`sam_3d_body/losses/nf_loss.py`)

Maximises **conditional log-likelihood of the GT residual** under both flows:
`L = -log p(Δβ_gt | c) - log p(Δθ_gt | c, Δβ_gt)`

For the `nf_ar` head type, the true shape residual is used to build the stage-2 context (teacher forcing).

### Multi-View Fusion via Importance Sampling (`sam_3d_body/models/meta_arch/nf_merging.py`)

At test time with multiple views of the same subject, shape/scale predictions are fused using the NF stage-1 likelihoods as importance weights:

For each view `i`, draw samples `β_i^k ~ p(β | I_i)`. Weight by likelihood under all other views:

```
w_i^k ∝ ∏_{j ≠ i} p(β_i^k | I_j)
```

Merged shape `β*` = importance-weighted mean over all `V × S` candidates (softmax weights).

**Only shape and scale are merged** — pose is view-dependent and kept per-view.

### Key Design Choices

- Flows operate on **residuals from the mean prediction**, not absolute parameters — keeps the flow's job tractable
- Shape/scale factored before pose — shape is view-invariant, pose is not
- `EGL_DEVICE_ID` must match `CUDA_VISIBLE_DEVICES` (set in `scripts/train.py`) to keep pyrender on the training GPU

## Key Files


| File                                         | Role                                                            |
| -------------------------------------------- | --------------------------------------------------------------- |
| `sam_3d_body/models/heads/prohmr_ar_head.py` | Factorised autoregressive NF head (NFARHead)                    |
| `sam_3d_body/models/meta_arch/nf_merging.py` | Multi-view IS merging + MHR output computation                  |
| `sam_3d_body/losses/nf_loss.py`              | NLL training loss for the NF head                               |
| `sam_3d_body/trainer.py`                     | PyTorch Lightning training loop, visualisation, multi-view eval |
| `sam_3d_body/models/meta_arch/sam3d_body.py` | Core model architecture                                         |
| `sam_3d_body/visualization/renderer.py`      | pyrender-based mesh renderer (EGL offscreen)                    |
| `scripts/train.py`                           | Training entry point (`--gpus`, `--dev`, `--plot` flags)        |

## Environment

pixi, via `pixi.toml` + `pixi.lock` at the repo root — Python 3.12.13, torch 2.13.0 (CUDA 13.0),
pytorch3d 0.7.9 (`cuda130` build), lightning 2.6.5.

```bash
pixi run python scripts/train.py -E exp/exp_NNN_tag --gpus 2
```

Use `pixi run`, **not** the interpreter path. conda-forge patches triton to find Blackwell's `ptxas`
through `$CONDA_PREFIX`; calling `.pixi/envs/default/bin/python` directly leaves it unset and fails with
`Cannot find ptxas-blackwell`. No `.claude/settings.json` is wired here for the same reason.

The conda env `prohmr` still runs this repo and stays as a fallback. It is not the env of record.

## Running

`scripts/train.py` — `-E/--experiment_dir`, `-R/--resume_training_states`, `-L/--load_from_ckpt`,
`--gpus`, `-C/--config`, `--lr`, `--dev`, `--plot`. `--gpus` is required in practice: it is assigned
straight into `CUDA_VISIBLE_DEVICES`.

`scripts/test.py` mirrors those; `scripts/merging.py` adds `-D/--dataset_name`, `--method`,
`--num_samples`, `--noplot`. Also `scripts/demo_nf_multiview.py`.

**juban is shared** — check `nvidia-smi` before claiming a GPU. 8 × RTX PRO 6000 Blackwell, 96 GB each.

## Config

Defaults in `sam_3d_body/configs/config.py` (`get_config_defaults`). `-C` is merged unconditionally
right after them (`scripts/train.py:32`), so an override YAML is the way to change a run.
`TRAIN.MAX_STEPS` is wired to `pl.Trainer(max_steps=...)` (`scripts/train.py:181`), default `-1` — use
it to bound smoke tests rather than a shell timeout.

`--dev` is not merely a shorter run: it forces `exp_dir` to `exp/exp_test`, batch 2, 4 workers, and the
`static-hdri-bbox44-smplx` dataset.

## Data

`paths.py` sets `BEDLAM2_PATH = /scratch/cq244/BEDLAM2/`, which **does not exist** — BEDLAM2 moved,
labels to `/scratch/cq244/datasets/BEDLAM2_labels/` and images to `/scratch/cq244/BEDLAM/b2/`. The
constant serves both, so no single symlink fixes it; the BEDLAM2 refactor is half-done.

This does not block training: commit `c1fc0f4` renamed the BEDLAM1 ids to `-bbox44-smplx`, so the
default `DATASETS_AND_RATIOS` is BEDLAM1, resolving to `data/training_images/<seq>/png` and
`data/training_labels/all_npz_12_training_mhr_conditioned/<seq>.npz` — all present. Only the
moyo/citysample entries are affected.

## Outputs

`exp/<run>/` — `saved_models/`, `lightning_logs/`, `vis/`, `merge_vis_bedlam/`, `merge_vis_4d-dress/`,
`viz_top_dims/`, plus the dumped `config.yaml`.

## Operational notes

Short-horizon metrics are **not reproducible run-to-run**. Two runs with identical seed and config,
shuffling off, workers 0, augmentation disabled gave `train_loss` -11.43 vs -7.91 at 12 steps.
Ablations need enforced determinism or enough repeats to clear that noise floor.

`nflows` is the ProHMR fork (`nkolot/nflows@26388ed`), pinned in `pixi.toml`. It alters `ActNorm`,
`LULinear` and the arity of `sample_and_log_prob`; moving to upstream would change numerics, not just
imports.
