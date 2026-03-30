# Claude Code Context

## Environment

To activate the Python environment for this project:

```bash
source /home/mifs/cq244/miniconda3/etc/profile.d/conda.sh && conda activate prohmr
```

**Why `source ~/.bashrc` doesn't work:** `.bashrc` exits early for non-interactive shells (line 6-9 `case $-`), so conda never gets initialised. Source `conda.sh` directly instead.

## GPU Usage

This is a shared cluster. Always prioritise GPUs 0 and 1 for any runs. Set this via:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

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

| File | Role |
|------|------|
| `sam_3d_body/models/heads/prohmr_ar_head.py` | Factorised autoregressive NF head (NFARHead) |
| `sam_3d_body/models/meta_arch/nf_merging.py` | Multi-view IS merging + MHR output computation |
| `sam_3d_body/losses/nf_loss.py` | NLL training loss for the NF head |
| `sam_3d_body/trainer.py` | PyTorch Lightning training loop, visualisation, multi-view eval |
| `sam_3d_body/models/meta_arch/sam3d_body.py` | Core model architecture |
| `sam_3d_body/visualization/renderer.py` | pyrender-based mesh renderer (EGL offscreen) |
| `scripts/train.py` | Training entry point (`--gpus`, `--dev`, `--plot` flags) |
