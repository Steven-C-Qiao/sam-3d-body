# NF Sample Diversity Analysis

## Background

The `NFARHead` (`sam_3d_body/models/heads/prohmr_ar_head.py`) models a conditional distribution over MHR body parameters via a factorised autoregressive normalising flow. In practice, samples from the trained model collapse toward the mean prediction and lack diversity, especially in scenarios where the image evidence is ambiguous.

## Two Ambiguity Modes

The NF samples should capture two distinct types of ambiguity:

**Mode 1 — Occlusion**
Parts of the body are not visible in the image. The model has no image evidence for those regions, so the distribution should be broad (prior-like) for the corresponding parameters. Samples should vary freely for unobserved joints/shape.

**Mode 2 — Single-view 3D ambiguity**
Even with a fully visible person, monocular 3D reconstruction is fundamentally ill-posed. Multiple distinct 3D configurations of pose and shape can project to the same 2D image and are equally consistent with the image evidence. The flow should capture this ambiguity as genuinely different modes in 3D, not just noise around a single solution.

---

## Problems (Ranked by Importance)

### 1. KP3D loss directly suppresses 3D ambiguity

**Location**: `sam_3d_body/losses/nf_loss.py:120–141`, config `LOSS.KP3D_WEIGHT = 50.0`

The 3D keypoint loss penalises all N samples against a single GT 3D annotation:

```python
loss_kp3d_samples = kp3d_loss.mean()  # averaged over [B, N, K]
```

This is the most damaging loss for **mode 2**: multiple valid 3D configurations are equally consistent with the image, yet all are penalised for deviating from the one specific GT 3D solution. The 2D reprojection loss (`KP2D`) is consistent with 3D ambiguity (many 3D configs project similarly), but `KP3D` collapses the distribution to one specific 3D configuration by design.

**Fix**: Remove or heavily downweight `KP3D_WEIGHT`. If 3D supervision is needed, apply it only to the mean prediction head output — not to the NF samples.

---

### 2. KP losses averaged over all samples collapse both modes

**Location**: `sam_3d_body/losses/nf_loss.py:114, 138`

Both KP2D and KP3D losses are averaged uniformly over all N samples:

```python
loss_kp2d_samples = kp2d_loss.mean()  # [B, N, K] → scalar
loss_kp3d_samples = kp3d_loss.mean()
```

Every sample is pulled toward the same GT target with equal force. This suppresses:
- **Mode 1**: samples for occluded regions cannot vary freely — they are all penalised against the same GT regardless of visibility.
- **Mode 2**: alternative 3D configurations consistent with the image are penalised equally with configurations inconsistent with it.

Note: the loss weights are not as imbalanced as the raw ratios suggest. KP2D is in normalised image space (raw values ~0.02–0.05) and NLL for a ~128D flow is large in magnitude (~100–300), so weighted contributions are roughly comparable. The structure of the loss (averaged over samples) is the more important issue.

**Fix**: Switch KP2D to a **best-of-N** (min-over-N) formulation. Only penalise the sample closest to GT; allow the rest to be diverse:

```python
kp2d_per_sample = kp2d_loss.mean(dim=-1)     # [B, N, K] → [B, N]
best_sample_loss, _ = kp2d_per_sample.min(dim=1)  # [B]
loss_kp2d_samples = best_sample_loss.mean()
```

This preserves image consistency (at least one sample must explain the image) while removing the mode-collapsing pressure on all other samples.

---

### 3. No explicit diversity objective

**Location**: `sam_3d_body/losses/nf_loss.py`

Nothing in the training loss directly rewards sample spread. The NLL loss (`PARAM_NLL_WEIGHT`) encourages the flow to assign probability to the GT residual, but does not penalise distribution collapse. A narrow distribution centred near the GT residual achieves low NLL while still having poor diversity.

**Fix**: Add an entropy bonus on the latent z-samples:

```python
# z_samples: [B, N, D] — reward spread in latent space
z_variance_bonus = z_samples.var(dim=1).mean()
loss -= entropy_weight * z_variance_bonus
```

This is complementary to the above fixes rather than a standalone solution — it adds a direct gradient signal for diversity.

---

## Summary Table

## Note on Stage 2 Conditioning

The loss function uses GT shape to build the stage-2 pose context (`nf_loss.py:69`). This is **correct and intentional**: the loss computes the factorised joint log-likelihood `log p(Δβ_gt | c) + log p(Δθ_gt | c, Δβ_gt)`, and the second term requires conditioning on the true Δβ. This is not teacher forcing in the harmful sense — it is the exact factorisation needed to maximise likelihood of the GT joint parameters.

The forward/sampling path (`prohmr_ar_head.py:311–312`) correctly conditions stage 2 on stage-1 shape samples, so there is no train/test mismatch. GT shape appears in `initialize_actnorm` only, where it is appropriate to mirror the loss distribution.

---

## Summary Table

| # | Problem | Modes Affected | Fix |
|---|---------|---------------|-----|
| 1 | KP3D loss collapses 3D distribution to single GT | Mode 2 | Remove from NF samples; apply to mean pred only |
| 2 | KP losses averaged over all samples | Mode 1 & 2 | Best-of-N (min-over-N) for KP2D |
| 3 | No diversity reward | Mode 1 & 2 | Entropy bonus on z-samples |
