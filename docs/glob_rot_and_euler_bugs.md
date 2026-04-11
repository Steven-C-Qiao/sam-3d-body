# Global Rotation & Euler Convention Bugs — Discovery and Fixes

**Date:** 2026-04-10
**Branch:** `refactor`
**Commits:** `d280b01`, `24019dc`, `8503e5f`

---

## Overview

Three interrelated bugs were discovered in the global rotation and pose handling,
found by tracing the debug GT-override path end-to-end and comparing predicted
samples against ground truth keypoints/vertices.

---

## Bug 1: GT vs Prediction Coordinate Frame Mismatch (`flip_global_rot`)

**Commit:** `24019dc`

### Discovery
When the debug block overrode all samples with GT values, the rendered mesh was
upside down. The GT global rotation in `model_params[:, 3:6]` is in **camera frame**
(mesh already correctly oriented), but model predictions are in an **"upright" frame**
where the mesh is subsequently flipped by `verts[..., [1, 2]] *= -1` in `sam3d_body.py`.

The NLL residual `gt_AA - mean_AA` conflated these two conventions, teaching the flow
to produce ~π rotations that would then be double-flipped by the downstream verts flip.

### Fix
Added `flip_global_rot` parameter to `convert_mhr_params_to_flow_params` that
pre-multiplies the GT rotation by `diag(1, -1, -1)` (the YZ-flip matrix) before
converting to axis-angle. Applied to all GT call sites:
- `nf_loss.py` — NLL loss GT
- `prohmr_ar_head.py` — actnorm GT and debug GT
- `prohmr_head.py` — non-AR actnorm GT

Mean prediction calls are unchanged (already in upright convention).

### Files
- `sam_3d_body/models/modules/mhr_utils.py` — new `flip_global_rot` parameter
- `sam_3d_body/losses/nf_loss.py` — pass `flip_global_rot=True`
- `sam_3d_body/models/heads/prohmr_ar_head.py` — pass `flip_global_rot=True`
- `sam_3d_body/models/heads/prohmr_head.py` — pass `flip_global_rot=True`

---

## Bug 2: `roma.rotmat_to_euler("ZYX")` Returns `(z, y, x)`, Not `(x, y, z)`

**Commit:** `24019dc`

### Discovery
Verified empirically:
```
batch6DFromXYZ input (x,y,z): [0.5, 0.3, 0.7]
roma ZYX output:               [0.7, 0.3, 0.5]   ← reversed!
batchXYZfrom6D output:         [0.5, 0.3, 0.7]   ← matches input
```

The MHR C++ backend interprets `model_params[3:6]` via the `batch6DFromXYZ` convention,
expecting `(x, y, z)`. But `roma.rotmat_to_euler("ZYX")` returns `(z, y, x)`. The
sampling path used roma, so the kp2d/kp3d reprojection losses during training were
computed on incorrectly rotated meshes.

### Impact
- **Affects training** when `MODEL_GLOB_ROT=True`: reprojection losses (kp2d, kp3d) on
  NF samples see wrong rotation → flow learns compensating (wrong) residuals.
- **NLL loss unaffected**: operates in AA space, never touches euler conversion.
- **Pre-trained model** (`MODEL_GLOB_ROT=False`): unaffected, since `glob_rot_samples`
  is `None` and the default path uses `roma.rotmat_to_euler("ZYX")` which the pre-trained
  model has learned to compensate for.

### Fix
Replaced `roma.rotmat_to_euler("ZYX", rotmat)` with `batchXYZfrom6D(rotmat[:, :, 0:1] ++ rotmat[:, :, 1])`
in the sampling path. The default non-sampled path (`sam3d_body.py:837`) keeps roma
to preserve pre-trained model compatibility.

### Pre-existing convention issue (not fixed, by design)
`mhr_head.py:297` and `sam3d_body.py:837` use `roma.rotmat_to_euler("ZYX")` for the
mean prediction's global rotation. This produces `(z, y, x)` at `model_params[3:6]`.
The pre-trained model was trained with this "wrong" convention and has learned to
compensate — fixing it would break existing checkpoints. When `MODEL_GLOB_ROT=True`
(new training), the flow sampling path now uses the correct `batchXYZfrom6D` convention.

### Files
- `sam_3d_body/models/heads/prohmr_ar_head.py` — sampling path and debug block

---

## Bug 3: Euler↔AA Branch-Cut Bias in NLL Loss

**Commit:** `8503e5f`

### Discovery
The NLL loss and sampling path computed the mean prediction's 3DOF pose axis-angle
via **different paths**:

**Sampling path** (correct, used at inference):
```
pred_pose_raw (6D continuous) → Gram-Schmidt → rotmat → matrix_to_axis_angle → aa
```

**NLL loss path** (biased):
```
pred_pose_raw → compact_cont_to_model_params_body → body_pose (euler via atan2)
  → batch6DFromXYZ → rotmat' → matrix_to_axis_angle → aa'
```

The extra euler roundtrip in the NLL path can land on a different `atan2` branch
(e.g., `(x, y, z)` vs `(x+π, π-y, z+π)` for the same rotation), causing
`rotmat' ≠ rotmat` and therefore `aa' ≠ aa`.

**Measured bias:**
```
3DOF AA diff:  mean=0.095  max=2.82  (radians)
1DOF diff:     mean=0.000  max=0.000
```
The bias is ~95% of a typical residual magnitude. 1DOF is unaffected (scalar angles,
no branch-cut issue).

### Impact
The flow is trained on residuals `gt_AA - mean_AA_euler` (NLL path) but at inference
adds residuals to `mean_AA_6D` (sampling path). The mismatch `mean_AA_euler ≠ mean_AA_6D`
introduces a systematic bias in the NLL training target.

### Fix
Replaced `convert_mhr_params_to_flow_params` on `mean_pred["body_pose"]` (euler path)
with direct use of `convert_pose_cont_to_flow_context` on `mean_pred["pred_pose_raw"]`
(same function the sampling path uses). Applied to:
- `nf_loss.py:forward()` — NLL loss mean computation
- `prohmr_ar_head.py:initialize_actnorm()` — actnorm mean computation
- `prohmr_ar_head.py` debug block — GT residual computation

**Verified:** after fix, loss and sampling paths produce identical mean AA
(error < 1e-7, floating-point precision).

### Files
- `sam_3d_body/losses/nf_loss.py` — replace euler-path mean with direct 6D→AA
- `sam_3d_body/models/heads/prohmr_ar_head.py` — same in actnorm and debug block

---

## Debug GT-Override Block

The debug block in `prohmr_ar_head.py:forward()` overrides all NF samples with GT
values to verify the pipeline produces exact GT keypoints.

### Evolution
1. **Original**: used mean+residual via euler path → ~0.12 mean / 1.48 max pose error
   from branch-cut bias, plus upside-down mesh from bugs 1 and 2.
2. **Direct override** (temporary): bypassed residuals, used GT values directly →
   exact results (body 3D err ~0.0003, limited by hand pose from mean prediction).
3. **Final**: mean+residual via direct 6D→AA path → exact results (error < 1e-7),
   exercises the same code path that training uses.

### Remaining expected error in debug mode
Hand joints show ~0.004-0.006 3D error because the flow does not model hand pose.
`mhr_forward` uses the mean prediction's hand pose (`output_mhr["hand"]`), not GT.
This is by design — hands are handled by a separate PCA-based predictor.

---

## Key Functions Reference

| Function | File | Convention |
|----------|------|-----------|
| `batch6DFromXYZ(euler)` | `mhr_utils.py` | Takes `(x, y, z)`, builds `Rz(z) @ Ry(y) @ Rx(x)` |
| `batchXYZfrom6D(6d)` | `mhr_utils.py` | Returns `(x, y, z)` from rotmat columns |
| `batch9Dfrom6D(6d)` | `mhr_utils.py` | Returns flattened 9D rotmat (Gram-Schmidt) |
| `roma.rotmat_to_euler("ZYX", R)` | roma lib | Returns `(z, y, x)` — **NOT** `(x, y, z)` |
| `convert_mhr_params_to_flow_params` | `mhr_utils.py` | GT euler → AA (uses `batch6DFromXYZ`) |
| `convert_pose_cont_to_flow_context` | `mhr_utils.py` | 6D continuous → AA (direct, no euler) |
| `convert_flow_samples_to_mhr_params` | `mhr_utils.py` | AA → euler → 130D MHR params |
