# NF Conversion Pipeline Reference

## Representations

| Name | Dim | Description |
|------|-----|-------------|
| **MHR model_params** | 204 | `[global_trans(3), global_rot_euler(3), body_pose(130), scale_68D(68)]` |
| **Continuous 6D** (pred_pose_raw) | 260 | Network output: 6D rotation (3-DOF joints) + sin/cos (1-DOF) + translations. Global = first 6 dims. |
| **MHR body_pose** | 133 | Euler angles: 23×3-DOF + 58×1-DOF + 6 translations. Indexed by `all_param_3dof_rot_idxs`, etc. |
| **Flow params** | variable | `[glob_rot_aa?(3), aa_3dofs(39), params_1dofs(34), shape?(45), scale?(10)]` |
| **Full flow residual** (samples) | theta_dim + beta_dim | `[theta: glob?(3) + 3dof(39) + 1dof(34) + cam?(3), beta: shape(45) + scale(10)]` |

## Key Conversion Functions

### 1. `compact_cont_to_model_params_body` (6D cont → 133D Euler)

- **Input:** `pred_pose_raw[:, 6:]` (254D, excluding 6D global)
- Splits into 3-DOF (6D each → XYZ Euler via `batchXYZfrom6D`), 1-DOF (sin/cos → atan2), translations
- Scatters into 133D via index arrays
- **Used in:** `NFARHead.forward` to get `pose_params_mhr` as base for `convert_flow_samples_to_mhr_params`

### 2. `convert_pose_cont_to_flow_context` (6D cont → axis-angle context)

- **Input:** same 254D continuous pose
- 3-DOF: 6D → 9D rotmat via `batch9Dfrom6D` → `matrix_to_axis_angle` → select 13 joints (`indices_3dof`) → flatten to 39D
- 1-DOF: sin/cos → atan2 → select 34 joints (`indices_1dof`)
- **Output:** `{"aa_3dofs": (B,39), "params_1dofs": (B,34)}`
- **Used in:** NFARHead.forward (stage-2 context), nf_loss.py (teacher-forced context)

### 3. `convert_mhr_params_to_flow_params` (204D model_params + 45D shape → flow space)

- **Input:** `model_params` (204D), `shape_params` (45D)
- Extracts `pose = model_params[:, 6:-68]` (130D), `scale = model_params[:, -68:]` (68D)
- 3-DOF: XYZ Euler → rotmat via `batch6DFromXYZ(return_9D=True)` → `matrix_to_axis_angle` → select 13 joints → 39D
- 1-DOF: select 34 joints (excl. hands) via `all_param_1dof_rot_idxs_except_hands`
- Scale: select 10 indices via `scale_indices`
- **Output:** `[glob_aa?(3), aa_3dofs(39), params_1dofs(34), shape?(45), scale?(10)]`
- **Note:** Does NOT include camera params.
- **Used in:** nf_loss.py (GT residual), NFARHead.initialize_actnorm

### 4. `convert_flow_samples_to_mhr_params` (flow AA samples → 133D Euler)

- **Input:** `aa_3dof_samples` (B,N,39), `params_1dofs_samples` (B,N,34), `pose_mean` (B,133)
- AA → rotmat via `axis_angle_to_matrix` → Euler via atan2 decomposition
- Scatters 3-DOF Euler into 133D at `all_param_3dof_rot_idxs_except_hands`
- Scatters 1-DOF at `all_param_1dof_rot_idxs_except_hands`
- Zeros out hands and jaw (last 3 dims)
- **Used in:** NFARHead.forward to convert sampled pose back to MHR

## Data Flow: Training (NLL Loss)

```
GT:  batch["model_params"] (204D) ──→ convert_mhr_params_to_flow_params ──→ gt_flow_params
Mean: mean_pred["body_pose"][:,:130] + scale_68D ──→ convert_mhr_params_to_flow_params ──→ mean_flow_params
                                                                                    ↓
                                                                          true_residual = gt - mean
                                                                                    ↓
                                                                    (splice cam residual if MODEL_CAM)
                                                                                    ↓
                                                              split: theta_params | beta_params
                                                                                    ↓
                                                              flow_beta.log_prob(beta_params, ctx_beta)
                                                              flow_theta.log_prob(theta_params, ctx_theta)
```

## Data Flow: Sampling (NFARHead.forward)

```
Mean pred ──→ beta_context_proj([flow_ctx, shape_mean, scale_mean_selected])
         ──→ flow_beta.sample(N) ──→ beta_residual_samples (B,N,55)
                                            ↓
                              shape_samples = shape_mean + shape_residual
                              scale_samples_68D[selected] = scale_mean[selected] + scale_residual
                                            ↓
         ──→ theta_context_proj([flow_ctx, shape_samples, scale_samples_selected, aa_3dofs, params_1dofs, cam?])
         ──→ flow_theta.sample(1, per-sample context) ──→ theta_residual (B,N,theta_dim)
                                            ↓
                              aa_3dof_samples = aa_3dofs_mean + pose_3dof_residual
                              params_1dofs_samples = params_1dofs_mean + pose_1dof_residual
                                            ↓
                              convert_flow_samples_to_mhr_params(aa_samples, 1dof_samples, pose_mean_133D)
                                            ↓
                              pose_samples (B,N,133) — MHR Euler for body decoder
```

## Data Flow: Multi-View Merging (nf_merging.py)

```
Per-view: shape_samples (absolute), scale_samples (absolute), flow_context_beta
                                            ↓
For each view i, sample k:
    beta_i^k = [shape_sample, scale_sample_selected]  (absolute, 55D)
    For each other view j:
        residual_j = beta_i^k - mean_beta_j           (re-residualise w.r.t. view j)
        logp_j = flow_beta.log_prob(residual_j, context_j)
    logw_i^k = Σ_{j≠i} logp_j
                                            ↓
merged_beta = softmax(logw) · candidate_beta           (importance-weighted mean)
```

## Joint Selection Details

- **13 3-DOF joints** (`indices_3dof = [0..11, 22]`): 12 body joints + jaw (index 22, always zero in flow). From the 23 total 3-DOF joints (22 body excl. jaw from `all_param_3dof_rot_idxs[:-1]` + 1 jaw dummy zero-padded).
- **34 1-DOF joints** (`indices_1dof = [0..25, 50..57]`): Body joints excluding hands (indices 26-49 in the 58-joint ordering).
- **10 scale params** (`scale_indices = [3,4,5,6,7,10,11,12,13,14]`): Selected from 68D scale.

## Known Issues

### BUG: Camera residual is zero in NLL loss when MODEL_CAM=True

**Location:** `nf_loss.py:188-198`

The loss splices `cam_zeros` into `true_residual` instead of the actual GT camera residual. This trains the flow's camera component to always output zero residual, making camera modelling useless.

`initialize_actnorm` in `prohmr_ar_head.py:152-154` handles this correctly:
```python
cam_residual = batch["gt_pred_cam"] - mean_pred["pred_cam"]
```

**Fix:** Replace `cam_zeros` with `batch["gt_pred_cam"] - mean_pred["pred_cam"]` in nf_loss.py.

### FIXED: Global rotation was treated as absolute instead of residual

The mean prediction head does predict global rotation, but the code was passing zeros for global rotation when assembling the mean prediction model_params for `convert_mhr_params_to_flow_params`. This made the flow learn the absolute global rotation instead of a residual from the mean.

Three places were fixed:
1. **nf_loss.py** — mean model_params assembly now includes `batchXYZfrom6D(pred_pose_raw[:, :6])` as the mean global rotation Euler
2. **prohmr_ar_head.py `initialize_actnorm`** — same fix
3. **prohmr_ar_head.py `forward`** — sampled global rotation now adds mean AA + residual AA before converting to Euler (consistent with how body joints are handled)
