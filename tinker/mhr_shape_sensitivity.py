"""Per-parameter sensitivity of the MHR mesh.

For each of the 45 shape coefficients and each of the 10 selected scale dims
(indices into the 68-D scale vector), perturb the parameter from its zero/mean
value and measure the resulting per-vertex displacement of the MHR mesh
(rest pose, no global rot/trans, neutral expression).

Two perturbations are reported per parameter:
  - delta = +1 unit  (raw linear sensitivity)
  - delta = +1 sigma (population std from shape_scale_std.pt)
where the population std comes from the MHR-conditioned BEDLAM training set.

Outputs mean and max L2 vertex displacement in millimetres.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MHR_MODEL_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
SCALE_MEAN_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/scale_mean.npy"
SCALE_COMPS_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/scale_comps.npy"
STD_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/shape_scale_std.pt"

SCALE_INDICES = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
SCALE_LABELS = {
    3: "torso length",
    4: "neck length",
    5: "shoulder width",
    6: "lower arm length",
    7: "lower arm/hand scale",
    10: "pelvis width",
    11: "leg length",
    12: "pelvis fwd offset",
    13: "lower calf length",
    14: "lower calf length",
}


def mhr_verts(mhr, shape, model_params, expr):
    """Run the JIT MHR forward and return vertices in metres."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        verts, _ = mhr(shape.float(), model_params.float(), expr.float())
    return verts / 100.0  # cm -> m


def displacement_mm(verts_a, verts_b):
    """Return (mean, max) per-vertex L2 distance in millimetres."""
    d = (verts_a - verts_b).norm(dim=-1) * 1000.0  # m -> mm
    return d.mean().item(), d.max().item(), int(d.argmax().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading MHR JIT module from {MHR_MODEL_PATH}…")
    mhr = torch.jit.load(str(MHR_MODEL_PATH), map_location=device)
    for p in mhr.parameters():
        p.requires_grad = False

    scale_mean = torch.tensor(np.load(SCALE_MEAN_PATH), device=device, dtype=torch.float32)  # (68,)
    scale_comps = torch.tensor(np.load(SCALE_COMPS_PATH), device=device, dtype=torch.float32)  # (28, 68)
    stds = torch.load(STD_PATH, weights_only=False, map_location="cpu")
    shape_std = stds["shape_std"].to(device)  # (45,)
    scale_std = stds["scale_std"].to(device)  # (10,)
    assert stds["scale_indices"] == SCALE_INDICES

    # Base inputs: rest pose, zero shape, mean scale, neutral expr.
    B = 1
    shape0 = torch.zeros(B, 45, device=device)
    expr0 = torch.zeros(B, 72, device=device)
    # model_params layout: [transl(3), rots(3), pose(130), scale(68)] = (B, 204)
    mp0 = torch.zeros(B, 204, device=device)
    mp0[:, -68:] = scale_mean

    base_verts = mhr_verts(mhr, shape0, mp0, expr0)  # (1, 18439, 3) in metres

    # Helpful per-vertex height range for context
    height = (base_verts[0, :, 1].max() - base_verts[0, :, 1].min()).item()
    print(f"Base mesh: {base_verts.shape[1]} verts, "
          f"y-extent (height) = {height * 1000:.1f} mm")

    # ---------------- Shape coefficients (45) ----------------
    print("\n" + "=" * 90)
    print("SHAPE coefficients (45-D identity)")
    print("=" * 90)
    print(f"{'idx':>3}  {'sigma':>7}  | "
          f"{'mean(+1u) mm':>12}  {'max(+1u) mm':>12}  | "
          f"{'mean(+1σ) mm':>12}  {'max(+1σ) mm':>12}  argmax_v")

    shape_rows = []
    for i in range(45):
        sigma = shape_std[i].item()

        s = shape0.clone()
        s[0, i] = 1.0
        v = mhr_verts(mhr, s, mp0, expr0)
        m_unit, mx_unit, am_unit = displacement_mm(v, base_verts)

        s = shape0.clone()
        s[0, i] = sigma
        v = mhr_verts(mhr, s, mp0, expr0)
        m_sig, mx_sig, am_sig = displacement_mm(v, base_verts)

        shape_rows.append((i, sigma, m_unit, mx_unit, m_sig, mx_sig, am_sig))
        print(f"{i:>3}  {sigma:>7.3f}  | "
              f"{m_unit:>12.2f}  {mx_unit:>12.2f}  | "
              f"{m_sig:>12.2f}  {mx_sig:>12.2f}  {am_sig}")

    # ---------------- Scale dims (10 selected of 68) ----------------
    print("\n" + "=" * 90)
    print("SCALE dims (10 selected of the 68-D scale vector)")
    print("=" * 90)
    print(f"{'idx':>3}  {'label':<22}  {'sigma':>7}  | "
          f"{'mean(+1u) mm':>12}  {'max(+1u) mm':>12}  | "
          f"{'mean(+1σ) mm':>12}  {'max(+1σ) mm':>12}  argmax_v")

    scale_rows = []
    for k, idx in enumerate(SCALE_INDICES):
        sigma = scale_std[k].item()
        label = SCALE_LABELS.get(idx, "")

        mp = mp0.clone()
        mp[0, -68 + idx] += 1.0
        v = mhr_verts(mhr, shape0, mp, expr0)
        m_unit, mx_unit, am_unit = displacement_mm(v, base_verts)

        mp = mp0.clone()
        mp[0, -68 + idx] += sigma
        v = mhr_verts(mhr, shape0, mp, expr0)
        m_sig, mx_sig, am_sig = displacement_mm(v, base_verts)

        scale_rows.append((idx, label, sigma, m_unit, mx_unit, m_sig, mx_sig, am_sig))
        print(f"{idx:>3}  {label:<22}  {sigma:>7.3f}  | "
              f"{m_unit:>12.2f}  {mx_unit:>12.2f}  | "
              f"{m_sig:>12.2f}  {mx_sig:>12.2f}  {am_sig}")

    # ---------------- Ranked summaries ----------------
    print("\n" + "=" * 90)
    print("Top-10 SHAPE dims by mean displacement at +1 sigma")
    print("=" * 90)
    for i, sig, mu, mxu, ms, mxs, _ in sorted(shape_rows, key=lambda r: -r[4])[:10]:
        print(f"  shape[{i:>2}]  σ={sig:.3f}  mean_disp={ms:6.2f} mm  max_disp={mxs:6.2f} mm")

    print("\nTop-10 SHAPE dims by mean displacement at +1 unit (raw sensitivity)")
    for i, sig, mu, mxu, ms, mxs, _ in sorted(shape_rows, key=lambda r: -r[2])[:10]:
        print(f"  shape[{i:>2}]  unit_mean={mu:6.2f} mm  unit_max={mxu:6.2f} mm  σ={sig:.3f}")

    print("\nAll SCALE dims ranked by mean displacement at +1 sigma")
    for idx, lbl, sig, mu, mxu, ms, mxs, _ in sorted(scale_rows, key=lambda r: -r[5]):
        print(f"  scale[{idx:>2}] {lbl:<22} σ={sig:.3f}  "
              f"mean_disp={ms:6.2f} mm  max_disp={mxs:6.2f} mm")

    print("\nAll SCALE dims ranked by mean displacement at +1 unit")
    for idx, lbl, sig, mu, mxu, ms, mxs, _ in sorted(scale_rows, key=lambda r: -r[3]):
        print(f"  scale[{idx:>2}] {lbl:<22} unit_mean={mu:7.1f} mm  "
              f"unit_max={mxu:7.1f} mm  σ={sig:.3f}")


if __name__ == "__main__":
    main()
