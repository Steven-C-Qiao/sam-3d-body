# Environment activation

Personal reference — not repo-wide guidance.

The shell Claude Code runs in already has the right Python on `PATH`, so most commands just need:

```bash
python scripts/train.py ...
```

## Fallback: manual conda activation

If an import fails (stale PATH, fresh shell, etc.), activate the `prohmr` env explicitly:

```bash
source /home/mifs/cq244/miniconda3/etc/profile.d/conda.sh && conda activate prohmr
```

**Why `source ~/.bashrc` doesn't work:** `.bashrc` exits early for non-interactive shells (see its `case $-` guard around lines 6–9), so `conda init`'s block never runs. Source `conda.sh` directly instead.

## Quick test commands

```bash
# Test training run
python scripts/train.py -E exp/exp_claude_test --gpus 0 --dev

# Test multi-view merging
python scripts/merging.py -E exp/<exp_test> --gpus 0 -D 4d-dress -L exp/<EXP_NAME>/saved_models/last.ckpt
```

## GPU

Shared cluster — prefer GPUs 0 and 1:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```
