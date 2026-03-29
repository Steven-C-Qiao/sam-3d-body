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
