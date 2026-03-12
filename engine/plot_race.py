"""
Plot val_bpb learning curves: Rust depth=32 vs Python reference.
Reads from log files; missing data is skipped gracefully.

Usage:
  python3 engine/plot_race.py                         # reads from default log paths
  RUST_LOG=/tmp/train_d32.log PYTHON_LOG=/tmp/train_py1000.log python3 engine/plot_race.py
"""

import re
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Colors (dark mode) ────────────────────────────────────────────────────────
BG        = "#0d1117"
AX_BG     = "#161b22"
GRID      = "#21262d"
RUST_CLR  = "#f78166"   # warm red-orange
PY_CLR    = "#79c0ff"   # cool blue
TEXT      = "#e6edf3"
MUTED     = "#8b949e"

# ── Parse helpers ──────────────────────────────────────────────────────────────
EVAL_RE = re.compile(r"\[eval\] step (\d+) \| val_bpb ([\d.]+)")

def parse_log(path):
    steps, bpb = [], []
    if not path or not os.path.exists(path):
        return steps, bpb
    with open(path) as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                bpb.append(float(m.group(2)))
    return steps, bpb


def parse_log_ssh(host, port, remote_path):
    """Read a remote log file over SSH and parse it."""
    import subprocess
    try:
        result = subprocess.run(
            ["ssh", "-p", str(port), host, f"cat {remote_path}"],
            capture_output=True, text=True, timeout=30
        )
        steps, bpb = [], []
        for line in result.stdout.splitlines():
            m = EVAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                bpb.append(float(m.group(2)))
        return steps, bpb
    except Exception:
        return [], []


# ── Load data ──────────────────────────────────────────────────────────────────
rust_log   = os.environ.get("RUST_LOG", "")
python_log = os.environ.get("PYTHON_LOG", "")

if rust_log and os.path.exists(rust_log):
    rust_steps, rust_bpb = parse_log(rust_log)
else:
    # Fetch from inst2 where the depth=32 run lives
    rust_steps, rust_bpb = parse_log_ssh("root@ssh2.vast.ai", 37220, "/tmp/train_d32.log")

if python_log and os.path.exists(python_log):
    py_steps, py_bpb = parse_log(python_log)
else:
    py_steps, py_bpb = parse_log_ssh("root@ssh3.vast.ai", 32620, "/tmp/train_py1000.log")

if not rust_steps and not py_steps:
    print("No data found. Set RUST_LOG / PYTHON_LOG or ensure SSH access.")
    sys.exit(1)


# ── Plot ───────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)

for spine in ax.spines.values():
    spine.set_color(GRID)

ax.tick_params(colors=TEXT, labelsize=11)
ax.xaxis.label.set_color(TEXT)
ax.yaxis.label.set_color(TEXT)
ax.title.set_color(TEXT)

ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
ax.set_axisbelow(True)

if rust_steps:
    ax.plot(rust_steps, rust_bpb,
            color=RUST_CLR, linewidth=2.0, label=f"Rust depth=32  (best {min(rust_bpb):.4f})",
            zorder=3)
    # Final marker
    ax.scatter([rust_steps[-1]], [rust_bpb[-1]],
               color=RUST_CLR, s=60, zorder=4)

if py_steps:
    ax.plot(py_steps, py_bpb,
            color=PY_CLR, linewidth=2.0, label=f"Python reference  (best {min(py_bpb):.4f})",
            zorder=3, alpha=0.9)
    ax.scatter([py_steps[-1]], [py_bpb[-1]],
               color=PY_CLR, s=60, zorder=4)
else:
    ax.text(0.98, 0.92, "Python run in progress…",
            transform=ax.transAxes, ha="right", va="top",
            color=PY_CLR, fontsize=10, alpha=0.7)

# Axis labels & title
ax.set_xlabel("Optimizer step", fontsize=13, labelpad=8)
ax.set_ylabel("val bpb  (lower = better)", fontsize=13, labelpad=8)
ax.set_title("Learning Curve: Rust depth=32 vs Python reference", fontsize=15, pad=14)

# Shade cooldown region (steps 500–1000 for 1000-step runs)
max_step = max((rust_steps or [0])[-1], (py_steps or [0])[-1], 1)
if max_step >= 500:
    ax.axvspan(500, max_step, color="#ffffff", alpha=0.03, label="Cooldown phase")
    ax.axvline(500, color=MUTED, linewidth=0.9, linestyle="--", alpha=0.5)
    ax.text(502, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] < 2 else 1.9,
            "cooldown", color=MUTED, fontsize=9, va="top")

legend = ax.legend(fontsize=12, facecolor=AX_BG, edgecolor=GRID,
                   labelcolor=TEXT, loc="upper right")

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

plt.tight_layout()
fig.subplots_adjust(left=0.08, right=0.97)  # compress horizontal whitespace to ~75%
out = os.environ.get("OUT", "results/learning_curve.png")
os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")

# Print summary
if rust_steps:
    print(f"Rust:   {len(rust_steps)} eval points, best {min(rust_bpb):.4f} @ step {rust_steps[rust_bpb.index(min(rust_bpb))]}")
if py_steps:
    print(f"Python: {len(py_steps)} eval points, best {min(py_bpb):.4f} @ step {py_steps[py_bpb.index(min(py_bpb))]}")
