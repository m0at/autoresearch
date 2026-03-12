#!/usr/bin/env python3
"""
Successive Halving sweep orchestrator.

Runs a list of experiments in rounds (default: 200 → 500 → 1000 steps),
cutting the bottom half after each round. Uses checkpoints to continue
survivors rather than re-running from scratch.

Usage:
  python3 engine/sweep.py                     # run phase4 experiments
  python3 engine/sweep.py --dry-run           # print plan, no execution
  python3 engine/sweep.py --rounds 200,500    # custom round budgets
  python3 engine/sweep.py --keep 2            # keep top-N per round (default: half)

Experiments are defined in EXPERIMENTS at the bottom of this file.
Each experiment specifies:
  - config_overrides: dict of const NAME -> value to patch into config.rs
  - env_vars: dict of env vars passed to the engine binary at runtime
  - name: short label

The build instance (BUILD_HOST) compiles all binaries. Run instances
get binaries + run them. Results are polled via SSH.
"""

import os
import re
import sys
import json
import time
import math
import shutil
import argparse
import textwrap
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Infrastructure ─────────────────────────────────────────────────────────────

BUILD_HOST = ("root@ssh3.vast.ai", 32620)   # has cargo + source

RUN_INSTANCES = [
    ("root@ssh3.vast.ai", 32620),
    ("root@ssh2.vast.ai", 37220),
    ("root@ssh7.vast.ai", 17364),
    ("root@ssh4.vast.ai", 17364),
]

REMOTE_REPO   = "/root/autoresearch/engine"
REMOTE_BIN    = "/root/autoresearch-engine-sweep-{name}"
REMOTE_DATA   = "/root/.cache/autoresearch/shards_packed"
REMOTE_CKPT   = "/tmp/sweep_ckpt_{name}"
REMOTE_LOG    = "/tmp/sweep_{name}.log"
REMOTE_FEEDER = "/root/feeder.py"    # new instances; inst1/2 have it at engine path

TOTAL_BATCH   = 524288
DEFAULT_BATCH = 64
SEED          = 42

# ── Config.rs template ─────────────────────────────────────────────────────────

BASE_CONFIG = """\
pub const VOCAB: usize = 8192;
pub const SEQ: usize = 2048;
pub const D_MODEL: usize = {D_MODEL};
pub const N_HEAD: usize = 4;
pub const N_KV_HEAD: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const MLP_DIM: usize = {MLP_DIM};
pub const N_LAYER: usize = {N_LAYER};
pub const VE_GATE_CH: usize = 32;
pub const SOFTCAP: f32 = 15.0;
pub const EPS: f32 = 1e-5;
pub const ROPE_BASE: f64 = 200_000.0;
pub const INIT_SCALE: f64 = {INIT_SCALE};

pub const VE_LAYERS: [usize; {VE_COUNT}] = [{VE_LAYERS}];

pub const WINDOW_SIZES: [usize; N_LAYER] = [{WINDOW_SIZES}];

pub fn has_ve(layer: usize) -> bool {{
    {HAS_VE_BODY}
}}
"""

def make_ssssl_windows(n_layer: int, s: int, full: int = 2048) -> list[int]:
    """Generate SSSSL repeating window pattern for n_layer layers."""
    pattern = [s, s, s, s, full]
    windows = []
    for i in range(n_layer):
        windows.append(pattern[i % len(pattern)])
    return windows

def make_config_rs(overrides: dict) -> str:
    defaults = dict(
        N_LAYER=30, D_MODEL=512, MLP_DIM=2048, INIT_SCALE=0.68,
        S=256,  # local window size
        VE_LAYERS_LIST=[1, 3, 5, 7],
    )
    cfg = {**defaults, **overrides}

    n_layer     = cfg["N_LAYER"]
    s           = cfg["S"]
    ve_layers   = cfg["VE_LAYERS_LIST"]
    windows     = make_ssssl_windows(n_layer, s)

    # Format window sizes with line breaks every 5
    ws_rows = []
    for i in range(0, len(windows), 5):
        row = windows[i:i+5]
        ws_rows.append("    " + ", ".join(str(w) for w in row) + ",")
    window_str = "\n" + "\n".join(ws_rows) + "\n"

    ve_str = ", ".join(str(l) for l in ve_layers)
    if ve_layers:
        has_ve_body = "matches!(layer, " + " | ".join(str(l) for l in ve_layers) + ")"
    else:
        has_ve_body = "false"

    return BASE_CONFIG.format(
        D_MODEL=cfg["D_MODEL"],
        MLP_DIM=cfg["MLP_DIM"],
        N_LAYER=n_layer,
        INIT_SCALE=cfg["INIT_SCALE"],
        VE_COUNT=len(ve_layers),
        VE_LAYERS=ve_str,
        WINDOW_SIZES=window_str,
        HAS_VE_BODY=has_ve_body,
    )

# ── SSH helpers ────────────────────────────────────────────────────────────────

def ssh(host, port, cmd, check=True, capture=True) -> subprocess.CompletedProcess:
    full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), host, cmd]
    return subprocess.run(full_cmd, capture_output=capture, text=True, check=check)

def scp_to(local, host, port, remote):
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port), local, f"{host}:{remote}"],
        check=True, capture_output=True,
    )

def scp_from(host, port, remote, local):
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port), f"{host}:{remote}", local],
        check=True, capture_output=True,
    )

# ── Experiment definition ──────────────────────────────────────────────────────

@dataclass
class Experiment:
    name: str
    config_overrides: dict = field(default_factory=dict)  # compile-time (config.rs)
    env_vars: dict = field(default_factory=dict)           # runtime env vars
    batch_size: int = DEFAULT_BATCH

    # filled in during run
    binary_path: Optional[str] = None        # local path to built binary
    results: dict = field(default_factory=dict)  # round -> val_bpb

    def label(self):
        parts = []
        for k, v in self.config_overrides.items():
            parts.append(f"{k}={v}")
        for k, v in self.env_vars.items():
            parts.append(f"{k}={v}")
        return self.name + (f" ({', '.join(parts)})" if parts else "")

# ── Build ──────────────────────────────────────────────────────────────────────

def build_experiment(exp: Experiment, dry_run: bool = False) -> str:
    """Build binary for experiment on BUILD_HOST. Returns local binary path."""
    config_rs = make_config_rs(exp.config_overrides)
    local_config = f"/tmp/sweep_config_{exp.name}.rs"
    local_bin    = f"/tmp/sweep_bin_{exp.name}"

    Path(local_config).write_text(config_rs)

    if dry_run:
        print(f"  [dry] would build {exp.name}")
        return local_bin

    print(f"  building {exp.name}...", end=" ", flush=True)
    bhost, bport = BUILD_HOST

    scp_to(local_config, bhost, bport, f"{REMOTE_REPO}/src/config.rs")

    result = ssh(bhost, bport,
        f'cd {REMOTE_REPO} && '
        f'export PATH="/root/.cargo/bin:/usr/local/cuda/bin:$PATH" && '
        f'FLASH_ATTN_V3_BUILD_DIR=fa3/build /root/.cargo/bin/cargo build --release 2>&1 | '
        f'grep -E "^error|Finished"',
        check=False)

    if "error" in result.stdout:
        print(f"FAILED\n{result.stdout}")
        sys.exit(1)

    # save binary on build host and download
    remote_bin = f"/root/sweep_bin_{exp.name}"
    ssh(bhost, bport, f"cp {REMOTE_REPO}/target/release/autoresearch-engine {remote_bin}")
    scp_from(bhost, bport, remote_bin, local_bin)
    print("done")
    return local_bin

# ── Run ────────────────────────────────────────────────────────────────────────

def run_experiment_on(exp: Experiment, instance: tuple, max_steps: int,
                      cooldown_steps: int, load_ckpt: bool, dry_run: bool):
    host, port = instance
    log   = REMOTE_LOG.format(name=exp.name)
    ckpt  = REMOTE_CKPT.format(name=exp.name)
    binp  = f"/root/sweep_bin_{exp.name}"

    # upload binary
    if not dry_run:
        scp_to(exp.binary_path, host, port, binp)
        ssh(host, port, f"chmod +x {binp}")

    # feeder path depends on instance
    feeder = REMOTE_FEEDER if port == 17364 else f"{REMOTE_REPO}/feeder.py"

    env = {
        "BATCH_SIZE": exp.batch_size,
        "MAX_STEPS": max_steps,
        "COOLDOWN_STEPS": cooldown_steps,
        "EVAL_EVERY": 10,
        **exp.env_vars,
    }
    env_str = " ".join(f"{k}={v}" for k, v in env.items())

    ckpt_arg = f"--load-checkpoint {ckpt}/model_step{load_ckpt}.safetensors" if load_ckpt else ""
    ckpt_dir_arg = f"CHECKPOINT_DIR={ckpt}"

    cmd = (
        f'nohup bash -c "'
        f'NUM_TRAIN_SHARDS=794 python3 {feeder} --stream --prefetch 4 '
        f'  2>/tmp/sweep_feeder_{exp.name}.log '
        f'| {env_str} {ckpt_dir_arg} {binp} train '
        f'  --stream-input --data-dir {REMOTE_DATA} --seed {SEED} '
        f'  {ckpt_arg} '
        f'  > {log} 2>&1'
        f'" > /dev/null 2>&1 &'
    )

    if dry_run:
        print(f"  [dry] {host}:{port} -> {exp.name} steps={max_steps}")
        return

    ssh(host, port, f"pkill -9 -f 'sweep_bin_{exp.name}' 2>/dev/null; "
                    f"mkdir -p {ckpt}; {cmd}")

def poll_result(exp: Experiment, instance: tuple, max_steps: int,
                poll_interval: int = 30) -> Optional[float]:
    """Block until final eval appears in log, return val_bpb."""
    host, port = instance
    log = REMOTE_LOG.format(name=exp.name)

    print(f"  polling {exp.name} on {host}:{port}...", flush=True)
    while True:
        r = ssh(host, port, f'grep "\\[eval\\] final" {log} 2>/dev/null', check=False)
        if r.returncode == 0 and r.stdout.strip():
            m = re.search(r"val_bpb ([\d.]+)", r.stdout)
            if m:
                return float(m.group(1))
        # check for crash
        r2 = ssh(host, port, f'grep -c "panicked\\|OOM\\|CUDA_ERROR" {log} 2>/dev/null', check=False)
        if r2.stdout.strip() not in ("", "0"):
            print(f"  WARNING: {exp.name} may have crashed")
            return None
        time.sleep(poll_interval)

# ── Successive Halving ─────────────────────────────────────────────────────────

def run_round(experiments: list[Experiment], instances: list[tuple],
              max_steps: int, cooldown_steps: int, dry_run: bool,
              prev_steps: int = 0) -> dict[str, float]:
    """
    Run all experiments for max_steps steps, return {name: val_bpb}.
    Distributes across instances round-robin.
    """
    print(f"\n{'='*60}")
    print(f"Round: {max_steps} steps  ({len(experiments)} experiments, {len(instances)} GPUs)")
    print(f"{'='*60}")

    assign = {}
    for i, exp in enumerate(experiments):
        inst = instances[i % len(instances)]
        assign[exp.name] = inst
        run_experiment_on(exp, inst, max_steps, cooldown_steps,
                          load_ckpt=False, dry_run=dry_run)

    if dry_run:
        return {exp.name: 0.9 for exp in experiments}

    # poll in parallel
    results = {}
    threads = []
    lock = threading.Lock()

    def poll_one(exp):
        val = poll_result(exp, assign[exp.name], max_steps)
        with lock:
            results[exp.name] = val
            status = f"{val:.4f}" if val else "FAILED"
            print(f"  {exp.name}: {status}")

    for exp in experiments:
        t = threading.Thread(target=poll_one, args=(exp,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return results

def successive_halving(experiments: list[Experiment], rounds: list[int],
                       keep_top: Optional[int], dry_run: bool):
    instances = RUN_INSTANCES[:len(experiments)]  # only use what we need

    survivors = experiments[:]
    all_results = {}  # name -> {round: bpb}

    for r_idx, max_steps in enumerate(rounds):
        cooldown = max_steps // 2
        results = run_round(survivors, instances, max_steps, cooldown, dry_run)

        for name, bpb in results.items():
            all_results.setdefault(name, {})[max_steps] = bpb

        # rank and cut
        valid = [(name, bpb) for name, bpb in results.items() if bpb is not None]
        valid.sort(key=lambda x: x[1])

        print(f"\nRound {max_steps} ranking:")
        for rank, (name, bpb) in enumerate(valid):
            marker = ""
            exp = next(e for e in survivors if e.name == name)
            print(f"  {rank+1}. {exp.label():<40} {bpb:.4f} {marker}")

        is_last = (r_idx == len(rounds) - 1)
        if is_last:
            break

        n_keep = keep_top or max(1, len(valid) // 2)
        promoted = [name for name, _ in valid[:n_keep]]
        survivors = [e for e in survivors if e.name in promoted]

        print(f"\nPromoting top {n_keep}: {', '.join(promoted)}")

    # final summary
    winner = min(all_results, key=lambda n: all_results[n].get(rounds[-1], 9.9))
    winner_exp = next(e for e in experiments if e.name == winner)
    print(f"\n{'='*60}")
    print(f"WINNER: {winner_exp.label()}")
    print(f"  val_bpb = {all_results[winner][rounds[-1]]:.4f}")
    print(f"{'='*60}")
    return winner_exp, all_results

# ── Phase 4 experiments ────────────────────────────────────────────────────────
# Edit this list to define your sweep.
# Baseline is depth=30, S=256, all current defaults.

EXPERIMENTS = [
    # ── Baseline (control) ──────────────────────────────────────────────
    Experiment(
        name="baseline",
        config_overrides=dict(N_LAYER=30, S=256),
    ),

    # ── LR variants ─────────────────────────────────────────────────────
    Experiment(
        name="lr_low",
        config_overrides=dict(N_LAYER=30, S=256),
        env_vars=dict(PEAK_LR=0.02, EMBEDDING_LR=0.45),
    ),
    Experiment(
        name="lr_high",
        config_overrides=dict(N_LAYER=30, S=256),
        env_vars=dict(PEAK_LR=0.06, EMBEDDING_LR=1.35),
    ),

    # ── VE coverage: every odd layer ────────────────────────────────────
    Experiment(
        name="ve_all_odd",
        config_overrides=dict(
            N_LAYER=30, S=256,
            VE_LAYERS_LIST=[i for i in range(30) if i % 2 == 1],
        ),
    ),

    # ── MLP ratio: 8/3× (≈683) instead of 4× (2048) ───────────────────
    Experiment(
        name="mlp_8_3",
        config_overrides=dict(N_LAYER=30, S=256, MLP_DIM=1365),
    ),

    # ── SSSSL pattern: SSSL (3S + 1L) ──────────────────────────────────
    # Override window_sizes manually by patching S to use a different pattern
    # We'll use a custom approach: SSSL = every 4th layer is full
    # For now use S=256 with different pattern via a special key
    Experiment(
        name="sssl_pattern",
        config_overrides=dict(N_LAYER=30, S=256),
        # SSSL: 3 local + 1 full repeating
        # handled below via post-init
    ),

    # ── Init scale: smaller for deeper model ────────────────────────────
    Experiment(
        name="init_0.5",
        config_overrides=dict(N_LAYER=30, S=256, INIT_SCALE=0.5),
    ),

    # ── Lower weight decay ───────────────────────────────────────────────
    Experiment(
        name="wd_0.1",
        config_overrides=dict(N_LAYER=30, S=256),
        env_vars=dict(WEIGHT_DECAY=0.1),
    ),
]

# SSSL pattern: override make_ssssl_windows result
_sssl_exp = next(e for e in EXPERIMENTS if e.name == "sssl_pattern")
_sssl_exp.config_overrides["_custom_windows"] = [
    256 if (i % 4) != 3 else 2048 for i in range(30)
]

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rounds", default="200,500,1000",
                        help="comma-separated step budgets per round")
    parser.add_argument("--keep", type=int, default=None,
                        help="how many to promote per round (default: half)")
    parser.add_argument("--experiments", default=None,
                        help="comma-separated experiment names to run (default: all)")
    args = parser.parse_args()

    rounds = [int(x) for x in args.rounds.split(",")]
    exps = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments.split(","))
        exps = [e for e in exps if e.name in names]

    print(f"Sweep: {len(exps)} experiments × {rounds} steps")
    print(f"Instances: {len(RUN_INSTANCES)} GPUs")
    print()

    # Build all binaries
    print("Building binaries...")
    for exp in exps:
        # handle custom windows
        if "_custom_windows" in exp.config_overrides:
            cw = exp.config_overrides.pop("_custom_windows")
            # patch make_config_rs to use custom windows
            original = make_ssssl_windows
            def patched(*a, **kw): return cw
            import builtins
            globals()["make_ssssl_windows"] = patched
            exp.binary_path = build_experiment(exp, dry_run=args.dry_run)
            globals()["make_ssssl_windows"] = original
        else:
            exp.binary_path = build_experiment(exp, dry_run=args.dry_run)

    print()

    # Run successive halving
    winner, all_results = successive_halving(exps, rounds, args.keep, args.dry_run)

    # Save results
    out = {
        "rounds": rounds,
        "experiments": [
            {
                "name": e.name,
                "label": e.label(),
                "results": all_results.get(e.name, {}),
            }
            for e in exps
        ],
        "winner": winner.name,
    }
    Path("results/sweep_results.json").parent.mkdir(exist_ok=True)
    Path("results/sweep_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to results/sweep_results.json")


if __name__ == "__main__":
    main()
