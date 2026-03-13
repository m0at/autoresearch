#!/bin/bash
# Fetch latest evals from H100 and regenerate comparison plot
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p 17364 root@ssh7.vast.ai"

$SSH "grep 'val_bpb' /tmp/train_run2.log" > /tmp/run2_evals.txt 2>/dev/null
if [ ! -s /tmp/run2_evals.txt ]; then
  echo "ERROR: couldn't fetch evals (SSH down or no data yet)"
  exit 1
fi

python3 << 'PY'
import json, re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse(path):
    s, b = [], []
    with open(path) as f:
        for l in f:
            m = re.search(r'step\s+(\d+)\s+\|\s+val_bpb\s+([\d.]+)', l)
            if m:
                s.append(int(m.group(1)))
                b.append(float(m.group(2)))
    return s, b

r1s, r1b = parse('/tmp/run1_evals.txt')
r2s, r2b = parse('/tmp/run2_evals.txt')
with open('/Users/andy/autoresearch/results/karpathy_val_history.json') as f:
    kd = json.load(f)
ks = [d['step'] for d in kd if not d.get('final')]
kb = [d['val_bpb'] for d in kd if not d.get('final')]

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.plot(ks, kb, color='#888', lw=2, marker='o', ms=3, label=f'Karpathy baseline (final: {kb[-1]:.4f} bpb)')
ax.plot(r1s, r1b, color='#00bfff', lw=1.2, alpha=0.5, label=f'Run 1 (died step {r1s[-1]}, best {min(r1b):.4f})')
ax.plot(r2s, r2b, color='#00ff88', lw=2, marker='o', ms=3, label=f'Run 2 (step {r2s[-1]}, latest {r2b[-1]:.4f} bpb)')
ax.axvline(x=8200, color='#444', ls=':')
ax.set_xlabel('Training Step', color='w', fontsize=12)
ax.set_ylabel('Validation BPB (lower is better)', color='w', fontsize=12)
ax.set_title('autoresearch: Rust brain vs Karpathy baseline\n8200 steps, same dataset & seed', color='w', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, facecolor='#2a2a2a', edgecolor='#444', labelcolor='w')
ax.set_xlim(0, 8500)
ax.set_ylim(0.8, 1.8)
ax.grid(True, alpha=0.2)
ax.tick_params(colors='w')
for s in ax.spines.values():
    s.set_color('#444')
plt.tight_layout()
plt.savefig('/Users/andy/autoresearch/brain_vs_karpathy.png', dpi=150, facecolor='#1a1a1a')
print(f'Run2: {len(r2s)} evals, latest {r2b[-1]:.4f} @ step {r2s[-1]}')
PY
