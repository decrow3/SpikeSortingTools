#!/usr/bin/env python3
"""
patch_kilosort_claimmask.py

Minimal patcher for Kilosort (e.g., 4.0.27) to add the cross-peel claim mask feature.
- Applies, dry-runs, or reverses the patch in-place in the current Python environment.
- Backs up originals as .bak before patching.

Usage:
  python patch_kilosort_claimmask.py [--dry-run] [--reverse]

Options:
  --dry-run   Show what would be changed, but do not modify files.
  --reverse   Restore from .bak backups (undo patch).

Tested with spikeinterface==0.102.1 and kilosort==4.0.27.
"""
import sys
import os
import argparse
import shutil
import difflib
from pathlib import Path
import importlib.util

# --- Patch content for parameters.py ---
PARAMETERS_PATCH = '''    'cross_peel_claim_ms': {
        'gui_name': 'cross-peel claim ms', 'type': float, 'min': 0, 'max': np.inf,
        'exclude': [], 'default': 0.0, 'step': 'spike detection',
        'description':
        """
        Suppress later matching-pursuit candidates that fall within this many
        milliseconds of an already accepted spike. A value of 0 disables the
        cross-peel claim rule.
        """
    },

    'cross_peel_claim_um': {
        'gui_name': 'cross-peel claim um', 'type': float, 'min': 0, 'max': np.inf,
        'exclude': [], 'default': 0.0, 'step': 'spike detection',
        'description':
        """
        Spatial radius paired with cross_peel_claim_ms. Later matching-pursuit
        candidates are suppressed only if they are also within this many
        microns of a previously accepted spike. A value of 0 applies the claim
        rule using time alone.
        """
    },
'''

# --- Patch content for template_matching.py (inserted in run_matching) ---
RUN_MATCHING_PATCH = '''    claim_ms = float(ops.get('cross_peel_claim_ms', 0.0))
    claim_um = float(ops.get('cross_peel_claim_um', 0.0))
    claim_enabled = claim_ms > 0
    claim_bins = int(np.ceil(claim_ms * ops['fs'] / 1000.0)) if claim_enabled else 0
    if claim_enabled:
        template_main_chan = torch.argmax((U**2).sum(-1), dim=1)
        xc = torch.as_tensor(ops['xc'], device=device, dtype=torch.float32)
        yc = torch.as_tensor(ops['yc'], device=device, dtype=torch.float32)
        template_x = xc[template_main_chan]
        template_y = yc[template_main_chan]
        claimed_t = []
        claimed_x = []
        claimed_y = []
'''

PEEL_LOOP_PATCH = '''        if claim_enabled and t > 0 and len(claimed_t) > 0:
            cand_t = iX[:, 0].to(torch.float32)
            prev_t = torch.cat(claimed_t)
            dt = torch.abs(cand_t[:, None] - prev_t[None, :])

            if claim_um > 0:
                cand_x = template_x[iY[:, 0]]
                cand_y = template_y[iY[:, 0]]
                prev_x = torch.cat(claimed_x)
                prev_y = torch.cat(claimed_y)
                d2 = (cand_x[:, None] - prev_x[None, :])**2 + (cand_y[:, None] - prev_y[None, :])**2
                claimed = (dt <= claim_bins) & (d2 <= claim_um**2)
            else:
                claimed = dt <= claim_bins

            keep = ~claimed.any(dim=1)
            if not keep.all():
                iX = iX[keep]
                iY = iY[keep]

            if len(iX) == 0:
                break
'''

AFTER_ACCEPT_PATCH = '''        if claim_enabled:
            claimed_t.append(iX[:, 0].to(torch.float32))
            claimed_x.append(template_x[iY[:, 0]])
            claimed_y.append(template_y[iY[:, 0]])
'''

def find_kilosort_path():
    spec = importlib.util.find_spec("kilosort")
    if not spec or not spec.submodule_search_locations:
        sys.exit("Could not locate installed kilosort package.")
    return Path(spec.submodule_search_locations[0])

def patch_parameters_py(path, dry_run, reverse):
    file = path / "parameters.py"
    bak = path / "parameters.py.bak"
    if reverse:
        if bak.exists():
            shutil.copy2(bak, file)
            print(f"Restored {file} from backup.")
        else:
            print(f"No backup found for {file}.")
        return
    with open(file) as f:
        lines = f.readlines()
    if not bak.exists():
        shutil.copy2(file, bak)
    # Insert after 'max_peels' param
    idx = None
    for i, line in enumerate(lines):
        if "'max_peels'" in line:
            idx = i
            break
    if idx is None:
        sys.exit("Could not find 'max_peels' in parameters.py.")
    # Find end of max_peels param (next closing '},')
    for j in range(idx+1, len(lines)):
        if lines[j].strip().startswith('},'):
            insert_idx = j+1
            break
    else:
        sys.exit("Could not find end of 'max_peels' param block.")
    # Check if already patched
    if any('cross_peel_claim_ms' in l for l in lines):
        print(f"Already patched: {file}")
        return
    new_lines = lines[:insert_idx] + [PARAMETERS_PATCH] + lines[insert_idx:]
    if dry_run:
        print(f"--- {file} (dry-run diff) ---")
        for l in difflib.unified_diff(lines, new_lines, fromfile=str(file), tofile=str(file)+'.patched'):
            print(l, end='')
    else:
        with open(file, 'w') as f:
            f.writelines(new_lines)
        print(f"Patched {file}")

def patch_template_matching_py(path, dry_run, reverse):
    file = path / "template_matching.py"
    bak = path / "template_matching.py.bak"
    if reverse:
        if bak.exists():
            shutil.copy2(bak, file)
            print(f"Restored {file} from backup.")
        else:
            print(f"No backup found for {file}.")
        return
    with open(file) as f:
        lines = f.readlines()
    if not bak.exists():
        shutil.copy2(file, bak)
    # Insert at start of run_matching
    idx = None
    for i, line in enumerate(lines):
        if 'def run_matching' in line:
            idx = i
            break
    if idx is None:
        sys.exit("Could not find run_matching in template_matching.py.")
    # Insert after function def and docstring (if present)
    insert_idx = idx+1
    while insert_idx < len(lines) and (lines[insert_idx].strip().startswith('"""') or lines[insert_idx].strip() == ''):
        insert_idx += 1
    # Check if already patched
    if any('cross_peel_claim_ms' in l for l in lines):
        print(f"Already patched: {file}")
        return
    new_lines = lines[:insert_idx] + [RUN_MATCHING_PATCH] + lines[insert_idx:]
    # Insert claim mask filter inside peel loop (look for 'for t in range(max_peels):')
    for_loop_idx = None
    for i, line in enumerate(new_lines):
        if 'for t in range(max_peels):' in line:
            for_loop_idx = i
            break
    if for_loop_idx is None:
        sys.exit("Could not find 'for t in range(max_peels):' in template_matching.py.")
    # Find where iX/iY are formed (look for 'iX = xs[:,:1]')
    ix_idx = None
    for i in range(for_loop_idx, len(new_lines)):
        if 'iX = xs[:,:1]' in new_lines[i]:
            ix_idx = i
            break
    if ix_idx is None:
        sys.exit("Could not find 'iX = xs[:,:1]' in template_matching.py.")
    # Insert after iX/iY
    after_ix = ix_idx + 2  # iX and iY
    new_lines = new_lines[:after_ix] + [PEEL_LOOP_PATCH] + new_lines[after_ix:]
    # Insert after accepting spikes (look for 'k+= nsp')
    k_idx = None
    for i in range(after_ix, len(new_lines)):
        if 'k+= nsp' in new_lines[i]:
            k_idx = i
            break
    if k_idx is None:
        sys.exit("Could not find 'k+= nsp' in template_matching.py.")
    new_lines = new_lines[:k_idx+1] + [AFTER_ACCEPT_PATCH] + new_lines[k_idx+1:]
    if dry_run:
        print(f"--- {file} (dry-run diff) ---")
        for l in difflib.unified_diff(lines, new_lines, fromfile=str(file), tofile=str(file)+'.patched'):
            print(l, end='')
    else:
        with open(file, 'w') as f:
            f.writelines(new_lines)
        print(f"Patched {file}")

def main():
    parser = argparse.ArgumentParser(description="Patch Kilosort for cross-peel claim mask.")
    parser.add_argument('--dry-run', action='store_true', help='Show changes but do not modify files')
    parser.add_argument('--reverse', action='store_true', help='Restore from .bak backups (undo patch)')
    args = parser.parse_args()

    ks_path = find_kilosort_path()
    print(f"Located kilosort at: {ks_path}")
    patch_parameters_py(ks_path, args.dry_run, args.reverse)
    patch_template_matching_py(ks_path, args.dry_run, args.reverse)
    print("Done.")

if __name__ == '__main__':
    main()
