"""One-off uploader for finished external-reference patched sets.

Edit HOST and FINISHED_PIPELINES below, run once in dry-run mode, then set
EXECUTE = True when the printed plan looks correct.
"""

from __future__ import annotations

import shlex
import subprocess
from datetime import datetime
from pathlib import Path


HOST = "declan@192.168.195.206"
EXECUTE = True
ARCHIVE_SUFFIX = datetime.now().strftime("%Y-%m-%d")

LOCAL_ROOT = Path("/media/huklaban5/Data/Patched")
REMOTE_ROOT = "/mnt/ssd2/RowleyMarmoV1V2/raw"

# Hardcoded from batch_ext_ref_patching.py entries marked "done".
# uploaded 2026-05-05:
# FINISHED_PIPELINES = [
#     #("Luke", "2026-03-16", "patched_pipeline_results_Luke03162026_V2V1_RH_g0_imec1"),
#     ("Luke", "2026-03-16", "patched_pipeline_results_Luke03162026_V2V1_RH_g0_imec0"),
#     ("Luke", "2026-03-13", "patched_pipeline_results_Luke03132026_V2V1_RH_g0_imec1"),
#     ("Luke", "2026-03-13", "patched_pipeline_results_Luke03132026_V2V1_RH_g0_imec0"),
#     ("Luke", "2026-03-11", "patched_pipeline_results_Luke03112026_V2V1_RH_g0_imec1"),
#     ("Luke", "2026-03-11", "patched_pipeline_results_Luke03112026_V2V1_RH_g0_imec0"),
#     ("Luke", "2026-03-09", "patched_pipeline_results_Luke03092026_V1_RH_g0_imec1"),
#     ("Luke", "2026-03-09", "patched_pipeline_results_Luke03092026_V1_RH_g0_imec0"),
#     ("Luke", "2026-03-08", "patched_pipeline_results_Luke03082026_V1_RH_g0_imec0"),
#     ("Luke", "2026-03-02", "patched_pipeline_results_Luke03022026_V2V1_RH_g0_imec1"),
#     ("Luke", "2026-03-02", "patched_pipeline_results_Luke03022026_V2V1_RH_g0_imec0"),
#     ("Luke", "2026-03-01", "patched_pipeline_results_Luke03012026_V2V1_RH_g0_imec1"),
#     ("Luke", "2026-03-01", "patched_pipeline_results_Luke03012026_V2V1_RH_g0_imec0"),
#     #("Luke", "2025-12-05", "patched_pipeline_results_Luke12052025_V1_RH_g0_imec1"),
#     ("Luke", "2025-12-05", "patched_pipeline_results_Luke12052025_V1_RH_g0_imec0"),
#     #("Luke", "2025-08-05", "patched_pipeline_results_Luke0805_V2V1_g0_imec1"),
#     ("Luke", "2025-08-05", "patched_pipeline_results_Luke0805_V2V1_g0_imec0"),
#     ("Luke", "2025-08-04", "patched_pipeline_results_Luke0804_V2V1_g0_imec1"),
#     ("Luke", "2025-08-04", "patched_pipeline_results_Luke0804_V2V1_g0_imec0"),
#     ("Luke", "2025-07-30", "patched_pipeline_results_Luke0730_V2V1_g0_imec1"),
# ]

#todo
FINISHED_PIPELINES = [
    # ("Luke", "2025-07-30", "patched_pipeline_results_Luke0730_V2V1_g0_imec0"),
    # ("Luke", "2025-07-30", "patched_pipeline_results_Luke0724_V2V1_g0_imec1"),
    # ("Luke", "2025-07-24", "patched_pipeline_results_Luke0724_V2V1_g0_imec0"),
    ("Luke", "2025-07-24", "patched_pipeline_results_Luke0724_V2V1_g0_imec1"),
    # ("Luke", "2025-07-17", "patched_pipeline_results_Luke0717_V1_g0_imec0"),
]

def build_sftp_commands(local_pipeline_dir: Path, remote_pipeline_dir: str) -> list[str]:
    local_cur = local_pipeline_dir / "cur"
    local_qc = local_pipeline_dir / "qc"
    remote_deprecated = f"{remote_pipeline_dir}/deprecated"
    return [
        f"mkdir {shlex.quote(remote_deprecated)}",
        f"rename {shlex.quote(remote_pipeline_dir + '/cur')} {shlex.quote(remote_deprecated + '/cur_' + ARCHIVE_SUFFIX)}",
        f"rename {shlex.quote(remote_pipeline_dir + '/qc')} {shlex.quote(remote_deprecated + '/qc_' + ARCHIVE_SUFFIX)}",
        f"put -r {shlex.quote(str(local_cur))} {shlex.quote(remote_pipeline_dir)}",
        f"put -r {shlex.quote(str(local_qc))} {shlex.quote(remote_pipeline_dir)}",
    ]


def run_command(command: list[str], *, input_text: str | None = None) -> None:
    subprocess.run(command, input=input_text, text=True, check=True)


def main() -> int:
    missing = []
    planned_batches = []

    for subject, session_date, pipeline_name in FINISHED_PIPELINES:
        local_pipeline_dir = LOCAL_ROOT / pipeline_name
        local_cur = local_pipeline_dir / "cur"
        local_qc = local_pipeline_dir / "qc"
        if not local_cur.exists() or not local_qc.exists():
            missing.append(str(local_pipeline_dir))
            continue

        remote_pipeline_dir = f"{REMOTE_ROOT}/{subject}_{session_date}/{pipeline_name}"
        sftp_commands = build_sftp_commands(local_pipeline_dir, remote_pipeline_dir)
        planned_batches.append((pipeline_name, local_pipeline_dir, remote_pipeline_dir, sftp_commands))

    if missing:
        print()
        print("Missing local cur/qc for:")
        for path in missing:
            print(f"  {path}")
        if not planned_batches:
            return 1

    all_sftp_commands = []
    for pipeline_name, local_pipeline_dir, remote_pipeline_dir, sftp_commands in planned_batches:
        print()
        print(pipeline_name)
        print(f"  local : {local_pipeline_dir}")
        print(f"  remote: {remote_pipeline_dir}")
        print("  sftp:")
        for line in sftp_commands:
            print(f"    {line}")
        all_sftp_commands.extend(sftp_commands)

    if EXECUTE:
        print()
        print("Opening one sftp session for all uploads")
        run_command(["sftp", HOST], input_text="\n".join(all_sftp_commands) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())