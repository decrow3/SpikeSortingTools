# H1 candidate3 transition readiness recommendation

Verdict: **ABANDON_H1_FOR_CANDIDATE3_TRANSITION_USE_EXISTING_H5_SUPERVISED_PATH_LATER**.

The condition is structural on the project timescale.  Thirty-five Python
threads have remained uninterruptible on remote-file page waits for nearly 168
days.  Six `sync` calls have been stuck from four to 152 days, including an
explicit `/mnt` packet sync.  Ten still-running whole-filesystem `bfs` workers
add FUSE inode-lock waiters and have persisted 19–39 days.  At the terminal
snapshot, `vmstat` held 36 blocked tasks and 77–79% I/O wait while showing zero
or negligible block traffic.  Passive waiting is therefore not a bounded or
credible route to launch readiness.

The September filesystem searches remain causative contributors and should not
be repeated.  They are not the original or dominant cause: the April SpikeGLX
processes predate them by about five months and account for 35 of 52 D-state
threads.  Both families converge on remote/FUSE/CIFS wait paths.  The main
processes hold the same `/mnt/NPX/Luke/...ap.bin` file, and the `/mnt` CIFS mount
shows reconnects and failed writes despite an established SMB socket and no
current outstanding request.

Do not launch transition v7 on H1 and do not schedule another passive readiness
poll.  Use the existing reviewed H5 supervised path later when the coordinator
prioritizes it.  H1 should be reconsidered only after separately authorized
host-administrator remediation of the old Code scope, stuck searches/syncs and
CIFS client/mount state.  A fresh readiness check would then require all of:

- no project-relevant D-state task on `/mnt` in two snapshots ten minutes apart;
- `vmstat b=0` and I/O wait below 10% in three one-second samples at each snapshot;
- no surviving whole-root search or blocked sync;
- a bounded `/mnt` metadata probe completes within its declared timeout;
- the candidate3 managed-launch resource/dedup prestart is rerun from scratch.

No process, service, mount, priority, or file was changed by this diagnosis.

Implementation checks
- Done: host namespace process/thread state and wait channels identify the
  blocked families and ages.
- Done: representative open-file descriptors bind the April SpikeGLX workers to
  the exact `/mnt/NPX/Luke/...ap.bin` remote input.
- Done: cgroup inspection binds them and the searches to the long-lived VS Code
  application scope; CIFS and socket statistics bind `/mnt` to 10.44.239.58.
- Done: repeated `vmstat` and disk-counter samples distinguish high wait from
  active local-device throughput.
- Not done: kernel stack dump, server-side SMB diagnosis, process termination,
  unmount/remount or reboot; all are outside this read-only task.
- Can establish: H1 is not safely launchable and passive waiting has no bounded
  completion condition.
- Cannot establish: the exact server/kernel defect or whether administrator
  remediation can recover H1 without reboot.
