#%%
#!/usr/bin/env python3
"""
compare_sortings_after_qc.py

Compares two pipeline outputs (post-curation) using
spikeinterface.comparison.compare_two_sorters().

Interactive usage:
    python compare_sortings_after_qc.py
    # then follow the prompts

By default it loads:
    <pipeX>/cur/cur_sorter_output
will attempt to keep only units labeled "good" in cluster info (if present).
QC-based filtering is intentionally left as a hook because your QC code
computes QC artifacts but does not define a single "good unit" threshold.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

import os
import matplotlib

import matplotlib.pyplot as plt

SHOW_PLOTS = True

# Depth grouping for plotting subsets of units.
# This relies on Phy/Kilosort exporting a per-unit depth-like property (often "depth").
DEPTH_GROUPS = 6
MAX_TICKLABEL_UNITS = 60


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


IN_IPYTHON = _in_ipython()

# If you want interactive windows, you need a GUI backend and an available display.
# In VS Code #%% / Jupyter (IPython), figures are best shown via rich display in the
# Interactive window; do not force a GUI backend.
# In a plain terminal, fall back to GUI backends when a display is available; otherwise save PNGs only.
if not IN_IPYTHON:
    if SHOW_PLOTS and os.environ.get("DISPLAY"):
        for _backend in ("QtAgg", "Qt5Agg", "TkAgg"):
            try:
                matplotlib.use(_backend, force=True)
                break
            except Exception:
                pass
    else:
        matplotlib.use("Agg")

import spikeinterface.extractors as se
import spikeinterface.comparison as scmp
import spikeinterface.widgets as sw
import spikeinterface as si


def compute_quality_metrics_from_sorting(
    sorting,
    *,
    recording,
    folder: Path,
    overwrite: bool = False,
    label: str = "sorting",
    max_spikes_per_unit: int = 500,
    prefer_sparse: bool = True,
    verbose: bool = True,
):
    """Compute SpikeInterface quality metrics using the SortingAnalyzer API.

    Newer SpikeInterface versions require `compute_quality_metrics()` to be called with a
    `SortingAnalyzer` (not a Sorting). This helper builds an analyzer, computes the
    minimal prerequisite extensions when possible, and returns the metrics dataframe.
    """

    try:
        import spikeinterface as si
        import spikeinterface.qualitymetrics as sqm
    except Exception as e:
        raise RuntimeError(f"Failed to import spikeinterface/qualitymetrics: {e}")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    analyzer = None
    attempt_errors: list[str] = []

    # In SI 0.102.x, create_sorting_analyzer signature is:
    # (sorting, recording, format='memory', folder=None, sparse=True, ..., overwrite=False, ...)
    # A common failure when using ad-hoc `read_binary()` recordings is missing probe/channel locations;
    # in that case, try `sparse=False`.
    sparse_order = [True, False] if prefer_sparse else [False, True]
    formats = ["binary_folder", "memory"]

    for sparse in sparse_order:
        for fmt in formats:
            for call_style in ("kwargs", "args"):
                try:
                    if call_style == "kwargs":
                        analyzer = si.create_sorting_analyzer(
                            sorting=sorting,
                            recording=recording,
                            format=fmt,
                            folder=str(folder),
                            sparse=sparse,
                            overwrite=overwrite,
                        )
                    else:
                        analyzer = si.create_sorting_analyzer(
                            sorting,
                            recording,
                            fmt,
                            str(folder),
                            sparse,
                            None,
                            True,
                            overwrite,
                        )
                    break
                except Exception as e:
                    attempt_errors.append(f"format={fmt} sparse={sparse} style={call_style}: {type(e).__name__}: {e}")
            if analyzer is not None:
                break
        if analyzer is not None:
            break

    if analyzer is None:
        msg = (
            "Could not create a SortingAnalyzer. "
            "This is often due to missing probe/channel location metadata when `sparse=True`. "
        )
        if verbose and attempt_errors:
            msg += "\nCreate attempts failed with:\n- " + "\n- ".join(attempt_errors[-8:])
        raise RuntimeError(msg)

    analyzer_nn = analyzer  # make non-optional for type checkers

    # Try to compute prerequisites (best-effort; API varies across versions).
    def _compute(ext: str, kwargs_list: list[dict] | None = None):
        kwargs_list = kwargs_list or [{}]
        for kwargs in kwargs_list:
            try:
                analyzer_nn.compute(ext, **kwargs)
                return
            except Exception:
                continue

    _compute(
        "random_spikes",
        [
            {"method": "uniform", "max_spikes_per_unit": max_spikes_per_unit, "seed": 0},
            {"max_spikes_per_unit": max_spikes_per_unit, "seed": 0},
            {},
        ],
    )
    _compute(
        "waveforms",
        [
            {"max_spikes_per_unit": max_spikes_per_unit},
            {},
        ],
    )
    _compute("templates", [{}, {"operators": "average"}])
    _compute("noise_levels", [{}, {"method": "mad"}])

    try:
        qm = sqm.compute_quality_metrics(analyzer_nn)
    except Exception as e:
        raise RuntimeError(
            "compute_quality_metrics() failed even with a SortingAnalyzer. "
            f"Original error: {e}"
        )

    # Normalize return to a DataFrame
    if isinstance(qm, pd.DataFrame):
        qm_df = qm
    else:
        try:
            qm_df = pd.DataFrame(qm)
        except Exception:
            qm_df = pd.DataFrame()

    try:
        qm_df.to_csv(folder / f"quality_metrics_{label}.csv")
    except Exception:
        pass

    return qm_df


def _series_for_unit_ids(values_by_cluster_id: pd.Series, unit_ids: list) -> pd.Series:
    """Reindex a Series keyed by cluster_id to the sorting's unit_ids.

    Handles minor dtype mismatches (e.g., int vs np.int64 vs str) by trying int casts.
    """
    try:
        unit_ids_int = [int(x) for x in unit_ids]
        values_idx = values_by_cluster_id.copy()
        try:
            values_idx.index = values_idx.index.map(int)
        except Exception:
            pass
        out = pd.Series(values_idx.reindex(unit_ids_int).to_numpy(), index=unit_ids, dtype="float64")
        return out
    except Exception:
        return pd.Series(values_by_cluster_id.reindex(unit_ids).to_numpy(), index=unit_ids, dtype="float64")


def _try_depths_from_cluster_info(phy_folder: Path, unit_ids: list) -> pd.Series | None:
    path = phy_folder / "cluster_info.tsv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return None

    # Identify the cluster id column
    id_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"cluster_id", "clusterid", "id", "cluster"}:
            id_col = c
            break
    if id_col is not None:
        df = df.set_index(id_col)

    # Identify the depth column
    depth_col = None
    depth_candidates = {
        "depth",
        "depth_um",
        "depth_µm",
        "y",
        "ypos",
        "y_pos",
        "y_position",
        "center_y",
        "pos_y",
        "position_y",
    }
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in depth_candidates:
            depth_col = c
            break
    if depth_col is None:
        return None

    s = pd.to_numeric(df[depth_col], errors="coerce")
    s.name = "depth"
    out = _series_for_unit_ids(s, unit_ids)
    if out.notna().any():
        return out
    return None


def _try_depths_from_cluster_depths_npy(phy_folder: Path, unit_ids: list) -> pd.Series | None:
    depths_path = phy_folder / "cluster_depths.npy"
    ids_path = phy_folder / "cluster_ids.npy"
    if not depths_path.exists() or not ids_path.exists():
        return None

    try:
        depths = np.load(depths_path).ravel()
        ids = np.load(ids_path).ravel()
        s = pd.Series(depths, index=ids, dtype="float64")
        out = _series_for_unit_ids(s, unit_ids)
        if out.notna().any():
            return out
    except Exception:
        return None
    return None


def _try_depths_from_channel_positions(
    phy_folder: Path, sorting, unit_ids: list, *, name: str
) -> pd.Series | None:
    pos_path = phy_folder / "channel_positions.npy"
    if not pos_path.exists():
        return None

    props = set(sorting.get_property_keys())
    ch_key = None
    for k in ["peak_channel", "peak_channel_id", "ch", "channel"]:
        if k in props:
            ch_key = k
            break
    if ch_key is None:
        return None

    ch_values = sorting.get_property(ch_key)
    if ch_values is None or len(ch_values) != len(unit_ids):
        return None

    try:
        ch_positions = np.load(pos_path)
        if ch_positions.ndim != 2 or ch_positions.shape[1] < 2:
            return None
        y = ch_positions[:, 1]
    except Exception:
        return None

    # Optional mapping from channel ids to indices
    id_to_index: dict[int, int] | None = None
    ch_map_path = phy_folder / "channel_map.npy"
    if ch_map_path.exists():
        try:
            ch_map = np.load(ch_map_path).ravel()
            id_to_index = {int(cid): int(i) for i, cid in enumerate(ch_map)}
        except Exception:
            id_to_index = None

    depths = []
    any_ok = False
    for v in ch_values:
        try:
            ch = int(v)
        except Exception:
            depths.append(np.nan)
            continue

        idx = None
        if id_to_index is not None:
            idx = id_to_index.get(ch)
        if idx is None and 0 <= ch < len(y):
            idx = ch

        if idx is None or not (0 <= idx < len(y)):
            depths.append(np.nan)
            continue

        depths.append(float(y[idx]))
        any_ok = True

    if any_ok:
        print(
            f"[{name}] NOTE: using '{ch_key}' + channel y-positions from channel_positions.npy for depth grouping."
        )
        return pd.Series(depths, index=unit_ids, dtype="float64")
    return None


def _get_unit_depths(sorting, phy_folder: Path, *, name: str) -> pd.Series:
    """Return per-unit depths as a Series indexed by unit_id.

    Prefers a true depth/y property from spikeinterface properties; otherwise tries
    common Phy/Kilosort output files in the Phy folder.
    """
    unit_ids = list(sorting.unit_ids)
    props = set(sorting.get_property_keys())

    depth_keys = [
        "depth",
        "Depth",
        "y",
        "Y",
        "unit_depth",
        "center_y",
        "peak_channel_depth",
    ]
    for key in depth_keys:
        if key not in props:
            continue
        values = sorting.get_property(key)
        if values is None:
            continue
        if len(values) != len(unit_ids):
            continue
        s = pd.Series(values, index=unit_ids, dtype="float64")
        if s.notna().any():
            return s

    # Try Phy/Kilosort files
    for getter in (
        _try_depths_from_cluster_info,
        _try_depths_from_cluster_depths_npy,
    ):
        try:
            out = getter(phy_folder, unit_ids)
        except Exception:
            out = None
        if out is not None and out.notna().any():
            return out

    try:
        out = _try_depths_from_channel_positions(phy_folder, sorting, unit_ids, name=name)
    except Exception:
        out = None
    if out is not None and out.notna().any():
        return out

    # Debug info
    try:
        keys_preview = ", ".join(sorted(props))
    except Exception:
        keys_preview = "<unavailable>"
    try:
        depth_files = [
            p.name
            for p in phy_folder.iterdir()
            if p.name
            in {
                "cluster_info.tsv",
                "cluster_depths.npy",
                "cluster_ids.npy",
                "channel_positions.npy",
                "channel_map.npy",
            }
        ]
    except Exception:
        depth_files = []

    print(f"[{name}] WARNING: no depth-like property found; depth-group plots will be skipped.")
    print(f"[{name}] Available sorting properties: {keys_preview}")
    if len(depth_files):
        print(f"[{name}] Depth-related files present: {', '.join(sorted(depth_files))}")
    else:
        print(f"[{name}] No depth-related files found in: {phy_folder}")
    return pd.Series(index=unit_ids, dtype="float64")


def _plot_agreement_heatmap(
    sub_scores: pd.DataFrame,
    *,
    title: str,
    outpath: Path,
    show_inline: bool,
):
    import matplotlib.pyplot as plt

    n_rows, n_cols = sub_scores.shape
    figsize = (
        float(np.clip(2.0 + 0.18 * max(n_cols, 1), 6.0, 18.0)),
        float(np.clip(2.0 + 0.18 * max(n_rows, 1), 6.0, 18.0)),
    )
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    mat = sub_scores.to_numpy(dtype=float, copy=True)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Units (sorting 2)")
    ax.set_ylabel("Units (sorting 1)")

    if n_rows <= MAX_TICKLABEL_UNITS:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([str(x) for x in sub_scores.index])
    else:
        ax.set_yticks([])

    if n_cols <= MAX_TICKLABEL_UNITS:
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([str(x) for x in sub_scores.columns], rotation=90)
    else:
        ax.set_xticks([])

    fig.colorbar(im, ax=ax, shrink=0.85, label="agreement")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if show_inline and IN_IPYTHON:
        try:
            from IPython.display import display  # type: ignore

            display(fig)
        except Exception:
            pass

    plt.close(fig)

#%%
def read_phy_sorting(phy_folder: Path):
    if not phy_folder.exists():
        raise FileNotFoundError(f"Phy folder not found: {phy_folder}")
    # spikeinterface.extractors.read_phy is imported/used in your repo curation code
    # so this matches your existing workflow. :contentReference[oaicite:12]{index=12}
    return se.read_phy(folder_path=str(phy_folder), load_all_cluster_properties=True)


def maybe_filter_good_units_from_phy(sorting):
    """
    If Phy cluster labels exist, keep units labeled 'good'.

    This is conservative: if no labels/properties are found, it returns sorting unchanged.
    """
    props = sorting.get_property_keys()
    label_key = None
    for k in ["KSLabel", "group", "quality", "cluster_group"]:
        if k in props:
            label_key = k
            break

    if label_key is None:
        return sorting

    labels = sorting.get_property(label_key)
    if labels is None:
        return sorting

    # normalize to lowercase strings
    labels_norm = [str(x).strip().lower() for x in labels]
    unit_ids = sorting.unit_ids
    good_ids = [uid for uid, lab in zip(unit_ids, labels_norm) if lab == "good"]

    if len(good_ids) == 0:
        return sorting

    return sorting.select_units(good_ids)


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value == "" and default is not None:
        return str(default)
    return value


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    choices_lc = [c.lower() for c in choices]
    default_lc = default.lower()
    if default_lc not in choices_lc:
        raise ValueError(f"Default '{default}' not in choices {choices}")

    while True:
        value = _prompt_text(f"{prompt} ({'/'.join(choices)})", default=default).lower()
        if value in choices_lc:
            return choices[choices_lc.index(value)]
        print(f"Invalid choice '{value}'. Choose one of: {', '.join(choices)}")


def _prompt_float(
    prompt: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    while True:
        raw = _prompt_text(prompt, default=str(default))
        try:
            value = float(raw)
        except ValueError:
            print(f"Not a number: '{raw}'")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}")
            continue
        return value


def _prompt_path(prompt: str, default: str | None = None, *, must_exist: bool) -> Path:
    while True:
        raw = _prompt_text(prompt, default=default)
        if raw.strip() == "":
            print("Please enter a path.")
            continue

        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            print(f"Path not found: {path}")
            continue
        return path


def _resolve_phy_folder(pipe_dir: Path, stage: str) -> Path:
    if stage == "cur":
        return pipe_dir / "cur" / "cur_sorter_output"
    if stage == "sorter_output":
        return pipe_dir / "sorting" / "sorter_output"
    raise ValueError(f"Unexpected stage: {stage}")

#%%


print("Compare two spike sorting outputs (interactive)")
print("- Provide two pipeline result folders")
print("- Outputs will be written to an output folder you choose")
print("")

pipe1 = Path("/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec1")
pipe2 = Path("/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec1_an5")
# pipe2 = Path("/mnt/NPX/Luke/20250804/dredgetest_pipeline_results_Luke0804_V2V1_g0_imec1")

stage = "cur"

default_outdir = str(Path.cwd() / "compare_out")
outdir = Path("/mnt/NPX/Luke/20250804/compare_out")
outdir.mkdir(parents=True, exist_ok=True)

delta_ms = 0.4
min_agreement = 0.5

units = "all"  # or "good"

#%%
phy1 = _resolve_phy_folder(pipe1, stage)
phy2 = _resolve_phy_folder(pipe2, stage)
if not phy1.exists():
    raise FileNotFoundError(f"Expected folder not found: {phy1}")
if not phy2.exists():
    raise FileNotFoundError(f"Expected folder not found: {phy2}")

sorting1 = read_phy_sorting(phy1)
sorting2 = read_phy_sorting(phy2)

if units == "good":
    sorting1 = maybe_filter_good_units_from_phy(sorting1)
    sorting2 = maybe_filter_good_units_from_phy(sorting2)

# Compare two sorters (symmetric comparison).
cmp = scmp.compare_two_sorters(
    sorting1=sorting1,
    sorting2=sorting2,
    sorting1_name=pipe1.name,
    sorting2_name=pipe2.name,
    delta_time=delta_ms / 1000.0,
)

# Export key matrices
match_event_count = cmp.match_event_count.copy()
agreement_scores = cmp.agreement_scores.copy()

# Convert to CSV
match_event_count.to_csv(outdir / "match_event_count.csv")
agreement_scores.to_csv(outdir / "agreement_scores.csv")

# Matching (Hungarian; unmatched show as -1).
m1_to_2, m2_to_1 = cmp.get_matching()
if m1_to_2 is None:
    m1_to_2 = {}

# Build a tidy table of matched pairs with their agreement
rows = []
for u1, u2 in m1_to_2.items():
    if u2 == -1:
        rows.append({"unit_1": u1, "unit_2": -1, "agreement": 0.0})
    else:
        # agreement_scores is a dataframe indexed by unit ids
        agr = float(agreement_scores.at[u1, u2])  # type: ignore[index]
        rows.append({"unit_1": u1, "unit_2": u2, "agreement": agr})
df_match = pd.DataFrame(rows)
df_match.to_csv(outdir / "matched_units.csv", index=False)

# Summary JSON
summary = {
    "pipe1": str(pipe1),
    "pipe2": str(pipe2),
    "stage": stage,
    "delta_ms": delta_ms,
    "units_filter": units,
    "n_units_1": int(len(sorting1.unit_ids)),
    "n_units_2": int(len(sorting2.unit_ids)),
    "n_matched": int((df_match["unit_2"] != -1).sum()),
    "n_unmatched_1": int((df_match["unit_2"] == -1).sum()),
    "matched_ge_min_agreement": int(((df_match["unit_2"] != -1) & (df_match["agreement"] >= min_agreement)).sum()),
    "min_agreement_threshold": min_agreement,
}
(outdir / "summary.json").write_text(json.dumps(summary, indent=2))

#%%
# Plots
# Agreement matrix visualization is the canonical first look.
fig1 = sw.plot_agreement_matrix(cmp, ordered=True)
fig1.figure.savefig(outdir / "agreement_matrix_ordered.png", dpi=200, bbox_inches="tight")
# Note: sw.plot_agreement_matrix returns a widget-like object; use matplotlib to show windows.

#%%
fig2 = sw.plot_agreement_matrix(cmp, ordered=False)
fig2.figure.savefig(outdir / "agreement_matrix.png", dpi=200, bbox_inches="tight")

if SHOW_PLOTS:
    if IN_IPYTHON:
        # VS Code Interactive/Jupyter: show inline in the Interactive window
        try:
            from IPython.display import display  # type: ignore

            display(fig1.figure)
            display(fig2.figure)
        except Exception:
            pass
    elif os.environ.get("DISPLAY"):
        # Terminal with GUI display: pop up interactive windows
        import matplotlib.pyplot as plt

        fig1.figure.show()
        fig2.figure.show()
        plt.show()

#%%
# Depth-grouped plots (subsets)
depths1 = _get_unit_depths(sorting1, phy1, name="sorting1")
depths2 = _get_unit_depths(sorting2, phy2, name="sorting2")

all_depths = pd.concat([depths1, depths2], axis=0).dropna().to_numpy(dtype=float)
if all_depths.size > 0 and DEPTH_GROUPS > 1:
    qs = np.linspace(0.0, 1.0, DEPTH_GROUPS + 1)
    edges = np.quantile(all_depths, qs)
    # Ensure strictly increasing edges (quantiles can repeat)
    edges = np.unique(edges)
    if edges.size >= 3:
        print(f"Plotting depth-grouped agreement matrices ({edges.size - 1} groups)")
        for i in range(edges.size - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == edges.size - 2:
                u1 = depths1[(depths1 >= lo) & (depths1 <= hi)].sort_values().index.tolist()
                u2 = depths2[(depths2 >= lo) & (depths2 <= hi)].sort_values().index.tolist()
            else:
                u1 = depths1[(depths1 >= lo) & (depths1 < hi)].sort_values().index.tolist()
                u2 = depths2[(depths2 >= lo) & (depths2 < hi)].sort_values().index.tolist()

            if len(u1) == 0 or len(u2) == 0:
                continue

            sub = agreement_scores.reindex(index=u1, columns=u2)

            # Print a few simple metrics for this depth group.
            df_g1 = df_match[df_match["unit_1"].isin(u1)]
            n1 = int(len(u1))
            n2 = int(len(u2))
            n1_matched_any = int((df_g1["unit_2"] != -1).sum())
            n1_matched_within = int(((df_g1["unit_2"] != -1) & (df_g1["unit_2"].isin(u2))).sum())
            within = df_g1[(df_g1["unit_2"] != -1) & (df_g1["unit_2"].isin(u2))]
            within_ge = int((within["agreement"] >= min_agreement).sum()) if len(within) else 0
            within_mean = float(within["agreement"].mean()) if len(within) else float("nan")
            within_median = float(within["agreement"].median()) if len(within) else float("nan")
            sub_np = sub.to_numpy(dtype=float, copy=True)
            sub_max = float(np.nanmax(sub_np)) if np.isfinite(sub_np).any() else float("nan")
            sub_mean = float(np.nanmean(sub_np)) if np.isfinite(sub_np).any() else float("nan")

            print(
                f"Depth group {i+1:02d} ({lo:.0f}–{hi:.0f}): "
                f"n1={n1} n2={n2} "
                f"matched_any={n1_matched_any}/{n1} "
                f"matched_within={n1_matched_within}/{n1} "
                f"within_ge_{min_agreement:.2f}={within_ge} "
                f"within_mean={within_mean:.3f} within_median={within_median:.3f} "
                f"sub_mean={sub_mean:.3f} sub_max={sub_max:.3f}"
            )

            outpath = outdir / f"agreement_depth_group_{i+1:02d}_{int(round(lo))}_{int(round(hi))}.png"
            _plot_agreement_heatmap(
                sub,
                title=f"Agreement (depth group {i+1}: {lo:.0f}–{hi:.0f})",
                outpath=outpath,
                show_inline=SHOW_PLOTS,
            )
    else:
        print("Depth grouping skipped: not enough unique depth values.")
else:
    print("Depth grouping skipped: no depth information found.")

# Also save the "high agreement" subset for quick eyeballing
df_high = df_match[(df_match["unit_2"] != -1) & (df_match["agreement"] >= min_agreement)].sort_values("agreement", ascending=False)
df_high.to_csv(outdir / f"matched_units_agreement_ge_{min_agreement:.2f}.csv", index=False)

print("Wrote outputs to:", outdir)
print(json.dumps(summary, indent=2))




#%%
def describe_sorting(s, name):
    print(f"\n== {name} ==")
    try:
        fs = s.get_sampling_frequency()
    except Exception as e:
        fs = None
        print("sampling_frequency: ERROR", e)
    print("sampling_frequency:", fs)
    try:
        nseg = s.get_num_segments()
    except Exception as e:
        nseg = None
        print("num_segments: ERROR", e)
    print("num_segments:", nseg)

    # crude duration proxy: max spike time across a few units
    unit_ids = list(s.unit_ids)
    probe_units = unit_ids[:20]
    max_s = -1.0
    for u in probe_units:
        st = s.get_unit_spike_train(u)  # samples
        if len(st):
            max_s = max(max_s, st.max())
    if fs and max_s > 0:
        print("max spike sample (first 20 units):", int(max_s))
        print("~max time (s):", float(max_s) / float(fs))

describe_sorting(sorting1, "sorting1")
describe_sorting(sorting2, "sorting2")
# What you’re looking for

# If sampling_frequency differs: that’s almost certainly the culprit.

# If the “~max time (s)” differs a lot: one pipeline likely cropped or segmented differently.

#%% 2) Does relaxing delta_time rescue matches?
#This tells you whether you have a constant offset vs complete mismatch.

#%%
def count_matches(delta_ms):
    cmp_tmp = scmp.compare_two_sorters(
        sorting1=sorting1, sorting2=sorting2,
        sorting1_name="s1", sorting2_name="s2",
        delta_time=delta_ms/1000.0,
    )
    m12, _ = cmp_tmp.get_matching()
    if m12 is None:
        m12 = {}
    # count matched (u2 != -1)
    matched = sum(1 for u1,u2 in m12.items() if u2 != -1)
    print(f"delta_ms={delta_ms:>6.1f}  matched={matched}")
    return matched

for d in [0.4, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    count_matches(d)


#%%

import numpy as np

def pop_rate(sorting, bin_s=1.0, max_s=1800.0):  # default: first 30 minutes
    fs = float(sorting.get_sampling_frequency())
    n_bins = int(np.floor(max_s / bin_s))
    counts = np.zeros(n_bins, dtype=float)
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u).astype(np.float64) / fs  # seconds
        st = st[(st >= 0) & (st < max_s)]
        idx = (st / bin_s).astype(int)
        np.add.at(counts, idx, 1)
    return counts

bin_s = 1.0
max_s = 1800.0

r1 = pop_rate(sorting1, bin_s=bin_s, max_s=max_s)
r2 = pop_rate(sorting2, bin_s=bin_s, max_s=max_s)

# cross-correlation
x = (r1 - r1.mean())
y = (r2 - r2.mean())
corr = np.correlate(x, y, mode="full")
lags = np.arange(-len(x)+1, len(x)) * bin_s
best = int(np.argmax(corr))
print("Best lag (s):", float(lags[best]))
# If you get a clear non-zero “Best lag (s)”, that’s your offset.

# Fixes once you know which issue it is
# If sampling frequency differs
# Force them to the same fs before comparing. SpikeInterface sortings often allow setting sampling frequency:

#%%
fs1 = sorting1.get_sampling_frequency()
fs2 = sorting2.get_sampling_frequency()
print(fs1, fs2)

# If one is None or clearly wrong, set it explicitly:
# sorting2 = sorting2.set_sampling_frequency(fs1)  # if support

#%%
import numpy as np

fs = float(sorting1.get_sampling_frequency())
T_s = 600.0          # test first 10 minutes
bin_ms = 5.0
bin_s = bin_ms / 1000.0
nb = int(np.floor(T_s / bin_s))

def binned_train(sorting, unit_id, T_s, bin_s, fs):
    st = sorting.get_unit_spike_train(unit_id).astype(np.float64) / fs
    st = st[(st >= 0) & (st < T_s)]
    b = np.zeros(nb, dtype=np.float32)
    idx = (st / bin_s).astype(int)
    np.add.at(b, idx, 1.0)
    return b

u1 = sorting1.unit_ids[0]
x = binned_train(sorting1, u1, T_s, bin_s, fs)
x = x - x.mean()

scores = []
for u2 in sorting2.unit_ids[:2000] if len(sorting2.unit_ids) > 2000 else sorting2.unit_ids:
    y = binned_train(sorting2, u2, T_s, bin_s, fs)
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-9
    scores.append((u2, float(np.dot(x, y) / denom)))

scores = sorted(scores, key=lambda t: t[1], reverse=True)
print("Top 10 coarse correlations for unit", u1)
for u2, r in scores[:10]:
    print(u2, r)



#%%
def nearest_diffs_ms(st_a_s, st_b_s, max_window_ms=200.0):
    # For each spike in A, find nearest spike in B
    b = st_b_s
    diffs = []
    j = 0
    maxw = max_window_ms / 1000.0
    for t in st_a_s:
        # advance pointer
        while j+1 < len(b) and b[j+1] < t:
            j += 1
        cand = []
        if j < len(b): cand.append(b[j] - t)
        if j+1 < len(b): cand.append(b[j+1] - t)
        if not cand: 
            continue
        d = min(cand, key=lambda z: abs(z))
        if abs(d) <= maxw:
            diffs.append(d * 1000.0)
    return np.array(diffs, dtype=float)

u2_best = scores[0][0]
st1 = sorting1.get_unit_spike_train(u1).astype(np.float64) / fs
st2 = sorting2.get_unit_spike_train(u2_best).astype(np.float64) / fs
st1 = st1[st1 < 600.0]
st2 = st2[st2 < 600.0]

diffs = nearest_diffs_ms(st1, st2, max_window_ms=200.0)
print("n diffs:", len(diffs))
if len(diffs):
    print("median |diff| (ms):", float(np.median(np.abs(diffs))))
    print("pct |diff| < 1 ms:", float(np.mean(np.abs(diffs) < 1.0)))
    print("pct |diff| < 5 ms:", float(np.mean(np.abs(diffs) < 5.0)))
    print("pct |diff| < 20 ms:", float(np.mean(np.abs(diffs) < 20.0)))


#%%
u = sorting1.unit_ids[0]
st = sorting1.get_unit_spike_train(u)
print("dtype:", st.dtype, "min:", st.min() if len(st) else None, "max:", st.max() if len(st) else None)

u = sorting2.unit_ids[0]
st = sorting2.get_unit_spike_train(u)
print("dtype:", st.dtype, "min:", st.min() if len(st) else None, "max:", st.max() if len(st) else None)


#%%
def total_spikes(s):
    return sum(len(s.get_unit_spike_train(u)) for u in s.unit_ids)

print("sorting1 total spikes:", total_spikes(sorting1))
print("sorting2 total spikes:", total_spikes(sorting2))


#%%
#%%
import numpy as np

fs = float(sorting1.get_sampling_frequency())

def sample_event_times_seconds(sorting, n_events=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    # sample units proportional to their spike counts
    unit_ids = np.array(sorting.unit_ids)
    counts = np.array([len(sorting.get_unit_spike_train(u)) for u in unit_ids], dtype=np.int64)
    probs = counts / counts.sum()

    # draw unit indices, then draw random spikes from each
    chosen_units = rng.choice(len(unit_ids), size=min(2000, len(unit_ids)), replace=False, p=probs)
    times = []
    for idx in chosen_units:
        u = unit_ids[idx]
        st = sorting.get_unit_spike_train(u)
        if len(st) == 0:
            continue
        take = max(1, int(n_events / len(chosen_units)))
        sel = rng.integers(0, len(st), size=min(take, len(st)))
        times.append(st[sel].astype(np.int64))
        if sum(len(x) for x in times) >= n_events:
            break
    t = np.concatenate(times)[:n_events]
    t.sort()
    return t.astype(np.float64) / fs  # seconds

t1 = sample_event_times_seconds(sorting1, n_events=500_000, seed=1)
t2 = sample_event_times_seconds(sorting2, n_events=500_000, seed=2)

def overlap_fraction(tA, tB, tol_ms=1.0):
    # two-pointer overlap count
    tol = tol_ms / 1000.0
    i = j = m = 0
    while i < len(tA) and j < len(tB):
        d = tA[i] - tB[j]
        if abs(d) <= tol:
            m += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return m / len(tA), m / len(tB), m

for tol in [0.4, 1.0, 5.0, 10.0, 20, 50.0, 100.0]:
    a, b, m = overlap_fraction(t1, t2, tol_ms=tol)
    print(f"tol_ms={tol:>4}: matches={m:>7}  frac_of_A={a:.4f}  frac_of_B={b:.4f}")

#%%
import numpy as np

def nn_diffs(tA, tB, max_window_ms=100.0):
    tol = max_window_ms / 1000.0
    i = j = 0
    diffs = []
    while i < len(tA) and j < len(tB):
        d = tA[i] - tB[j]
        if abs(d) <= tol:
            diffs.append(d * 1000.0)
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return np.array(diffs)

diffs_ms = nn_diffs(t1, t2, max_window_ms=200.0)
print("n matched pairs:", len(diffs_ms))
print("median |diff| (ms):", float(np.median(np.abs(diffs_ms))))
print("p90 |diff| (ms):", float(np.percentile(np.abs(diffs_ms), 90)))
print("p99 |diff| (ms):", float(np.percentile(np.abs(diffs_ms), 99)))

#%%
#%%
import numpy as np

def nearest_neighbor_diffs_ms(tA, tB, max_window_ms=20.0):
    """
    For each time in tA, find nearest time in tB (by absolute difference).
    Returns diffs (tA - nearest_tB) in ms, keeping only matches within max_window_ms.
    Assumes tA and tB are sorted 1D arrays in seconds.
    """
    tA = np.asarray(tA, dtype=np.float64)
    tB = np.asarray(tB, dtype=np.float64)
    if len(tA) == 0 or len(tB) == 0:
        return np.array([], dtype=np.float64)

    maxw = max_window_ms / 1000.0
    diffs = np.empty(len(tA), dtype=np.float64)
    keep = np.zeros(len(tA), dtype=bool)

    j = 0
    for i, t in enumerate(tA):
        while j + 1 < len(tB) and tB[j + 1] < t:
            j += 1

        # candidates around t
        cand = []
        if j < len(tB):
            cand.append(tB[j])
        if j + 1 < len(tB):
            cand.append(tB[j + 1])

        if not cand:
            continue

        # nearest candidate
        c = cand[0]
        if len(cand) == 2 and abs(cand[1] - t) < abs(cand[0] - t):
            c = cand[1]

        d = t - c
        if abs(d) <= maxw:
            diffs[i] = d * 1000.0
            keep[i] = True

    return diffs[keep]

# Use your sampled pooled times t1, t2 (seconds, sorted)
for W in [0.4, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    diffs = nearest_neighbor_diffs_ms(t1, t2, max_window_ms=W)
    if len(diffs) == 0:
        print(f"W={W:>5} ms: no matches")
        continue
    print(
        f"W={W:>5} ms: n={len(diffs):>7}, "
        f"median|d|={np.median(np.abs(diffs)):.3f} ms, "
        f"p90|d|={np.percentile(np.abs(diffs),90):.3f} ms, "
        f"p99|d|={np.percentile(np.abs(diffs),99):.3f} ms"
    )

#%% recording is 
recordingfolder = "/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec1/Luke0730_V2V1_g0_t0.imec1.ap.bin"
recording=se.read_binary(str(recordingfolder), sampling_frequency=30000.0, num_chan=384, dtype="int16")
import spikeinterface.qualitymetrics as sqm

qm1 = compute_quality_metrics_from_sorting(
    sorting1,
    recording=recording,
    folder=outdir / "qm_sorting1",
    overwrite=False,
    label="sorting1",
)
qm2 = compute_quality_metrics_from_sorting(
    sorting2,
    recording=recording,
    folder=outdir / "qm_sorting2",
    overwrite=False,
    label="sorting2",
)


#%%
def plot_population_raster(sorting, title, *, max_units=250, t0=0.0, t1=600.0, seed=0):
    fs = float(sorting.get_sampling_frequency())
    rng = np.random.default_rng(seed)
    unit_ids = np.array(sorting.unit_ids)
    if len(unit_ids) > max_units:
        unit_ids = rng.choice(unit_ids, size=max_units, replace=False)

    fig = plt.figure(figsize=(11, 6))
    for i, u in enumerate(unit_ids):
        st = sorting.get_unit_spike_train(u).astype(np.float64) / fs
        st = st[(st >= t0) & (st < t1)]
        plt.plot(st, np.full_like(st, i), ".", markersize=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Unit index (subset)")
    plt.title(title)
    plt.tight_layout()
    return fig


fig = plot_population_raster(sorting1, "Sorting1 raster (subset)", seed=1)
fig.savefig(outdir / "raster_sorting1_subset.png", dpi=200)
if SHOW_PLOTS and IN_IPYTHON:
    try:
        from IPython.display import display  # type: ignore

        display(fig)
    except Exception:
        pass
plt.close(fig)

fig = plot_population_raster(sorting2, "Sorting2 raster (subset)", seed=2)
fig.savefig(outdir / "raster_sorting2_subset.png", dpi=200)
if SHOW_PLOTS and IN_IPYTHON:
    try:
        from IPython.display import display  # type: ignore

        display(fig)
    except Exception:
        pass
plt.close(fig)


#%%
WE1_DIR = outdir / "we_sorting1"
WE2_DIR = outdir / "we_sorting2"


def ensure_recording_has_geometry(recording, *, pitch_um: float = 20.0):
    """Ensure the recording has either a probe or channel locations.

    `si.create_sorting_analyzer(..., sparse=True)` requires geometry/probe in SI>=0.101.
    For ad-hoc `se.read_binary()` recordings, this is often missing.
    """
    # Already has a probe?
    try:
        pg = recording.get_probegroup()
        if pg is not None:
            return recording
    except Exception:
        pass

    # Already has channel locations?
    try:
        loc = recording.get_channel_locations()
        loc = np.asarray(loc)
        if loc.size and np.isfinite(loc).any():
            return recording
    except Exception:
        pass

    # Build simple linear geometry
    n = None
    for attr in ("get_num_channels",):
        if hasattr(recording, attr):
            try:
                n = int(getattr(recording, attr)())
                break
            except Exception:
                pass
    if n is None:
        try:
            n = int(len(recording.channel_ids))
        except Exception:
            n = 0
    if n <= 0:
        return recording

    positions = np.zeros((n, 2), dtype="float64")
    positions[:, 1] = np.arange(n, dtype="float64") * float(pitch_um)

    # Prefer setting channel locations directly if available
    if hasattr(recording, "set_channel_locations"):
        try:
            rec2 = recording.set_channel_locations(positions, in_place=False)
            print("[recording] NOTE: attached dummy channel locations for analyzer geometry")
            return rec2
        except TypeError:
            try:
                recording.set_channel_locations(positions)
                print("[recording] NOTE: attached dummy channel locations for analyzer geometry")
                return recording
            except Exception:
                pass
        except Exception:
            pass

    # Fallback: attach a dummy probe (requires probeinterface)
    try:
        import probeinterface as pi  # type: ignore

        probe = pi.Probe(ndim=2)
        probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
        probe.set_device_channel_indices(np.arange(n, dtype=int))

        if hasattr(recording, "set_probe"):
            rec2 = recording.set_probe(probe, in_place=False)
            print("[recording] NOTE: attached dummy probe for analyzer geometry")
            return rec2
        set_probe_fn = getattr(si, "set_probe", None)
        if callable(set_probe_fn):
            rec2 = set_probe_fn(recording, probe, in_place=False)  # type: ignore[misc]
            print("[recording] NOTE: attached dummy probe for analyzer geometry")
            return rec2
    except Exception:
        pass

    return recording


def get_or_make_we(recording, sorting, folder: Path):
    """Get a WaveformExtractor-like object (or compatible proxy).

    Tries to use legacy WaveformExtractor APIs if available; otherwise uses a SortingAnalyzer
    and computes the required extensions.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Legacy API (older SI)
    if folder.exists():
        try:
            if hasattr(si, "load_waveforms"):
                return si.load_waveforms(folder)
        except Exception:
            pass

        try:
            from spikeinterface.core.waveforms import WaveformExtractor  # type: ignore

            return WaveformExtractor.load_from_folder(folder)
        except Exception:
            pass

    try:
        if hasattr(si, "extract_waveforms"):
            return si.extract_waveforms(
                recording,
                sorting,
                folder=folder,
                overwrite=True,
                max_spikes_per_unit=500,
                sparse=True,
                n_jobs=-1,
                progress_bar=True,
            )
    except Exception:
        pass

    # Newer API: SortingAnalyzer
    rec = ensure_recording_has_geometry(recording)
    analyzer = None
    created_sparse = True
    attempt_errors: list[str] = []
    for sparse in (True, False):
        for fmt in ("binary_folder", "memory"):
            try:
                analyzer = si.create_sorting_analyzer(
                    sorting=sorting,
                    recording=rec,
                    format=fmt,
                    folder=str(folder),
                    sparse=sparse,
                    overwrite=True,
                )
                created_sparse = sparse
                break
            except Exception as e:
                attempt_errors.append(
                    f"format={fmt} sparse={sparse}: {type(e).__name__}: {e}"
                )
        if analyzer is not None:
            break
    if analyzer is None:
        raise RuntimeError(
            "Could not create SortingAnalyzer for waveforms/templates. "
            + ("\n- " + "\n- ".join(attempt_errors[-6:]) if attempt_errors else "")
        )

    try:
        analyzer.compute(
            "random_spikes",
            method="uniform",
            max_spikes_per_unit=500,
            seed=0,
        )
    except Exception:
        pass
    try:
        analyzer.compute(
            "waveforms",
            max_spikes_per_unit=500,
            sparse=created_sparse,
            n_jobs=-1,
            progress_bar=True,
        )
    except Exception:
        pass
    try:
        analyzer.compute("templates")
    except Exception:
        pass
    return analyzer


we1 = get_or_make_we(recording, sorting1, WE1_DIR)
we2 = get_or_make_we(recording, sorting2, WE2_DIR)


def _get_templates_and_unit_ids(we_or_analyzer):
    # WaveformExtractor path
    if hasattr(we_or_analyzer, "get_all_templates"):
        templates = we_or_analyzer.get_all_templates()
        unit_ids = list(we_or_analyzer.sorting.unit_ids)
        return templates, unit_ids

    # SortingAnalyzer path
    if hasattr(we_or_analyzer, "get_extension"):
        try:
            ext = we_or_analyzer.get_extension("templates")
        except Exception:
            ext = None

        templates = None
        if ext is not None:
            for attr in ("get_data", "templates", "data"):
                try:
                    v = getattr(ext, attr)
                    templates = v() if callable(v) else v
                    break
                except Exception:
                    continue

        if templates is None:
            raise RuntimeError("Could not access templates from SortingAnalyzer")

        # sorting
        try:
            unit_ids = list(we_or_analyzer.sorting.unit_ids)
        except Exception:
            unit_ids = list(we_or_analyzer.sorting_analyzer.sorting.unit_ids)  # type: ignore[attr-defined]
        return templates, unit_ids

    raise TypeError("Unsupported waveform/template container")


def plot_templates_by_depth(we, title, outpath, max_units=180):
    templates, unit_ids = _get_templates_and_unit_ids(we)
    templates = np.asarray(templates)
    if templates.ndim != 3:
        raise ValueError(f"Unexpected templates shape: {templates.shape}")

    n = min(max_units, len(unit_ids), templates.shape[0])

    # For each unit, choose peak channel by ptp
    peak_ch = []
    waves = []
    for i in range(n):
        temp = templates[i]
        ch = int(np.argmax(np.ptp(temp, axis=0)))
        peak_ch.append(ch)
        waves.append(temp[:, ch])

    peak_ch = np.array(peak_ch)
    waves = np.array(waves)

    order = np.argsort(peak_ch)
    waves = waves[order]

    fig = plt.figure(figsize=(7, 10))
    for i, w in enumerate(waves):
        w = w.astype(np.float64)
        w /= (np.max(np.abs(w)) + 1e-9)
        plt.plot(w + 2.2 * i, linewidth=0.8)

    plt.title(title + " (each unit: peak-channel waveform, normalized)")
    plt.xlabel("Samples")
    plt.ylabel("Units ordered by peak channel (depth proxy)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    if SHOW_PLOTS and IN_IPYTHON:
        try:
            from IPython.display import display  # type: ignore

            display(fig)
        except Exception:
            pass
    plt.close(fig)


plot_templates_by_depth(we1, "Sorting1 templates vs depth proxy", outdir / "templates_depth_sorting1.png")
plot_templates_by_depth(we2, "Sorting2 templates vs depth proxy", outdir / "templates_depth_sorting2.png")


#%%
def presence_ratio_per_unit(sorting, bin_s=10.0):
    fs = float(sorting.get_sampling_frequency())
    max_sample = 0
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u)
        if len(st):
            max_sample = max(max_sample, int(st.max()))
    T = max_sample / fs if max_sample > 0 else bin_s
    nbins = max(1, int(T // bin_s))

    ratios = np.zeros(len(sorting.unit_ids), dtype=np.float64)
    for i, u in enumerate(sorting.unit_ids):
        st = sorting.get_unit_spike_train(u).astype(np.float64) / fs
        bins = np.floor(st / bin_s).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < nbins)]
        ratios[i] = (len(np.unique(bins)) / nbins) if nbins else 0.0
    return ratios, nbins


def fraction_units_active_over_time(sorting, bin_s=10.0):
    fs = float(sorting.get_sampling_frequency())
    max_sample = 0
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u)
        if len(st):
            max_sample = max(max_sample, int(st.max()))
    T = max_sample / fs if max_sample > 0 else bin_s
    nbins = max(1, int(T // bin_s))

    active = np.zeros(nbins, dtype=np.int64)
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u).astype(np.float64) / fs
        bins = np.floor(st / bin_s).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < nbins)]
        if len(bins) == 0:
            continue
        active[np.unique(bins)] += 1

    frac = active / float(len(sorting.unit_ids))
    t = (np.arange(nbins) + 0.5) * bin_s
    return t, frac


bin_s = 20.0

pr1, _ = presence_ratio_per_unit(sorting1, bin_s=bin_s)
pr2, _ = presence_ratio_per_unit(sorting2, bin_s=bin_s)

fig = plt.figure(figsize=(7,4))
plt.hist(pr1, bins=40, alpha=0.5, label="sorting1")
plt.hist(pr2, bins=40, alpha=0.5, label="sorting2")
plt.xlabel("Presence ratio")
plt.ylabel("Units")
plt.title(
    f"Presence ratio per unit (bin={bin_s:.0f}s) | mean: s1={pr1.mean():.3f}, s2={pr2.mean():.3f}"
)
plt.legend()
plt.tight_layout()
fig.savefig(outdir / "presence_ratio_hist.png", dpi=200)
if SHOW_PLOTS and IN_IPYTHON:
    try:
        from IPython.display import display  # type: ignore

        display(fig)
    except Exception:
        pass
plt.close(fig)

t1, frac1 = fraction_units_active_over_time(sorting1, bin_s=bin_s)
t2, frac2 = fraction_units_active_over_time(sorting2, bin_s=bin_s)
T = min(len(frac1), len(frac2))

fig = plt.figure(figsize=(11,4))
plt.plot(t1[:T], frac1[:T], label="sorting1")
plt.plot(t2[:T], frac2[:T], label="sorting2")
plt.xlabel("Time (s)")
plt.ylabel("Fraction units active")
plt.title(f"Population stability over time (>=1 spike per unit per {bin_s:.0f}s bin)")
plt.legend()
plt.tight_layout()
fig.savefig(outdir / "fraction_units_active_over_time.png", dpi=200)
if SHOW_PLOTS and IN_IPYTHON:
    try:
        from IPython.display import display  # type: ignore

        display(fig)
    except Exception:
        pass
plt.close(fig)

print("Mean presence ratio:", float(pr1.mean()), float(pr2.mean()))


#%%
def pooled_isi_ms(sorting, max_units=200, max_spikes_per_unit=20000, seed=0):
    fs = float(sorting.get_sampling_frequency())
    rng = np.random.default_rng(seed)
    unit_ids = np.array(sorting.unit_ids)
    if len(unit_ids) > max_units:
        unit_ids = rng.choice(unit_ids, size=max_units, replace=False)

    isis = []
    for u in unit_ids:
        st = sorting.get_unit_spike_train(u).astype(np.int64)
        if len(st) < 2:
            continue
        if len(st) > max_spikes_per_unit:
            idx = rng.choice(len(st), size=max_spikes_per_unit, replace=False)
            st = np.sort(st[idx])
        isi = np.diff(st) / fs * 1000.0
        isis.append(isi)
    if len(isis) == 0:
        return np.array([], dtype=float)
    return np.concatenate(isis)


isi1 = pooled_isi_ms(sorting1, seed=1)
isi2 = pooled_isi_ms(sorting2, seed=2)

fig = plt.figure(figsize=(7,4))
plt.hist(isi1, bins=200, range=(0, 20), alpha=0.5, label="sorting1")
plt.hist(isi2, bins=200, range=(0, 20), alpha=0.5, label="sorting2")
plt.axvline(1.0, linewidth=1)
plt.xlabel("ISI (ms)")
plt.ylabel("Counts")
plt.title("Pooled ISI (subset of units/spikes), 0–20 ms")
plt.legend()
plt.tight_layout()
fig.savefig(outdir / "pooled_isi_0_20ms.png", dpi=200)
if SHOW_PLOTS and IN_IPYTHON:
    try:
        from IPython.display import display  # type: ignore

        display(fig)
    except Exception:
        pass
plt.close(fig)


#%%
def template_vectors(we, max_units=250):
    templates, _unit_ids = _get_templates_and_unit_ids(we)
    templates = np.asarray(templates)
    n = min(max_units, templates.shape[0])

    vecs = []
    for i in range(n):
        temp = templates[i]  # (samples, channels)
        ch = int(np.argmax(np.ptp(temp, axis=0)))
        w = temp[:, ch].astype(np.float64)
        w = w - w.mean()
        w /= (np.linalg.norm(w) + 1e-12)
        vecs.append(w)
    return np.vstack(vecs) if len(vecs) else np.zeros((0, 0), dtype=np.float64)


A = template_vectors(we1, max_units=250)
B = template_vectors(we2, max_units=250)

if A.size and B.size:
    S = A @ B.T  # cosine similarity

    fig = plt.figure(figsize=(7,6))
    plt.imshow(S, aspect="auto", vmin=-1, vmax=1, origin="lower")
    plt.xlabel("sorting2 units (subset)")
    plt.ylabel("sorting1 units (subset)")
    plt.title("Template similarity (peak-channel waveform cosine sim)")
    plt.colorbar(label="cosine sim")
    plt.tight_layout()
    fig.savefig(outdir / "template_similarity_heatmap.png", dpi=200)
    if SHOW_PLOTS and IN_IPYTHON:
        try:
            from IPython.display import display  # type: ignore

            display(fig)
        except Exception:
            pass
    plt.close(fig)

    print("Similarity summary: max per row median =", float(np.median(np.max(S, axis=1))))
else:
    print("Template similarity skipped: could not build template vectors")

