"""Plan or evaluate a longitudinal sorter-development comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testing.development_ladder import build_plan, evaluate_results, load_contract, pin_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare-strip", "run-arms", "compare-arms", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunk-duration", default="10s")
    args = parser.parse_args()
    contract = load_contract(args.config)
    if args.command == "plan":
        report = build_plan(contract)
    elif args.command == "prepare-strip":
        from testing.development_strip import materialize_development_strip

        if args.output_root is None:
            parser.error("prepare-strip requires --output-root")
        pin_plan(contract, args.output_root.parent / "comparison_plan.json")
        _, manifest = materialize_development_strip(
            contract.raw["recording"]["accepted_recording_path"],
            args.output_root,
            recording_spec=contract.raw["recording"],
            spatial_spec=contract.raw["spatial_contract"],
            n_jobs=args.n_jobs,
            chunk_duration=args.chunk_duration,
        )
        report = manifest
    elif args.command == "run-arms":
        from testing.development_runner import run_development_arms

        if args.recording_dir is None or args.output_root is None:
            parser.error("run-arms requires --recording-dir and --output-root")
        pin_plan(contract, args.output_root / "comparison_plan.json")
        report = run_development_arms(
            contract,
            recording_dir=args.recording_dir,
            output_root=args.output_root,
            require_cuda=not args.no_require_cuda,
        )
    elif args.command == "compare-arms":
        from testing.sort_comparison import compare_sorts, load_comparison_inputs

        if args.output_root is None or args.baseline is None or args.candidate is None:
            parser.error("compare-arms requires --output-root, --baseline, and --candidate")
        def arm(name):
            path = args.output_root / name / "candidate_manifest.json"
            if not path.is_file():
                parser.error(f"arm manifest not found: {path}")
            return json.loads(path.read_text())
        baseline_manifest = arm(args.baseline)
        candidate_manifest = arm(args.candidate)
        for manifest in (baseline_manifest, candidate_manifest):
            if manifest.get("contract_digest") != contract.digest:
                raise RuntimeError("arm output belongs to another development contract")
        fs = float(json.loads((Path(args.recording_dir) / "rescue_recording_manifest.json").read_text())["sampling_frequency_hz"]) if args.recording_dir else None
        if fs is None:
            parser.error("compare-arms requires --recording-dir to establish the clock")
        baseline_sort, baseline_qc = load_comparison_inputs(
            args.baseline, baseline_manifest["curated_output"], baseline_manifest["qc_directory"],
            sampling_frequency_hz=fs,
        )
        candidate_sort, candidate_qc = load_comparison_inputs(
            args.candidate, candidate_manifest["curated_output"], candidate_manifest["qc_directory"],
            sampling_frequency_hz=fs,
        )
        evaluation = {
            **contract.raw["evaluation"],
            "sampling_frequency_hz": fs,
            "duration_s": contract.raw["recording"]["duration_s"],
            "minimum_common_time_fraction": contract.raw["metrics"]["minimum_common_time_fraction"],
            "minimum_measurable_unit_fraction": contract.raw["metrics"]["minimum_measurable_unit_fraction"],
        }
        comparison_output = args.output_root / "comparisons" / f"{args.candidate}_vs_{args.baseline}"
        report = compare_sorts(
            baseline_sort, candidate_sort, baseline_qc, candidate_qc, evaluation,
            spatial_region=contract.raw["spatial_contract"], output_dir=comparison_output,
        )
        report = {
            "summary": report["summary"],
            "coverage_summary": report["coverage_summary"],
            "decision": report["decision"],
            "output_directory": str(comparison_output),
        }
    else:
        if args.results is None:
            parser.error("evaluate requires --results")
        report = evaluate_results(contract, json.loads(args.results.read_text()))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
