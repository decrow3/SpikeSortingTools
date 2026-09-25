"""Command-line entry points for motionqc."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .field import load_field
from .report import build_report, load_peaks


def main(argv=None):
    parser=argparse.ArgumentParser(prog="motionqc")
    sub=parser.add_subparsers(dest="command",required=True)
    report=sub.add_parser("report",help="build a matched-sample per-session field-QC report")
    report.add_argument("--field",action="append",required=True,help="saved field; repeat to compare")
    report.add_argument("--field-format",action="append",help="optional matching loader format")
    report.add_argument("--origin-offset-s",action="append",type=float,help="optional matching clock offset")
    report.add_argument("--peaks",required=True);report.add_argument("--windows",required=True)
    report.add_argument("--mask",required=True);report.add_argument("--units");report.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    formats=args.field_format or ["auto"]*len(args.field);offsets=args.origin_offset_s or [None]*len(args.field)
    if len(formats)!=len(args.field) or len(offsets)!=len(args.field):parser.error("field formats/offsets must match --field count")
    fields=[load_field(path,format=fmt,origin_offset_s=offset,source=Path(path).stem) for path,fmt,offset in zip(args.field,formats,offsets)]
    units=load_peaks(args.units) if args.units else None
    build_report(fields,load_peaks(args.peaks),pd.read_csv(args.windows),pd.read_csv(args.mask),args.out,units)


if __name__ == "__main__":main()
