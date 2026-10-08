#!/usr/bin/env python3
"""Source-only fixture for DARTsort gather/resume write ordering."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


SOURCE = Path("/home/huklab/Documents/DARTsort/src/dartsort/peel/peel_base.py")
EXPECTED_SHA256 = "ac357ade5b6f23b690a403e297a2260ea93c2a1b35df13ee7798d8fc8591a2d4"


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED_SHA256
    tree = ast.parse(source_bytes)
    gather = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "gather_chunk_result"
    )
    text = source_bytes.decode()
    marker_lines = []
    data_lines = []
    flush_lines = []
    for node in ast.walk(gather):
        segment = ast.get_source_segment(text, node) or ""
        if isinstance(node, ast.Assign) and (
            'output_h5["last_chunk_index"]' in segment
            or 'output_h5["last_chunk_start"]' in segment
        ):
            marker_lines.append(node.lineno)
        if isinstance(node, (ast.Assign, ast.Expr)) and (
            "residual_file" in segment and ".tofile(" in segment
            or "h5ds[" in segment
            or 'output_h5["residual"][' in segment
        ):
            data_lines.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "flush":
            flush_lines.append(node.lineno)
    assert marker_lines == [635, 638], marker_lines
    assert data_lines and min(data_lines) > max(marker_lines), data_lines
    assert not flush_lines, flush_lines
    print(json.dumps({
        "source_sha256": EXPECTED_SHA256,
        "marker_lines": marker_lines,
        "first_data_write_line": min(data_lines),
        "flush_lines": flush_lines,
        "establishes": "marker assignments precede data writes and no explicit flush occurs inside gather_chunk_result",
        "does_not_establish": "which bytes survive abrupt process or host termination",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
