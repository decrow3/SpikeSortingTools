"""Private worker for the exact packet-owned saved-format fixture."""

from pathlib import Path
import argparse

from testing.first_medium_saved_output_qualification import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = Path(__file__).resolve().parents[2]
    run(args.contract, args.request, args.output, fixture_packet_root=packet)


if __name__ == "__main__":
    main()

