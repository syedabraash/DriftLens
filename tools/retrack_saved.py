"""Re-track original cached detector candidates into a new result directory."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftlens.retrack import retrack_cached_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = retrack_cached_run(args.run, args.output)
    print(json.dumps({key: result[key] for key in ("clip_id", "processing_mode", "frame_count", "shot_count", "detected_camera_cuts_seconds", "processing_seconds")}, indent=2))


if __name__ == "__main__":
    main()
