from __future__ import annotations
import argparse
import json
from pathlib import Path
from .pipeline import analyze_video

def main():
    parser = argparse.ArgumentParser(description="Track a short Formula Drift video interval.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    result = analyze_video(args.source, args.output, args.start, args.end, args.tracker, args.size, args.fps, lambda progress, message: print(f"{progress:.0%}: {message}", flush=True))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
