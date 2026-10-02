"""Process the inspected catalog and compare trackers on held out shots."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftlens.pipeline import analyze_video

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--comparisons", action="store_true")
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--force", action="store_true", help="Regenerate existing results after a processing revision")
    args=parser.parse_args()
    catalog=json.loads((ROOT/"data/clip_catalog.json").read_text(encoding="utf-8"))
    source=ROOT/catalog["source"]["path"]
    for clip in catalog["clips"]:
        if args.only and clip["id"] not in args.only:
            continue
        out=ROOT/"outputs/runs"/clip["id"]
        jobs=[("bytetrack",out)]
        if args.comparisons and clip["split"]=="test":
            jobs.append(("botsort",out/"comparisons/botsort"))
        for tracker,destination in jobs:
            saved=destination/"summary.json"
            if not args.force and saved.exists() and json.loads(saved.read_text()).get("status")=="complete" and json.loads(saved.read_text()).get("pipeline_revision")==2:
                print("Already processed:",clip["id"],tracker,flush=True)
                continue
            print("PROCESS",clip["id"],tracker,clip["start_seconds"],clip["end_seconds"],flush=True)
            result=analyze_video(source,destination,clip["start_seconds"],clip["end_seconds"],tracker,args.size,10,lambda fraction,message:print(f"{fraction:.0%} {message}",flush=True))
            print(json.dumps({"clip_id":clip["id"],"tracker":tracker,"fps":result["processing_fps"],"ids":result["track_ids"],"shots":result["shot_count"]}),flush=True)
if __name__=="__main__":
    main()
