"""Build review sheets from actual exported full run replays and aligned source.

This helper does not assign car identities or roles. Its purpose is visual
inspection of local track IDs, omissions, and fragmentation after processing.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    for path in [Path(r"C:\Windows\Fonts\arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read exported frame {frame_index}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def selections(frames, interval):
    chosen = []
    target = 0.0
    last_time = float(frames[-1]["clip_time"])
    while target <= last_time + 1e-8:
        record = min(frames, key=lambda row: abs(float(row["clip_time"]) - target))
        if not chosen or record["frame_index"] != chosen[-1]["frame_index"]:
            chosen.append(record)
        target += interval
    if chosen[-1]["frame_index"] != frames[-1]["frame_index"]:
        chosen.append(frames[-1])
    return chosen


def add_caption(sheet, xy, width, height, title, detail):
    draw = ImageDraw.Draw(sheet)
    x, y = xy
    draw.text((x + 8, y + height + 5), title, fill="white", font=font(16))
    draw.text((x + 8, y + height + 26), detail, fill="#bfcad4", font=font(13))


def inspect_shot(project, shot_id, args):
    directory = project / "outputs" / "full_runs" / args.run_id / "shots" / shot_id
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    frames = csv_rows(directory / "frames.csv")
    observations = defaultdict(list)
    for row in csv_rows(directory / "observations.csv"):
        if row.get("observed", "True").lower() == "true":
            observations[int(row["frame_index"])].append(row)
    chosen = selections(frames, args.step)
    replay = cv2.VideoCapture(str(directory / "annotated.mp4"))
    source = cv2.VideoCapture(summary["source_path"]) if args.source_pairs else None
    if not replay.isOpened() or (source is not None and not source.isOpened()):
        raise RuntimeError(f"Cannot open source or replay for {shot_id}")
    width = args.width
    height = round(width * summary["height"] / summary["width"])
    caption_height = 54
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    images, records = [], []
    per_page = 4 if args.source_pairs else 8
    for page_start in range(0, len(chosen), per_page):
        page_records = chosen[page_start:page_start + per_page]
        rows = len(page_records) if args.source_pairs else (len(page_records) + 1) // 2
        sheet = Image.new("RGB", (width * 2, (height + caption_height) * rows), "#111822")
        for index, row in enumerate(page_records):
            frame_index = int(row["frame_index"])
            replay_image = read_frame(replay, frame_index).resize((width, height))
            ids = sorted({int(v["track_id"]) for v in observations[frame_index] if v.get("track_id")})
            source_time = float(row["source_time"])
            clip_time = float(row["clip_time"])
            title = f"{shot_id}  clip {clip_time:.2f}s  source {source_time:.3f}s"
            detail = f"Exported frame {frame_index}; observed track IDs: {','.join(map(str, ids)) or 'none'}"
            if args.source_pairs:
                y = index * (height + caption_height)
                source_index = round(source_time * summary["source_fps"])
                source_image = read_frame(source, source_index).resize((width, height))
                sheet.paste(source_image, (0, y))
                add_caption(sheet, (0, y), width, height, title, "Source image at exported observation timestamp")
                sheet.paste(replay_image, (width, y))
                add_caption(sheet, (width, y), width, height, title, detail)
            else:
                x, y = index % 2 * width, index // 2 * (height + caption_height)
                sheet.paste(replay_image, (x, y))
                add_caption(sheet, (x, y), width, height, title, detail)
            records.append(dict(frame_index=frame_index,clip_seconds=clip_time,source_seconds=source_time,
                                observed_track_ids=ids))
        suffix = "paired" if args.source_pairs else "replay"
        path = output / f"{args.prefix}{shot_id}_{suffix}_{page_start // per_page + 1:02d}.jpg"
        sheet.save(path, quality=95)
        images.append(str(path.resolve()))
    replay.release()
    if source is not None:
        source.release()
    tracks = defaultdict(list)
    for frame_rows in observations.values():
        for row in frame_rows:
            if row.get("track_id"):
                tracks[int(row["track_id"])].append(float(row["clip_time"]))
    track_ranges = {str(k): dict(first_clip_seconds=min(v),last_clip_seconds=max(v),observations=len(v))
                    for k, v in sorted(tracks.items())}
    return dict(shot_id=shot_id,pipeline_revision=summary.get("pipeline_revision"),
                source_start_seconds=summary["start_seconds"],source_end_seconds=summary["end_seconds"],
                exported_frame_count=len(frames),track_ranges=track_ranges,
                sampled_review_frames=records,contact_sheets=images,
                warning="Track ranges do not establish car identities or continuous visibility. Visual livery review is required before assigning lead and chase.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--run-id", default="full_run01")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "full_run_inspection" / "tracks")
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--source-pairs", action="store_true")
    parser.add_argument("--shots", default="shot01,shot02,shot03,shot04")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    if args.step <= 0 or args.width < 160:
        parser.error("Step must be positive and width must be at least160")
    results = [inspect_shot(args.project_dir, shot, args) for shot in args.shots.split(",")]
    path = args.output_dir / f"{args.prefix}inspection_index.json"
    path.write_text(json.dumps(dict(run_id=args.run_id,step_seconds=args.step,source_pairs=args.source_pairs,
                                   role_assignments="None; this helper only prepares visual inspection",shots=results),indent=2)+"\n",encoding="utf-8")
    print(path)
    for result in results:
        print(result["shot_id"], result["pipeline_revision"], result["track_ranges"])


if __name__ == "__main__":
    main()
