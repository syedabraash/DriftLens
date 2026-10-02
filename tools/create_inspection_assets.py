"""Export catalog thumbnails and visual annotation reference sheets."""

from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw
from prepare_footage import ROOT, SOURCE, INSPECTION, read_frame

CLIPS = [
    ("clip01", "White tandem launch", 408.5, 412.5, "clear then overlap", "tune", "battle02_white_red_blue"),
    ("clip02", "White tandem bridge", 414.2, 418.6, "smoke and overlap", "tune", "battle02_white_red_blue"),
    ("clip03", "Red yellow launch", 832.3, 837.5, "clear then overlap", "tune", "battle03_taguchi_minowa"),
    ("clip04", "Red yellow bridge", 840.0, 845.6, "smoke and overlap", "tune", "battle03_taguchi_minowa"),
    ("clip05", "White gold launch", 1142.4, 1146.8, "clear then overlap", "tune", "battle04_white_gold_black"),
    ("clip06", "White gold bridge", 1149.4, 1154.0, "smoke and partial occlusion", "tune", "battle04_white_gold_black"),
    ("clip07", "Green orange launch", 1906.0, 1910.3, "clear then overlap", "test", "battle05_green_orange"),
    ("clip08", "Green orange bridge", 1913.4, 1917.5, "smoke and overlap", "test", "battle05_green_orange"),
    ("clip09", "Black blue bridge", 2106.4, 2111.3, "smoke and overlap", "test", "battle06_black_blue"),
    ("clip10", "Black blue outer zone", 2111.6, 2115.0, "smoke and overlap", "test", "battle06_black_blue"),
    ("clip11", "Yellow red launch", 2692.0, 2697.0, "clear then overlap", "test", "battle07_yellow_red"),
    ("clip12", "Yellow red bridge", 2699.8, 2703.3, "smoke and overlap", "test", "battle07_yellow_red"),
]


def main():
    INSPECTION.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "annotations").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE))
    clips = []
    for clip_id, label, start, end, condition, split, group in CLIPS:
        thumb = f"outputs/inspection/{clip_id}.jpg"
        read_frame(cap, (start + end) / 2).resize((640, 360)).save(ROOT / thumb)
        clips.append(dict(id=clip_id, label=label, start_seconds=start, end_seconds=end,
                          condition=condition, split=split, run_group=group, thumbnail=thumb))
        if split != "test" and clip_id != "clip01":
            continue
        sheet = Image.new("RGB", (1280, 796), "#151515")
        draw = ImageDraw.Draw(sheet)
        for i, relative in enumerate([0.3, 1.3, 2.3, 3.3]):
            x, y = i % 2 * 640, i // 2 * 398
            frame = read_frame(cap, start + relative)
            frame.save(INSPECTION / f"{clip_id}_sample{i}.jpg")
            frame = frame.resize((640, 360))
            grid = ImageDraw.Draw(frame)
            for gx in range(0, 640, 80):
                grid.line([(gx, 0), (gx, 359)], fill=(170, 170, 170), width=1)
                grid.text((gx + 2, 0), str(gx * 2), fill=(255, 255, 255))
            for gy in range(0, 360, 60):
                grid.line([(0, gy), (639, gy)], fill=(170, 170, 170), width=1)
                grid.text((0, gy + 2), str(gy * 2), fill=(255, 255, 255))
            sheet.paste(frame, (x, y))
            draw.text((x + 8, y + 365), f"{clip_id} sample{i} relative={relative}s absolute={start + relative}s", fill="white")
        sheet.save(INSPECTION / f"{clip_id}_label_sheet.jpg")
    cap.release()
    descriptions = {
        "clip01": ("White body with red roof and black green side", "White body with blue accents"),
        "clip07": ("Green pink", "Orange blue"), "clip08": ("Green pink", "Orange blue"),
        "clip09": ("Black with multicolor livery", "Blue white"), "clip10": ("Black with multicolor livery", "Blue white"),
        "clip11": ("Yellow", "Red"), "clip12": ("Yellow", "Red"),
    }
    for clip in clips:
        if clip["id"] in descriptions:
            clip["lead_identity"] = "car_a"
            clip["chase_identity"] = "car_b"
            clip["lead_description"], clip["chase_description"] = descriptions[clip["id"]]
            clip["role_basis"] = "AI assistant visual inspection of travel order through initiation and continuous bridge shots; not inferred from starting lane alone; no human expert review"
    catalog = dict(source=dict(path="longbeach2024_action.mp4",url="https://www.youtube.com/watch?v=nobounLesY4",event="Long Beach 2024",title="Formula DRIFT Long Beach Top 16 ALL ACTION",reuse_status="not verified; local file supplied by user",width=1280,height=720,fps=30,duration_seconds=6682.4667),clips=clips,
                   selection_notes="Start, middle and end frames visually checked. Broadcast cut boundary trims checked at tenth-second resolution where necessary. Clips are grouped by whole battle so alternate runs and replays cannot cross tune and test. No replay footage is selected.")
    (ROOT / "data" / "clip_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(clips)} clips and 28 visual annotation reference frames")


if __name__ == "__main__":
    main()
