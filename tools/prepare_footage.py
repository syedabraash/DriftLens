"""Read the supplied footage to create inspection sheets and sparse label assets."""

from pathlib import Path
import argparse
import json
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"G:\DriftLens\longbeach2024_action.mp4")
INSPECTION = ROOT / "outputs" / "inspection"


def read_frame(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read {seconds:.3f}s")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def sheet(times, destination, columns=4, width=320):
    cap = cv2.VideoCapture(str(SOURCE))
    height = int(width * 720 / 1280)
    rows = (len(times) + columns - 1) // columns
    image = Image.new("RGB", (width * columns, (height + 28) * rows), "#141414")
    draw = ImageDraw.Draw(image)
    for index, seconds in enumerate(times):
        x = index % columns * width
        y = index // columns * (height + 28)
        image.paste(read_frame(cap, seconds).resize((width, height)), (x, y))
        draw.text((x + 8, y + height + 5), f"{seconds:.2f}s  {int(seconds // 60):02d}:{seconds % 60:05.2f}", fill="white")
    cap.release()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=600)
    parser.add_argument("--step", type=float, default=30)
    parser.add_argument("--name", default="overview.jpg")
    parser.add_argument("--times", default="", help="Comma separated absolute source times")
    args = parser.parse_args()
    times = [float(item) for item in args.times.split(",")] if args.times else []
    time = args.start
    while not args.times and time <= args.end:
        times.append(time)
        time += args.step
    sheet(times, INSPECTION / args.name)


if __name__ == "__main__":
    main()
