"""Extract only tuning battles for a small detector fine-tuning experiment."""
from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    catalog = json.loads((ROOT / 'data/clip_catalog.json').read_text(encoding='utf-8-sig'))
    destination = ROOT / 'outputs/finetuning/dataset'
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(ROOT / catalog['source']['path']))
    records = []
    for clip in catalog['clips']:
        if clip['split'] != 'tune':
            continue
        split = 'train' if clip['id'] in ['clip01', 'clip02', 'clip03', 'clip04'] else 'val'
        count = 6 if split == 'train' else 4
        thumbs = []
        for index in range(count):
            seconds = clip['start_seconds'] + .3 + index * (clip['end_seconds'] - clip['start_seconds'] - .6) / (count - 1)
            frame_number = round(seconds * 30)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                raise RuntimeError('Source frame decode failed')
            name = f"{clip['id']}_{index:02d}"
            image_path = destination / 'images' / split / f'{name}.jpg'
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)
            thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 360))
            canvas = Image.new('RGB', (640, 392), '#161d23')
            canvas.paste(thumb, (0, 32))
            ImageDraw.Draw(canvas).text((8, 8), f'{name} source {frame_number / 30:.3f}s | bbox coordinates source 1280x720', fill='white')
            thumbs.append(canvas)
            records.append({'id': name, 'clip_id': clip['id'], 'run_group': clip['run_group'], 'split': split, 'source_frame': frame_number, 'source_seconds': frame_number / 30, 'image': str(image_path.relative_to(ROOT))})
        sheet = Image.new('RGB', (1280, 392 * ((count + 1) // 2)), '#161d23')
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((i % 2) * 640, (i // 2) * 392))
        sheet.save(destination / f"{clip['id']}_source_sheet.jpg", quality=95)
    capture.release()
    (destination / 'frames.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'train_frames': 24, 'validation_frames': 8, 'test_frames_used_for_training': 0}))


if __name__ == '__main__':
    main()
