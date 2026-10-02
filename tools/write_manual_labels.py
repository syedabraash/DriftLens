"""Save sparse visual reference boxes drafted by the AI assistant.

These labels are approximate visible extents, independent of detector output.
They have no human expert review and do not establish positions in smoke.
"""

import json
from PIL import Image, ImageDraw
from prepare_footage import ROOT, INSPECTION

# Each pair is car_a and car_b in source pixels. v=visible, p=partial.
BOXES = {
    "clip01": [
        [([760,320,940,443],"v"),([278,294,451,423],"v")],
        [([654,274,837,406],"v"),([334,243,511,379],"v")],
        [([690,265,900,412],"v"),([414,224,628,366],"v")],
        [([591,251,946,422],"v"),([508,193,804,339],"p")],
    ],
    "clip07": [
        [([924,384,1076,484],"v"),([382,380,530,477],"v")],
        [([786,334,914,432],"v"),([406,330,532,417],"v")],
        [([594,259,725,356],"v"),([412,255,540,337],"v")],
        [([636,246,792,348],"v"),([514,234,650,325],"p")],
    ],
    "clip08": [
        [([594,196,890,320],"v"),([430,190,664,300],"p")],
        [([518,244,998,430],"v"),([388,220,724,380],"p")],
        [([320,204,676,467],"v"),([484,164,782,348],"p")],
        [([140,153,760,423],"v"),([288,90,794,322],"p")],
    ],
    "clip09": [
        [([306,172,505,258],"v"),([58,172,170,246],"p")],
        [([516,210,863,328],"v"),([334,208,578,310],"p")],
        [([414,216,952,402],"v"),([412,210,650,358],"p")],
        [([414,196,802,451],"v"),([645,169,904,320],"p")],
    ],
    "clip10": [
        [([494,264,617,315],"v"),([615,254,704,302],"v")],
        [([414,296,642,389],"v"),([595,290,739,355],"p")],
        [([217,373,613,526],"v"),([374,327,613,435],"p")],
        [([486,510,887,720],"p"),([261,478,599,622],"p")],
    ],
    "clip11": [
        [([916,320,1072,426],"v"),([416,310,560,414],"v")],
        [([790,298,939,391],"v"),([496,286,624,384],"v")],
        [([670,286,814,389],"v"),([514,272,648,366],"v")],
        [([736,236,901,351],"v"),([632,204,770,307],"p")],
    ],
    "clip12": [
        [([566,222,940,366],"v"),([332,198,610,334],"p")],
        [([376,230,810,444],"v"),([366,184,706,354],"p")],
        [([224,204,764,443],"v"),([550,121,882,326],"p")],
        [([402,257,998,484],"v"),([486,162,893,332],"p")],
    ],
}


def main():
    samples = []
    for clip_id, frames in BOXES.items():
        sheet = Image.new("RGB", (1280,796), "#161616")
        draw = ImageDraw.Draw(sheet)
        for i, cars in enumerate(frames):
            car_labels = []
            frame = Image.open(INSPECTION / f"{clip_id}_sample{i}.jpg")
            frame_draw = ImageDraw.Draw(frame)
            for k, (box, visibility) in enumerate(cars):
                identity = "car_a" if k == 0 else "car_b"
                car_labels.append(dict(identity=identity,bbox=box,visibility="visible" if visibility == "v" else "partial"))
                frame_draw.rectangle(box, outline="#33eab8" if k == 0 else "#ffc766", width=3)
                frame_draw.text((box[0]+4,max(0,box[1]-16)),identity+" "+visibility,fill="white")
            samples.append(dict(clip_id=clip_id,time_seconds=round(0.3 + i,1),shot_index=0,cars=car_labels,
                                split="tune" if clip_id == "clip01" else "test",
                                lead_identity="car_a",chase_identity="car_b",
                                frame=f"outputs/inspection/{clip_id}_sample{i}.jpg"))
            x,y = i % 2 * 640, i // 2 * 398
            sheet.paste(frame.resize((640,360)),(x,y))
            draw.text((x+8,y+365),f"Visual reference labels {clip_id} frame{i}, relative={0.3+i}s",fill="white")
        sheet.save(INSPECTION / f"{clip_id}_labels_qa.jpg")
    payload = dict(coordinate_space="source_pixels",width=1280,height=720,
                   annotation_method="Visual reference boxes drafted by AI assistant, independently of detector predictions; approximate visible extents; no human expert review",
                   identity_notes={"clip01":"car_a white red roof black green side; car_b white blue","clip07":"car_a green; car_b orange blue","clip08":"car_a green; car_b orange blue","clip09":"car_a black; car_b blue white","clip10":"car_a black; car_b blue white","clip11":"car_a yellow; car_b red","clip12":"car_a yellow; car_b red"},
                   limitations="Sparse frames from one event and one broadcast. Partial boxes use visible extents and are subjective. No fully hidden car positions are labeled. No human expert has validated the visual reference boxes or identity assignments. This is a small diagnostic evaluation, not a certified ground truth benchmark or dense identity audit.",samples=samples)
    (ROOT/"data"/"annotations"/"labels.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(f"Saved {len(samples)} samples and {sum(len(s['cars']) for s in samples)} car boxes")


if __name__ == "__main__":
    main()
