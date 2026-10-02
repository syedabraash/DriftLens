"""One bounded CPU fine-tune; keep original weights and frozen test references."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.settings'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib_cache'))
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('POLARS_FORCE_PKG', 'compat')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def score(model, samples, images, size=640, agnostic=True):
    from driftlens.evaluation import match_boxes
    tp = fp = fn = 0
    rows = []
    for sample, image in zip(samples, images):
        boxes = model.predict(image, imgsz=size, conf=.15, iou=.5, classes=[2, 7], agnostic_nms=agnostic, device='cpu', verbose=False)[0].boxes.xyxy.cpu().tolist()
        truth = [box['bbox'] for box in sample['boxes']]
        matched = len(match_boxes(truth, boxes, .5))
        tp += matched
        fp += len(boxes) - matched
        fn += len(truth) - matched
        rows.append({'id': sample['id'], 'tp': matched, 'fp': len(boxes)-matched, 'fn': len(truth)-matched})
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision+recall else 0, 'frames': len(samples), 'reference_boxes': tp+fn, 'per_frame': rows}


def main(evaluate_only=False):
    import cv2
    import torch
    import yaml
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    out = ROOT / 'outputs/finetuning'
    dataset = out / 'dataset'
    weights = ROOT / 'models/yolov8n.pt'
    backup = ROOT / 'models/yolov8n_baseline.pt'
    if not backup.exists():
        shutil.copyfile(weights, backup)
    baseline_hash = digest(backup)
    original_labels_hash = digest(ROOT / 'data/annotations/labels.json')
    status = out / 'status.json'
    save(status, {'status': 'preparing', 'baseline_sha256': baseline_hash})
    try:
        annotation = json.loads((ROOT / 'data/finetuning_labels.json').read_text(encoding='utf-8'))
        frames = json.loads((dataset / 'frames.json').read_text(encoding='utf-8'))
        samples = {sample['id']: sample for sample in annotation['samples']}
        assert set(samples) == {frame['id'] for frame in frames}
        assert not ({f['run_group'] for f in frames if f['split']=='train'} & {f['run_group'] for f in frames if f['split']=='val'})
        val_samples, val_images = [], []
        for frame in frames:
            lines = []
            for box in samples[frame['id']]['boxes']:
                x1,y1,x2,y2 = box['bbox']
                assert box['class_id'] in [2,7] and 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
                lines.append(f"{box['class_id']} {(x1+x2)/2560:.8f} {(y1+y2)/1440:.8f} {(x2-x1)/1280:.8f} {(y2-y1)/720:.8f}")
            label = dataset / 'labels' / frame['split'] / (frame['id']+'.txt')
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            if frame['split']=='val':
                val_samples.append(samples[frame['id']])
                val_images.append(str(ROOT / frame['image']))
        base = YOLO(str(backup))
        data_file = dataset / 'dataset.yaml'
        data_file.write_text(yaml.safe_dump({'path': dataset.as_posix(), 'train': 'images/train', 'val': 'images/val', 'names': base.names}), encoding='utf-8')
        baseline_val = score(base, val_samples, val_images)
        settings = dict(data=str(data_file), epochs=12, imgsz=416, batch=2, device='cpu', workers=0, freeze=10, optimizer='AdamW', lr0=.0001, lrf=.1, warmup_epochs=0, warmup_bias_lr=.0001, patience=4, seed=42, deterministic=True, pretrained=True, single_cls=False, amp=False, cache=False, mosaic=0, mixup=0, copy_paste=0, degrees=3, translate=.05, scale=.15, fliplr=.5, hsv_h=.015, hsv_s=.25, hsv_v=.2, close_mosaic=0, plots=False, save=True, project=str(out/'training'), name='cpu_adaptation', exist_ok=False, verbose=False)
        save(out / 'training_settings.json', settings)
        save(status, {'status': 'training', 'baseline_validation': baseline_val, 'epochs_requested': 12})
        candidate = ROOT / 'models/driftlens_finetuned.pt'
        if evaluate_only:
            elapsed = json.loads((out / 'training_outcome.json').read_text(encoding='utf-8'))['training_seconds']
        else:
            started = time.perf_counter()
            # Prediction fuses Conv/BN layers in-place. Train an unfused fresh
            # checkpoint so transfer learning loads every pretrained tensor.
            trainer_model = YOLO(str(backup))
            trainer_model.train(**settings)
            elapsed = time.perf_counter()-started
            trained = Path(trainer_model.trainer.save_dir) / 'weights/best.pt'
            shutil.copyfile(trained, candidate)
            save(out / 'training_outcome.json', {'training_seconds':elapsed, 'training_directory':str(trainer_model.trainer.save_dir), 'candidate_sha256':digest(candidate)})
        model = YOLO(str(candidate))
        candidate_val = score(model, val_samples, val_images)
        save(status, {'status': 'evaluating', 'training_seconds': elapsed, 'candidate_validation': candidate_val})
        # Test references are opened only after validation selects best.pt.
        refs = json.loads((ROOT / 'data/annotations/labels.json').read_text(encoding='utf-8-sig'))
        catalog = json.loads((ROOT / 'data/clip_catalog.json').read_text(encoding='utf-8-sig'))
        clips = {c['id']: c for c in catalog['clips']}
        capture = cv2.VideoCapture(str(ROOT / catalog['source']['path']))
        test_samples, test_images = [], []
        for index, sample in enumerate(s for s in refs['samples'] if s['split']=='test'):
            clip = clips[sample['clip_id']]
            assert clip['run_group'] not in {f['run_group'] for f in frames}
            with (ROOT / 'outputs/runs' / clip['id'] / 'frames.csv').open(encoding='utf-8-sig') as handle:
                recorded = list(csv.DictReader(handle))
            closest = min(recorded, key=lambda r: abs(float(r['clip_time'])-sample['time_seconds']))
            assert abs(float(closest['clip_time'])-sample['time_seconds']) <= .1
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(float(closest['source_time']) * capture.get(cv2.CAP_PROP_FPS)))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError('Test frame decode failed')
            test_samples.append({'id': f"{clip['id']}_{index}", 'boxes': [{'bbox': c['bbox']} for c in sample['cars'] if c.get('visibility') != 'hidden']})
            test_images.append(image)
        capture.release()
        original = YOLO(str(backup))
        baseline_test = score(original, test_samples, test_images)
        candidate_test = score(model, test_samples, test_images)
        improved = candidate_val['f1'] >= baseline_val['f1'] and candidate_test['recall'] > baseline_test['recall'] and candidate_test['precision'] >= baseline_test['precision']
        assert digest(ROOT/'data/annotations/labels.json') == original_labels_hash
        report = {'status':'complete', 'training_seconds':round(elapsed,3), 'training_frames':24, 'validation_frames':8, 'test_frames':24, 'train_groups': sorted({f['run_group'] for f in frames if f['split']=='train'}), 'validation_groups':sorted({f['run_group'] for f in frames if f['split']=='val'}), 'baseline_sha256': baseline_hash, 'candidate_sha256':digest(candidate), 'frozen_reference_sha256':original_labels_hash, 'annotation_method':annotation['annotation_method'], 'limitations':annotation['limitations'], 'evaluation_settings':{'imgsz':640,'confidence':.15,'iou':.5,'agnostic_nms':True,'classes':[2,7],'matching_iou':.5}, 'baseline_validation':baseline_val,'candidate_validation':candidate_val,'baseline_test':baseline_test,'candidate_test':candidate_test,'meets_promotion_gate':improved,'active_model_changed':False, 'selection':'best validation checkpoint only; one final test comparison, no test-driven retraining', 'note':'Detector quality only. Role identity, full-run pair availability and model deployment require fresh tracking and visual role review.'}
        save(out/'comparison.json',report)
        save(status,{'status':'complete','meets_promotion_gate':improved,'candidate':str(candidate)})
        print(json.dumps({k:report[k] for k in ['status','training_seconds','baseline_test','candidate_test','meets_promotion_gate']}),flush=True)
    except Exception as error:
        save(status,{'status':'failed','error':str(error)})
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true')
    main(parser.parse_args().evaluate_only)
