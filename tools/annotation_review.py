"""Create, explicitly confirm, import and export a separate annotation review queue."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driftlens.annotation_review import (build_review_queue, confirm_sample, export_detector_labels,
                                        export_reviewed_labels, import_review_queue,
                                        load_review_queue, save_review_queue)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _destination(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        path.relative_to(ROOT / "outputs" / "annotation_review")
    except ValueError as error:
        raise ValueError("Write reviews under outputs/annotation_review; the frozen labels remain separate") from error
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("queue", help="Copy frozen evaluator references into a pending human review queue")
    create.add_argument("--labels", type=Path, default=ROOT / "data/annotations/labels.json")
    create.add_argument("--catalog", type=Path, default=ROOT / "data/clip_catalog.json")
    create.add_argument("--out", default="outputs/annotation_review/queue.json")
    create.add_argument("--group-by", choices=["battle", "video"], default="battle")
    status = commands.add_parser("status")
    status.add_argument("--queue", type=Path, default=ROOT / "outputs/annotation_review/queue.json")
    confirm = commands.add_parser("confirm", help="Record a human's explicit review of one source frame")
    confirm.add_argument("--queue", type=Path, default=ROOT / "outputs/annotation_review/queue.json")
    confirm.add_argument("--sample", required=True)
    confirm.add_argument("--reviewer", required=True)
    confirm.add_argument("--edits", type=Path, required=True, help="JSON object containing cars and smoke/overlap tags")
    for name in ("image", "boxes", "identities"):
        confirm.add_argument(f"--confirm-{name}", action="store_true")
    merge = commands.add_parser("import")
    merge.add_argument("--queue", type=Path, default=ROOT / "outputs/annotation_review/queue.json")
    merge.add_argument("--input", type=Path, required=True)
    merge.add_argument("--out", default="outputs/annotation_review/queue.json")
    export = commands.add_parser("export")
    export.add_argument("--queue", type=Path, default=ROOT / "outputs/annotation_review/queue.json")
    export.add_argument("--out", required=True)
    export.add_argument("--split", choices=["tune", "train", "val", "test"])
    export.add_argument("--format", choices=["evaluator", "detector"], default="evaluator")
    arguments = parser.parse_args()
    try:
        if arguments.command == "queue":
            path = _destination(arguments.out)
            if path.exists():
                raise ValueError("The queue already exists; use status or import to retain its review history")
            source_path = arguments.labels.resolve().relative_to(ROOT).as_posix()
            queue = build_review_queue(_read(arguments.labels), _read(arguments.catalog),
                                       source_labels_path=source_path, group_by=arguments.group_by)
            save_review_queue(path, queue)
        else:
            queue = load_review_queue(arguments.queue)
            if arguments.command == "confirm":
                edits = _read(arguments.edits)
                queue = confirm_sample(queue, arguments.sample, edits["cars"], edits["tags"], arguments.reviewer,
                                       {name: getattr(arguments, f"confirm_{name}") for name in ("image", "boxes", "identities")})
                save_review_queue(_destination(arguments.queue), queue)
            elif arguments.command == "import":
                queue = import_review_queue(queue, _read(arguments.input))
                save_review_queue(_destination(arguments.out), queue)
            elif arguments.command == "export":
                if arguments.format == "detector" and arguments.split is None:
                    raise ValueError("Detector export requires an explicit split to keep test references separate")
                exported = (export_detector_labels(queue, arguments.split) if arguments.format == "detector"
                            else export_reviewed_labels(queue, arguments.split))
                path = _destination(arguments.out)
                if path.resolve() == arguments.queue.resolve():
                    raise ValueError("Export to a new file instead of replacing the review queue")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".tmp")
                try:
                    temporary.write_text(json.dumps(exported, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                print(json.dumps({"output": str(path), "reviewed_frames": exported["reviewed_frames"]}))
                return
        reviewed = sum(record["review"] is not None for record in queue["records"])
        print(json.dumps({"reviewed_frames": reviewed, "pending_frames": len(queue["records"]) - reviewed,
                          "split_grouping": queue["split_grouping"],
                          "samples": [{"sample_id": record["sample_id"], "image": record["image"],
                                       "split": record["split"], "provenance": record["provenance"]}
                                      for record in queue["records"]]}, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
