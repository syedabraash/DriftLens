import io
from pathlib import Path
import tempfile
import unittest

from driftlens.uploads import save_video_upload


class Upload(io.BytesIO):
    def __init__(self, content, name="myvideo.mp4", declared_size=None):
        super().__init__(content)
        self.name = name
        self.size = len(content) if declared_size is None else declared_size


class UploadTests(unittest.TestCase):
    def probe(self, path):
        self.assertEqual(path.read_bytes(), b"video-content")
        return dict(duration_seconds=2, source_fps=30, width=1280, height=720, total_frames=60)

    def test_safe_content_addressed_name_and_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = Upload(b"video-content", "../../outside.mp4")
            first = save_video_upload(source, root, self.probe)
            second = save_video_upload(Upload(b"video-content", "different.mp4"), root, self.probe)
            self.assertEqual(first["path"], second["path"])
            self.assertEqual(Path(first["path"]).parent, root / "data/uploads")
            self.assertEqual(len(list((root / "data/uploads").iterdir())), 1)
            self.assertEqual(source.tell(), 0)

    def test_actual_bytes_limit_even_if_declared_size_is_wrong(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                save_video_upload(Upload(b"oversized", declared_size=0), root, self.probe, max_bytes=3)
            self.assertEqual(list((root / "data/uploads").iterdir()), [])

    def test_invalid_video_leaves_no_published_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                save_video_upload(Upload(b"video-content"), root, lambda _: {"source_fps": 0})
            self.assertEqual(list((root / "data/uploads").iterdir()), [])

    def test_empty_and_unsupported_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for content, name in [(b"", "empty.mp4"), (b"video-content", "file.exe")]:
                with self.assertRaises(ValueError):
                    save_video_upload(Upload(content, name), Path(temporary), self.probe)


if __name__ == "__main__":
    unittest.main()
