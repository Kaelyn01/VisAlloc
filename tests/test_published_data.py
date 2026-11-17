import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("published_data", ROOT / "scripts/prepare_data.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="PNG")
    return stream.getvalue()


class PublishedDataTests(unittest.TestCase):
    def fixture(self, root):
        release, source = root / "release", root / "source"
        (release / "data").mkdir(parents=True)
        (release / "docs").mkdir()
        (source / "images").mkdir(parents=True)
        rows, upstream = [], []
        for i in range(3):
            question = f"Question {i}?"
            positive, negative = png("red"), png("blue")
            (source / f"images/{i}.png").write_bytes(png("white"))
            upstream.append({"problem": "<image>\n" + question, "answer": "A", "images": [f"images/{i}.png"], "original_images": [f"original/{i}.jpg"]})
            rows.append({
                "sample_id": i, "question": question, "answer": "A",
                "legacy_quality_label": "healthy" if i != 1 else "annotation_suspect",
                "alignment_verified": True, "generation_version": "2.0.0", "bbox_iou": 0.0,
                "source_image_path": f"original/{i}.jpg",
                "positive_image": {"bytes": positive, "path": None},
                "negative_image": {"bytes": negative, "path": None},
                "positive_sha256": hashlib.sha256(positive).hexdigest(),
                "negative_sha256": hashlib.sha256(negative).hexdigest(),
                "source_bbox": [0, 0, 4, 4], "negative_bbox": [8, 8, 12, 12],
                "output_size": [16, 16], "negative_crop_window": [8, 8, 16, 16],
                "original_size": [32, 32], "target_bbox_in_crop": [0, 0, 4, 4],
                "shared_frame_mask_sha256": "frame-hash",
            })
        (source / "train.jsonl").write_text("\n".join(json.dumps(row) for row in upstream))
        shard = release / "data/full-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), shard)
        ids = json.dumps([0, 2]).encode()
        (release / "docs/training_sample_ids.json").write_bytes(ids)
        stats = {"num_examples": 3, "training_subset": {"num_examples": 2, "sample_ids_file": "docs/training_sample_ids.json", "sample_ids_sha256": hashlib.sha256(ids).hexdigest()}, "shards": [{"path": "data/full-00000.parquet", "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}
        (release / "dataset_stats.json").write_text(json.dumps(stats))
        return release, source

    def test_selection_preserves_source_ids_and_aligns_all_three_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release, source = self.fixture(root)
            output = root / "training"
            DATA.materialize(release, source, output)
            rows = pq.read_table(output / "trainset_clean.parquet").to_pylist()
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["extra_info"]["source_extra_info"]["source_sample_id"] for r in rows], [0, 2])
            for i, row in enumerate(rows):
                for key in ["images", "bbox_images", "neg_bbox_images"]:
                    self.assertTrue(Path(row[key][0]["path"]).is_file())
                self.assertEqual(Path(row["neg_bbox_images"][0]["path"]).name, f"{i:06d}.png")
            self.assertEqual((output / "teacher_neg/000001.png").read_bytes(), png("blue"))

    def test_corrupt_keep_ids_are_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release, source = self.fixture(root)
            (release / "docs/training_sample_ids.json").write_text("[0,1]")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                DATA.materialize(release, source, root / "output")
            self.assertFalse((root / "output").exists())

    def test_corrupt_shard_cannot_be_published_as_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release, source = self.fixture(root)
            with (release / "data/full-00000.parquet").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "shard hash"):
                DATA.materialize(release, source, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
