#!/usr/bin/env python3
"""Materialize the released VisAlloc training subset and paired teacher views."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from build_candidate_views import convert_to_parquet, extract_tar_parts, safe_dataset_path

DATASET_ID = "Kaelyn01/VisAlloc-6K"
DATASET_REVISION = "e01c86c5c60abecede2d908aa9aeb6a84705b775"
UPSTREAM_ID = "yuanqianhao/Vision-OPD-6K"
UPSTREAM_REVISION = "eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./cache/visalloc-4b")
    parser.add_argument("--release-snapshot", type=Path, help="Existing pinned VisAlloc-6K snapshot")
    parser.add_argument("--source-dir", type=Path, help="Pinned upstream train.jsonl and extracted student images")
    return parser.parse_args()


def load_keep_ids(release):
    stats = json.loads((release / "dataset_stats.json").read_text())
    path = release / stats["training_subset"]["sample_ids_file"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != stats["training_subset"]["sample_ids_sha256"]:
        raise ValueError("Published training sample IDs failed their SHA-256 check")
    ids = json.loads(data)
    if len(ids) != stats["training_subset"]["num_examples"] or ids != sorted(set(ids)):
        raise ValueError("Training sample IDs must be unique, ordered, and match the declared count")
    return ids, stats


def image_bytes(row, key):
    data = row[key]["bytes"]
    digest_key = "positive_sha256" if key == "positive_image" else "negative_sha256"
    if hashlib.sha256(data).hexdigest() != row[digest_key]:
        raise ValueError(f"{key} hash mismatch at sample {row['sample_id']}")
    return data


def materialize(release, source, destination):
    """Keep published sample IDs; dense output indices align negative views and priors."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; use a fresh data directory")
    ids, stats = load_keep_ids(release)
    keep = set(ids)
    with (source / "train.jsonl").open() as stream:
        upstream = [json.loads(line) for line in stream if line.strip()]
    if len(upstream) != stats["num_examples"]:
        raise ValueError("Upstream row count does not match the pinned release")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".visalloc-data-", dir=destination.parent))
    results, selected = [], []
    try:
        for directory in ["images", "teacher_images", "teacher_neg"]:
            (stage / directory).mkdir()
        with (stage / "trainset_clean.jsonl").open("w") as output:
            for shard in stats["shards"]:
                shard_path = release / shard["path"]
                digest = hashlib.sha256()
                with shard_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != shard["sha256"]:
                    raise ValueError(f"Parquet shard hash mismatch: {shard_path.name}")
                for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        sample_id = row["sample_id"]
                        if sample_id not in keep:
                            continue
                        if row["legacy_quality_label"] != "healthy" or not row["alignment_verified"]:
                            raise ValueError(f"Published membership or alignment mismatch: {sample_id}")
                        if row["generation_version"] != "2.0.0" or row["bbox_iou"] >= 0.1:
                            raise ValueError(f"Paired-view geometry mismatch: {sample_id}")
                        original = upstream[sample_id]
                        question = original["problem"].replace("<image>", "").strip()
                        if question != row["question"] or original["answer"] != row["answer"]:
                            raise ValueError(f"Upstream question/answer mismatch: {sample_id}")
                        if original["original_images"][0] != row["source_image_path"]:
                            raise ValueError(f"Upstream image identity mismatch: {sample_id}")
                        index = len(selected)
                        student = Path(safe_dataset_path(str(source), original["images"][0]))
                        student_name = f"images/{index:06d}{student.suffix}"
                        shutil.copyfile(student, stage / student_name)
                        positive_name = f"teacher_images/{index:06d}.png"
                        (stage / positive_name).write_bytes(image_bytes(row, "positive_image"))
                        (stage / f"teacher_neg/{index:06d}.png").write_bytes(image_bytes(row, "negative_image"))
                        item = {
                            "problem": "<image>\n" + row["question"],
                            "images": [student_name],
                            "teacher_images": [positive_name],
                            "answer": row["answer"],
                            "extra_info": {"source_sample_id": sample_id, "answer": row["answer"], "question": row["question"]},
                        }
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
                        results.append({
                            "idx": index, "source_sample_id": sample_id, "ok": True,
                            "gt_bbox": row["source_bbox"], "neg_bbox": row["negative_bbox"],
                            "iou": row["bbox_iou"], "crop_size": row["output_size"],
                            "negative_crop_window": row["negative_crop_window"],
                            "original_size": row["original_size"], "target_bbox_in_crop": row["target_bbox_in_crop"],
                            "positive_sha256": row["positive_sha256"], "negative_sha256": row["negative_sha256"],
                            "shared_frame_mask_sha256": row["shared_frame_mask_sha256"],
                            "generation_version": row["generation_version"], "alignment_verified": True,
                        })
                        selected.append(sample_id)
        if selected != ids:
            raise ValueError("Selected rows do not exactly reproduce the ordered keep-ID list")
        (stage / "dataset_generation_manifest.json").write_text(json.dumps(results))
        (stage / "dataset_source.json").write_text(json.dumps({
            "dataset": DATASET_ID, "revision": DATASET_REVISION,
            "upstream": UPSTREAM_ID, "upstream_revision": UPSTREAM_REVISION,
            "sample_ids": selected,
        }, indent=2))
        os.replace(stage, destination)
        convert_to_parquet(str(destination), results)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Prepared {len(selected)} retained examples in {destination}")


def main():
    args = parse_args()
    release = args.release_snapshot or Path(snapshot_download(
        DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=["data/*.parquet", "dataset_stats.json", "docs/training_sample_ids.json"],
    ))
    if args.source_dir:
        source = args.source_dir
    else:
        source = Path(snapshot_download(
            UPSTREAM_ID, repo_type="dataset", revision=UPSTREAM_REVISION,
            allow_patterns=["train.jsonl", "images/*"],
        ))
        image_dir = source / "images"
        parts = sorted(image_dir.glob("images.tar.gz*"))
        if parts:
            extract_tar_parts(parts, str(image_dir))
    materialize(release, source, Path(args.data_dir))


if __name__ == "__main__":
    main()
