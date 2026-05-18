"""
Pipeline: S3 download (producer) || revisit-anything.py + benchmark.py (consumer).

Input: a txt file with one S3 key per line, e.g.
    v3/151/15525/2026/05/08/07:20:10.000000/data_number_0
The scene_name is parts[2] of the key (e.g. "15525").

For each key, downloads recursively:
    s3://<bucket>/<key>/cpu_prepro/level_and_locked_frames/  -> <base>/<scene>/images/
    s3://<bucket>/<key>/sfm/sfm_run_data/outputs/            -> <base>/<scene>/colmap_reference/
where <base> = <dataset_base_dir>/<dataset_name>.

A single producer thread feeds a queue (default maxsize=2) so up to N scenes
are downloaded ahead while the consumer runs revisit-anything.py then
benchmark.py for the current scene in separate subprocesses.
"""

import csv
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import typer
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger
import time

REPO_ROOT = Path(__file__).resolve().parent

SUBDIRS = [
    ("cpu_prepro/level_and_locked_frames", "images"),
    ("sfm/people_masks", "masks"),
    ("sfm/sfm_run_data/outputs", "colmap_reference"),
]
SENTINEL = None

app = typer.Typer(add_completion=False, help=__doc__)


def read_keys(path):
    keys = []
    with open(path) as f:
        for line in f:
            line = line.strip().strip("/")
            if line and not line.startswith("#"):
                keys.append(line)
    return keys


def scene_name_from_key(key):
    parts = key.split("/")
    if len(parts) < 3:
        raise ValueError(f"key has fewer than 3 path components, cannot extract scene_name: {key}")
    return parts[2]


def download_prefix(s3, bucket, prefix, local_dir, workers):
    """Download every object under s3://bucket/prefix/ into local_dir, preserving relative paths."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/") + "/"

    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            objects.append((key, rel))

    if not objects:
        logger.warning(f"no objects under s3://{bucket}/{prefix}")
        return 0

    def _one(key, rel):
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, k, r) for k, r in objects]
        for f in as_completed(futures):
            f.result()
    return len(objects)


def downloader(keys, bucket, base_dir, workers, q):
    s3 = boto3.client("s3")
    for key in keys:
        try:
            scene = scene_name_from_key(key)
        except ValueError as e:
            logger.error(str(e))
            continue

        scene_dir = base_dir / scene
        logger.info(f"[download] start {scene}  ({key})")
        ok = True
        try:
            for s3_sub, local_sub in SUBDIRS:
                src = f"{key}/{s3_sub}"
                n = download_prefix(s3, bucket, src, scene_dir / local_sub, workers)
                logger.info(f"[download]   {local_sub}: {n} files")
        except ClientError as e:
            logger.error(f"[download] {scene} failed: {e}")
            ok = False
        except Exception as e:
            logger.error(f"[download] {scene} failed: {e}")
            ok = False

        if ok:
            logger.info(f"[download] done {scene}")
            q.put(scene)

    q.put(SENTINEL)


def run_subprocess(cmd, scene):
    logger.info(f"[process] {scene}: $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        logger.error(f"[process] {scene}: command failed (exit {r.returncode}): {' '.join(cmd)}")
        return False
    return True


def process_scene(scene, dataset_base_dir, dataset_name, topn=10):
    workdir = Path(dataset_base_dir) / dataset_name / scene
    t_start = time.time()
    rev_cmd = [
        sys.executable, "revisit-anything.py",
        f"dataset_base_dir={dataset_base_dir}",
        f"dataset_name={dataset_name}",
        f"scene_name={scene}",
        f"top_n={topn}",
        f"pca_config.svd_solver=randomized",  # faster for large num_segments
    ]

    if not run_subprocess(rev_cmd, scene):
        return False

    t_end = time.time()
    logger.info(f"[process] {scene}: revisit-anything.py took {(t_end - t_start)}s")
    with open(str(workdir / "revisit-anything.time.txt"), "w") as f:
        f.write(f"{t_end - t_start}\n")

    bench_cmd = [sys.executable, "submodules/DetectorFreeSfM/benchmark.py", 
        f"dataset_base_dir={dataset_base_dir}",
        f"dataset_name={dataset_name}",
        f"scene_name={scene}",
        f"matcher_cfg.img_pair_strategy=pair_from_txt",
        f"matcher_cfg.img_pair_txt={str(workdir)}/outputs/pairs_top{topn}.txt",
    ]
    return run_subprocess(bench_cmd, scene)


def sum_dfsfm_time(csv_path):
    """Return the sum of process_time values where task_name == 'dfsfm' (seconds)."""
    total = 0.0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("task_name") or "").strip() == "dfsfm":
                total += float(row["process_time"])
    return total


@app.command()
def compare_times(
    keys_file: Path = typer.Option(..., "--keys-file", exists=True, dir_okay=False, readable=True, help="txt file, one S3 key per line"),
    bucket: str = typer.Option(..., "--bucket", help="S3 bucket name"),
    dataset_base_dir: str = typer.Option("/home/ubuntu/", "--dataset-base-dir"),
    dataset_name: str = typer.Option("search-data", "--dataset-name"),
    env_file: Path = typer.Option(REPO_ROOT / ".env", "--env-file", help="dotenv file with AWS credentials"),
):
    """Compare dfsfm process times: S3 reference (matching + sfm) vs. local outputs."""
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        logger.info(f"loaded env from {env_file}")
    else:
        logger.warning(f"env file {env_file} not found; relying on existing environment / IAM role")

    keys = read_keys(keys_file)
    if not keys:
        logger.error(f"no keys found in {keys_file}")
        raise typer.Exit(code=1)
    logger.info(f"loaded {len(keys)} keys from {keys_file}")

    base_dir = Path(dataset_base_dir) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)
    output_csv = base_dir / "times.csv"

    s3 = boto3.client("s3")
    rows = []
    for key in keys:
        try:
            scene = scene_name_from_key(key)
        except ValueError as e:
            logger.error(str(e))
            continue

        scene_dir = base_dir / scene
        matching_csv = scene_dir / "sfm" / "matching_run_data" / "process_time.csv"
        sfm_csv = scene_dir / "sfm" / "sfm_run_data" / "process_time.csv"
        local_csv = scene_dir / "outputs" / "process_time.csv"

        try:
            for s3_rel, dest in [
                ("sfm/matching_run_data/process_time.csv", matching_csv),
                ("sfm/sfm_run_data/process_time.csv", sfm_csv),
            ]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, f"{key}/{s3_rel}", str(dest))
        except ClientError as e:
            logger.error(f"[{scene}] reference download failed: {e}")
            continue

        try:
            reference_time = sum_dfsfm_time(matching_csv) + sum_dfsfm_time(sfm_csv)
        except Exception as e:
            logger.error(f"[{scene}] failed to parse reference CSVs: {e}")
            continue

        if not local_csv.is_file():
            logger.warning(f"[{scene}] local {local_csv} not found; skipping")
            continue
        try:
            local_time = sum_dfsfm_time(local_csv)
        except Exception as e:
            logger.error(f"[{scene}] failed to parse local CSV: {e}")
            continue

        logger.info(f"[{scene}] reference={reference_time:.4f}s  local={local_time:.4f}s")
        rows.append((scene, reference_time, local_time))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_name", "reference", "local"])
        for scene, ref, loc in rows:
            writer.writerow([scene, f"{ref:.4f}", f"{loc:.4f}"])

    logger.info(f"wrote {len(rows)} rows to {output_csv}")
    raise typer.Exit(code=0)


@app.command()
def main(
    keys_file: Path = typer.Option(..., "--keys-file", exists=True, dir_okay=False, readable=True, help="txt file, one S3 key per line"),
    bucket: str = typer.Option(..., "--bucket", help="S3 bucket name"),
    dataset_base_dir: str = typer.Option("/home/ubuntu/", "--dataset-base-dir"),
    dataset_name: str = typer.Option("search-data", "--dataset-name"),
    queue_size: int = typer.Option(2, "--queue-size", min=1, help="how many scenes to download ahead"),
    download_workers: int = typer.Option(8, "--download-workers", min=1, help="parallel file downloads per scene"),
    topn: int = typer.Option(10, "--topn", min=1, help="Top-N matches per image for revisit-anything.py"),
    env_file: Path = typer.Option(REPO_ROOT / ".env", "--env-file", help="dotenv file with AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ...)"),
):
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        logger.info(f"loaded env from {env_file}")
    else:
        logger.warning(f"env file {env_file} not found; relying on existing environment / IAM role")

    keys = read_keys(keys_file)
    if not keys:
        logger.error(f"no keys found in {keys_file}")
        raise typer.Exit(code=1)
    logger.info(f"loaded {len(keys)} keys from {keys_file}")

    base_dir = Path(dataset_base_dir) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue = queue.Queue(maxsize=queue_size)
    t = threading.Thread(
        target=downloader,
        args=(keys, bucket, base_dir, download_workers, q),
        daemon=True,
    )
    t.start()

    ok_count = 0
    fail_count = 0
    while True:
        scene = q.get()
        if scene is SENTINEL:
            break
        if process_scene(scene, dataset_base_dir, dataset_name, topn):
            ok_count += 1
        else:
            fail_count += 1

    t.join()
    logger.info(f"pipeline complete: {ok_count} ok, {fail_count} failed")
    raise typer.Exit(code=0 if fail_count == 0 else 1)


if __name__ == "__main__":
    app()
