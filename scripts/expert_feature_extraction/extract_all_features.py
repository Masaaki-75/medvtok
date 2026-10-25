#!/usr/bin/env python3
"""Split training CSVs into single-modality tables and run expert extraction.

Run from any working directory. Defaults: data/*.csv -> data/single_modality/
-> the adjacent 2D/3D extractors -> outputs/alignment/default/.

Examples (from the project root):
    python scripts/expert_feature_extraction/extract_all_features.py --dry-run
    python scripts/expert_feature_extraction/extract_all_features.py
    python scripts/expert_feature_extraction/extract_all_features.py --csv 'data/*.csv' --data-root /path/to/images
    python scripts/expert_feature_extraction/extract_all_features.py --modalities ct3d mr3d --nproc-per-node 4
"""
from __future__ import annotations

import argparse
import csv
import glob
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Match training dataset aliases, emitting names accepted by the extractors.
MODALITY_ALIASES = {
    "dermoscopy": "dermoscopy", "dermatology": "dermoscopy", "derm": "dermoscopy",
    "cfp": "cfp", "fundus": "cfp", "fundus-photography": "cfp",
    "histopathology": "histopathology", "pathology": "histopathology",
    "histo": "histopathology", "patho": "histopathology", "microscopy": "histopathology",
    "surgical": "surgical", "endoscopy": "endoscopy",
    "ophthalmoscope": "ophthalmoscope", "ophthalmoscopy": "ophthalmoscopy",
    "ct2d": "ct2d", "ct3d2d": "ct3d2d", "ct3d": "ct3d",
    "mr2d": "mr2d", "mr3d2d": "mr3d2d", "mr3d": "mr3d",
    "ultrasound": "ultrasound", "us": "ultrasound",
    "xray": "xray", "x-ray": "xray", "cxr": "xray", "mammography": "mammography",
    "oct": "oct", "pet": "pet", "unknown": "unknown", "other": "unknown",
}
THREE_D_MODALITIES = {"ct3d", "mr3d"}


def expand_csv_paths(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            raise FileNotFoundError(f"No CSV files match: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file() or path.suffix.lower() != ".csv":
                raise ValueError(f"Expected a CSV file: {path}")
            if path not in paths:
                paths.append(path)
    return paths


def normalize_modality(row: dict[str, str]) -> str:
    name = (row.get("submodality") or row.get("modality") or "").lower().strip().replace("_", "-")
    if name in ("ct", "mr"):
        # Prefer explicit ct3d/mr3d/ct3d2d labels. Legacy ct/mr labels use file type.
        is_volume = row["identifier"].lower().endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"))
        return name + ("3d" if is_volume else "2d")
    if name not in MODALITY_ALIASES:
        raise ValueError(f"Unsupported modality/submodality {name!r}; use an explicit supported name")
    return MODALITY_ALIASES[name]


def split_metadata(csv_paths: list[Path], data_root: Path, metadata_dir: Path, modalities=None) -> list[dict]:
    """Preserve metadata columns, normalize submodality, and deduplicate images.

    Original training CSVs (including upsampling rows) are untouched. Only the
    extraction tables are deduplicated: each image needs one feature.
    """
    groups, image_modalities = {}, {}
    columns = []
    total_rows = 0
    for source in csv_paths:
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if "identifier" not in fields or not {"submodality", "modality"}.intersection(fields):
                raise ValueError(f"{source}: requires identifier and submodality (or modality) columns")
            for field in fields + ["submodality"]:
                if field not in columns:
                    columns.append(field)
            for line_number, row in enumerate(reader, 2):
                total_rows += 1
                if None in row:
                    raise ValueError(f"{source}:{line_number}: too many CSV fields")
                identifier = (row.get("identifier") or "").strip()
                if not identifier:
                    raise ValueError(f"{source}:{line_number}: empty identifier")
                row["identifier"] = identifier
                try:
                    modality = normalize_modality(row)
                except ValueError as error:
                    raise ValueError(f"{source}:{line_number}: {error}") from error
                if modalities and modality not in modalities:
                    continue
                # Mirror the source tree under feature_root; normalize absolute IDs.
                image_path = Path(identifier).expanduser()
                image_path = (image_path if image_path.is_absolute() else data_root / image_path).resolve()
                try:
                    relative = image_path.relative_to(data_root).as_posix()
                except ValueError as error:
                    raise ValueError(f"{source}:{line_number}: image is outside --data-root: {image_path}") from error
                previous = image_modalities.setdefault(relative, modality)
                if previous != modality:
                    raise ValueError(f"Conflicting modalities for {relative}: {previous}, {modality}")
                row["identifier"], row["submodality"] = relative, modality
                groups.setdefault(modality, {}).setdefault(relative, row)
    if not groups:
        raise ValueError("No rows remain after selecting modalities")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for modality, rows_by_path in sorted(groups.items()):
        metadata_path = metadata_dir / f"{modality}.csv"
        if metadata_path in csv_paths:
            raise ValueError(f"Extraction table would overwrite a source CSV: {metadata_path}")
        # Atomically replace complete derived tables, without appending duplicates.
        temporary = metadata_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows_by_path.values())
        temporary.replace(metadata_path)
        jobs.append({"modality": modality, "meta_path": metadata_path, "rows": len(rows_by_path)})
        print(f"{modality}: {len(rows_by_path)} unique images -> {metadata_path}", flush=True)
    print(f"Converted {total_rows} source rows into {sum(job['rows'] for job in jobs)} unique images "
          f"across {len(jobs)} modalities", flush=True)
    return jobs


def extraction_command(job: dict, args) -> list[str]:
    is_3d = job["modality"] in THREE_D_MODALITIES
    script = SCRIPT_DIR / ("extract_alignment_features_3d.py" if is_3d else "extract_alignment_features.py")
    if not script.is_file():
        raise FileNotFoundError(script)
    if args.nproc_per_node > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc_per_node={args.nproc_per_node}", str(script)]
    else:
        command = [sys.executable, str(script)]
    command += ["--modality", job["modality"], "--meta_path", str(job["meta_path"]),
                "--data_root", str(args.data_root), "--feature_root", str(args.feature_root),
                "--batch_size", str(args.batch_size_3d if is_3d else args.batch_size),
                "--num_workers", str(args.num_workers_3d if is_3d else args.num_workers)]
    if args.device != "auto":
        command += ["--device", args.device]
    if args.max_samples is not None:
        command += ["--end_idx" if is_3d else "--end", str(args.max_samples)]
    if args.force:
        command.append("--force")
    if is_3d and args.use_2d_encoder:
        command.append("--use_2d_encoder")
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", default=[str(PROJECT_ROOT / "data" / "*.csv")],
                        help="Source CSV paths or quoted glob patterns (default: project data/*.csv)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "data" / "single_modality")
    parser.add_argument("--feature-root", type=Path, default=PROJECT_ROOT / "outputs" / "alignment")
    parser.add_argument("--modalities", nargs="+", choices=sorted(set(MODALITY_ALIASES.values())))
    parser.add_argument("--batch-size", type=int, default=128, help="2D batch size per process")
    parser.add_argument("--batch-size-3d", type=int, default=1, help="3D batch size per process")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-workers-3d", type=int, default=10)
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Processes/GPUs for each extractor")
    parser.add_argument("--device", default="auto", help="Single-process device, e.g. cpu or cuda:0")
    parser.add_argument("--max-samples", type=int, help="Extract at most this many pending images per modality")
    parser.add_argument("--use-2d-encoder", action="store_true", help="Use MII for 3D ablation instead of SPECTRE/HLIP")
    parser.add_argument("--force", action="store_true", help="Regenerate existing features")
    parser.add_argument("--dry-run", action="store_true", help="Write derived tables and print commands without extraction")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.batch_size_3d, args.nproc_per_node) < 1:
        parser.error("batch sizes and --nproc-per-node must be positive")
    if min(args.num_workers, args.num_workers_3d) < 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("worker counts must be nonnegative and --max-samples must be positive")
    if args.nproc_per_node > 1 and args.device not in ("auto", "cuda"):
        parser.error("Multi-GPU extraction requires --device auto or cuda; torchrun assigns each local device")
    for name in ("data_root", "metadata_dir", "feature_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.data_root.is_dir():
        parser.error(f"Image data root does not exist: {args.data_root}")
    return args


def main(argv=None):
    args = parse_args(argv)
    csv_paths = expand_csv_paths(args.csv)
    jobs = split_metadata(csv_paths, args.data_root, args.metadata_dir, args.modalities)
    commands = [extraction_command(job, args) for job in jobs]
    for job, command in zip(jobs, commands):
        print(f"\n[{job['modality']}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            # No shell: pass spaces/glob characters literally, and resolve repo imports via cwd too.
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("Conversion and command preview complete." if args.dry_run else "All requested feature extractions completed.",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
