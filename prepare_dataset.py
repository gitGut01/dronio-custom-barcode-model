import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
import gdown

DEFAULT_GDRIVE_FILE_ID = "1OkgxBTgSOse2uKqYf306bJYmvSq-eC5T"

def _die(message: str, exit_code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)

def _is_dir_nonempty(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())

def download_from_google_drive(file_id: str, destination: Path) -> None:
    """Uses gdown to handle large files and virus scan warnings automatically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Attempting to download from Google Drive ID: {file_id}")
    output = gdown.download(url, str(destination), quiet=False)
    
    if output is None:
        _die("Download failed. Ensure the file is public or the ID is correct.")

def unzip(zip_path: Path, extract_dir: Path, *, force: bool) -> None:
    if not zip_path.exists():
        _die(f"Zip file not found: {zip_path}")
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

def _find_split_dirs(extract_dir: Path) -> dict[str, Path]:
    split_dirs: dict[str, Path] = {}
    for split in ("train", "test", "val"):
        candidates = []
        for p in extract_dir.rglob(split):
            if not p.is_dir():
                continue
            # Look for markers of a dataset split
            if (p / "labels.csv").exists() or (p / "images").is_dir() or any(p.iterdir()):
                candidates.append(p)
        if len(candidates) == 1:
            split_dirs[split] = candidates[0]
        elif len(candidates) > 1:
            # Pick the shortest path (usually the top-level one)
            candidates_sorted = sorted(candidates, key=lambda x: len(str(x)))
            split_dirs[split] = candidates_sorted[0]
    return split_dirs

def _move_contents(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        target = dst_dir / child.name
        if target.exists():
            if target.is_dir() and child.is_dir():
                for nested in child.iterdir():
                    if (target / nested.name).exists():
                        os.remove(target / nested.name) # Overwrite files if they exist
                    shutil.move(str(nested), str(target / nested.name))
                continue
            # If it's a file collision, remove the old one and move the new one
            os.remove(target)
        shutil.move(str(child), str(target))

def flatten_dataset(extract_dir: Path, output_dir: Path, *, force: bool) -> None:
    split_dirs = _find_split_dirs(extract_dir)
    missing = [s for s in ("train", "test", "val") if s not in split_dirs]
    
    if missing:
        _die(f"Could not locate split directories {missing} inside {extract_dir}")

    for split, src in split_dirs.items():
        dst = output_dir / split
        if _is_dir_nonempty(dst) and not force:
            print(f"Warning: {dst} already exists. Use --force to overwrite.")
            continue
        if dst.exists() and force:
            shutil.rmtree(dst)
        
        print(f"Moving {split} data to {dst}...")
        _move_contents(src, dst)

def replace_labels(output_dir: Path, labels_dir: Path) -> None:
    """
    Replaces my_dataset/<split>/labels.csv with labels/<split>/labels.csv
    """
    for split in ("train", "test", "val"):
        # The source of new labels
        src = labels_dir / split / "labels.csv"
        # The target destination in the flattened dataset
        dst = output_dir / split / "labels.csv"
        
        if src.exists():
            if (output_dir / split).exists():
                print(f"Replacing {dst} with labels from {src}")
                shutil.copy2(src, dst)
            else:
                print(f"Skipping label replacement for {split}: Split folder not in output.")
        else:
            print(f"Note: No replacement labels found at {src}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", default=DEFAULT_GDRIVE_FILE_ID)
    parser.add_argument("--output-dir", default="my_dataset")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--zip-path", default="")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset folders")
    parser.add_argument("--keep-extracted", action="store_true", help="Keep the raw extracted folder")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    labels_dir = (repo_root / args.labels_dir).resolve()

    # Determine where the zip is or should be
    if args.zip_path:
        zip_path = Path(args.zip_path).expanduser().resolve()
    else:
        zip_path = "dataset.zip"

    extract_dir = "my_dataset"

    # 1. Download
    if not args.skip_download:
        download_from_google_drive(args.file_id, zip_path)
    
    return
    # 2. Unzip
    unzip(zip_path, extract_dir, force=True)

    # 3. Flatten
    print(f"Flattening dataset into: {output_dir}")
    flatten_dataset(extract_dir, output_dir, force=args.force)

    # 4. Replace Labels
    print(f"Replacing labels.csv from {labels_dir}...")
    replace_labels(output_dir, labels_dir)

    # 5. Cleanup
    if not args.keep_extracted and extract_dir.exists():
        shutil.rmtree(extract_dir)
    
    # Optional: remove the zip after successful extraction
    # if zip_path.exists(): zip_path.unlink()

    print("\nProcess Complete!")
    return 0

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)