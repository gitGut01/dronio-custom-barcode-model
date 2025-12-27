

# Download dataset from https://drive.google.com/file/d/1OkgxBTgSOse2uKqYf306bJYmvSq-eC5T/view?usp=drive_link

# Unzip dataset to my_dataset 

# flatten it so its my_dataset/train and my_dataset/test and my_dataset/val

# replcae the labels.csv from the ./labels folder with the respective labels.csv files in my_dataset
# So e.g replace the my_dataset/test/labels.csv with the test_labels.csv from labels/test/labels.csv


import argparse
import http.cookiejar
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_GDRIVE_FILE_ID = "1OkgxBTgSOse2uKqYf306bJYmvSq-eC5T"


def _die(message: str, exit_code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _is_dir_nonempty(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _build_opener() -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    handler = urllib.request.HTTPCookieProcessor(cookie_jar)
    opener = urllib.request.build_opener(handler)
    opener.addheaders = [
        ("User-Agent", "Mozilla/5.0 (prepare_dataset.py)")
    ]
    return opener


def download_from_google_drive(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    opener = _build_opener()
    base_url = "https://drive.google.com/uc?export=download"
    url = f"{base_url}&id={urllib.parse.quote(file_id)}"

    def _fetch(url_to_fetch: str) -> bytes:
        with opener.open(url_to_fetch) as resp:
            return resp.read()

    data = _fetch(url)
    text_snippet = data[:20000].decode("utf-8", errors="ignore")

    confirm_token = None
    for marker in ("confirm=", "confirm%3D"):
        if marker in text_snippet:
            idx = text_snippet.find(marker)
            token_start = idx + len(marker)
            token = []
            for ch in text_snippet[token_start:token_start + 128]:
                if ch.isalnum() or ch in ("_", "-"):
                    token.append(ch)
                else:
                    break
            if token:
                confirm_token = "".join(token)
                break

    if confirm_token:
        url = f"{base_url}&confirm={urllib.parse.quote(confirm_token)}&id={urllib.parse.quote(file_id)}"
        with opener.open(url) as resp, open(destination, "wb") as f:
            shutil.copyfileobj(resp, f)
        return

    if data.startswith(b"PK\x03\x04"):
        with open(destination, "wb") as f:
            f.write(data)
        return

    _die(
        "Google Drive download did not return a zip file. "
        "The file might require permission access or the confirmation token parse failed. "
        f"(Downloaded HTML snippet starts with: {text_snippet[:120]!r})"
    )


def unzip(zip_path: Path, extract_dir: Path, *, force: bool) -> None:
    if not zip_path.exists():
        _die(f"Zip file not found: {zip_path}")
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def _find_split_dirs(extract_dir: Path) -> dict[str, Path]:
    split_dirs: dict[str, Path] = {}
    for split in ("train", "test", "val"):
        candidates = []
        for p in extract_dir.rglob(split):
            if not p.is_dir():
                continue
            if (p / "labels.csv").exists() or (p / "images").is_dir():
                candidates.append(p)
        if len(candidates) == 1:
            split_dirs[split] = candidates[0]
        elif len(candidates) > 1:
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
                    shutil.move(str(nested), str(target / nested.name))
                continue
            _die(f"Collision while flattening dataset: {target} already exists")
        shutil.move(str(child), str(target))


def flatten_dataset(extract_dir: Path, output_dir: Path, *, force: bool) -> None:
    split_dirs = _find_split_dirs(extract_dir)
    missing = [s for s in ("train", "test", "val") if s not in split_dirs]
    if missing:
        _die(
            "Could not locate dataset split directories inside extracted zip: "
            f"missing {missing}. Looked under: {extract_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, src in split_dirs.items():
        dst = output_dir / split
        if _is_dir_nonempty(dst) and not force:
            _die(
                f"Target split directory already exists and is not empty: {dst}. "
                "Re-run with --force to overwrite."
            )
        if dst.exists() and force:
            shutil.rmtree(dst)
        _move_contents(src, dst)


def replace_labels(output_dir: Path, labels_dir: Path) -> None:
    for split in ("train", "test", "val"):
        src = labels_dir / split / "labels.csv"
        dst = output_dir / split / "labels.csv"
        if not src.exists():
            _die(f"Missing labels file: {src}")
        if not (output_dir / split).exists():
            _die(f"Missing dataset split folder: {output_dir / split}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", default=DEFAULT_GDRIVE_FILE_ID)
    parser.add_argument("--output-dir", default="my_dataset")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--zip-path", default="")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    labels_dir = (repo_root / args.labels_dir).resolve()

    zip_path = Path(args.zip_path).expanduser().resolve() if args.zip_path else (output_dir / "dataset.zip")
    extract_dir = output_dir / "_extracted"

    if not args.skip_download:
        print(f"Downloading dataset zip to: {zip_path}")
        download_from_google_drive(args.file_id, zip_path)
    else:
        if not zip_path.exists():
            _die(f"--skip-download was set but zip does not exist: {zip_path}")

    print(f"Unzipping to: {extract_dir}")
    unzip(zip_path, extract_dir, force=args.force)

    print(f"Flattening dataset into: {output_dir}")
    flatten_dataset(extract_dir, output_dir, force=args.force)

    print(f"Replacing labels.csv using: {labels_dir}")
    replace_labels(output_dir, labels_dir)

    if not args.keep_extracted and extract_dir.exists():
        shutil.rmtree(extract_dir)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
