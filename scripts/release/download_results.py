"""Download and verify the data required by the paper analysis notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(destination: Path, name: str) -> Path:
    """Reject archive paths that could escape the selected checkout."""

    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    path = destination.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Archive path leaves destination: {name}")
    return path


def same_existing_file(path: Path, entry: dict[str, object], staged: Path) -> bool:
    """Accept identical data and equivalent JSON already supplied by a checkout."""

    if not path.is_file():
        return False
    if sha256(path) == entry["sha256"]:
        return True
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text()) == json.loads(staged.read_text())
        except (ValueError, UnicodeError):
            return False
    return False


def extract_verified(path: Path, destination: Path) -> int:
    """Verify every file before installing it, never replacing different local data."""

    with tarfile.open(path, "r:gz") as archive, tempfile.TemporaryDirectory(prefix="gcs-data-") as temporary:
        manifest_file = archive.extractfile("MANIFEST.json")
        if manifest_file is None:
            raise ValueError("Archive has no file manifest")
        entries = json.load(manifest_file)["files"]
        expected = {entry["path"]: entry for entry in entries}
        if len(expected) != len(entries):
            raise ValueError("Duplicate manifest paths")
        staging = Path(temporary)
        seen = set()
        for member in archive:
            if member.name == "MANIFEST.json":
                continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            staged = safe_path(staging, member.name)
            staged.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive entry: {member.name}")
            with staged.open("wb") as handle:
                shutil.copyfileobj(stream, handle)
            entry = expected[member.name]
            if staged.stat().st_size != entry["size"] or sha256(staged) != entry["sha256"]:
                raise ValueError(f"File checksum failed: {member.name}")
            target = safe_path(destination, member.name)
            if target.exists() and not same_existing_file(target, entry, staged):
                raise FileExistsError(f"Different local file exists: {target}; choose an empty --destination")
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError("Archive is missing files listed in its manifest")
        for name in sorted(seen):
            target = safe_path(destination, name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging / name), target)
        return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("data_manifest.json"))
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--from-directory", type=Path, help="Verify and install locally prepared release archives")
    parser.add_argument("--base-url", help="Override the GitHub release download URL")
    parser.add_argument("--assets", nargs="+", help="Download only these named studies")
    args = parser.parse_args()
    catalog = json.loads(args.manifest.read_text())
    if not catalog["published"] and args.from_directory is None and args.base_url is None:
        parser.error("The data release is prepared but not published; use --from-directory for local verification")
    names = {asset["name"] for asset in catalog["assets"]}
    if args.assets and not set(args.assets).issubset(names):
        parser.error(f"Choose assets from: {', '.join(sorted(names))}")
    base_url = args.base_url or f"https://github.com/{catalog['repository']}/releases/download/{catalog['release_tag']}"
    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gcs-download-") as temporary:
        for asset in catalog["assets"]:
            if args.assets and asset["name"] not in args.assets:
                continue
            filename = asset["filename"]
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe asset filename: {filename}")
            if args.from_directory:
                path = args.from_directory / filename
            else:
                path = Path(temporary) / filename
                print(f"Downloading {filename}", flush=True)
                with urlopen(f"{base_url.rstrip('/')}/{filename}", timeout=60) as response, path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if path.stat().st_size != asset["archive_size_bytes"] or sha256(path) != asset["sha256"]:
                raise ValueError(f"Archive checksum failed: {filename}")
            count = extract_verified(path, args.destination)
            if count != asset["file_count"]:
                raise ValueError(f"Unexpected file count: {filename}")
            print(f"Verified {asset['name']}: {count} files", flush=True)


if __name__ == "__main__":
    main()
