"""Package code and measured evidence, excluding weights, caches and uploads."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    excluded={"weights", ".triton-cache", ".cuda-cache", ".hf-cache", "__pycache__"}
    files=[p for p in Path(".").rglob("*") if p.is_file() and not excluded.intersection(p.parts)
           and p.suffix not in (".zip", ".pyc") and p.name!="SHA256SUMS.json"]
    manifest={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    Path("SHA256SUMS.json").write_text(json.dumps(manifest,indent=2))
    files.append(Path("SHA256SUMS.json"))
    target=Path("MoE-P0-P3.zip")
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.as_posix())
    print("Packaged",len(files),"files",target.stat().st_size,"bytes")
    print("SHA256",hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
