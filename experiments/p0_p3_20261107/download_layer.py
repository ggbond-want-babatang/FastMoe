"""Download only pretrained OLMoE embedding + decoder layer 0 using HTTP ranges.

No remote Python code is executed. A valid standalone subset safetensors file is
written, with fixed revision, byte-range checks, and a provenance manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import requests

REPO = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(session, name, start=None, end=None):
    for attempt in range(4):
        try:
            url = BASE + name
            headers = {}
            if start is not None:
                url += f"?download=true&moe_range={start}-{end}"
                headers["Range"] = f"bytes={start}-{end}"
            with session.get(url, headers=headers, timeout=(15, 90), stream=True) as r:
                r.raise_for_status()
                if start is not None:
                    prefix = f"bytes {start}-{end}/"
                    if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(prefix):
                        raise RuntimeError(f"Server did not honor exact range: {r.status_code} {r.headers.get('Content-Range')}")
                    expected = end-start+1
                    data = b"".join(r.iter_content(1 << 20))
                    if len(data) != expected:
                        raise RuntimeError(f"Truncated range {len(data)} != {expected}")
                else:
                    data = r.content
            return data
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"retry {name} {start}-{end}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(attempt+1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/olmoe-layer0")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "layer0.safetensors"
    manifest_path = root / "manifest.json"
    if target.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if digest(target) == manifest["subset_sha256"] and manifest["revision"] == REVISION:
            print("Verified existing pretrained subset", flush=True)
            return
        raise RuntimeError("Existing weights differ from manifest; preserve and investigate")
    session = requests.Session()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md", "model.safetensors.index.json"):
        (root / name).write_bytes(get(session, name))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    selected = [k for k in index if k.startswith("model.layers.0.") or k == "model.embed_tokens.weight"]
    shards = sorted({index[k] for k in selected})
    metadata = {}
    layouts = {}
    for shard in shards:
        size = struct.unpack("<Q", get(session, shard, 0, 7))[0]
        header = json.loads(get(session, shard, 8, 7+size))
        for key in selected:
            if index[key] == shard:
                info = dict(header[key])
                a, b = info["data_offsets"]
                layouts[key] = dict(shard=shard, start=8+size+a, end=8+size+b)
                metadata[key] = info
    keys = sorted(selected, key=lambda k:(layouts[k]["shard"], layouts[k]["start"]))
    position = 0
    segments = []
    for key in keys:
        layout = layouts[key]
        length = layout["end"]-layout["start"]
        metadata[key]["data_offsets"] = [position, position+length]
        if segments and segments[-1]["shard"] == layout["shard"] and segments[-1]["end"] == layout["start"]:
            segments[-1]["end"] = layout["end"]
        else:
            segments.append(dict(shard=layout["shard"], start=layout["start"], end=layout["end"], destination=position))
        position += length
    metadata["__metadata__"] = dict(format="pt", source=REPO, revision=REVISION, subset="embedding and decoder layer 0 only")
    header_bytes = json.dumps(metadata, separators=(",", ":")).encode()
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    partial = root / "layer0.safetensors.partial"
    started = time.monotonic()
    done = 0
    with partial.open("wb") as output:
        output.write(struct.pack("<Q", len(header_bytes)))
        output.write(header_bytes)
        base = output.tell()
        for segment in segments:
            for start in range(segment["start"], segment["end"], 32 << 20):
                end = min(start+(32 << 20), segment["end"])-1
                data = get(session, segment["shard"], start, end)
                output.seek(base+segment["destination"]+start-segment["start"])
                output.write(data)
                output.flush()
                done += len(data)
                print(f"Downloaded {done/2**20:.1f}/{position/2**20:.1f} MiB ({time.monotonic()-started:.1f}s)", flush=True)
    partial.rename(target)
    manifest = dict(repo=REPO, revision=REVISION, downloaded_bytes=position,
                    subset_sha256=digest(target), tensor_count=len(keys), tensors=keys,
                    source_tensor_ranges=layouts, metadata_sha256={
                        p.name:digest(p) for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".md")},
                    method="Verified HTTP 206 exact ranges; no complete checkpoint download", elapsed_seconds=time.monotonic()-started)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {target}: {len(keys)} genuine pretrained tensors", flush=True)


if __name__ == "__main__":
    main()
