"""scp manifest utilities.

Manifest formats used across the URGENT projects:

- kv scp (2 columns):    ``uid value``            (wav.scp, spk1.scp, utt2fs, ...)
- source scp (3 columns): ``uid fs audio_path``   (speech/noise/rir/wind-noise pools)

Paths inside the URGENT 2025 repo manifests are relative to the repo root;
`abspath_source_scp` / `abspath_kv_scp` rewrite them to absolute paths so that
manifests can live anywhere (e.g. BaseUSE/data/) while audio stays in place.
"""

import os
from collections import defaultdict


def read_kv_scp(scp):
    rtv = {}
    with open(scp, "r") as f:
        for line in f:
            uid, value = line.strip().split()
            rtv[uid] = value
    return rtv


def read_source_scp(scp):
    source_dict = defaultdict(dict)
    with open(scp, "r") as f:
        for line in f:
            uid, fs, audio_path = line.strip().split()
            source_dict[int(fs)][uid] = audio_path
    return source_dict


def abspath_kv_scp(in_scp, out_scp, base_dir):
    """Rewrite a 2-column scp so that relative paths are resolved against base_dir."""
    base_dir = os.path.abspath(base_dir)
    n = 0
    with open(in_scp, "r") as fin, open(out_scp, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, value = line.split()
            if not os.path.isabs(value):
                value = os.path.join(base_dir, value)
            fout.write(f"{uid} {value}\n")
            n += 1
    return n


def abspath_source_scp(in_scp, out_scp, base_dir):
    """Rewrite a 3-column (uid fs path) scp so that relative paths resolve against base_dir."""
    base_dir = os.path.abspath(base_dir)
    n = 0
    with open(in_scp, "r") as fin, open(out_scp, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, fs, path = line.split()
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path.lstrip("./"))
            fout.write(f"{uid} {fs} {path}\n")
            n += 1
    return n


def concat_scps(in_scps, out_scp, base_dir=None, abspath=False):
    """Concatenate 3-column scps into one file, optionally rewriting relative paths.

    Duplicate uids raise an AssertionError (same convention as the original
    read_source_scp which asserts uid uniqueness per sampling rate).
    """
    seen = set()
    total = 0
    with open(out_scp, "w") as fout:
        for in_scp in in_scps:
            with open(in_scp, "r") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    uid, fs, path = line.split()
                    assert uid not in seen, f"duplicate uid: {uid} ({in_scp})"
                    seen.add(uid)
                    if abspath and base_dir is not None and not os.path.isabs(path):
                        path = os.path.join(os.path.abspath(base_dir), path.lstrip("./"))
                    fout.write(f"{uid} {fs} {path}\n")
                    total += 1
    return total
