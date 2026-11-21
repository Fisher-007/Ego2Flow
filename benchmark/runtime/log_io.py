"""Read complete state logs in JSONL or gzip-compressed JSONL format."""
import gzip
import json


def iter_frames(directory):
    compressed = directory / 'frames.jsonl.gz'
    stream = gzip.open(compressed, 'rt') if compressed.exists() else (directory / 'frames.jsonl').open()
    with stream:
        for line in stream:
            yield json.loads(line)
