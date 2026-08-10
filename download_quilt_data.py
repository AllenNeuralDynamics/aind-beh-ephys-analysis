"""
Download all objects under a prefix from the aind-scratch-data S3 bucket
(browsable at https://open.quiltdata.com/b/aind-scratch-data/tree/...)
to a local directory, preserving the folder structure.

The bucket permits anonymous (unsigned) S3 reads, so no AWS credentials
are required. Requires: pip install boto3
"""

import os
import re
import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "aind-scratch-data"
PREFIX = "sue_su/LC_beh_physiology/results/"
DEST_DIR = r"C:\Users\zhixi\OneDrive - Allen Institute\LCpaper\submission\review_07052026\figures\0720"

# Characters not allowed in Windows file/directory names
_INVALID_CHARS = re.compile(r'[<>:"|?*]')


def sanitize_windows_path(rel_path):
    parts = rel_path.split("/")
    parts = [_INVALID_CHARS.sub("_", p) for p in parts]
    return os.path.join(*parts)


def main():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):  # skip folder placeholders
                keys.append((obj["Key"], obj["Size"]))

    total = len(keys)
    total_bytes = sum(size for _, size in keys)
    print(f"Found {total} files ({total_bytes / 1024 / 1024:.1f} MB) under s3://{BUCKET}/{PREFIX}")
    print(f"Downloading to: {DEST_DIR}\n")

    for i, (key, size) in enumerate(keys, 1):
        rel_path = key[len(PREFIX):]
        local_rel_path = sanitize_windows_path(rel_path)
        local_path = os.path.join(DEST_DIR, local_rel_path)
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(local_path) and os.path.getsize(local_path) == size:
            print(f"[{i}/{total}] SKIP (already exists): {rel_path}")
            continue

        print(f"[{i}/{total}] Downloading: {rel_path} ({size / 1024:.1f} KB)")
        s3.download_file(BUCKET, key, local_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
