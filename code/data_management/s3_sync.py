# %%
import concurrent.futures
import logging
import os
import subprocess
from threading import local

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def sync_directory(local_dir, destination, if_copy=False, if_dry_run=True):
    """
    Sync the local directory with the given S3 destination using aws s3 sync.
    Returns a status string based on the command output.
    """
    try:
        if if_copy:
            cmd = ["aws", "s3", "cp", local_dir, destination]
            if os.path.isdir(local_dir):
                cmd.append("--recursive")

            if if_dry_run:
                cmd.append("--dryrun")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        else:
            # Run aws s3 sync command and capture the output
            cmd = ["aws", "s3", "sync", local_dir, destination]
            if if_dry_run:
                cmd.append("--dryrun")
            result = subprocess.run(
                cmd, capture_output=True, text=True
            )
        output = result.stdout + result.stderr

        # Check output: if "upload:" appears, files were sent;
        # otherwise, assume that nothing needed uploading.
        if "upload:" in output:
            logger.info(f"Uploaded {local_dir} to {destination}!")
            return "successfully uploaded"
        else:
            logger.info(output)
            logger.info(f"Already exists, skip {local_dir}.")
            return "already exists, skip"
    except Exception as e:
        return f"error during sync: {e}"


# %%
if __name__ == "__main__":
    s3_bucket_dest = "s3://aind-scratch-data/sue_su/LC_beh_physiology/"
    local_dir = "/root/capsule/scratch/"
    combine_only = False
    manuscript = True
    if combine_only:
        s3_bucket_dest += "combined/"
        local_dir += "combined/"
    elif manuscript:
        s3_bucket_dest += "manuscript/"
        local_dir += "manuscript/"

    out = sync_directory(local_dir, s3_bucket_dest, if_copy=False, if_dry_run=False)
    print(out)


