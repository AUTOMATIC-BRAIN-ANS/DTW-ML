"""
Recording of the inputs rejected by a pipeline stage: the rejected file or directory is copied (never moved) to the
directory of the spoiled data and the reason is appended to its log.
"""

import csv
import os
import shutil
from datetime import datetime

# log of the rejected inputs, kept in every directory of the spoiled data
FAILURES_FILE = "_failures.csv"


def copy_to_spoiled(src: str, spoiled_dir: str, reason: str) -> str:
    """
    Function to copy a rejected file or directory to the directory of the spoiled data and to log the reason.
    The input itself is left in place.
    :param src: path of the rejected file or directory.
    :param spoiled_dir: directory of the spoiled data (created if missing).
    :param reason: reason of the rejection.
    :return: path of the copy.
    """
    os.makedirs(spoiled_dir, exist_ok=True)
    dst = os.path.join(spoiled_dir, os.path.basename(os.path.normpath(src)))
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    log_path = os.path.join(spoiled_dir, FAILURES_FILE)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if write_header:
            writer.writerow(["Time", "Source", "Reason"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), src, reason])
    return dst
