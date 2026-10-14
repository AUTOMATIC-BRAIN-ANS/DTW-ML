"""
Unit tests for utils/spoiled.py.
"""

import csv

from ddtw_ml.utils.spoiled import FAILURES_FILE, copy_to_spoiled


def read_log(spoiled_dir):
    with open(spoiled_dir / FAILURES_FILE, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter=";"))


def test_file_is_copied_and_logged(tmp_path):
    src = tmp_path / "input" / "V1_B6_PP_F_M.csv"
    src.parent.mkdir()
    src.write_text("a;b\n1;2\n", encoding="utf-8")
    spoiled_dir = tmp_path / "spoiled" / "B6"
    dst = copy_to_spoiled(str(src), str(spoiled_dir), "ValueError: bad data")
    # the input stays in place, the copy is identical
    assert src.exists()
    assert dst == str(spoiled_dir / src.name)
    assert (spoiled_dir / src.name).read_text(encoding="utf-8") == "a;b\n1;2\n"
    rows = read_log(spoiled_dir)
    assert rows[0] == ["Time", "Source", "Reason"]
    assert rows[1][1:] == [str(src), "ValueError: bad data"]


def test_directory_is_copied_and_log_is_appended(tmp_path):
    src = tmp_path / "metrics" / "B6" / "V1_B6_PP_F"
    src.mkdir(parents=True)
    (src / "ABP.csv").write_text("x\n", encoding="utf-8")
    (src / "CBFV.csv").write_text("y\n", encoding="utf-8")
    spoiled_dir = tmp_path / "spoiled" / "B6"
    copy_to_spoiled(str(src), str(spoiled_dir), "first")
    copy_to_spoiled(str(src), str(spoiled_dir), "second")
    assert (src / "ABP.csv").exists()
    assert sorted(p.name for p in (spoiled_dir / src.name).iterdir()) == ["ABP.csv", "CBFV.csv"]
    # one header, one row per rejection
    assert [row[2] for row in read_log(spoiled_dir)] == ["Reason", "first", "second"]
