"""Small helpers shared by the training, inference and preprocessing scripts."""

import ast
import argparse

import pandas as pd

ID_COL = "case_id"

# Column names used by the original research code, renamed on read so its CSVs keep working.
LEGACY_COLUMNS = {
    "RadAccessionID": ID_COL,
    "t2_image_path": "t2_path",
    "dwi_image_path": "dwi_path",
    "adc_image_path": "adc_path",
}


def read_table(path, required=()):
    """Reads a CSV, renames legacy columns, and checks that `required` (plus case_id) are present."""
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in LEGACY_COLUMNS.items() if k in df.columns and v not in df.columns})
    missing = [c for c in (ID_COL, *required) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; has {list(df.columns)}")
    df[ID_COL] = df[ID_COL].astype(str)
    return df


def parse_tuple(value):
    return tuple(int(x) for x in str(value).split(","))


def parse_float_tuple(value):
    return tuple(float(x) for x in str(value).split(","))


def parse_str_list(value):
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


def parse_dict(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a dict literal, got {value!r}")
    return parsed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {v!r}")


def whitelist_str(v):
    """
    Canonical string of a raw label value, so 3, 3.0 and "3" all compare equal (pandas upcasts an
    int column with missing values to float).
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else str(f)


def log_print(msg, log_path=None, mode="a"):
    print(msg, flush=True)
    if log_path is not None:
        with open(log_path, mode) as f:
            f.write(msg + "\n")


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"
