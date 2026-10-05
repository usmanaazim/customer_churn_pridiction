#!/usr/bin/env python3
"""Download the public Telco Customer Churn dataset if it is missing."""

from __future__ import annotations

import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, DATA_PATH, DATASET_URLS, REQUIRED_COLUMNS
from src.data_loader import dataset_exists, validate_columns
import pandas as pd


def download_csv(url: str, dest: Path) -> None:
    payload = b""
    try:
        with urlopen(url, timeout=45, context=SSL_CONTEXT) as response:
            payload = response.read()
    except Exception:
        import subprocess

        result = subprocess.run(["curl", "-fsSL", url], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:300] or "curl failed")
        payload = result.stdout
    if not payload or payload.startswith(b"404") or b"Not Found" in payload[:80]:
        raise RuntimeError("Downloaded file was empty or not found.")
    dest.write_bytes(payload)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if dataset_exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        missing = validate_columns(df)
        if missing:
            print("Existing file is missing columns:", ", ".join(missing))
            return 1
        print(f"Dataset already present: {DATA_PATH} ({len(df)} rows)")
        return 0

    print("Dataset not found. Attempting download of the public Telco Customer Churn dataset...")
    last_error = None
    for url in DATASET_URLS:
        try:
            print(f"Trying {url}")
            download_csv(url, DATA_PATH)
            df = pd.read_csv(DATA_PATH)
            df.columns = df.columns.str.strip()
            missing = validate_columns(df)
            if missing:
                DATA_PATH.unlink(missing_ok=True)
                raise RuntimeError("Downloaded file is missing columns: " + ", ".join(missing))
            print(f"Saved {DATA_PATH} with {len(df)} rows.")
            return 0
        except (URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_error = exc
            print(f"  Failed: {exc}")

    print("Could not download the dataset automatically.")
    print("Place the IBM/Kaggle Telco Customer Churn CSV at data/customer_churn.csv")
    print("See data/README.md for details.")
    if last_error:
        print(f"Last error: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
