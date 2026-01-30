
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import sqlite3
import pytz




# Create a timezone object for Melbourne, Australia
MELB_TZ = pytz.timezone("Australia/Melbourne")


class RetryableError(Exception):
    """Raised for transient failures that should trigger a retry."""
    pass


def should_skip(file_path, temp_path, min_bytes):
    if temp_path.exists():
        print(f"Removing stale temp file: {temp_path.name}")
        temp_path.unlink(missing_ok=True)

    if not file_path.exists():
        return False

    size = file_path.stat().st_size
    if size >= min_bytes:
        print(f"File exists and looks OK ({size} bytes), skipping: {file_path.name}")
        return True

    print(f"Existing file too small ({size} bytes), re-downloading: {file_path.name}")

    # unlink because you don't want to keep this incomplete file
    file_path.unlink(missing_ok=True)
    return False





def read_write_data(r, filename, temp_path, file_path, min_bytes):
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        raise RetryableError(f"Content-Type is HTML: {filename}")
    
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    with open(temp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    if bytes_written < min_bytes:
        print(f"File too small ({bytes_written} bytes), retrying: {filename}")
        temp_path.unlink(missing_ok=True)
        raise RetryableError(f"File too small ({bytes_written} bytes): {filename}")

    # Check if temp file is actually HTML
    with open(temp_path, "rb") as f:
        first = f.read(256).lower()
        if b"<html" in first or b"<!doctype" in first:
            print(f"Response was HTML, not CSV: {filename}")
            temp_path.unlink(missing_ok=True)
            raise RetryableError(f"Response was HTML, not CSV: {filename}")

    temp_path.replace(file_path)
    return True


def now_melbourne():
    return datetime.now(MELB_TZ)



def download_month(
    session,
    file_url,
    filename,
    file_path,
    temp_path,
    min_bytes,
    sleep_time,
    max_retries,
    connect_timeout,
    read_timeout,
):
    for attempt in range(max_retries):
        try:
            with session.get(
                file_url,
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as r:
                r.raise_for_status()
                read_write_data(r, filename, temp_path, file_path, min_bytes)
                print(f"Downloaded -> {filename}")
            
            time.sleep(sleep_time)
            return True
            
        except Exception as e:
            # Don't retry non-retryable HTTP errors (404, 403, etc.)
            if isinstance(e, requests.exceptionsHTTPError):
                status = e.response.status_code if e.response else None
                if status and 400 <= status < 500 and status != 429:
                    print(f"Non-retryable HTTP {status}: {filename}")
                    return False
            
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Retry {attempt + 1}/{max_retries} for {filename} after {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f"Failed after {max_retries} attempts: {filename} ({e})")
                temp_path.unlink(missing_ok=True)

    return False


def build_sql_db(
    file_path,
    table,
    connection,
    cursor,
    cols,
    chunksize=200_000,
):
    file_df = pd.read_csv(file_path, chunksize=chunksize)

    for chunk in file_df:
        missing = set(cols) - set(chunk.columns)
        extra = set(chunk.columns) - set(cols)
        if missing or extra:
            raise ValueError(
                f"Schema mismatch in {file_path.name}.\nMissing: {sorted(missing)}\nExtra: {sorted(extra)}"
            )

        rows = chunk.where(chunk.notna(), None).itertuples(index=False, name=None)
        placeholders = ", ".join(["?"] * len(cols))  
        col_list = ", ".join([f'"{c}"' for c in cols])
        # Prepare SQL INSERT statement
        # INSERT OR IGNORE means insert row unless it violates UNIQUE constraint (duplicate)
        insert_sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders});"
        cursor.executemany(insert_sql, rows)

    connection.commit()


def collect_data(
    start_year,
    end_year,
    state="VIC1",
    out_dir="data",
    max_retries=3,
    connect_timeout=10,
    read_timeout=30,
    sleep_time=0.2,
    min_bytes=1024,
):
    """Download monthly AEMO CSVs for the given state within the year range."""

    base_url = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_dt = now_melbourne()
    current_yyyymm = int(current_dt.strftime("%Y%m"))
    end_year = min(end_year, current_dt.year)

    session = requests.Session()
    session.headers.update({"User-Agent": "aemo-downloader/1.0 (+data-project)"})

    try:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                yyyymm = f"{year}{month:02d}"
                if int(yyyymm) > current_yyyymm:
                    continue
                
                filename = f"PRICE_AND_DEMAND_{yyyymm}_{state}.csv"
                file_url = f"{base_url}/{filename}"
                file_path = out_dir / filename
                temp_path = out_dir / f"{filename}.tmp"

                if should_skip(file_path, temp_path, min_bytes):
                    continue

                download_month(
                    session,
                    file_url,
                    filename,
                    file_path,
                    temp_path,
                    min_bytes,
                    sleep_time,
                    max_retries,
                    connect_timeout,
                    read_timeout,
                )
    finally:
        session.close()


def merge_monthly_files_sql(
    state="VIC1",
    in_dir="data",
    out_file="PRICE_AND_DEMAND_FULL_VIC1.csv",
    db_file="aemo_merge.sqlite",
    chunksize=200_000,
):
    """
    Memory-safe merge + duplicate removal + sort using SQLite.
    - Loads CSVs in chunks
    - Duplicate removal enforced by UNIQUE index
    - Exports sorted CSV
    - Might want to add some helper functions to check for missing months
    """

    in_dir = Path(in_dir)


    # the * means it can be anything in between (the YYYYMM part) and still match 
    pattern = f"PRICE_AND_DEMAND_*_{state}.csv"  
    
    # find all files that match the expected pattern and return iterator of Path objects
    files = sorted(in_dir.glob(pattern))

    if not files:
        print("No files found to merge.")
        return

    cols = pd.read_csv(files[0], nrows=0).columns.tolist()

    # SETTLEMENTDATE is the timestamp for each data point
    if "SETTLEMENTDATE" not in cols:
        raise ValueError("Expected SETTLEMENTDATE column not found.")


    # columns that shouldd have unique values, used for duplicate removal
    dedup_cols = ["SETTLEMENTDATE"]
    if "REGIONID" in cols:
        dedup_cols.append("REGIONID")

    # Delete old database if rerunning this function
    db_path = Path(db_file)
    if db_path.exists():  

        db_path.unlink()  

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor() 

    table = "price_demand"  

    col_defs = ", ".join([f'"{c}" TEXT' for c in cols])
    cursor.execute(f"CREATE TABLE {table} ({col_defs});")

    # UNIQUE index to prevent dupes 
    # If trying to insert a row with same SETTLEMENTDATE (and REGIONID), it will be ignored
    uniq = ", ".join([f'"{c}"' for c in dedup_cols])
    cursor.execute(f"CREATE UNIQUE INDEX uniq_rows ON {table} ({uniq});")

    # index on SETTLEMENTDATE for faster sorting later
    cursor.execute(f'CREATE INDEX idx_time ON {table} ("SETTLEMENTDATE");')
    connection.commit()

    print(f"Merging {len(files)} files into SQLite (chunksize={chunksize})...")

    for file_path in files:
        print(f"Processing file: {file_path.name}...")
        build_sql_db(file_path, table, connection, cursor, cols, chunksize)


    print("Exporting sorted, duplicate removed CSV...")
    order_by = '"SETTLEMENTDATE"' 
    if "REGIONID" in cols:
        order_by += ', "REGIONID"'  

    out_path = Path(out_file)  
    col_list = ", ".join([f'"{c}"' for c in cols])

    # Query to select all data, sorted
    query = f"SELECT {col_list} FROM {table} ORDER BY {order_by};"
    
    first_chunk = True  

    # exporting data in chunks to avoid memory issues
    for df in pd.read_sql_query(query, connection, chunksize=chunksize):
        # mode="w" for first chunk, mode="a" for rest cos you want to append
        # header=True for first chunk
        df.to_csv(out_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False

    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    n_rows = cursor.fetchone()[0]  
    connection.close()  

    print(f"Merged {len(files)} files into {out_file} ({n_rows} rows after duplicate removal).")

def missing_months_in_merged(start_year, end_year, merged_file="PRICE_AND_DEMAND_FULL_VIC1.csv"):
    """Returns list of (year, month) tuples not present in the merged CSV."""

    existing_months = set()

    try:
        for chunk in pd.read_csv(
            merged_file,
            usecols=["SETTLEMENTDATE"],
            parse_dates=["SETTLEMENTDATE"],
            chunksize=200_000,
        ):
            years = chunk["SETTLEMENTDATE"].dt.year
            months = chunk["SETTLEMENTDATE"].dt.month
            existing_months.update(zip(years, months))
    except FileNotFoundError:
        print("Merged CSV not found. Assuming all months are missing.")
        existing_months = set()

    # Build complete set of months in the specified range
    all_months = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            all_months.add((year, month))

    # Find missing months by subtracting existing from all
    missing = sorted(all_months - existing_months)
    return missing


def main():
    """Main function: downloads data then merges it."""
    collect_data(start_year=2023, end_year=2026, state="VIC1", out_dir="aemo_vic1")

    merge_monthly_files_sql(
        state="VIC1",
        in_dir="aemo_vic1",
        out_file="PRICE_AND_DEMAND_FULL_VIC1.csv",
        db_file="aemo_merge.sqlite",
    )
    print("Missing months in merged data:", missing_months_in_merged(2023, 2026, "PRICE_AND_DEMAND_FULL_VIC1.csv"))



if __name__ == "__main__":
    main()
