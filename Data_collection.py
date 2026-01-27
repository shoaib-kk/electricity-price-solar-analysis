
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import sqlite3
import pytz

# KEY NOTE: All datetime operations are done in Melbourne timezone
# Personal note: Consider refactoring functions into smaller helper functions for clarity



# Create a timezone object for Melbourne, Australia
MELB_TZ = pytz.timezone("Australia/Melbourne")
def should_retry(attempt, max_retries, e, filename, temp_path):
    if attempt < max_retries:
        # Exponential backoff: wait 1s, then 2s, then 4s, etc.
        wait = 2 ** (attempt - 1)
        print(
            f"Retry {attempt}/{max_retries} for {filename} after {wait}s ({e})"
        )
        time.sleep(wait)
        return True  # Try again
    else:
        # All retries failed, give up on this file
        print(f"Failed after {max_retries} attempts: {filename} ({e})")
        temp_path.unlink(missing_ok=True)  # Clean up partial download
        return False
def should_skip(file_path, temp_path, min_bytes):
    # Remove any stale temp file before deciding
    if temp_path.exists():
        print(f"Removing stale temp file: {temp_path.name}")
        temp_path.unlink(missing_ok=True)

    if not file_path.exists():
        return False

    size = file_path.stat().st_size
    if size >= min_bytes:
        print(f"File exists and looks OK ({size} bytes), skipping: {file_path.name}")
        return True

    # Existing file is too small; re-download
    print(f"Existing file too small ({size} bytes), re-downloading: {file_path.name}")
    file_path.unlink(missing_ok=True)
    return False
def skip_future_months(yyyymm, current_yyyymm, filename):
    if int(yyyymm) > current_yyyymm:
        print(f"Future month, skipping: {filename}")
        return True
    return False   
def read_n_write_data(r, filename, temp_path, file_path, min_bytes):
    # 404 == file doesn't exist on server
    if r.status_code == 404:
        print(f"Missing (404): {filename}")

        # No point retrying if file doesn't exist
        return False  

    # Raise an exception if the HTTP request returned an error status
    # (4xx client errors, 5xx server errors)
    r.raise_for_status()

    # Check the Content-Type header to make sure we're getting CSV, not HTML error page
    content_type = (r.headers.get("Content-Type") or "").lower()
    if "html" in content_type:
        print(f"Skipping HTML response for {filename}")
        return False  # Server sent HTML instead of CSV, skip this file

    # Download the file in chunks and write to temporary file
    bytes_written = 0

    # 'wb' mode = write binary (important for downloading binary/CSV data)
    with open(temp_path, "wb") as f:
        bytes_written = write_chunks(r,f,bytes_written)
    
    # If file is tiny, it's probably an error message
    if bytes_written < min_bytes:
        print(
            f"Skipping suspiciously small file ({bytes_written} bytes): {filename}"
        )
        # Path.unlink() deletes the file (missing_ok=True means no error if already gone)
        temp_path.unlink(missing_ok=True)
        return False

    # Double-check: read first 256 bytes to make sure it's not HTML
    if validate_not_html(temp_path, filename):
        return False 

    # move temp file to final location
    # Path.replace() all-or-nothing so partial files don't get copied
    temp_path.replace(file_path)
    return True


def now_melbourne():
    """Get current datetime in Melbourne timezone."""
    return datetime.now(MELB_TZ)


def write_chunks(r,f,bytes_written):
    # iter_content() streams the file in chunks (memory efficient for large files)
    # chunk_size=8192 means 8KB at a time
    for chunk in r.iter_content(chunk_size=8192):
        
        # Skip empty chunks
        if not chunk:
            continue

        # Write chunk to file
        f.write(chunk) 

        # Keep track of total bytes written
        bytes_written += len(
            chunk
        )  
    return bytes_written
def build_sql_db(file_path, table, connection, cursor, cols, col_list, placeholders, chunksize=200_000):
        # Read CSV in chunks (memory efficient - doesn't load entire file at once)
        # chunksize=200000 means process 200k rows at a time
        file_df = pd.read_csv(file_path, chunksize=chunksize)

        for chunk in file_df:

            # Check missing columns 
            missing = set(cols) - set(chunk.columns)

            # Check extra columns   
            extra = set(chunk.columns) - set(cols) 
            if missing or extra:
                raise ValueError(
                    f"Schema mismatch in {file_path.name}.\nMissing: {sorted(missing)}\nExtra: {sorted(extra)}"
                )

            # Convert DataFrame chunk to rows (tuples) for bulk insert
            # astype(str) converts everything to string for consistent storage
            # where(notna(), None) converts NaN to None (SQL NULL)
            rows = (
                chunk.astype(str)
                .where(chunk.notna(), None)
                .itertuples(index=False, name=None)
            )

            # insert query
            insert_sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders});"

            # executemany() is efficient bulk insert of multiple rows
            cursor.executemany(insert_sql, rows)
            connection.commit()

def validate_not_html(temp_path, filename):
    with open(temp_path, "rb") as f:  # 'rb' = read binary
        first = f.read(256).lower()  # Read first 256 bytes
        # Check if file starts with HTML tags (means we got error page, not CSV)
        if b"<html" in first or b"<!doctype" in first:
            print(
                f"Skipping HTML payload masquerading as CSV: {filename}"
            )
            temp_path.unlink(missing_ok=True)  # Delete the bad file
            return True
    return False
def collect_data(
    start_year,
    end_year,
    state="VIC1",
    out_dir="data",
    max_retries=3,
    connect_timeout=10,
    read_timeout=30,
    polite_sleep=0.2,
    min_bytes=1024,
):
    """
    Downloads AEMO PRICE_AND_DEMAND monthly CSVs for a given state (e.g., VIC1)
    for all months in the year range [start_year, end_year].
    """
    # Base URL where AEMO hosts the CSV files
    base_url = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"

    # Convert string path to Path object
    out_dir = Path(out_dir)

    # Create the output directory if it doesn't exist
    # parents=True: creates parent directories if needed (like 'mkdir -p' in Unix)
    # exist_ok=True: doesn't raise error if directory already exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    current_dt = now_melbourne()
    current_yyyymm = int(current_dt.strftime("%Y%m"))
    end_year = min(end_year, current_dt.year)

    def file_features(year, month):
        yyyymm_local = f"{year}{month:02d}"
        filename_local = f"PRICE_AND_DEMAND_{yyyymm_local}_{state}.csv"
        file_url_local = f"{base_url}/{filename_local}"
        file_path_local = out_dir / filename_local
        temp_path_local = out_dir / f"{filename_local}.tmp"
        return yyyymm_local, filename_local, file_url_local, file_path_local, temp_path_local

    # Create a requests Session object - this reuses the same TCP connection for multiple requests
    # This is more efficient than making individual requests.get() calls
    session = requests.Session()

    # Set a User-Agent header to identify our script (polite web scraping practice)
    session.headers.update({"User-Agent": "aemo-downloader/1.0 (+data-project)"})

    try:
        # Download loop
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):

                # Generate file features
                yyyymm, filename, file_url, file_path, temp_path = file_features(year, month)

                # Skip future months
                if skip_future_months(yyyymm, current_yyyymm, filename):
                    continue

                # Skip existing files (or stale temp/small files)
                if should_skip(file_path, temp_path, min_bytes):
                    continue

                # Retry loop for downloading the file
                for attempt in range(1, max_retries + 1):
                    try:
                        # Make HTTP GET request to download the file
                        # stream=True: downloading in chunks is memory efficient
                        # timeout=(connect, read): max seconds to wait for connection and reading data
                        with session.get(file_url, stream=True, timeout=(connect_timeout, read_timeout)) as r:

                            if not read_n_write_data(r, filename, temp_path, file_path, min_bytes):
                                break
                            print(f"Downloaded: {year}-{month:02d} -> {filename}")

                        # Polite pause after successful download (avoid hammering server)
                        time.sleep(polite_sleep)
                        break

                    # Catch any network/HTTP errors from requests library
                    except requests.RequestException as e:

                        if not should_retry(attempt, max_retries, e, filename, temp_path):
                            break
    finally:
        # Close the session to free up resources
        session.close()


def merge_monthly_files_sqlite(
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

    # Convert directory path to Path object
    in_dir = Path(in_dir)

    # Create a glob pattern to match all monthly CSV files for this state
    # the * means it can be anything in between (the YYYYMM part)
    pattern = f"PRICE_AND_DEMAND_*_{state}.csv"  
    
    # find all files that match the expected pattern and return iterator of Path objects
    # Sort to ensure consistent order
    files = sorted(in_dir.glob(pattern))

    if not files:
        print("No files found to merge.")
        return

    # Read just the header row (0 data rows) from first file to get column names
    cols = pd.read_csv(files[0], nrows=0).columns.tolist()

    # SETTLEMENTDATE is the timestamp for each data point, essential for time series
    if "SETTLEMENTDATE" not in cols:
        raise ValueError("Expected SETTLEMENTDATE column not found.")


    # Build list of columns that uniquely identify a row (for removing duplicates)
    dedup_cols = ["SETTLEMENTDATE"]
    if "REGIONID" in cols:
        dedup_cols.append("REGIONID")

    # Delete old database if rerunning this function
    db_path = Path(db_file)
    if db_path.exists():  

        # Unlinking deletes file 
        db_path.unlink()  

    # Create new SQLite database connection
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor() 

    table = "price_demand"  

    # Create the table with columns matching the CSV
    # Store everything as TEXT 
    col_defs = ", ".join(
        [f'"{c}" TEXT' for c in cols]
    )  # e.g., "COL1" TEXT, "COL2" TEXT
    cursor.execute(f"CREATE TABLE {table} ({col_defs});")

    # Create UNIQUE index - this prevents duplicate rows from being inserted
    # If we try to insert a row with same SETTLEMENTDATE (and REGIONID), it will be ignored
    uniq = ", ".join([f'"{c}"' for c in dedup_cols])
    cursor.execute(f"CREATE UNIQUE INDEX uniq_rows ON {table} ({uniq});")

    # Create index on SETTLEMENTDATE for faster sorting later
    cursor.execute(f'CREATE INDEX idx_time ON {table} ("SETTLEMENTDATE");')

    # Commit changes 
    connection.commit()

    # Prepare SQL INSERT statement
    # INSERT OR IGNORE means: insert row unless it violates UNIQUE constraint (duplicate)
    placeholders = ", ".join(["?"] * len(cols))  
    col_list = ", ".join([f'"{c}"' for c in cols])


    print(f"Merging {len(files)} files into SQLite (chunksize={chunksize})...")

    # Loop through each CSV file and add its data to the database
    for file_path in files:
        print(f"Processing file: {file_path.name}...")
        build_sql_db(file_path, table, connection, cursor, cols, col_list, placeholders, chunksize)

    # Export the merged, duplicate removed data back to CSV, sorted by time
    print("Exporting sorted, duplicate removed CSV...")
    order_by = '"SETTLEMENTDATE"' 
    if "REGIONID" in cols:

        # Secondary sort: by region
        order_by += ', "REGIONID"'  

    # Output path as Path object
    out_path = Path(out_file)  


    # Query to select all data, sorted
    query = f"SELECT {col_list} FROM {table} ORDER BY {order_by};"
    
    # Export in chunks to avoid loading entire dataset into memory
    first_chunk = True  

    for df in pd.read_sql_query(query, connection, chunksize=chunksize):
        # mode="w" for first chunk (overwrite), mode="a" for rest (append)
        # header=True for first chunk, header=False for rest
        df.to_csv(out_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False

    # Count total rows in final merged dataset
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    n_rows = cursor.fetchone()[0]  
    connection.close()  

    print(f"Merged {len(files)} files into {out_file} ({n_rows} rows after duplicate removal).")

def missing_months_in_merged(start_year, end_year, merged_file="PRICE_AND_DEMAND_FULL_VIC1.csv"):
    """Returns list of (year, month) tuples not present in the merged CSV."""

    # Read existing merged CSV to find which months are present
    try:
        df = pd.read_csv(merged_file, parse_dates=["SETTLEMENTDATE"])
    except FileNotFoundError:
        print("Merged CSV not found. Assuming all months are missing.")
        existing_months = set()
    else:
        # Extract year and month from SETTLEMENTDATE
        df["Year"] = df["SETTLEMENTDATE"].dt.year
        df["Month"] = df["SETTLEMENTDATE"].dt.month

        # Create set of (year, month) tuples present in the data
        # zip pairs up Year and Month columns
        existing_months = set(zip(df["Year"], df["Month"]))

    # Build complete set of months in the specified range
    all_months = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            all_months.add((year, month))

    # Find missing months by subtracting existing from all
    missing = sorted(all_months - existing_months)
    return missing


# Backward-compatible alias
def missing_months(start_year, end_year):
    return missing_months_in_merged(start_year, end_year)

def main():
    """Main function: downloads data then merges it."""
    # Step 1: Download monthly CSV files from AEMO
    collect_data(start_year=2023, end_year=2026, state="VIC1", out_dir="aemo_vic1")

    # Step 2: Merge all monthly files into one sorted, duplicate removed CSV
    merge_monthly_files_sqlite(
        state="VIC1",
        in_dir="aemo_vic1",
        out_file="PRICE_AND_DEMAND_FULL_VIC1.csv",
        db_file="aemo_merge.sqlite",
    )


if __name__ == "__main__":
    main()
