import os
import requests
import concurrent.futures
import datetime
from pathlib import Path
from settings import Config
from logger import setup_logger

logger = setup_logger()

def file_download(base_url, file_id, file_path):
    """
    Downloads a file from the specified URL and saves it to a given path.
    Returns a message indicating success or failure.
    """
    download_url = f"{base_url}{file_id}/file"
    headers = Config.get_auth_headers()

    zip_file_path = f"{file_path}.zip"

    try:
        logger.info(f"Downloading {zip_file_path}")
        with requests.get(download_url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(zip_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=Config.CHUNK_SIZE):
                    f.write(chunk)
        return f"SUCCESS: Downloaded to {zip_file_path}"

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {file_id}: {e}")
        return f"ERROR: Download failed for {file_id}: {e}"
    except IOError as e:
        logger.error(f"File write error for {zip_file_path}: {e}")
        return f"ERROR: File write failed for {zip_file_path}: {e}"

def download_customer_files(valid_entries_all_sites, customer_name):
    """
    Filters valid entries by customer name and downloads files concurrently.
    """
    customer_entries = [
        entry for entry in valid_entries_all_sites
        if customer_name.lower() in entry.get('name', '').lower()
    ]

    if not customer_entries:
        logger.warning(f"No entries found for customer: {customer_name}")
        return None

    # Create customer folder in data directory
    customer_folder = Config.DOWNLOAD_DIR / customer_name
    customer_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting concurrent downloads for {customer_name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        download_tasks = []
        for entry in customer_entries:
            site_id = entry.get('site')
            file_name = entry.get('name')
            file_id = entry.get('id')

            if all([site_id, file_name, file_id]):
                base_url = f'https://wrightbus-{site_id}.odosolutions.com/api/v1/downloads/'
                save_path = customer_folder / file_name
                download_tasks.append((base_url, file_id, save_path))
            else:
                logger.warning(f"Skipping incomplete entry: {entry}")

        for future in concurrent.futures.as_completed(
            [executor.submit(file_download, *task) for task in download_tasks]
        ):
            try:
                result = future.result()
                logger.info(result)
            except Exception as exc:
                logger.error(f"Download task failed: {exc}")

    logger.info(f"Completed downloads for {customer_name}")
    return customer_folder