import shutil
from pathlib import Path
from logger import setup_logger

logger = setup_logger()

def cleanup_download_folder(folder_path):
    """
    Completely removes the download folder and all its contents
    
    Args:
        folder_path (str/Path): Path to the folder to be deleted
    Returns:
        bool: True if deletion succeeded, False otherwise
    """
    try:
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning(f"Nothing to clean - folder doesn't exist: {folder}")
            return True 

        logger.info(f"Starting complete cleanup of {folder}")

        # Remove the entire directory tree
        shutil.rmtree(folder)
        logger.info(f"Successfully deleted folder and all contents: {folder}")
        return True

    except Exception as e:
        logger.error(f"Failed to cleanup {folder}: {str(e)}")
        return False