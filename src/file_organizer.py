import os
import zipfile
import pandas as pd
from pathlib import Path
from logger import setup_logger

logger = setup_logger()

class FileOrganizer:
    def __init__(self, vehicle_df):
        """
        Args:
            vehicle_df: Pre-loaded vehicle dataframe from assign_paths
        """
        self.vehicle_df = vehicle_df
        logger.info(f"FileOrganizer initialized with {len(vehicle_df)} vehicle entries")
        
    def unzip_and_organize(self, working_folder):
        """
        Unzips files and organizes them by bus number in the same folder
        Args:
            working_folder: Path to folder containing zip files
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            working_folder = Path(working_folder)
            logger.info(f"Starting organization in: {working_folder}")

            if not working_folder.exists():
                logger.error(f"Working folder does not exist: {working_folder}")
                return False

            # Process each zip file
            for zip_file in working_folder.glob("*.zip"):
                logger.info(f"Processing {zip_file.name}")
                
                try:
                    # Extract directly to working folder
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(working_folder)
                        logger.info(f"Extracted {zip_file.name}")
                    
                    # Process extracted MAT files
                    self._organize_mat_files(working_folder)
                    
                    # Remove processed zip file
                    zip_file.unlink()
                    logger.info(f"Deleted processed zip: {zip_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {zip_file.name}: {str(e)}")
                    continue
                
            logger.info("File organization completed")
            return True
            
        except Exception as e:
            logger.error(f"Fatal error during organization: {str(e)}")
            return False

    def _organize_mat_files(self, working_folder):
        """Organizes MAT files into bus number folders within the same directory"""
        for mat_file in working_folder.glob("*Z.mat"):
            try:
                # Extract IMEI from filename (first part before underscore)
                imei = mat_file.name.split("_")[0]
                
                # Find matching vehicle
                vehicle_info = self.vehicle_df[
                    self.vehicle_df["IMEI"].astype(str) == imei
                ]
                
                if not vehicle_info.empty:
                    bus_no = vehicle_info.iloc[0]["Bus No"]
                    dest_folder = working_folder / bus_no
                    dest_folder.mkdir(exist_ok=True)
                    
                    # Move file to appropriate folder
                    dest_path = dest_folder / mat_file.name
                    mat_file.rename(dest_path)
                    logger.debug(f"Moved {mat_file.name} to {bus_no} folder")
                else:
                    logger.warning(f"No vehicle found for IMEI: {imei}")
                    
            except Exception as e:
                logger.error(f"Error processing {mat_file.name}: {str(e)}")