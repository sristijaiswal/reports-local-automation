# src/customer_processor.py
from format_excel_files import format_excel_file
from logger import setup_logger
from api_client import api_call_all_sites
from file_downloader import download_customer_files
from file_organizer import FileOrganizer
from busOpVarsEV import *
from SummaryEV import *
from assign_paths import *
from cleanup import *
from s3_helper import *
import os

logger = setup_logger()

def process_customer(customer_name):

    downloaded_folder = None

    try:
        """Process a single customer"""
        logger.info(f"Starting processing for: {customer_name}")

       # Setup paths and load vehicle data
        FolderDir, MainDir, Results_Final, Results_CC, VehicleNamesEV = assign_paths(customer_name)
        
        # Get valid sites and download files
        valid_entries = api_call_all_sites()
        downloaded_folder = download_customer_files(valid_entries, customer_name)
        
        # Organize and process files
        organizer = FileOrganizer(VehicleNamesEV)
        organizer.unzip_and_organize(downloaded_folder)
        
        # Generate reports
        trips, veh_names, veh_count, trip_summaries, daily_summaries, report_name, vehicle_names_ev = busOpVarsEV(
            MainDir, downloaded_folder, Results_Final, VehicleNamesEV
        )
        
        SummaryTable, FleetEfficiency = SummaryEV(
            daily_summaries, trip_summaries, vehicle_names_ev, report_name, Results_Final, Results_CC
        )
        
        # Format and save
        format_excel_file(os.path.join(Results_CC, report_name.replace('.xlsx', '_CC.xlsx')))
        save_reports_to_s3(customer_name, Results_Final, Results_CC, report_name)

    finally:
        logger.info(f"Cleaning up temporary files for {customer_name}")
        cleanup_download_folder(downloaded_folder)
        logger.info(f"Cleanup completed for {customer_name}")

        
    logger.info(f"Completed processing for: {customer_name}")