from format_excel_files import format_excel_file
from settings import Config
from logger import setup_logger
from api_client import api_call_all_sites
from file_downloader import download_customer_files
from file_organizer import FileOrganizer
from cleanup import cleanup_download_folder
from busOpVarsEV import *
from SummaryEV import *
from assign_paths import *
import time
from s3_helper import *
logger = setup_logger()

from multiprocessing import Process

def run_for_customer(customer_name):
    try:
        logger.info("Starting report generator...")
        
        # Step 1: Assign paths based on customer name
        FolderDir, MainDir, Results_Final, Results_CC, VehicleNamesEV = assign_paths(customer_name)
        
        # Step 2: Make API call to get valid sites
        logger.info("Making API call to get valid sites...")
        valid_entries = api_call_all_sites()
        
        # Step 3: Download customer files
        logger.info(f"Downloading files for {customer_name}...")
        downloaded_folder = download_customer_files(valid_entries, customer_name)
        
        # Step 4: Organize downloaded files
        logger.info("Organizing downloaded files...")
        organizer = FileOrganizer(VehicleNamesEV) 
        organizer.unzip_and_organize(downloaded_folder)

       # Step 5: Process bus operations data
        logger.info("Processing bus operations data...")
        trips, veh_names, veh_count, trip_summaries, daily_summaries, report_name, vehicle_names_ev = busOpVarsEV(
            MainDir, downloaded_folder , Results_Final, VehicleNamesEV
        )

        # Step 6: Generate summary
        logger.info("Generating summary report...")
        SummaryTable, FleetEfficiency = SummaryEV(
            daily_summaries, trip_summaries, vehicle_names_ev, report_name, Results_Final, Results_CC
        )
        
        # Step 7: Format the final Excel file 
        logger.info("Formatting final Excel file...")
        format_excel_file(os.path.join(Results_CC, report_name.replace('.xlsx', '_CC.xlsx')))

        # Step 8: Save files to S3

        logger.info("Saving reports to S3...")
        save_reports_to_s3(customer_name, Results_Final, Results_CC, report_name)


        # Step 9: Cleanup temporary files
        logger.info("Cleaning up download folder...")
        
        #cleanup_download_folder(downloaded_folder)
        logger.info("Process completed successfully")

    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise


def main():
    #customers = ["KMB", "Oxford_City_SD","Oxford_Brookes_Low_Height","Oxford_City_Sightseeing"]
    customers = ["Oxford_City_Sightseeing"]
    processes = []

    for customer in customers:
        p = Process(target=run_for_customer, args=(customer,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all to finish

if __name__ == "__main__":
    main()
