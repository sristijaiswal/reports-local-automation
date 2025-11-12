from pathlib import Path
import datetime
import os
import pandas as pd
from cloud_config import CloudConfig

def assign_paths(customer_name):
    """
    Args:
        customer_name: Name of customer (e.g., "KMB", "WestCoast")
    Returns: FolderDir, MainDir, Results_Final, Results_CC, VehicleNamesEV
    """
    # Use cloud-friendly main directory
    MainDir = Path('/app')
    
    # Use /tmp for data folder
    data_folder = CloudConfig.TMP_DIR / "data" / customer_name
    data_folder.mkdir(parents=True, exist_ok=True)
    FolderDir = str(data_folder)
    
    # Get current date and calculate previous week 
    today = datetime.date.today()
    previous_week = today - datetime.timedelta(days=7)
    week_number = previous_week.isocalendar()[1]
    year_number = previous_week.year
    
    # Create Results folder for Final files 
    Results_Final = CloudConfig.TMP_DIR / 'Results' / f'Week{week_number}_{year_number}' / 'Final'
    Results_Final.mkdir(parents=True, exist_ok=True)
    
    # Create Results folder for Customer Copy files 
    Results_CC = CloudConfig.TMP_DIR / 'Results' / f'Week{week_number}_{year_number}' / 'Customer Copy'
    Results_CC.mkdir(parents=True, exist_ok=True)
    
    # Load Vehicle Details from Excel
    excel_path = MainDir / "src" / "VehicleNamesEV.xlsx"
    VehicleNamesEV = pd.read_excel(excel_path)
    
    return FolderDir, MainDir, Results_Final, Results_CC, VehicleNamesEV