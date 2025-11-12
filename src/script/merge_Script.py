import os
import pandas as pd
from datetime import datetime

def merge_bus_data(input_folder, sheet_name="Daily Summary", output_folder=None):
    """
    Merge bus trip data from multiple Excel files with specific columns.
    
    Args:
        input_folder (str): Path to folder containing Excel files
        sheet_name (str): Name of the sheet to merge (default: 'Trip Summary')
        output_folder (str, optional): Output directory (defaults to input folder)
    """
    
    # Define the exact columns we need (in the order you specified)
    REQUIRED_COLUMNS = [
        'VIN', 'Bus Name', 'Start Time (dd/mm/yyyy hh:mm:ss)',
        'Stop Time (dd/mm/yyyy hh:mm:ss)', 'Day No.', 'Drive Time (mins)',
        'Total Time (mins)', 'Idle Time (%)', 'Average Speed (kph)',
        'Distance (km)', 'Odometer Reading (km)', 'Estimated Range (km)',
        'State Of Health (%)', 'Efficiency (kWh/km)', '24V System Energy (kWh)',
        'HV System Energy (kWh)', 'Charge Energy (kWh)', 'Drive Energy (kWh)',
        'Battery Energy (kWh)', 'Battery Regen Energy (kWh)',
        'Battery Discharge Energy (kWh)', 'Motor Energy (kWh)',
        'Motor Discharge Energy (kWh)', 'Min SOC (%)', 'Max SOC (%)',
        'Start SOC (%)', 'End SOC (%)', 'Max Cell Temp (°C)',
        'Min Cell Temp (°C)', 'Min Ambient Temp (°C)', 'Max Ambient Temp (°C)',
        'Mean Ambient Temp (°C)'
    ]
    
    # Set output folder if not provided
    output_folder = output_folder or input_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all Excel files
    excel_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.xlsx', '.xls', '.xlsm')) and not f.startswith('~$')]
    
    if not excel_files:
        print("No Excel files found in the specified folder.")
        return
    
    print(f"Found {len(excel_files)} Excel files to process.")
    print(f"Looking for sheet: '{sheet_name}'")
    
    # Process files
    all_data = []
    missing_columns = set()
    
    for file in excel_files:
        file_path = os.path.join(input_folder, file)
        try:
            with pd.ExcelFile(file_path) as xls:
                # Check if sheet exists
                if sheet_name not in xls.sheet_names:
                    available = ", ".join(xls.sheet_names)
                    print(f"'{file}' is missing sheet '{sheet_name}'. Available sheets: {available}")
                    continue
                
                # Read the sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Handle efficiency column - check for either version
                efficiency_found = False
                if 'Efficiency (kWh/km)' in df.columns:
                    efficiency_found = True
                elif 'Efficiency (kWh)' in df.columns:
                    # Rename to standardize column name
                    df = df.rename(columns={'Efficiency (kWh)': 'Efficiency (kWh/km)'})
                    efficiency_found = True
                
                # Track missing columns
                missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                
                # Don't report efficiency as missing if we found either version
                if efficiency_found and 'Efficiency (kWh/km)' in missing:
                    missing.remove('Efficiency (kWh/km)')
                
                if missing:
                    missing_columns.update(missing)
                    print(f"'{file}' is missing columns: {', '.join(missing)}")
                
                # Select only the required columns that exist
                available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
                df = df[available_cols]
                
                # Add source file tracking
                df['Source_File'] = os.path.basename(file)
                all_data.append(df)
                
        except Exception as e:
            print(f"Error processing '{file}': {str(e)}")
            continue
    
    if not all_data:
        print(f"No valid data found in files with sheet '{sheet_name}'.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Report on missing columns
    if missing_columns:
        print("\nWarning: These columns were missing in some files:")
        for col in sorted(missing_columns):
            print(f"- {col}")
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Consolidated_Bus_Data_{timestamp}.xlsx"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save to Excel (only with columns that exist in the data)
    combined_df.to_excel(output_path, index=False, sheet_name=sheet_name)
    
    print(f"\nSuccessfully processed {len(all_data)} files.")
    print(f"Final dataset contains {len(combined_df)} rows.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Get user inputs
    input_folder = input("Enter the path to the folder containing Excel files: ").strip()
    
    if not os.path.isdir(input_folder):
        print("Error: The specified folder does not exist.")
    else:
        sheet_name = input("Enter the sheet name to merge (default: 'Trip Summary'): ").strip() or "Trip Summary"
        output_folder = input("Enter output folder path (leave blank to use input folder): ").strip() or None
        
        merge_bus_data(
            input_folder=input_folder,
            sheet_name=sheet_name,
            output_folder=output_folder
        )