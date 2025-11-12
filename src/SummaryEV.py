import os
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def SummaryEV(DailySummaries, TripSummaries, VehicleNamesEV, report_name, Results_Final, Results_CC):
   
    logger.info('Summary Code Started')

    # Replacing NaN with zeros
    numeric_cols_daily = DailySummaries.select_dtypes(include=[np.number]).columns
    numeric_cols_trip = TripSummaries.select_dtypes(include=[np.number]).columns
    
    DailySummaries.loc[:, numeric_cols_daily] = DailySummaries[numeric_cols_daily].fillna(0)
    TripSummaries.loc[:, numeric_cols_trip] = TripSummaries[numeric_cols_trip].fillna(0)

    # Preallocating the Summary Table
    Names = ['Bus ID', 'IMEI Number', 'Bus Job Number', 'Bus Fleet Number', 'Bus Reg', 'Distance Driven (km)',
             'Energy Consumption (kWh)', 'Efficiency (kWh/km)', 'Regen Energy (kWh)', 'Total Time(min)', 
             'Total Drive Time(min)', 'Total Charge Time (min)', 'Total Preconditioning Time (min)', 
             'Total Idle Time (min)', 'Average Idle Time (%)', 'Average Speed (kph)', 'Mean Ambient Temp (°C)', 
             'Distance Driven >1 (km)', 'Drive Energy >1 (kWh)', 'Energy Consumption >1 (kWh/km)', 
             'Distance Driven <1 (km)', 'Drive Energy <1 (kWh)', 'Energy Consumption <1 (kWh/km)', 
             'Odometer Reading (km)']

    
    #UniqueBus = DailySummaries['VIN'].unique()
    #UniqueBus = VehicleNamesEV.iloc[:, 1].unique()

    first_vin = DailySummaries['VIN'].iloc[0]
    vin_match = VehicleNamesEV.iloc[:, 1] == first_vin
    customer_name = VehicleNamesEV.loc[vin_match, VehicleNamesEV.columns[3]].values[0] 
    customer_vehicles = VehicleNamesEV[VehicleNamesEV.iloc[:, 3] == customer_name]
    UniqueBus = customer_vehicles.iloc[:, 1].unique()   
    
    rows = len(UniqueBus)
    SummaryTable = pd.DataFrame(columns=Names)
    SummaryTable['Bus ID'] = UniqueBus

    for i in range(len(UniqueBus)):
        UniqueBusID = str(UniqueBus[i])
        
        # Use pandas DataFrame indexing
        indiEV = VehicleNamesEV.iloc[:, 1] == UniqueBusID
        
        if any(indiEV):
            SummaryTable.loc[i, 'IMEI Number'] = str(VehicleNamesEV.loc[indiEV, VehicleNamesEV.columns[2]].values[0])
            SummaryTable.loc[i, 'Bus Job Number'] = VehicleNamesEV.loc[indiEV, VehicleNamesEV.columns[0]].values[0]
            SummaryTable.loc[i, 'Bus Fleet Number'] = VehicleNamesEV.loc[indiEV, VehicleNamesEV.columns[4]].values[0]
            SummaryTable.loc[i, 'Bus Reg'] = VehicleNamesEV.loc[indiEV, VehicleNamesEV.columns[5]].values[0]
        else:
            SummaryTable.loc[i, 'IMEI Number'] = ""
            SummaryTable.loc[i, 'Bus Job Number'] = ""
            SummaryTable.loc[i, 'Bus Fleet Number'] = ""
            SummaryTable.loc[i, 'Bus Reg'] = ""

        # Get data for this specific bus
        indices = DailySummaries['VIN'] == UniqueBusID
        
        if np.sum(indices) > 0:
            SummaryTable.loc[i, 'Distance Driven (km)'] = DailySummaries.loc[indices, 'Distance (km)'].sum()
            SummaryTable.loc[i, 'Energy Consumption (kWh)'] = DailySummaries.loc[indices, 'Drive Energy (kWh)'].sum()
            
            if SummaryTable.loc[i, 'Distance Driven (km)'] > 0:
                SummaryTable.loc[i, 'Efficiency (kWh/km)'] = round(SummaryTable.loc[i, 'Energy Consumption (kWh)'] / SummaryTable.loc[i, 'Distance Driven (km)'], 2)
            else:
                SummaryTable.loc[i, 'Efficiency (kWh/km)'] = 0
                
            SummaryTable.loc[i, 'Regen Energy (kWh)'] = DailySummaries.loc[indices, 'Battery Regen Energy (kWh)'].sum()
            
            SummaryTable.loc[i, 'Total Time(min)'] = DailySummaries.loc[indices, 'Total Time (mins)'].sum()
            SummaryTable.loc[i, 'Total Drive Time(min)'] = DailySummaries.loc[indices, 'Drive Time (mins)'].sum()
            
            SummaryTable.loc[i, 'Total Charge Time (min)'] = DailySummaries.loc[indices, 'Charge Time (mins)'].sum()
            
            # Check if column exists before accessing it
            if 'Precon Time (mins)' in DailySummaries.columns:
                SummaryTable.loc[i, 'Total Preconditioning Time (min)'] = DailySummaries.loc[indices, 'Precon Time (mins)'].sum()
            else:
                SummaryTable.loc[i, 'Total Preconditioning Time (min)'] = 0
            
            # Calculate idle time from percentage
            bus_daily = DailySummaries[indices]
            if len(bus_daily) > 0 and SummaryTable.loc[i, 'Total Time(min)'] > 0:
                idle_time_total = (bus_daily['Total Time (mins)'] * (bus_daily['Idle Time (%)'] / 100)).sum()
                SummaryTable.loc[i, 'Total Idle Time (min)'] = round(idle_time_total, 2)
                SummaryTable.loc[i, 'Average Idle Time (%)'] = round((idle_time_total / SummaryTable.loc[i, 'Total Time(min)']) * 100, 2)
            else:
                SummaryTable.loc[i, 'Total Idle Time (min)'] = 0
                SummaryTable.loc[i, 'Average Idle Time (%)'] = 0
               
            # Calculate average speed
            if SummaryTable.loc[i, 'Total Drive Time(min)'] > 0:
                SummaryTable.loc[i, 'Average Speed (kph)'] = round(SummaryTable.loc[i, 'Distance Driven (km)'] / (SummaryTable.loc[i, 'Total Drive Time(min)'] / 60), 2)
            else:
                SummaryTable.loc[i, 'Average Speed (kph)'] = 0
                
            # Calculate weighted ambient temperature
            if len(bus_daily) > 0 and SummaryTable.loc[i, 'Total Drive Time(min)'] > 0:
                ambient_temp_weighted = (bus_daily['Drive Time (mins)'] * bus_daily['Mean Ambient Temp (°C)']).sum()
                SummaryTable.loc[i, 'Mean Ambient Temp (°C)'] = round(ambient_temp_weighted / SummaryTable.loc[i, 'Total Drive Time(min)'], 2)
            else:
                SummaryTable.loc[i, 'Mean Ambient Temp (°C)'] = 0
            
            # Trips >1 km
            trip_indices = (TripSummaries['VIN'] == UniqueBusID) & (TripSummaries['Distance (km)'] >= 1)
            SummaryTable.loc[i, 'Distance Driven >1 (km)'] = TripSummaries.loc[trip_indices, 'Distance (km)'].sum()
            SummaryTable.loc[i, 'Drive Energy >1 (kWh)'] = TripSummaries.loc[trip_indices, 'Drive Energy (kWh)'].sum()
            
            if SummaryTable.loc[i, 'Distance Driven >1 (km)'] > 0:
                SummaryTable.loc[i, 'Energy Consumption >1 (kWh/km)'] = round(SummaryTable.loc[i, 'Drive Energy >1 (kWh)'] / SummaryTable.loc[i, 'Distance Driven >1 (km)'], 2)
            else:
                SummaryTable.loc[i, 'Energy Consumption >1 (kWh/km)'] = 0
            
            # Trips <1 km
            trip_indices_lt1 = (TripSummaries['VIN'] == UniqueBusID) & (TripSummaries['Distance (km)'] < 1)
            SummaryTable.loc[i, 'Distance Driven <1 (km)'] = TripSummaries.loc[trip_indices_lt1, 'Distance (km)'].sum()
            SummaryTable.loc[i, 'Drive Energy <1 (kWh)'] = TripSummaries.loc[trip_indices_lt1, 'Drive Energy (kWh)'].sum()
            
            if SummaryTable.loc[i, 'Distance Driven <1 (km)'] > 0:
                efficiency_lt1 = SummaryTable.loc[i, 'Drive Energy <1 (kWh)'] / SummaryTable.loc[i, 'Distance Driven <1 (km)']
                if efficiency_lt1 == float('inf'):
                    SummaryTable.loc[i, 'Energy Consumption <1 (kWh/km)'] = "-"
                else:
                    SummaryTable.loc[i, 'Energy Consumption <1 (kWh/km)'] = round(efficiency_lt1, 2)
            else:
                SummaryTable.loc[i, 'Energy Consumption <1 (kWh/km)'] = 0
            
            # Odometer reading
            odometer_indices = (TripSummaries['VIN'] == UniqueBusID) & (TripSummaries['Odometer Reading (km)'] > 0)
            if np.any(odometer_indices):
                SummaryTable.loc[i, 'Odometer Reading (km)'] = TripSummaries.loc[odometer_indices, 'Odometer Reading (km)'].max()
            else:
                SummaryTable.loc[i, 'Odometer Reading (km)'] = "-"
                
        else:
            # Fill all numeric columns with zeros if no data
            numeric_columns = ['Distance Driven (km)', 'Energy Consumption (kWh)', 'Efficiency (kWh/km)', 
                              'Regen Energy (kWh)', 'Total Time(min)', 'Total Drive Time(min)', 
                              'Total Charge Time (min)', 'Total Preconditioning Time (min)', 'Total Idle Time (min)',
                              'Average Idle Time (%)', 'Average Speed (kph)', 
                              'Mean Ambient Temp (°C)', 'Distance Driven >1 (km)', 'Drive Energy >1 (kWh)', 
                              'Energy Consumption >1 (kWh/km)', 'Distance Driven <1 (km)', 
                              'Drive Energy <1 (kWh)', 'Energy Consumption <1 (kWh/km)', 'Odometer Reading (km)']
            
            for col in numeric_columns:
                if col in SummaryTable.columns:
                    SummaryTable.loc[i, col] = 0

    # Creating Fleet Efficiency table
    FENames = ['Total Distance (km)', 'Total Energy Consumption (kWh)', 'Total Charge Energy (kWh)', 
               'Mean Ambient Temp (°C)', 'Min Ambient Temp (°C)', 'Max Ambient Temp (°C)', 
               'Average Speed (kph)', 'Efficiency (kWh/km)']
    
    FleetEfficiency = pd.DataFrame(columns=FENames)
    
    FleetEfficiency.loc[0, 'Total Distance (km)'] = SummaryTable['Distance Driven (km)'].sum()
    FleetEfficiency.loc[0, 'Total Energy Consumption (kWh)'] = SummaryTable['Energy Consumption (kWh)'].sum()
    FleetEfficiency.loc[0, 'Total Charge Energy (kWh)'] = DailySummaries['Charge Energy (kWh)'].sum()
    
    # Calculate weighted ambient temperature for fleet
    drive_time_total = DailySummaries['Drive Time (mins)'].sum()
    if drive_time_total > 0:
        ambient_weighted = (DailySummaries['Mean Ambient Temp (°C)'] * DailySummaries['Drive Time (mins)']).sum()
        FleetEfficiency.loc[0, 'Mean Ambient Temp (°C)'] = round(ambient_weighted / drive_time_total, 2)
    else:
        FleetEfficiency.loc[0, 'Mean Ambient Temp (°C)'] = 0
        
    # Min and max temperatures
    min_temp_nonzero = DailySummaries.loc[DailySummaries['Min Ambient Temp (°C)'] != 0, 'Min Ambient Temp (°C)']
    FleetEfficiency.loc[0, 'Min Ambient Temp (°C)'] = min_temp_nonzero.min() if len(min_temp_nonzero) > 0 else 0
    FleetEfficiency.loc[0, 'Max Ambient Temp (°C)'] = DailySummaries['Max Ambient Temp (°C)'].max()
    
    # Average speed for fleet
    total_drive_time_min = SummaryTable['Total Drive Time(min)'].sum()
    if total_drive_time_min > 0:
        FleetEfficiency.loc[0, 'Average Speed (kph)'] = round(FleetEfficiency.loc[0, 'Total Distance (km)'] / (total_drive_time_min / 60), 2)
    else:
        FleetEfficiency.loc[0, 'Average Speed (kph)'] = 0
        
    # Fleet efficiency
    if FleetEfficiency.loc[0, 'Total Distance (km)'] > 0:
        FleetEfficiency.loc[0, 'Efficiency (kWh/km)'] = round(FleetEfficiency.loc[0, 'Total Energy Consumption (kWh)'] / FleetEfficiency.loc[0, 'Total Distance (km)'], 2)
    else:
        FleetEfficiency.loc[0, 'Efficiency (kWh/km)'] = 0

    # Creating summary rows (Total, Average, Efficiency)
    nanTable = pd.DataFrame(np.nan, index=range(4), columns=SummaryTable.columns)
    nanTable['Bus ID'] = nanTable['Bus ID'].astype(object)
    nanTable.iloc[0, 0] = np.nan
    nanTable.iloc[1, 0] = 'Total'
    nanTable.iloc[2, 0] = 'Average'
    nanTable.iloc[3, 0] = 'Efficiency (kWh/km)'

    # Calculate sum and average for numeric columns (starting from column index 5)
    for col_idx in range(5, len(SummaryTable.columns)):
        col_name = SummaryTable.columns[col_idx]
        numeric_col = pd.to_numeric(SummaryTable[col_name], errors='coerce')
        # Check if we have any valid numeric values
        if not numeric_col.isna().all():
            nanTable.iloc[1, col_idx] = round(numeric_col.sum(), 2)
            nanTable.iloc[2, col_idx] = round(numeric_col.mean(), 2)

    # FIRST set specific columns to NaN
        nanTable.iloc[:, 7] = np.nan  # Efficiency column (Var8)
        nanTable.iloc[:, 14] = np.nan  # Var15
        nanTable.iloc[:, 15] = np.nan  # Var16  
        nanTable.iloc[:, 16] = np.nan  # Var17
        nanTable.iloc[:, 19] = np.nan  # Var20
        nanTable.iloc[:, 22] = np.nan  # Var23
        nanTable.iloc[:, 23] = np.nan  # Var24

        # NOW do the specific calculations
        if nanTable.iloc[1, 5] > 0:  # Total Distance (column index 5)
            nanTable.iloc[3, 7] = round(nanTable.iloc[1, 6] / nanTable.iloc[1, 5], 2)  # Efficiency (only row 3)

        if nanTable.iloc[1, 10] > 0:  # Total Drive Time (column index 10)
            nanTable.iloc[2, 15] = round(nanTable.iloc[1, 5] / (nanTable.iloc[1, 10] / 60), 2)  # Average Speed (only row 2)

        # Weighted ambient temperature (only row 2)
        ambient_weighted_total = (DailySummaries['Mean Ambient Temp (°C)'] * DailySummaries['Drive Time (mins)']).sum()
        drive_time_total = DailySummaries['Drive Time (mins)'].sum()
        if drive_time_total > 0:
            nanTable.iloc[2, 16] = round(ambient_weighted_total / drive_time_total, 2)

        if nanTable.iloc[1, 17] > 0:  # Distance >1 km (column index 17)
            nanTable.iloc[3, 19] = round(nanTable.iloc[1, 18] / nanTable.iloc[1, 17], 2)  # Efficiency >1 km (only row 3)

        combinedTable = pd.concat([SummaryTable, nanTable], ignore_index=True)
    
    with pd.ExcelWriter(os.path.join(Results_Final, report_name)) as writer:
        combinedTable.to_excel(writer, sheet_name="SummaryTable", index=False)
        DailySummaries.to_excel(writer, sheet_name="Daily Summary", index=False)
        TripSummaries.to_excel(writer, sheet_name="Trip Summary", index=False)
        FleetEfficiency.to_excel(writer, sheet_name="Fleet Efficiency", index=False)

    # Create customer copy versions by removing specific columns
    CombinedTable_CC = combinedTable.copy()
    DailySummaries_CC = DailySummaries.copy()
    TripSummaries_CC = TripSummaries.copy()

    # Remove columns from CombinedTable_CC
    columns_to_remove_combined = ['Total Charge Time (min)', 'Total Preconditioning Time (min)', 'Total Idle Time (min)']
    CombinedTable_CC = CombinedTable_CC.drop(columns=[col for col in columns_to_remove_combined if col in CombinedTable_CC.columns])

    # Remove columns from DailySummaries_CC
    columns_to_remove_daily = ['Motor Regen Energy (kWh)', 'Charge Time (mins)', 'AvgAPP (%)', 'AvgBPP (%)', 
                            'Mean LSE Temp (°C)', 'Min LSE Temp (°C)', 'Max LSE Temp (°C)', 
                            'Mean UDC Temp (°C)', 'Min UDC Temp (°C)', 'Max UDC Temp (°C)', 
                            'Door 1', 'Door 2']
    DailySummaries_CC = DailySummaries_CC.drop(columns=[col for col in columns_to_remove_daily if col in DailySummaries_CC.columns])

    # Remove columns from TripSummaries_CC
    columns_to_remove_trip = ['Day No.', 'Efficiency (kWh/km)', '24V System Energy (kWh)', 'HV System Energy (kWh)', 
                            'Motor Regen Energy (kWh)', 'Charge Time (mins)', 'AvgAPP (%)', 'AvgBPP (%)', 
                            'Mean LSE Temp (°C)', 'Min LSE Temp (°C)', 'Max LSE Temp (°C)', 
                            'Mean UDC Temp (°C)', 'Min UDC Temp (°C)', 'Max UDC Temp (°C)', 
                            'Door 1', 'Door 2']
    
    TripSummaries_CC = TripSummaries_CC.drop(columns=[col for col in columns_to_remove_trip if col in TripSummaries_CC.columns])

    # Create customer copy report name
    report_name_CC = report_name.replace('.xlsx', '_CC.xlsx')

    with pd.ExcelWriter(os.path.join(Results_CC, report_name_CC)) as writer:  
        CombinedTable_CC.to_excel(writer, sheet_name="SummaryTable", index=False)
        DailySummaries_CC.to_excel(writer, sheet_name="Daily Summary", index=False)
        TripSummaries_CC.to_excel(writer, sheet_name="Trip Summary", index=False)
        FleetEfficiency.to_excel(writer, sheet_name="Fleet Efficiency", index=False)

    return SummaryTable, FleetEfficiency