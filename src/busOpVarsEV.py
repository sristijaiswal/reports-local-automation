import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.io as sio
from typing import Dict, List, Tuple, Any
from mat2odos import mat2odos
from odos2mat import odos2mat
import logging
from safe_nan_statistic import safe_nan_statistic
from MBMSSOCCalc_2 import *

logger = logging.getLogger(__name__)

def busOpVarsEV(main_dir: str, folder_dir: str, results_dir: str, vehicle_names_ev: pd.DataFrame) -> Tuple:
    
    logger.info("Busop Vars EV Code started")

    # Define trip variable names and types
    trip_var_names = [
        "VIN", "Start Time (dd/mm/yyyy hh:mm:ss)", "Stop Time (dd/mm/yyyy hh:mm:ss)", "Day No.", "Distance (km)", "Odometer Reading (km)",
        "Battery Energy (kWh)", "Drive Energy (kWh)", "Charge Energy (kWh)", "Motor Energy (kWh)", "24V System Energy (kWh)", "HV System Energy (kWh)",
        "Idle Time (%)", "Efficiency (kWh/km)", "AvgAPP (%)", "Total Time (mins)", "Drive Time (mins)", "Charge Time (mins)",
        "Average Speed (kph)", "Mean Ambient Temp (°C)", "Min Ambient Temp (°C)", "Max Ambient Temp (°C)", "Max Cell Temp (°C)", "Min Cell Temp (°C)",
        "Min SOC (%)", "Max SOC (%)", "Start SOC (%)", "End SOC (%)", "Charge Started Time (min)", "ChargeMode Interrupted Shutdown",
        "ChargeMode Failure", "ChargeMode Charging Completed", "Lowest State Of Charge", "AvgBPP (%)", "Precon Time (mins)", "Neutral Time (mins)",
        "Motor Regen Energy (kWh)", "Battery Regen Energy (kWh)", "Motor Discharge Energy (kWh)", "Battery Discharge Energy (kWh)", "Odometer Reading start", "Bus Name",
        "State Of Health (%)", "Throughput", "Throughput (%)", "Estimated Range (km)", "Mean LSE Temp (°C)", "Min LSE Temp (°C)",
        "Max LSE Temp (°C)", "Mean UDC Temp (°C)", "Min UDC Temp (°C)", "Max UDC Temp (°C)", "Door 1", "Door 2",
        "Displayed Min SOC (%)", "Displayed Max SOC (%)", "Precon Energy (kWh)", "Act Charge Time (mins)", "Charge Energy (kW)",
        "LAT_Start","LAT_End","LONG_Start","LONG_End","GPS_ALTITUDE_Start","GPS_ALTITUDE_End"
    ]

    trip_var_types = [
        'str', 'datetime64[ns]', 'datetime64[ns]', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'datetime64[ns]', 'datetime64[ns]',
        'datetime64[ns]', 'datetime64[ns]', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64', 'str',
        'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64','float64'
    ]

    # Create empty DataFrame with specified columns and types
    trip_summaries = pd.DataFrame(columns=trip_var_names)
    for col, dtype in zip(trip_var_names, trip_var_types):
        trip_summaries[col] = trip_summaries[col].astype(dtype)

    # Initialize data structures
    trips = {}
    total_no_trips = 0

    veh_names = [d for d in os.listdir(folder_dir) 
                if os.path.isdir(os.path.join(folder_dir, d))]
    
    veh_count = len(veh_names)

    for veh_no, veh_name in enumerate(veh_names, 1):
        veh_path = os.path.join(folder_dir, veh_name)
        trip_files = [f for f in os.listdir(veh_path) 
              if f.endswith('.mat') and 'Z' in f]

        # Initialize vehicle in trips dictionary
        trips[veh_name] = {}

        for trip_file in trip_files:
            trip_name = 'T' + os.path.splitext(trip_file)[0]
            mat_file_path = os.path.join(veh_path, trip_file)
            
            try:
                mat_data = sio.loadmat(mat_file_path, simplify_cells=True)
            except Exception as e:
                logger.error(f"Error loading {mat_file_path}: {str(e)}")
                continue

            # Store variable names and data
            trips[veh_name][trip_name] = {
                'var_names': list(mat_data.keys()),
                #'data': mat_data
            }

            logger.info(f"Vehicle No: {veh_no}  Trip No: {len(trips[veh_name])}")

            # Create new row in trip_summaries
            row_idx = len(trip_summaries)
            
            trip_summaries.at[row_idx, 'VIN'] = mat_data.get('VIN', '')

            ## Vehicle matching
            imei = str(mat_data.get('IMEI', '')).strip()
            vehicle_imeis = vehicle_names_ev.iloc[:, 2].astype(str).str.strip() 
            matching_rows = vehicle_names_ev[vehicle_imeis == imei]

            if matching_rows.empty:
                logger.warning(f"No match found for IMEI: {imei}")
            else:
                vin = matching_rows.iloc[0, 1] 
                trip_summaries.at[row_idx, 'VIN'] = vin
                
                vin_match = vehicle_names_ev[vehicle_names_ev.iloc[:, 1] == vin]
                if not vin_match.empty:
                    trip_summaries.at[row_idx, 'Bus Name'] = vin_match.iloc[0, 4]
                    Customer = vin_match['Customer'].iloc[0]


            
            ##  datetime conversion
            start_time = mat_data.get('START_TIME', '')
            stop_time = mat_data.get('STOP_TIME', '')
            
            try:
                trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)'] = (
                    datetime.strptime(start_time, "%d/%m/%Y %H:%M:%S") if start_time else pd.NaT
                )
                trip_summaries.at[row_idx, 'Stop Time (dd/mm/yyyy hh:mm:ss)'] = (
                    datetime.strptime(stop_time, "%d/%m/%Y %H:%M:%S") if stop_time else pd.NaT
                )
                
                # Calculate day number
                if not pd.isna(trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)']):
                    trip_summaries.at[row_idx, 'Day No.'] = (
                        trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)'].day
                    )
                
                # Calculate total time in minutes
                if (not pd.isna(trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)'])) and \
                   (not pd.isna(trip_summaries.at[row_idx, 'Stop Time (dd/mm/yyyy hh:mm:ss)'])):
                    time_diff = (
                        trip_summaries.at[row_idx, 'Stop Time (dd/mm/yyyy hh:mm:ss)'] - 
                        trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)']
                    )
                    trip_summaries.at[row_idx, 'Total Time (mins)'] = round(time_diff.total_seconds() / 60, 2)
                
            except Exception as e:
                logger.error(f"Error processing datetime for {trip_file}: {str(e)}")
        
            try:
                # Calculate time array in ODOS format 
                start_dt = trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)'].to_pydatetime()
                stop_dt = trip_summaries.at[row_idx, 'Stop Time (dd/mm/yyyy hh:mm:ss)'].to_pydatetime()
                
                start_odos = mat2odos(start_dt)
                stop_odos = mat2odos(stop_dt)
                time = np.arange(start_odos, stop_odos + 1e7, 1e7)  # 1e7 ns = 1 second steps
                
            except Exception as e:
                logger.error(f"Error creating time array: {str(e)}")
                time = np.array([])

            ## GPS data processing
            gps_signals = ['LAT', 'LONG', 'GPS_ALTITUDE']

            for signal in gps_signals:
                data_field = signal
                time_field = f'{signal}_X'
                
                # Initialize with default values
                trip_summaries.at[row_idx, f'{signal}_Start'] = 0.0
                trip_summaries.at[row_idx, f'{signal}_End'] = 0.0
                
                if data_field in mat_data and time_field in mat_data:
                    raw_data = np.atleast_1d(mat_data[data_field])
                    raw_time = np.atleast_1d(mat_data[time_field])
                    
                    # Remove NaN values
                    valid_indices = ~np.isnan(raw_data)
                    clean_data = raw_data[valid_indices]
                    clean_time = raw_time[valid_indices]
                    
                    # Get unique time points
                    if len(clean_data) > 2:
                        unique_times, unique_indices = np.unique(clean_time, return_index=True)
                        
                        if len(unique_indices) > 2:
                            # Get first and last values
                            start_value = clean_data[unique_indices[0]]
                            end_value = clean_data[unique_indices[-1]]
                            
                            trip_summaries.at[row_idx, f'{signal}_Start'] = start_value
                            trip_summaries.at[row_idx, f'{signal}_End'] = end_value

            ## Distance calculation
            if 'VDHR_TotalVehDistance' in mat_data:
                distance_data = mat_data['VDHR_TotalVehDistance']
                if isinstance(distance_data, (list, np.ndarray)) and len(distance_data) > 0:
                    distance = distance_data[-1] - distance_data[0]
                    if distance < 0:
                        distance = 0
                    trip_summaries.at[row_idx, 'Distance (km)'] = distance
                    trip_summaries.at[row_idx, 'Odometer Reading (km)'] = distance_data[-1]
                    trip_summaries.at[row_idx, 'Odometer Reading start'] = distance_data[0]

            ## HV Current
            if 'ActualBatterySystemCurrentAcc' in mat_data:
                battCurr1 = np.atleast_1d(mat_data['ActualBatterySystemCurrentAcc'])
                battCurrTime1 = np.atleast_1d(mat_data['ActualBatterySystemCurrentAcc_X'])
                
                valid_entries = (battCurr1 < 4444) & (~np.isnan(battCurr1))
                battCurr2 = battCurr1[valid_entries]
                battCurrTime = battCurrTime1[valid_entries]

                #battCurr1 = battCurr1[battCurr1<4444]
                #battCurr2 = battCurr1[~np.isnan(battCurr1)]
                #battCurrTime = battCurrTime1[~np.isnan(battCurr1)]
                _, ind = np.unique(battCurrTime, return_index=True)

                if len(battCurr2) > 2:
                    battCurr = np.interp(time, battCurrTime[ind],battCurr2[ind],left =0 , right = 0)

                else:
                    battCurr = np.zeros_like(time)
                    battCurrTime = time

            elif 'B2VST2PackCurrent' in mat_data:
                battCurr1 = np.atleast_1d(mat_data['B2VST2PackCurrent'])
                battCurrTime1 = np.atleast_1d(mat_data['B2VST2PackCurrent_X'])

                # Apply both filters sequentially (like MATLAB)
                valid_indices = (battCurr1 < 4444) & (~np.isnan(battCurr1))
                battCurr2 = battCurr1[valid_indices]
                battCurrTime = battCurrTime1[valid_indices]
                
                _, ind = np.unique(battCurrTime, return_index=True)

                if len(battCurr2) > 2:
                    battCurr = np.interp(time, battCurrTime[ind], battCurr2[ind], left=0, right=0)
                else:
                    battCurr = np.zeros_like(time)
                    battCurrTime = time
            
            else:
                battCurr = np.zeros_like(time)
                battCurrTime = time

            ## HV Voltage

            #if 'ActualBatterySystemVoltageaccuracy' in mat_data:
            if 'terminalVoltageActual' in mat_data:

                #battVolt1 = np.atleast_1d(mat_data['ActualBatterySystemVoltageaccuracy'])
                #battVoltTime1 = np.atleast_1d(mat_data['ActualBatterySystemVoltageaccuracy_X'])
                battVolt1 = np.atleast_1d(mat_data['terminalVoltageActual'])
                battVoltTime1 = np.atleast_1d(mat_data['terminalVoltageActual_X'])
                
                battVolt2 = battVolt1[~np.isnan(battVolt1)]
                battVoltTime = battVoltTime1[~np.isnan(battVolt1)]

                _, ind = np.unique(battVoltTime, return_index=True)

                if len(battVolt2) > 2:
                    battVolt = np.interp(time, battVoltTime[ind],battVolt2[ind],left =0 , right = 0)

                else:
                    battVolt = np.zeros_like(time)
                    battVoltTime = time


            elif 'B2VST2PackInsideVolt' in mat_data:

                battVolt1 = np.atleast_1d(mat_data['B2VST2PackInsideVolt'])
                battVoltTime1 = np.atleast_1d(mat_data['B2VST2PackInsideVolt_X'])

                battVolt2 = battVolt1[~np.isnan(battVolt1)]
                battVoltTime = battVoltTime1[~np.isnan(battVolt1)]

                _, ind = np.unique(battVoltTime, return_index = True)

                if len(battVolt2) > 2:
                    battVolt = np.interp(time,battVoltTime[ind], battVolt2[ind],left =0, right=0)
                else:
                    battVolt = np.zeros_like(time)
                    battVoltTime = time
            
            else:
                battVolt = np.zeros_like(time)
                battVoltTime = time

            ## VEDS State

            if 'VEDS_State__numerical_' in mat_data:
                veds1 = np.atleast_1d(mat_data['VEDS_State__numerical_'])
                vedsTime1 = np.atleast_1d(mat_data['VEDS_State__numerical__X'])
                
                veds2 = veds1[~np.isnan(veds1)]
                vedsTime2 = vedsTime1[~np.isnan(veds1)]
                
                vedsTime3 = vedsTime2[~np.isnan(vedsTime2)]
                veds3 = veds2[~np.isnan(vedsTime2)]  
                
                _, ind = np.unique(vedsTime3, return_index=True)
                
                if len(veds3) > 2:
                    veds = np.interp(time, vedsTime3[ind], veds3[ind], left=0, right=0)
                    # veds = np.round(np.interp(time, vedsTime3[ind], veds3[ind], left=0, right=0))

                else:
                    veds = np.zeros_like(time)
                    vedsTime = time
            else:
                veds = np.zeros_like(time)
                vedsTime = time


            vedsDrive = (veds >= 20) & (veds < 50)
            vedsCharge = (veds >= 50) & (veds < 255)
            vedsPrecon = (veds == 52)
            vedsNeut = (veds >= 0) & (veds < 20)

            timeDrive = np.sum(vedsDrive)  # drive time in seconds (s)
            timePrecon = np.sum(vedsPrecon)  # Preconditioning time in seconds (s)
            timeNeut = np.sum(vedsNeut)  # Neutral time in seconds (s)
            timeCharge = np.sum(vedsCharge)  # charge time in seconds (s)

            ## Accelerator Pedal Position

            if 'Acltr_Pedal_AP_Position' in mat_data:
                accPedal1 = np.atleast_1d(mat_data['Acltr_Pedal_AP_Position'])
                accPedalTime1 = np.atleast_1d(mat_data['Acltr_Pedal_AP_Position_X'])
                
                accPedal2 = accPedal1[~np.isnan(accPedal1)]
                accPedalTime = accPedalTime1[~np.isnan(accPedal1)]

                _, ind = np.unique(accPedalTime, return_index=True)
                
                if len(accPedal2) > 2:
                    accPedal = np.interp(time, accPedalTime[ind], accPedal2[ind], left=0, right=0)

                else:
                    accPedal = np.zeros_like(time)
                    accPedalTime = time

            else:
                accPedal = np.zeros_like(time)
                accPedalTime = time

            driveAccPedal = accPedal[vedsDrive]    # Values during driving mode
            engagedAccPedal = accPedal[accPedal > 0] # Values when pedal is engaged

            ## Brake Pedal Position

            if 'Brake_Pedal_Position' in mat_data:
                bccPedal1 = np.atleast_1d(mat_data['Brake_Pedal_Position'])
                bccPedalTime1 = np.atleast_1d(mat_data['Brake_Pedal_Position_X'])
                
                bccPedal2 = bccPedal1[~np.isnan(bccPedal1)]
                bccPedalTime = bccPedalTime1[~np.isnan(bccPedal1)]

                _, ind = np.unique(bccPedalTime, return_index=True)
                
                if len(bccPedal2) > 2:
                    bccPedal = np.interp(time, bccPedalTime[ind], bccPedal2[ind], left=0, right=0)

                else:
                    bccPedal = np.zeros_like(time)
                    bccPedalTime = time

            else:
                bccPedal = np.zeros_like(time)
                bccPedalTime = time

            driveBccPedal = bccPedal[vedsDrive]    # Values for brake pedal position in driving mode
            engagedBccPedal = bccPedal[bccPedal > 0] # Values for brake pedal position when engaged


            ## Vehicle speed

            if 'TCO1_TachographVehicleSpeed__numerical_' in mat_data:
                vehSpeed1 = np.atleast_1d(mat_data['TCO1_TachographVehicleSpeed__numerical_'])
                vehSpeedTime1 = np.atleast_1d(mat_data['TCO1_TachographVehicleSpeed__numerical__X'])
                
                vehSpeed2 = vehSpeed1[~np.isnan(vehSpeed1)]
                vehSpeedTime = vehSpeedTime1[~np.isnan(vehSpeed1)]

                _, ind = np.unique(vehSpeedTime, return_index=True)
                
                if len(vehSpeed2) > 2:
                    vehSpeed = np.interp(time, vehSpeedTime[ind], vehSpeed2[ind], left=0, right=0)

                else:
                    vehSpeed = np.zeros_like(time)
                    vehSpeedTime = time

            else:
                vehSpeed = np.zeros_like(time)
                vehSpeedTime = time

            avgDriveSpeed = np.mean(vehSpeed)

            ## Idle Time

            timeDiff = np.concatenate(([0], np.diff(time)))  # [0, diff(time)]

            idleTimeIdx = vehSpeed == 0                     # Indices where vehicle is stationary
            driveDuration = np.sum(timeDiff[~idleTimeIdx])   # Time moving (s)
            idleTimeDuration = np.sum(timeDiff[idleTimeIdx]) # Time stationary (s)
            totalTime = np.sum(timeDiff)                     # Total time (s)

            pcIdleTime = (idleTimeDuration - (timeCharge * 1e7)) / totalTime

            ## Motor Power

            if 'TMP_TractionMotorPower' in mat_data:
                motorPower1 = np.atleast_1d(mat_data['TMP_TractionMotorPower'])
                motorPowerTime1 = np.atleast_1d(mat_data['TMP_TractionMotorPower_X'])
                
                motorPower2 = motorPower1[~np.isnan(motorPower1)]
                motorPowerTime2 = motorPowerTime1[~np.isnan(motorPower1)]
                
                motorPower3 = motorPower2 * (motorPower2 < 300000)
                
                _, ind = np.unique(motorPowerTime2, return_index=True)
                
                if len(motorPower3) > 2:
                    motorPower = np.interp(time, motorPowerTime2[ind], motorPower3[ind], left=0, right=0)
                else:
                    motorPower = np.zeros_like(time)
                    motorPowerTime = time
            else:
                motorPower = np.zeros_like(time)
                motorPowerTime = time


            ## Ambient air temperature
            if 'AmbientAirTemp' in mat_data:
                ambientAirTemp1 = np.atleast_1d(mat_data['AmbientAirTemp'])  
                ambientAirTempTime1 = np.atleast_1d(mat_data['AmbientAirTemp_X'])  
                
                ambientAirTemp2 = ambientAirTemp1[~np.isnan(ambientAirTemp1)]
                ambientAirTempTime2 = ambientAirTempTime1[~np.isnan(ambientAirTemp1)]
                
                ambientAirTemp3 = ambientAirTemp2[ambientAirTemp2 != 0]
                ambientAirTempTime = ambientAirTempTime2[ambientAirTemp2 != 0]
                
                ambientAirTempTime[ambientAirTempTime <= -16] = 0
                
                if len(ambientAirTemp3) > 2:
                    ambientAirTemp = np.interp(time, ambientAirTempTime, ambientAirTemp3, left=0, right=0)
                elif len(ambientAirTemp3) == 0:
                    ambientAirTemp = np.zeros_like(time)
                else:
                    ambientAirTemp = np.full_like(time, ambientAirTemp3[0])
                    ambientAirTempTime = time
            else:
                ambientAirTemp = np.full_like(time, np.nan) 
                # ambientAirTemp = np.empty((0, len(time))) 
                ambientAirTempTime = time

            ## Batt Max Cell Temp

            if 'Signal_BMS02_Max_Cell_Temperature' in mat_data:
                MaxCellTemp1 = np.atleast_1d(mat_data['Signal_BMS02_Max_Cell_Temperature'])
                MaxCellTempTime1 = np.atleast_1d(mat_data['Signal_BMS02_Max_Cell_Temperature_X'])
                
                MaxCellTemp1[MaxCellTemp1 > 60] = 28
                MaxCellTemp2 = MaxCellTemp1[~np.isnan(MaxCellTemp1)]
                MaxCellTempTime = MaxCellTempTime1[~np.isnan(MaxCellTemp1)]
                
                _, ind = np.unique(MaxCellTempTime, return_index=True)
                
                if len(MaxCellTemp2) > 2:
                    MaxCellTemp = np.interp(time, MaxCellTempTime[ind], MaxCellTemp2[ind], left=0, right=0)
                else:
                    MaxCellTemp = np.zeros_like(time)
                    MaxCellTempTime = time

            elif 'Signal_Battery_Max_Cell_Temperature_02' in mat_data:
                MaxCellTemp1 = np.atleast_1d(mat_data['Signal_Battery_Max_Cell_Temperature_02'])
                MaxCellTempTime1 = np.atleast_1d(mat_data['Signal_Battery_Max_Cell_Temperature_02_X'])
                
                MaxCellTemp1[MaxCellTemp1 > 60] = 28
                MaxCellTemp2 = MaxCellTemp1[~np.isnan(MaxCellTemp1)]
                MaxCellTempTime = MaxCellTempTime1[~np.isnan(MaxCellTemp1)]
                
                _, ind = np.unique(MaxCellTempTime, return_index=True)
                
                if len(MaxCellTemp2) > 2:
                    MaxCellTemp = np.interp(time, MaxCellTempTime[ind], MaxCellTemp2[ind], left=0, right=0)
                else:
                    MaxCellTemp = np.zeros_like(time)
                    MaxCellTempTime = time

            elif 'B2VST4MaxTemp' in mat_data:
                MaxCellTemp1 = np.atleast_1d(mat_data['B2VST4MaxTemp'])
                MaxCellTempTime1 = np.atleast_1d(mat_data['B2VST4MaxTemp_X'])
                
                MaxCellTemp1[MaxCellTemp1 > 60] = 28
                MaxCellTemp2 = MaxCellTemp1[~np.isnan(MaxCellTemp1)]
                MaxCellTempTime = MaxCellTempTime1[~np.isnan(MaxCellTemp1)]
                
                _, ind = np.unique(MaxCellTempTime, return_index=True)
                
                if len(MaxCellTemp2) > 2:
                    MaxCellTemp = np.interp(time, MaxCellTempTime[ind], MaxCellTemp2[ind], left=0, right=0)
                else:
                    MaxCellTemp = np.zeros_like(time)
                    MaxCellTempTime = time

            else:
                MaxCellTemp = np.zeros_like(time)
                MaxCellTempTime = time

            ## Batt Min Cell Temp
            if 'Signal_BMS02_Min_Cell_Temperature' in mat_data:
                MinCellTemp1 = np.atleast_1d(mat_data['Signal_BMS02_Min_Cell_Temperature'])
                MinCellTempTime1 = np.atleast_1d(mat_data['Signal_BMS02_Min_Cell_Temperature_X'])
                
                MinCellTemp2 = MinCellTemp1[~np.isnan(MinCellTemp1)]
                MinCellTempTime = MinCellTempTime1[~np.isnan(MinCellTemp1)]  # Fixed this line
                
                _, ind = np.unique(MinCellTempTime, return_index=True)
                
                if len(MinCellTemp2) > 2:
                    MinCellTemp = np.interp(time, MinCellTempTime[ind], MinCellTemp2[ind], left=0, right=0)
                else:
                    MinCellTemp = np.zeros_like(time)
                    MinCellTempTime = time

            elif 'Signal_Battery_Min_Cell_Temperature_02' in mat_data:
                MinCellTemp1 = np.atleast_1d(mat_data['Signal_Battery_Min_Cell_Temperature_02'])
                MinCellTempTime1 = np.atleast_1d(mat_data['Signal_Battery_Min_Cell_Temperature_02_X'])
                
                MinCellTemp2 = MinCellTemp1[~np.isnan(MinCellTemp1)]
                MinCellTempTime = MinCellTempTime1[~np.isnan(MinCellTemp1)]
                
                _, ind = np.unique(MinCellTempTime, return_index=True)
                
                if len(MinCellTemp2) > 2:
                    MinCellTemp = np.interp(time, MinCellTempTime[ind], MinCellTemp2[ind], left=0, right=0)
                else:
                    MinCellTemp = np.zeros_like(time)
                    MinCellTempTime = time

            elif 'B2VST4MinTemp' in mat_data:
                MinCellTemp1 = np.atleast_1d(mat_data['B2VST4MinTemp'])
                MinCellTempTime1 = np.atleast_1d(mat_data['B2VST4MinTemp_X'])
                
                MinCellTemp2 = MinCellTemp1[~np.isnan(MinCellTemp1)]
                MinCellTempTime = MinCellTempTime1[~np.isnan(MinCellTemp1)]
                
                _, ind = np.unique(MinCellTempTime, return_index=True)
                
                if len(MinCellTemp2) > 2:
                    MinCellTemp = np.interp(time, MinCellTempTime[ind], MinCellTemp2[ind], left=0, right=0)
                else:
                    MinCellTemp = np.zeros_like(time)
                    MinCellTempTime = time

            else:
                MinCellTemp = np.zeros_like(time)
                MinCellTempTime = time

            ## Lower Saloon Temperature
            if 'LowerSaloonExtTemperature' in mat_data:
                LowerSaloonExtTemp1 = np.atleast_1d(mat_data['LowerSaloonExtTemperature'])
                LowerSaloonExtTempTime1 = np.atleast_1d(mat_data['LowerSaloonExtTemperature_X'])
                
                LowerSaloonExtTemp2 = LowerSaloonExtTemp1[~np.isnan(LowerSaloonExtTemp1)]
                LowerSaloonExtTempTime2 = LowerSaloonExtTempTime1[~np.isnan(LowerSaloonExtTemp1)]
                
                LowerSaloonExtTemp3 = LowerSaloonExtTemp2[LowerSaloonExtTemp2 != 0]
                LowerSaloonExtTempTime = LowerSaloonExtTempTime2[LowerSaloonExtTemp2 != 0]
                
                if len(LowerSaloonExtTemp3) > 2:
                    LowerSaloonExtTemp = np.interp(time, LowerSaloonExtTempTime, LowerSaloonExtTemp3, left=0, right=0)
                elif len(LowerSaloonExtTemp3) == 0:
                    LowerSaloonExtTemp = np.full_like(time, np.nan)  
                    LowerSaloonExtTempTime = time
                else:
                    LowerSaloonExtTemp = np.full_like(time, LowerSaloonExtTemp3[0])  
                    LowerSaloonExtTempTime = time
            else:
                LowerSaloonExtTemp = np.full_like(time, np.nan)  
                LowerSaloonExtTempTime = time

            LowerSaloonExtTemp[LowerSaloonExtTemp > 100] = np.nan 

         ## Upper Deck Cabin Temperature
            if 'UpperDeckCabinTemp' in mat_data:
                UpperDeckCabinTemp1 = np.atleast_1d(mat_data['UpperDeckCabinTemp'])
                UpperDeckCabinTempTime1 = np.atleast_1d(mat_data['UpperDeckCabinTemp_X'])
                
                UpperDeckCabinTemp2 = UpperDeckCabinTemp1[~np.isnan(UpperDeckCabinTemp1)]
                UpperDeckCabinTempTime2 = UpperDeckCabinTempTime1[~np.isnan(UpperDeckCabinTemp1)]
                
                UpperDeckCabinTemp3 = UpperDeckCabinTemp2[UpperDeckCabinTemp2 != 0]
                UpperDeckCabinTempTime = UpperDeckCabinTempTime2[UpperDeckCabinTemp2 != 0]
                
                if len(UpperDeckCabinTemp3) > 2:
                    UpperDeckCabinTemp = np.interp(time, UpperDeckCabinTempTime, UpperDeckCabinTemp3, left=0, right=0)
                elif len(UpperDeckCabinTemp3) == 0:
                    UpperDeckCabinTemp = np.full_like(time, np.nan)  
                    UpperDeckCabinTempTime = time
                else:
                    UpperDeckCabinTemp = np.full_like(time, UpperDeckCabinTemp3[0])  
                    UpperDeckCabinTempTime = time
            else:
                UpperDeckCabinTemp = np.full_like(time, np.nan)  
                UpperDeckCabinTempTime = time

            UpperDeckCabinTemp[UpperDeckCabinTemp > 50] = np.nan 

            ## Door 1 open/close count 

            if 'OpenStatus_Door1' in mat_data:
                OpenStatus_Door11 = np.atleast_1d(mat_data['OpenStatus_Door1'])
                OpenStatus_Door21 = np.atleast_1d(mat_data['OpenStatus_Door2'])
                
                OpenStatus_Door1 = OpenStatus_Door11
                OpenStatus_Door2 = OpenStatus_Door21
            else:
                OpenStatus_Door1 = np.array([0]) 
                OpenStatus_Door2 = np.array([0])

            # Find the indices where the array changes from 0 to 1
            Door1ChangeIndices = np.where(np.diff(np.concatenate(([0], OpenStatus_Door1))) == 1)[0]
            Door2ChangeIndices = np.where(np.diff(np.concatenate(([0], OpenStatus_Door2))) == 1)[0]

            # Count the number of transitions
            Door1Noofoperations = len(Door1ChangeIndices)
            Door2Noofoperations = len(Door2ChangeIndices)

            ## 24V Current

            if 'CB24_CurrentVehicleBattery24V' in mat_data:
                batt24Curr1 = np.atleast_1d(mat_data['CB24_CurrentVehicleBattery24V'])
                batt24CurrTime1 = np.atleast_1d(mat_data['CB24_CurrentVehicleBattery24V_X'])
                
                batt24Curr2 = batt24Curr1[~np.isnan(batt24Curr1)]
                batt24CurrTime = batt24CurrTime1[~np.isnan(batt24Curr1)]

                _, ind = np.unique(batt24CurrTime, return_index=True)
                
                if len(batt24Curr2) > 2:
                    batt24Curr = np.interp(time, batt24CurrTime[ind], batt24Curr2[ind], left=0, right=0)
                else:
                    batt24Curr = np.zeros_like(time)
                    batt24CurrTime = time
            else:
                batt24Curr = np.zeros_like(time)
                batt24CurrTime = time
                    
        

            Throughput = 100000

            ## 24 V Voltage

            batt24Volt = np.ones_like(time) * 27.5
            batt24VoltTime = time

            ## SOC Lowest and Mean

            if 'MBMSStat1_DisplayedSOC' in mat_data:
                SOC1 = np.atleast_1d(mat_data['MBMSStat1_DisplayedSOC'])
                SOCTime1 = np.atleast_1d(mat_data['MBMSStat1_DisplayedSOC_X'])

                SOC2 = SOC1[~np.isnan(SOC1)]
                SOCTime = SOCTime1[~np.isnan(SOC1)]

                _, ind = np.unique(SOCTime, return_index=True)

                if len(SOC2) > 2:
                    DSOC = np.interp(time, SOCTime[ind], SOC2[ind], left=0, right=0)
                else:
                    DSOC = np.full_like(time, np.nan) 
                    DSOCTime = time

            if ('MBMSStat1_GrossSOC' in mat_data) and ('Signal_BMS01_SOC' in mat_data):

                if ('Signal_BMS04_SOC' in mat_data) and ('Signal_BMS02_SOC' in mat_data) and ('Signal_BMS03_SOC' in mat_data):

                    SOC = MBMSSOCCalc_2(trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)'], trip_summaries.at[row_idx, 'Stop Time (dd/mm/yyyy hh:mm:ss)'], # pyright: ignore[reportUndefinedVariable]
                                    np.atleast_1d(mat_data['MBMSStat1_GrossSOC']), np.atleast_1d(mat_data['MBMSStat1_GrossSOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS01_SOC']), np.atleast_1d(mat_data['Signal_BMS01_SOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS02_SOC']), np.atleast_1d(mat_data['Signal_BMS02_SOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS03_SOC']), np.atleast_1d(mat_data['Signal_BMS03_SOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS04_SOC']), np.atleast_1d(mat_data['Signal_BMS04_SOC_X']))
                elif 'Signal_BMS03_SOC' in mat_data:
                    SOC = MBMSSOCCalc_2(trip_summaries.at[row_idx, 'Start Time (dd/mm/yyyy hh:mm:ss)'], trip_summaries.at[row_idx, 'Stop Time (dd/mm/yyyy hh:mm:ss)'], # pyright: ignore[reportUndefinedVariable]
                                    np.atleast_1d(mat_data['MBMSStat1_GrossSOC']), np.atleast_1d(mat_data['MBMSStat1_GrossSOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS01_SOC']), np.atleast_1d(mat_data['Signal_BMS01_SOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS02_SOC']), np.atleast_1d(mat_data['Signal_BMS02_SOC_X']),
                                    np.atleast_1d(mat_data['Signal_BMS03_SOC']), np.atleast_1d(mat_data['Signal_BMS03_SOC_X']))
                else:
                    SOC = np.nan

            elif 'B2VST2SOC' in mat_data:
                SOC1 = np.atleast_1d(mat_data['B2VST2SOC'])
                SOCTime1 = np.atleast_1d(mat_data['B2VST2SOC_X'])
                
                SOC2 = SOC1[~np.isnan(SOC1)]
                SOCTime = SOCTime1[~np.isnan(SOC1)]
                
                _, ind = np.unique(SOCTime, return_index=True)
                
                if len(SOC2) > 2: 
                    SOC = np.interp(time, SOCTime[ind], SOC2[ind], left=0, right=0)
                    # SOC = np.interp(time, SOCTime, SOC2, left=np.nan, right=np.nan)
                else:
                    SOC = np.full_like(time, np.nan)  
                    SOCTime = time
                
                DSOC = SOC.copy()  
            else:
                SOC = np.full_like(time, np.nan)  
                DSOC = np.full_like(time, np.nan)


            SOC = SOC[(~np.isnan(SOC)) & (SOC != 0)]
            DSOC = DSOC[(~np.isnan(DSOC)) & (DSOC != 0)]

            if len(SOC) == 0: 
                MINSOC = np.nan
                MAXSOC = 0
                startSOC = 0
                endSOC = 0
            else:
                MINSOC = round(np.min(SOC), 1)
                MAXSOC = round(np.max(SOC), 1)
                startSOC = SOC[0]  
                endSOC = SOC[-1]  

            if len(DSOC) == 0: 
                MinDSOC = 0
                MaxDSOC = 0
                startDSOC = 0
                endDSOC = 0
            else:
                MinDSOC = round(np.min(DSOC), 1)
                MaxDSOC = round(np.max(DSOC), 1)
                startDSOC = DSOC[0] 
                endDSOC = DSOC[-1]   

            ## Charge Start 

            if 'VEDS_State__numerical_' in mat_data:
                veds_state = np.atleast_1d(mat_data['VEDS_State__numerical_'])
                
                if np.any(veds_state == 51):
                    ChargeStartID = np.where(veds_state == 51)[0][0] 
                    Time_ChargeStart = odos2mat(mat_data['VEDS_State__numerical__X'][ChargeStartID])[1] 
                else:
                    Time_ChargeStart = np.datetime64('NaT')  
            else:
                Time_ChargeStart = np.datetime64('NaT')

            ## ChargeMode Interrupted Shutdown
            if 'VEDS_State__numerical_' in mat_data:
                veds_state = np.atleast_1d(mat_data['VEDS_State__numerical_'])
                
                if np.any(veds_state == 55):
                    ChargeStartID = np.where(veds_state == 55)[0][0]  # find(..., 1, 'first')
                    _, ChargeMode_Interrupted_Shutdown, _ = odos2mat(mat_data['VEDS_State__numerical__X'][ChargeStartID])
                else:
                    ChargeMode_Interrupted_Shutdown = np.datetime64('NaT')
            else:
                ChargeMode_Interrupted_Shutdown = np.datetime64('NaT')

            ## ChargeMode Failure
            if 'VEDS_State__numerical_' in mat_data:
                veds_state = np.atleast_1d(mat_data['VEDS_State__numerical_'])
                
                if np.any(veds_state == 53):
                    ChargeStartID = np.where(veds_state == 53)[0][0]
                    _, ChargeMode_Failure, _ = odos2mat(mat_data['VEDS_State__numerical__X'][ChargeStartID])
                else:
                    ChargeMode_Failure = np.datetime64('NaT')
            else:
                ChargeMode_Failure = np.datetime64('NaT')

            ## ChargeMode Charging Completed
            if 'VEDS_State__numerical_' in mat_data:
                veds_state = np.atleast_1d(mat_data['VEDS_State__numerical_'])
                
                if np.any(veds_state == 54):
                    ChargeStartID = np.where(veds_state == 54)[0][0]
                    _, ChargeMode_ChargingCompleted, _ = odos2mat(mat_data['VEDS_State__numerical__X'][ChargeStartID])
                else:
                    ChargeMode_ChargingCompleted = np.datetime64('NaT')
            else:
                ChargeMode_ChargingCompleted = np.datetime64('NaT')

            ## Precondition time calculation

            if 'PreConditioningMode' in mat_data:
                Precon1 = np.atleast_1d(mat_data['PreConditioningMode'])
                PreconTime1 = np.atleast_1d(mat_data['PreConditioningMode_X'])
                
                Precon2 = Precon1[~np.isnan(Precon1)]
                PreconTime = PreconTime1[~np.isnan(Precon1)]
                
                _, ind = np.unique(PreconTime, return_index=True)
                
                if np.sum(Precon2) > 2:  
                    PreconStatus = np.interp(time, PreconTime[ind], Precon2[ind], left=0, right=0)
                else:
                    PreconStatus = np.zeros_like(time)
                    PreconTime = time
            else:
                PreconStatus = np.zeros_like(time)
                PreconTime = time

            Active_PreconTime = np.sum(PreconStatus)
            PCEnergy = (PreconStatus > 0).astype(int)  

            ## Power and Energy Calcs

            battPower = battCurr * battVolt  # Battery Power (W)
            battEnergy = np.sum(battPower)/3.6e6  # Battery Energy (kWh)

            # Power during different modes
            driveBattPower = battPower[vedsDrive]  # battery power during driving
            chargeBattPower = battPower[vedsCharge]  # battery power during charging

            ActChargeTime = np.sum(vedsCharge & (battCurr < 0))  # Actual charge time in seconds

            # Energy calculations
            driveBattEnergy = np.sum(driveBattPower)/3.6e6  # battery energy during driving (kWh)
            chargeBattEnergy = np.sum(chargeBattPower)/3.6e6  # battery energy during charging (kWh)
            motorEnergy = np.sum(motorPower)/3.6e6  # energy used by the motor (kWh)

            if ActChargeTime == 0:
                chargeBattEnergykW = 0
            else:
                chargeBattEnergykW = chargeBattEnergy / (ActChargeTime / 3600)  

            # Energy efficiency calculations
            if trip_summaries.at[row_idx, 'Distance (km)'] > 0:  # Only calculate if distance > 0
                effDriveEnergyBatt = round(driveBattEnergy / trip_summaries.at[row_idx, 'Distance (km)'], 2)
            else:
                effDriveEnergyBatt = 0.0  # or np.nan if you prefer

            # 24V battery calculations
            batt24Power = batt24Curr * batt24Volt  # Battery Power (J)
            batt24DrivePower = batt24Power[vedsDrive]  # 24V power while driving (J)
            batt24Energy = round(np.sum(batt24DrivePower)) / 3.6e6  # 24V System Energy (kWh)

            # Temperature calculations
            ambientAirTemp_nonzero = ambientAirTemp[ambientAirTemp != 0]
            meanAmbientAirTemp = safe_nan_statistic(np.nanmean, ambientAirTemp_nonzero, 0)
            maxAmbientAirTemp = safe_nan_statistic(np.nanmax, ambientAirTemp_nonzero, 0)
            minAmbientAirTemp = safe_nan_statistic(np.nanmin, ambientAirTemp_nonzero, 0)
            #minAmbientAirTemp = np.nanmin(ambientAirTemp)

            meanLowerSaloonExtTemp = safe_nan_statistic(np.nanmean, LowerSaloonExtTemp, 0)
            maxLowerSaloonExtTemp = safe_nan_statistic(np.nanmax, LowerSaloonExtTemp, 0)
            minLowerSaloonExtTemp = safe_nan_statistic(np.nanmin, LowerSaloonExtTemp, 0)

            meanUpperDeckCabinTemp = safe_nan_statistic(np.nanmean, UpperDeckCabinTemp, 0)
            maxUpperDeckCabinTemp = safe_nan_statistic(np.nanmax, UpperDeckCabinTemp, 0)
            minUpperDeckCabinTemp = safe_nan_statistic(np.nanmin, UpperDeckCabinTemp, 0)

            # Energy regeneration and discharge calculations
            motorregen = round(np.sum(motorPower[motorPower < 0]) / 3.6e6, 2)
            batregen = round(np.sum(driveBattPower[driveBattPower < 0]) / 3.6e6, 2)

            motordis = round(np.sum(motorPower[motorPower > 0]) / 3.6e6, 2)  # Sum of the motor discharge in kWh
            batdis = round(np.sum(driveBattPower[driveBattPower > 0]) / 3.6e6, 2)  # Sum of the bat discharge in kWh

            # Battery cell temperature calculations
            valid_cell_temps = MaxCellTemp[MaxCellTemp < 100]
            MaxCellTemp = np.nanmax(valid_cell_temps) if len(valid_cell_temps) > 0 else np.nan
            MinCellTemp = np.nanmean(MinCellTemp)

            # Odometer and preconditioning energy
            odometer = round(trip_summaries.at[row_idx, 'Odometer Reading (km)'], 2)

            PCPower01 = battPower[PCEnergy.astype(bool)] if isinstance(PCEnergy, np.ndarray) else np.array([])
            PreconnditionEnergy = round(np.sum(PCPower01) / 3.6e6, 2) if len(PCPower01) > 0 else 0.0

            # Distance and odometer
            trip_summaries.at[row_idx, 'Distance (km)'] = round(trip_summaries.at[row_idx, 'Distance (km)'], 2) 
            trip_summaries.at[row_idx, 'Odometer Reading (km)'] = round(odometer,2) 

            # Energy calculations
            trip_summaries.at[row_idx, 'Battery Energy (kWh)'] = round(battEnergy, 2)
            trip_summaries.at[row_idx, 'Drive Energy (kWh)'] = round(driveBattEnergy, 2)
            trip_summaries.at[row_idx, 'Charge Energy (kWh)'] = round(chargeBattEnergy, 2)
            trip_summaries.at[row_idx, 'Motor Energy (kWh)'] = round(motorEnergy, 2)
            trip_summaries.at[row_idx, '24V System Energy (kWh)'] = round(batt24Energy, 2)

            # HV System Energy calculation
            hv_energy = round(driveBattEnergy - motorEnergy - batt24Energy, 2)
            trip_summaries.at[row_idx, 'HV System Energy (kWh)'] = 0 if hv_energy < 0 else hv_energy

            # Other metrics
            trip_summaries.at[row_idx, 'Idle Time (%)'] = round(pcIdleTime * 100, 2)

            # Efficiency calculation
            distance = trip_summaries.at[row_idx, 'Distance (km)']
            if distance > 0:
                trip_summaries.at[row_idx, 'Efficiency (kWh/km)'] = round(driveBattEnergy / distance, 2)
            else:
                trip_summaries.at[row_idx, 'Efficiency (kWh/km)'] = 0

            # Pedal and time metrics
            trip_summaries.at[row_idx, 'AvgAPP (%)'] = round(np.mean(engagedAccPedal), 2) if len(engagedAccPedal) > 0 else 0
            trip_summaries.at[row_idx, 'Drive Time (mins)'] = round(timeDrive / 60, 2)
            trip_summaries.at[row_idx, 'Charge Time (mins)'] = round(timeCharge / 60, 2)
            trip_summaries.at[row_idx, 'Precon Time (mins)'] = round(Active_PreconTime / 60, 2)
            trip_summaries.at[row_idx, 'Neutral Time (mins)'] = round(timeNeut / 60, 2)
            trip_summaries.at[row_idx, 'Average Speed (kph)'] = round(avgDriveSpeed, 2)

            # Temperature metrics
            trip_summaries.at[row_idx, 'Mean Ambient Temp (°C)'] = round(meanAmbientAirTemp, 2)
            trip_summaries.at[row_idx, 'Min Ambient Temp (°C)'] = round(minAmbientAirTemp, 2)
            trip_summaries.at[row_idx, 'Max Ambient Temp (°C)'] = round(maxAmbientAirTemp, 2)
            trip_summaries.at[row_idx, 'Max Cell Temp (°C)'] = round(MaxCellTemp, 2)
            trip_summaries.at[row_idx, 'Min Cell Temp (°C)'] = round(MinCellTemp, 2)

            # SOC metrics
            trip_summaries.at[row_idx, 'Lowest State Of Charge'] = MINSOC
            trip_summaries.at[row_idx, 'State Of Health (%)'] = 100  
            trip_summaries.at[row_idx, 'AvgBPP (%)'] = round(np.mean(engagedBccPedal), 2) if len(engagedBccPedal) > 0 else 0
            trip_summaries.at[row_idx, 'Start SOC (%)'] = round(startSOC, 2)
            trip_summaries.at[row_idx, 'End SOC (%)'] = round(endSOC, 2)
            trip_summaries.at[row_idx, 'Min SOC (%)'] = round(MINSOC, 2)
            trip_summaries.at[row_idx, 'Max SOC (%)'] = round(MAXSOC, 2)

            # Charge metrics
            trip_summaries.at[row_idx, 'Charge Started Time (min)'] = Time_ChargeStart
            trip_summaries.at[row_idx, 'ChargeMode Interrupted Shutdown'] = ChargeMode_Interrupted_Shutdown
            trip_summaries.at[row_idx, 'ChargeMode Failure'] = ChargeMode_Failure
            trip_summaries.at[row_idx, 'ChargeMode Charging Completed'] = ChargeMode_ChargingCompleted

            # Energy regeneration metrics
            trip_summaries.at[row_idx, 'Motor Regen Energy (kWh)'] = round(motorregen,2)
            trip_summaries.at[row_idx, 'Battery Regen Energy (kWh)'] = round(batregen,2)
            trip_summaries.at[row_idx, 'Motor Discharge Energy (kWh)'] = round(motordis,2)
            trip_summaries.at[row_idx, 'Battery Discharge Energy (kWh)'] = round(batdis,2)

            # Preconditioning
            trip_summaries.at[row_idx, 'Precon Energy (kWh)'] = round(PreconnditionEnergy,2)

            # Additional metrics
            trip_summaries.at[row_idx, 'Odometer Reading start'] = round(odometer, 2)
            trip_summaries.at[row_idx, 'Bus Name'] = trip_summaries.at[row_idx, 'Bus Name']  

            # Temperature metrics continued
            trip_summaries.at[row_idx, 'Mean LSE Temp (°C)'] = round(meanLowerSaloonExtTemp, 1)
            trip_summaries.at[row_idx, 'Min LSE Temp (°C)'] = round(minLowerSaloonExtTemp, 1)
            trip_summaries.at[row_idx, 'Max LSE Temp (°C)'] = round(maxLowerSaloonExtTemp, 1)

            trip_summaries.at[row_idx, 'Mean UDC Temp (°C)'] = round(meanUpperDeckCabinTemp, 1)
            trip_summaries.at[row_idx, 'Min UDC Temp (°C)'] = round(minUpperDeckCabinTemp, 1)
            trip_summaries.at[row_idx, 'Max UDC Temp (°C)'] = round(maxUpperDeckCabinTemp, 1)

            # Door operations
            trip_summaries.at[row_idx, 'Door 1'] = Door1Noofoperations
            trip_summaries.at[row_idx, 'Door 2'] = Door2Noofoperations

            # Displayed SOC
            trip_summaries.at[row_idx, 'Displayed Min SOC (%)'] = round(MinDSOC, 2)
            trip_summaries.at[row_idx, 'Displayed Max SOC (%)'] = round(MaxDSOC, 2)

            # Charge metrics continued
            trip_summaries.at[row_idx, 'Act Charge Time (mins)'] = round(ActChargeTime / 60, 2)
            trip_summaries.at[row_idx, 'Charge Energy (kW)'] = round(chargeBattEnergykW, 2)

            arrays_to_clear = [
                'mat_data', 'time', 'battCurr', 'battVolt', 'veds', 
                'accPedal', 'bccPedal', 'vehSpeed', 'motorPower',
                'ambientAirTemp', 'MaxCellTemp', 'MinCellTemp', 
                'LowerSaloonExtTemp', 'UpperDeckCabinTemp', 'batt24Curr',
                'battPower', 'PreconStatus', 'SOC', 'DSOC'
            ]

            for arr_name in arrays_to_clear:
                if arr_name in locals():
                    del locals()[arr_name]

            import gc
            gc.collect()

    

    # Daily summary table 
    if not trip_summaries.empty:
        
        grouped = trip_summaries.groupby(['VIN', 'Day No.'])
        
        daily_summaries_list = []
        
        for (vin, day), daily_data in grouped:
            i = len(daily_summaries_list)
            daily_row = {}
            
           
            daily_row['VIN'] = vin
            daily_row['Day No.'] = day
            
            # Start and stop times
            start_times = daily_data['Start Time (dd/mm/yyyy hh:mm:ss)']
            stop_times = daily_data['Stop Time (dd/mm/yyyy hh:mm:ss)']
            daily_row['Start Time (dd/mm/yyyy hh:mm:ss)'] = start_times.iloc[0]
            daily_row['Stop Time (dd/mm/yyyy hh:mm:ss)'] = stop_times.iloc[-1]
            
            # Distance and Odometer 
            daily_row['Distance (km)'] = round(daily_data['Distance (km)'].sum(), 2)
            daily_row['Odometer Reading (km)'] = daily_data['Odometer Reading (km)'].max()
            
            # Energy calculations 
            daily_row['Battery Energy (kWh)'] = round(daily_data['Battery Energy (kWh)'].sum(), 2)
            daily_row['Drive Energy (kWh)'] = round(daily_data['Drive Energy (kWh)'].sum(), 2)
            daily_row['Charge Energy (kWh)'] = round(daily_data['Charge Energy (kWh)'].sum(), 2)
            daily_row['Motor Energy (kWh)'] = round(daily_data['Motor Energy (kWh)'].sum(), 2)
            daily_row['24V System Energy (kWh)'] = round(daily_data['24V System Energy (kWh)'].sum(), 2)
            daily_row['HV System Energy (kWh)'] = round(daily_data['HV System Energy (kWh)'].sum(), 2)

            # Efficiency calculation 
            if daily_data['Distance (km)'].sum() == 0:
                daily_row['Efficiency (kWh/km)'] = 0
            else:
                efficiency = daily_data['Drive Energy (kWh)'].sum() / daily_data['Distance (km)'].sum()
                daily_row['Efficiency (kWh/km)'] = 0 if efficiency == float('inf') else round(efficiency, 2)

            # Time calculations
            daily_row['Total Time (mins)'] = daily_data['Total Time (mins)'].sum()
            daily_row['Drive Time (mins)'] = daily_data['Drive Time (mins)'].sum()
            daily_row['Precon Time (mins)'] = daily_data['Precon Time (mins)'].sum()
            daily_row['Neutral Time (mins)'] = daily_data['Neutral Time (mins)'].sum()
            daily_row['Charge Time (mins)'] = daily_data['Charge Time (mins)'].sum()

            # Charge metrics
            daily_row['Charge Started Time (min)'] = daily_data['Charge Started Time (min)'].iloc[0]
            daily_row['ChargeMode Interrupted Shutdown'] = daily_data['ChargeMode Interrupted Shutdown'].iloc[0]
            daily_row['ChargeMode Failure'] = daily_data['ChargeMode Failure'].iloc[0]
            daily_row['ChargeMode Charging Completed'] = daily_data['ChargeMode Charging Completed'].iloc[0]

            # State Of Health
            daily_row['State Of Health (%)'] = 100

            # Weighted averages
            total_drive_time = daily_row['Drive Time (mins)']
            total_total_time = daily_row['Total Time (mins)']
            
            if total_drive_time > 0:
                daily_row['AvgAPP (%)'] = round((daily_data['AvgAPP (%)'] * daily_data['Drive Time (mins)']).sum() / total_drive_time, 2)
                daily_row['Idle Time (%)'] = round((daily_data['Idle Time (%)'] * daily_data['Drive Time (mins)']).sum() / total_drive_time, 2)
                daily_row['Average Speed (kph)'] = round((daily_data['Average Speed (kph)'] * daily_data['Drive Time (mins)']).sum() / total_drive_time, 2)
                daily_row['AvgBPP (%)'] = round((daily_data['AvgBPP (%)'] * daily_data['Drive Time (mins)']).sum() / total_drive_time, 2)
                daily_row['Start SOC (%)'] = round((daily_data['Start SOC (%)'] * daily_data['Drive Time (mins)']).sum() / total_drive_time, 2)
                daily_row['End SOC (%)'] = round((daily_data['End SOC (%)'] * daily_data['Drive Time (mins)']).sum() / total_drive_time, 2)
            else:
                daily_row['AvgAPP (%)'] = 0
                daily_row['Idle Time (%)'] = 0
                daily_row['Average Speed (kph)'] = 0
                daily_row['AvgBPP (%)'] = 0
                daily_row['Start SOC (%)'] = 0
                daily_row['End SOC (%)'] = 0
            
            daily_row['Mean Ambient Temp (°C)'] = round((daily_data['Mean Ambient Temp (°C)'] * daily_data['Total Time (mins)']).sum() / total_total_time, 1)
            daily_row['Min Cell Temp (°C)'] = round((daily_data['Min Cell Temp (°C)'] * daily_data['Total Time (mins)']).sum() / total_total_time, 2)
            daily_row['Max Cell Temp (°C)'] = round((daily_data['Max Cell Temp (°C)'] * daily_data['Total Time (mins)']).sum() / total_total_time, 2)
            daily_row['Mean LSE Temp (°C)'] = round((daily_data['Mean LSE Temp (°C)'] * daily_data['Total Time (mins)']).sum() / total_total_time, 1)
            daily_row['Mean UDC Temp (°C)'] = round((daily_data['Mean UDC Temp (°C)'] * daily_data['Total Time (mins)']).sum() / total_total_time, 1)
           

            # Min/Max calculations
            daily_row['Min Ambient Temp (°C)'] = round(daily_data['Min Ambient Temp (°C)'].min(), 2)
            daily_row['Max Ambient Temp (°C)'] = round(daily_data['Max Ambient Temp (°C)'].max(), 2)
            daily_row['Lowest State Of Charge'] = daily_data['Lowest State Of Charge'].min()
            daily_row['Max SOC (%)'] = round(daily_data['Max SOC (%)'].max(), 2)

            # Min SOC (%) with >0 filter (exact MATLAB logic)
            min_soc_positive = daily_data['Min SOC (%)'][daily_data['Min SOC (%)'] > 0]
            if min_soc_positive.empty:
                daily_row['Min SOC (%)'] = 0
            else:
                daily_row['Min SOC (%)'] = round(min_soc_positive.min(), 2)

            # Energy regeneration and discharge
            daily_row['Motor Regen Energy (kWh)'] = daily_data['Motor Regen Energy (kWh)'].sum()
            daily_row['Battery Regen Energy (kWh)'] = daily_data['Battery Regen Energy (kWh)'].sum()
            daily_row['Motor Discharge Energy (kWh)'] = daily_data['Motor Discharge Energy (kWh)'].sum()
            daily_row['Battery Discharge Energy (kWh)'] = daily_data['Battery Discharge Energy (kWh)'].sum()

            # Odometer Reading start and Bus Name
            daily_row['Odometer Reading start'] = daily_data['Odometer Reading start'].min()
            daily_row['Bus Name'] = daily_data['Bus Name'].iloc[0]

            # Temperature min/max
            daily_row['Min LSE Temp (°C)'] = round(daily_data['Min LSE Temp (°C)'].min(), 2)
            daily_row['Max LSE Temp (°C)'] = round(daily_data['Max LSE Temp (°C)'].max(), 2)
            daily_row['Min UDC Temp (°C)'] = round(daily_data['Min UDC Temp (°C)'].min(), 2)
            daily_row['Max UDC Temp (°C)'] = round(daily_data['Max UDC Temp (°C)'].max(), 2)

            # Door operations
            daily_row['Door 1'] = daily_data['Door 1'].sum()
            daily_row['Door 2'] = daily_data['Door 2'].sum()

            # Add these lines in the daily summary loop, after other aggregations:
            daily_row['LAT_Start'] = daily_data['LAT_Start'].iloc[0]  # First trip's start
            daily_row['LAT_End'] = daily_data['LAT_End'].iloc[-1]     # Last trip's end
            daily_row['LONG_Start'] = daily_data['LONG_Start'].iloc[0]
            daily_row['LONG_End'] = daily_data['LONG_End'].iloc[-1]
            daily_row['GPS_ALTITUDE_Start'] = daily_data['GPS_ALTITUDE_Start'].iloc[0]
            daily_row['GPS_ALTITUDE_End'] = daily_data['GPS_ALTITUDE_End'].iloc[-1]

            # Displayed Min SOC (%) handling
            displayed_min_soc_data = daily_data['Displayed Min SOC (%)'][daily_data['Displayed Min SOC (%)'] > 0]
            if displayed_min_soc_data.empty:
                daily_row['Displayed Min SOC (%)'] = 0
            else:
                daily_row['Displayed Min SOC (%)'] = round(displayed_min_soc_data.min(), 2)
                if 0 < daily_row['Displayed Min SOC (%)'] < 10:
                    daily_row['Displayed Min SOC (%)'] *= 10 

            # Displayed Max SOC (%)
            daily_row['Displayed Max SOC (%)'] = round(daily_data['Displayed Max SOC (%)'].max(), 2)

            # Estimated Range calculation
            efficiency = daily_row['Efficiency (kWh/km)']
            if efficiency > 0:
                daily_row['Estimated Range (km)'] = round(444 * 0.8 / efficiency, 2)
            else:
                daily_row['Estimated Range (km)'] = 0

            # Act Charge Time and Charge Energy
            daily_row['Act Charge Time (mins)'] = daily_data['Act Charge Time (mins)'].sum()
            daily_row['Charge Energy (kW)'] = round(daily_data['Charge Energy (kW)'].sum(), 2)

            daily_summaries_list.append(daily_row)
        
        # Create DataFrame from all daily summaries
        daily_summaries = pd.DataFrame(daily_summaries_list)
        
    else:
        daily_summaries = pd.DataFrame()


    # Selection of columns for Trip Summaries
    desired_columns_trip_summaries = [
        "VIN", "Bus Name", "Start Time (dd/mm/yyyy hh:mm:ss)", "Stop Time (dd/mm/yyyy hh:mm:ss)", 
        "Day No.", "Drive Time (mins)", "Total Time (mins)", "Idle Time (%)", "Average Speed (kph)", 
        "Distance (km)", "Odometer Reading (km)", "State Of Health (%)", "Efficiency (kWh/km)", 
        "24V System Energy (kWh)", "HV System Energy (kWh)", "Charge Energy (kWh)", "Drive Energy (kWh)", 
        "Battery Energy (kWh)", "Battery Regen Energy (kWh)", "Battery Discharge Energy (kWh)", 
        "Motor Energy (kWh)", "Motor Regen Energy (kWh)", "Motor Discharge Energy (kWh)", "Min SOC (%)", 
        "Max SOC (%)", "Start SOC (%)", "End SOC (%)", "Max Cell Temp (°C)", "Min Cell Temp (°C)", 
        "Min Ambient Temp (°C)", "Max Ambient Temp (°C)", "Mean Ambient Temp (°C)", "Charge Time (mins)", 
        "AvgAPP (%)", "AvgBPP (%)", "Mean LSE Temp (°C)", "Min LSE Temp (°C)", "Max LSE Temp (°C)", 
        "Mean UDC Temp (°C)", "Min UDC Temp (°C)", "Max UDC Temp (°C)", "Door 1", "Door 2",
        "LAT_Start", "LAT_End", "LONG_Start", "LONG_End","GPS_ALTITUDE_Start", "GPS_ALTITUDE_End"
    ]

    # Select only the specified columns in the specified order
    trip_summaries = trip_summaries[desired_columns_trip_summaries]

    desired_columns_daily_summaries = [
    "VIN", "Bus Name", "Start Time (dd/mm/yyyy hh:mm:ss)", "Stop Time (dd/mm/yyyy hh:mm:ss)", 
    "Day No.", "Drive Time (mins)", "Total Time (mins)", "Idle Time (%)", "Average Speed (kph)", 
    "Distance (km)", "Odometer Reading (km)", "Estimated Range (km)", "State Of Health (%)", "Efficiency (kWh/km)", 
    "24V System Energy (kWh)", "HV System Energy (kWh)", "Charge Energy (kWh)", "Drive Energy (kWh)", 
    "Battery Energy (kWh)", "Battery Regen Energy (kWh)", "Battery Discharge Energy (kWh)", 
    "Motor Energy (kWh)", "Motor Regen Energy (kWh)", "Motor Discharge Energy (kWh)", "Min SOC (%)", 
    "Max SOC (%)", "Start SOC (%)", "End SOC (%)", "Max Cell Temp (°C)", "Min Cell Temp (°C)", 
    "Min Ambient Temp (°C)", "Max Ambient Temp (°C)", "Mean Ambient Temp (°C)", "Charge Time (mins)", 
    "AvgAPP (%)", "AvgBPP (%)", "Mean LSE Temp (°C)", "Min LSE Temp (°C)", "Max LSE Temp (°C)", 
    "Mean UDC Temp (°C)", "Min UDC Temp (°C)", "Max UDC Temp (°C)", "Door 1", "Door 2",
    "LAT_Start", "LAT_End", "LONG_Start", "LONG_End","GPS_ALTITUDE_Start", "GPS_ALTITUDE_End"
    ]

    daily_summaries = daily_summaries[desired_columns_daily_summaries]
    
    daily_summaries = daily_summaries.sort_values(['Bus Name', 'Start Time (dd/mm/yyyy hh:mm:ss)'], ascending=True)

    #Format Report Names
    start_date = daily_summaries['Start Time (dd/mm/yyyy hh:mm:ss)'].iloc[0]
    end_date = daily_summaries['Stop Time (dd/mm/yyyy hh:mm:ss)'].iloc[-1]
        
    start_day = start_date.strftime('%d')
    start_month = start_date.strftime('%b')  
    end_day = end_date.strftime('%d')
    end_month = end_date.strftime('%b')      
        
    report_name = f"{Customer}EV_{start_day}_{start_month}-{end_day}_{end_month}.xlsx"

   
    return trips, veh_names, veh_count, trip_summaries, daily_summaries, report_name, vehicle_names_ev