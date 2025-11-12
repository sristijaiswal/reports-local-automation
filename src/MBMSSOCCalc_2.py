import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

def odosTime2MatTimeVar(odosTime, startTime, stopTime):
    dur = (stopTime.timestamp() - startTime.timestamp())
    OdosDur = odosTime[-1] - odosTime[0]
    # Use safe division:
    if OdosDur != 0:
        OdosScale = dur / OdosDur
    else:
        OdosScale = 0
    OdosOffset = odosTime[0]
    newTime = (odosTime - OdosOffset) * OdosScale if OdosScale != 0 else 0
    return newTime

def MBMSSOCCalc_2(START_TIME, STOP_TIME, MBMSStat1_GrossSOC, MBMSStat1_GrossSOC_X,
                  Signal_BMS01_SOC, Signal_BMS01_SOC_X,
                  Signal_BMS02_SOC, Signal_BMS02_SOC_X,
                  Signal_BMS03_SOC=None, Signal_BMS03_SOC_X=None,
                  Signal_BMS04_SOC=None, Signal_BMS04_SOC_X=None):

    startTime = START_TIME
    stopTime = STOP_TIME
    
    args_count = len([arg for arg in [Signal_BMS01_SOC, Signal_BMS01_SOC_X, 
                                     Signal_BMS02_SOC, Signal_BMS02_SOC_X,
                                     Signal_BMS03_SOC, Signal_BMS03_SOC_X,
                                     Signal_BMS04_SOC, Signal_BMS04_SOC_X] 
                     if arg is not None])
    
    # Master BMS SOC
    MSOC = np.atleast_1d(MBMSStat1_GrossSOC)
    MSOC_X = np.atleast_1d(MBMSStat1_GrossSOC_X)
    MSOC = MSOC[~np.isnan(MSOC)]
    MSOC_X = MSOC_X[~np.isnan(MSOC)]
    MSOC = MSOC[~np.isnan(MSOC_X)]
    MSOC_X = MSOC_X[~np.isnan(MSOC_X)]
    MSOC_t1 = odosTime2MatTimeVar(MSOC_X, startTime, stopTime)

    if args_count == 8:  
        # String 1 SOC
        if Signal_BMS01_SOC is None:
            SOC1_X = MSOC_t1
            SOC1 = np.zeros_like(SOC1_X)
        else:
            SOC1a = np.isnan(Signal_BMS01_SOC)
            SOC1 = Signal_BMS01_SOC[~SOC1a]
            SOC1_Xa = np.isnan(Signal_BMS01_SOC_X)
            SOC1_X = Signal_BMS01_SOC_X[~SOC1_Xa]

        # String 2 SOC
        if Signal_BMS02_SOC is None:
            SOC2_X = MSOC_t1
            SOC2 = np.zeros_like(SOC2_X)
        else:
            SOC2a = np.isnan(Signal_BMS02_SOC)
            SOC2 = Signal_BMS02_SOC[~SOC2a]
            SOC2_Xa = np.isnan(Signal_BMS02_SOC_X)
            SOC2_X = Signal_BMS02_SOC_X[~SOC2_Xa]

        # String 3 SOC
        if Signal_BMS03_SOC is None:
            SOC3_X = MSOC_t1
            SOC3 = np.zeros_like(SOC3_X)
        else:
            SOC3a = np.isnan(Signal_BMS03_SOC)
            SOC3 = Signal_BMS03_SOC[~SOC3a]
            SOC3_Xa = np.isnan(Signal_BMS03_SOC_X)
            SOC3_X = Signal_BMS03_SOC_X[~SOC3_Xa]

        # String 4 SOC
        if Signal_BMS04_SOC is None:
            SOC4_X = MSOC_t1
            SOC4 = np.zeros_like(SOC4_X)
        else:
            SOC4a = np.isnan(Signal_BMS04_SOC)
            SOC4 = Signal_BMS04_SOC[~SOC4a]
            SOC4_Xa = np.isnan(Signal_BMS04_SOC_X)
            SOC4_X = Signal_BMS04_SOC_X[~SOC4_Xa]

        SOC1_t1 = odosTime2MatTimeVar(SOC1_X, startTime, stopTime)
        SOC2_t1 = odosTime2MatTimeVar(SOC2_X, startTime, stopTime)
        SOC3_t1 = odosTime2MatTimeVar(SOC3_X, startTime, stopTime)
        SOC4_t1 = odosTime2MatTimeVar(SOC4_X, startTime, stopTime)

        SOCTime = MSOC_t1

        # String 1 interpolation
        if np.sum(SOC1 > 0) > 2:
            SOC1_interp1 = SOC1[SOC1 > 0]
            SOC1_interp1 = SOC1_interp1[~np.isnan(SOC1_interp1)]
            SOC1_t2 = SOC1_t1[SOC1 > 0]
            _, ind = np.unique(SOC1_t2, return_index=True)
            SOC1_interp2 = np.interp(SOCTime, SOC1_t2[ind], SOC1_interp1[ind], 
                                   left=SOC1_interp1[0], right=SOC1_interp1[-1])
        else:
            # Match MATLAB: double.empty(0,length(SOCTime))
            #SOC1_interp2 = np.full(len(SOCTime), np.nan)
            SOC1_interp2 = np.full(len(np.atleast_1d(SOCTime)), np.nan)

        # String 2 interpolation
        if np.sum(SOC2 > 0) > 2:
            SOC2_interp1 = SOC2[SOC2 > 0]
            SOC2_interp1 = SOC2_interp1[~np.isnan(SOC2_interp1)]
            SOC2_t2 = SOC2_t1[SOC2 > 0]
            _, ind = np.unique(SOC2_t2, return_index=True)
            SOC2_interp2 = np.interp(SOCTime, SOC2_t2[ind], SOC2_interp1[ind], 
                                    left=SOC2_interp1[0], right=SOC2_interp1[-1])
        else:
            SOC2_interp2 = np.full(len(np.atleast_1d(SOCTime)), np.nan)


        # String 3 interpolation
        if np.sum(SOC3 > 0) > 2:
            SOC3_interp1 = SOC3[SOC3 > 0]
            SOC3_interp1 = SOC3_interp1[~np.isnan(SOC3_interp1)]
            SOC3_t2 = SOC3_t1[SOC3 > 0]
            _, ind = np.unique(SOC3_t2, return_index=True)
            SOC3_interp2 = np.interp(SOCTime, SOC3_t2[ind], SOC3_interp1[ind], 
                                   left=SOC3_interp1[0], right=SOC3_interp1[-1])
        else:
            #SOC3_interp2 = np.full(len(SOCTime), np.nan)
            SOC3_interp2 = np.full(len(np.atleast_1d(SOCTime)), np.nan)

        # String 4 interpolation
        if np.sum(SOC4 > 0) > 2:
            SOC4_interp1 = SOC4[SOC4 > 0]
            SOC4_interp1 = SOC4_interp1[~np.isnan(SOC4_interp1)]
            SOC4_t2 = SOC4_t1[SOC4 > 0]
            _, ind = np.unique(SOC4_t2, return_index=True)
            SOC4_interp2 = np.interp(SOCTime, SOC4_t2[ind], SOC4_interp1[ind], 
                                   left=SOC4_interp1[0], right=SOC4_interp1[-1])
        else:
            #SOC4_interp2 = np.full(len(SOCTime), np.nan)
            SOC4_interp2 = np.full(len(np.atleast_1d(SOCTime)), np.nan)

       
        # Create 2D array where each row is one SOC array
        soc_arrays = np.vstack([SOC1_interp2, SOC2_interp2, SOC3_interp2, SOC4_interp2])
        # Calculate mean along axis 0 (columns), ignoring NaN values like MATLAB
        SOC = np.nanmean(soc_arrays, axis=0)

    elif args_count == 6:  # 3-string case 
        # String 1 SOC
        if Signal_BMS01_SOC is None:
            SOC1_X = MSOC_t1
            SOC1 = np.zeros_like(SOC1_X)
        else:
            SOC1a = np.isnan(Signal_BMS01_SOC)
            SOC1 = Signal_BMS01_SOC[~SOC1a]
            SOC1_Xa = np.isnan(Signal_BMS01_SOC_X)
            SOC1_X = Signal_BMS01_SOC_X[~SOC1_Xa]

        # String 2 SOC
        if Signal_BMS02_SOC is None:
            SOC2_X = MSOC_t1
            SOC2 = np.zeros_like(SOC2_X)
        else:
            SOC2a = np.isnan(Signal_BMS02_SOC)
            SOC2 = Signal_BMS02_SOC[~SOC2a]
            SOC2_Xa = np.isnan(Signal_BMS02_SOC_X)
            SOC2_X = Signal_BMS02_SOC_X[~SOC2_Xa]

        # String 3 SOC
        if Signal_BMS03_SOC is None:
            SOC3_X = MSOC_t1
            SOC3 = np.zeros_like(SOC3_X)
        else:
            SOC3a = np.isnan(Signal_BMS03_SOC)
            SOC3 = Signal_BMS03_SOC[~SOC3a]
            SOC3_Xa = np.isnan(Signal_BMS03_SOC_X)
            SOC3_X = Signal_BMS03_SOC_X[~SOC3_Xa]

        SOC1_t1 = odosTime2MatTimeVar(SOC1_X, startTime, stopTime)
        SOC2_t1 = odosTime2MatTimeVar(SOC2_X, startTime, stopTime)
        SOC3_t1 = odosTime2MatTimeVar(SOC3_X, startTime, stopTime)
        SOCTime = MSOC_t1

        # String 1 interpolation
        if np.sum(SOC1 > 0) > 2:
            SOC1_interp1 = SOC1[SOC1 > 0]
            SOC1_interp1 = SOC1_interp1[~np.isnan(SOC1_interp1)]
            SOC1_t2 = SOC1_t1[SOC1 > 0]
            _, ind = np.unique(SOC1_t2, return_index=True)
            SOC1_interp2 = np.interp(SOCTime, SOC1_t2[ind], SOC1_interp1[ind], 
                                   left=SOC1_interp1[0], right=SOC1_interp1[-1])
        else:
            SOC1_interp2 = np.full(len(SOCTime), np.nan)

        # String 2 interpolation
        if np.sum(SOC2 > 0) > 2:
            SOC2_interp1 = SOC2[SOC2 > 0]
            SOC2_interp1 = SOC2_interp1[~np.isnan(SOC2_interp1)]
            SOC2_t2 = SOC2_t1[SOC2 > 0]
            _, ind = np.unique(SOC2_t2, return_index=True)
            SOC2_interp2 = np.interp(SOCTime, SOC2_t2[ind], SOC2_interp1[ind], 
                                   left=SOC2_interp1[0], right=SOC2_interp1[-1])
        else:
            SOC2_interp2 = np.full(len(SOCTime), np.nan)

        # String 3 interpolation
        if np.sum(SOC3 > 0) > 2:
            SOC3_interp1 = SOC3[SOC3 > 0]
            SOC3_interp1 = SOC3_interp1[~np.isnan(SOC3_interp1)]
            SOC3_t2 = SOC3_t1[SOC3 > 0]
            _, ind = np.unique(SOC3_t2, return_index=True)
            SOC3_interp2 = np.interp(SOCTime, SOC3_t2[ind], SOC3_interp1[ind], 
                                   left=SOC3_interp1[0], right=SOC3_interp1[-1])
        else:
            SOC3_interp2 = np.full(len(SOCTime), np.nan)

        soc_arrays = np.vstack([SOC1_interp2, SOC2_interp2, SOC3_interp2])
        SOC = np.nanmean(soc_arrays, axis=0)

    else:
        raise ValueError("Wrong number of inputs to function")

    return SOC