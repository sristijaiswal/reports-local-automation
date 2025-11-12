import numpy as np
from datetime import datetime, timedelta

def odos2mat(times):
    """
    Converts ODOS times to Python datetime objects
    Input: ODOS time scalar or vector (numpy array or single value)
    Returns: (vec, dt, t0)
      - vec: datetime objects (equivalent to MATLAB datevec)
      - dt: numpy array of datetime64 objects
      - t0: array of seconds since first timestamp
    """
    # Convert input to numpy array to handle both scalars and arrays
    times = np.atleast_1d(times)
    
    # MATLAB: datenum(2000,1,1,0,0,0) + datenum(0,0,0,0,0,times(1,:)/1e7)
    base_date = datetime(2000, 1, 1)
    delta = timedelta(seconds=float(times[0]/1e7))
    first_dt = base_date + delta
    
    # Create datetime object (equivalent to MATLAB's datetime(vec))
    dt = np.array([first_dt], dtype='datetime64[us]')
    
    # Create date vector (equivalent to MATLAB's datevec)
    vec = np.array([[first_dt.year, first_dt.month, first_dt.day, 
                    first_dt.hour, first_dt.minute, first_dt.second]])
    
    # Calculate t0 (times since first timestamp in seconds)
    t0 = (times - times[0]) / 1e7
    
    return vec, dt, t0