from datetime import datetime
import numpy as np

def mat2odos(dt):
    """
    Convert Python datetime to ODOS time (nanoseconds since 2000-01-01)
    Equivalent to MATLAB's mat2odos function
    """
    epoch = datetime(2000, 1, 1)
    delta = dt - epoch
    return int(delta.total_seconds() * 1e7)  # Convert to nanoseconds