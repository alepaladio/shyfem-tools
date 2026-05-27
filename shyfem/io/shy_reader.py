"""
Reader for SHYFEM formatted files
"""

import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Union
import re

class SHYReader:
    """Reader for SHYFEM tide/observation files"""
    
    def __init__(self, filename: str):
        """
        Initialize reader with file path
        
        Parameters
        ----------
        filename : str
            Path to SHYFEM tide file
        """
        self.filename = filename
        
    def read_observations(self) -> Tuple[List[datetime], List[float]]:
        """
        Read observed sea level data from tide file
        
        Returns
        -------
        tuple
            (dates, values) lists of datetime objects and float values
        """
        dates = []
        values = []
        
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                try:
                    # Split on any whitespace (space, tab, multiple spaces)
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 2:
                        date_str, value_str = parts[0], parts[1]
                        # Parse date: YYYY-MM-DD::HH:MM:SS
                        if '::' in date_str:
                            date_part, time_part = date_str.split('::')
                            dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
                        elif ' ' in date_str:
                            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        else:
                            continue
                            
                        value = float(value_str)
                        if not np.isnan(value):
                            dates.append(dt)
                            values.append(value)
                except Exception:
                    continue
                    
        return dates, values
    
    def read_station_info(self) -> Optional[str]:
        """
        Read station name from header
        
        Returns
        -------
        str or None
            Station name if found
        """
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('# Station:'):
                    return line.split(':', 1)[1].strip()
        return None


# Convenience function
def read_shy_tide_file(filename: str) -> Tuple[List[datetime], List[float]]:
    """
    Convenience function to read SHYFEM tide file
    
    Parameters
    ----------
    filename : str
        Path to SHYFEM tide file
    
    Returns
    -------
    tuple
        (dates, values) lists of datetime objects and float values
    """
    reader = SHYReader(filename)
    return reader.read_observations()