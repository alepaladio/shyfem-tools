"""
Reader for SHYFEM formatted files
"""

import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Union
import re

class SHYReader:
    """Reader for SHYFEM time series files"""
    
    def __init__(self, filename: str):
        """
        Initialize reader with file path
        
        Parameters
        ----------
        filename : str
            Path to SHYFEM file
        """
        self.filename = filename
        
    def read(self) -> Tuple[List[datetime], List[float]]:
        """
        Read data from SHYFEM file
        
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
    
    def read_header(self) -> Optional[str]:
        """
        Read first comment line from file
        
        Returns
        -------
        str or None
            First comment line if found
        """
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    return line.strip()
        return None


def read_shy_file(filename: str) -> Tuple[List[datetime], List[float]]:
    """
    Read SHYFEM file
    
    Parameters
    ----------
    filename : str
        Path to SHYFEM file
    
    Returns
    -------
    tuple
        (dates, values) lists of datetime objects and float values
    """
    reader = SHYReader(filename)
    return reader.read()