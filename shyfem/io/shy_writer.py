"""
Writer for SHYFEM formatted files
"""

import numpy as np
from datetime import datetime
from typing import List, Union, Optional

class SHYWriter:
    """Writer for SHYFEM time series files"""
    
    def __init__(self, filename: str, station_name: str, 
                 comment: Optional[str] = None):
        """
        Initialize writer with file path and metadata
        
        Parameters
        ----------
        filename : str
            Output file path
        station_name : str
            Name of station or location
        comment : str, optional
            Additional comment for header
        """
        self.filename = filename
        self.station_name = station_name
        self.comment = comment
        
    def write(self, times: List[datetime], values: List[float]) -> None:
        """
        Write time series to SHYFEM format
        
        Parameters
        ----------
        times : list of datetime
            Time values
        values : list of float
            Corresponding values
        """
        with open(self.filename, 'w') as f:
            f.write(f"# Station: {self.station_name}\n")
            if self.comment:
                f.write(f"# {self.comment}\n")
            
            for t, v in zip(times, values):
                if not np.isnan(v):
                    date_str = t.strftime("%Y-%m-%d::%H:%M:%S")
                    f.write(f"{date_str}\t{v:.3f}\n")


def write_shy_file(output_file: str, times: List[datetime], 
                   values: List[float], station_name: str,
                   comment: Optional[str] = None) -> None:
    """
    Write SHYFEM file
    
    Parameters
    ----------
    output_file : str
        Output file path
    times : list of datetime
        Time values
    values : list of float
        Corresponding values
    station_name : str
        Name of station or location
    comment : str, optional
        Additional comment for header
    """
    writer = SHYWriter(output_file, station_name, comment)
    writer.write(times, values)