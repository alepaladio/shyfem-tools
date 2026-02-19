"""
Utility functions for plotting
"""

import os
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from datetime import datetime
from typing import List, Optional
import matplotlib.pyplot as plt

def save_figure(fig, folder: str, filename: str, dpi: int = 100, close: bool = True) -> str:
    """
    Save figure to file and optionally close it.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    folder : str
        Output folder
    filename : str
        Filename (without path)
    dpi : int
        Resolution
    close : bool
        Whether to close the figure after saving
    
    Returns
    -------
    str
        Full path to saved file
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    if close:
        plt.close(fig)
    return filepath

def create_video(image_paths: List[str], output_path: str, fps: int = 4) -> str:
    """
    Create video from list of images.
    
    Parameters
    ----------
    image_paths : List[str]
        List of paths to images
    output_path : str
        Path for output video
    fps : int
        Frames per second
    
    Returns
    -------
    str
        Path to created video
    """
    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
    return output_path

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points."""
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c