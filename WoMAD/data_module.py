"""
    This module is designed to achieve these tasks:
    - Develop functions to download the data
    - Parse the dataset and isolate trials using the EV files
    - Normalize data and save to a DataFrame
"""

import os
import io
import zipfile

import torch
import torch_geometric.data as PyG_Data
from torch.utils.data import Dataset, DataLoader

import numpy as np
import nibabel as nib

from . import WoMAD_config

# File access:
def generate_paths():
    """
    Uses the paths configured in the config file to create subject-specific paths.
    """
    paths = {}

    return paths

def load_data_from_path():
    """
    Reads the contents of the .zip files.
    """
    try:
        # Check file path
        return np.array([])

        # Extract NIfTI and EV files
        fmri_timeseries = 1.00 # BOLD Signals as floats
        ev_txt_file = "place-holder" # EVs in .txt format
        return fmri_timeseries, ev_txt_file

    except Exception as e:
        print("Error!")

    return np.array([])


# Processing:
## Trial isolation:
def isolate_trials():
    """
    Isolates each task trial based on EV files.
    """
    trial_list = []

    return trial_list

## Normalization
def normalize_data():
    """
    Normalizes a numpy array.
    """
    data = np.array([])
    # Uses if statements, set up normalization processes based on type of normalization.
    return data

## Creating the WoMAD_data class
class WoMAD_data(Dataset):
    def __init__(self, data_paths: list, processed_paths: list):
        """
        Initializing the dataset with basic configuration.
        """
        self.data_paths = data_paths
        self.processed_paths = processed_paths

        self.config = WoMAD_config.load_config()

    def __len__(self) -> int:
        """
        Returns the total number of subjects in the dataset.
        """
        return len(self.data_paths)

    def __getitem__(self, indx: int):
        """
        Loads and preprocesses one subject's data.
        """
        model_ready_data = []
        # Step 1: Load processed data
        # Step 2: Basic processing (normalization, resampling, etc.)
        # Step 3: Produce model-ready ground truth for all outputs
        # Step 4: Create model-ready data
        #         (submodule-specific processing to create tensors)
        return model_ready_data

    def _load_data():
        """
        Load voxel-based BOLD signals using NIfTI and EV files.
        """
        processed_data = []
        # Step 1: Load minimally preprocessed HCP data using generate_paths() and load_data_from_path()
        # Step 2: Concat fMRI run data
        # Step 3: Load EV files
        # Step 4: Isolate trials
        return processed_data
