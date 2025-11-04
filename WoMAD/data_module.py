"""
    This module is designed to achieve these tasks:
    - Develop functions to access or download the data
    - Parse the dataset and isolate trials using the EV files
    - Normalize data and save to a DataFrame
    - Define task-specific baselines and references
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

# File Access:
def generate_paths():
    """
    Uses the paths configured in WoMAD_config to create subject-specific paths.
    """
    paths = {}

    return paths

def load_data_from_path():
    """
    Reads the contents of each subject's files.
    """
    try:
        # Use paths from generate_paths()
        # to create np.array of the data.

        # Extract NIfTI and EV files
        fmri_timeseries = 1.00 # BOLD Signals as floats (should be a np.array)
        ev_file = "place-holder" # EVs in .txt format
        return fmri_timeseries, ev_file

    except Exception as e:
        print("Error!")

# Preprocessing:
## Parse and Isolate Trials
def isolate_trials():
    """
    Parses through the data and isolates each task trial using EV files.
    """
    trial_list = []

    return trial_list

## Normalization
def normalize_data(data, norm_mode):
    """
    Normalizes a np.array.
    """
    data = np.array([])
    # Use if-statements based on norm_mode and normalizes data.
    return data

## Save to Pandas DataFrame
def save_to_df():
    """
    """
    # Save data to model-ready folder.

# Initial Processing (with the WoMAD_data class):
class WoMAD_data(Dataset):
    def __init__(self, data_paths: list, processed_paths: list):
        """
        Basic configuration of the dataset.
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
        Loads one subject's data.
        """
        return self.data[indx]

    def _load_data():
        """
        Load and parse the data using NIfTI and EV files.
        """
        parsed_data = []
        # Load minimally preprocessed HCP data using generate_paths() and load_data_from_path()
        # Concat fMRI run data
        # Load EV files
        # Isolate trials
        return parsed_data

    def basic_processing():
        """
        Normalization and saving with normalize_data() and save_to_df().
        """
        # Normalize the parsed data.
        # Saved normalized data to "processed" folder.

    def initial_processing():
        """
        Basic statistical analysis of the task-based activity.
        """
        # Step 1: Load parsed and preprocessed data.
        # Step 2: Basic analysis
        # Step 3: Correlation matrices and functional connectivity
        # Step 4: Visualize the whole-brain, network-level, and voxel-wise activity

# TODO: Add function for "validation set processing" which can process non-HCP data.
