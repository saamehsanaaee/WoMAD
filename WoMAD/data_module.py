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
from typing import List, Dict, Any

import torch
import torch_geometric.data as PyG_Data
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import nibabel as nib

from . import WoMAD_config

# File Access:
def generate_paths(task : str = WoMAD_config.target_tasks[0],
                   run  : str = WoMAD_config.run_direction[0],
                   sub_list : list  = WoMAD_config.dev_subjects):
    """
    Uses the paths configured in WoMAD_config to create subject-specific paths.

    Arguments:
        task             (str): The target task from HCP -> [W]orking [M]emory as default
        run              (str): RL or LR -> RL as our arbitrary default
        sub_list (list of int): List of target subject ID's as defined in the config file

    Returns:
        Dictionary with this integer and tuple-of-strings format:
        paths = {
            subject_id (int) : ("Main 'Results' file path", "EV file path")
        }
    """
    paths = {}

    # General path format for each subject's directory:
    # (f"../data/HCP_zipped/{subject-ID}/MNINonLinear/Results/")

    # General path format for subjects' task EV files:
    # (f"../data/HCP_zipped/{subject-ID}/MNINonLinear/Results/tfMRI_{TASK}_{RUN}/EVs/")

    # List of target subjects: full_3T_task_subjects (imported from WoMAD_config)
    for subject in sub_list:
        subject_path    = f"../data/HCP_zipped/{subject}/MNINonLinear/Results/"
        subject_ev_path = f"../data/HCP_zipped/{subject}/MNINonLinear/Results/tfMRI_{task}_{run}/EVs/"
        paths[subject]  = (subject_path, subject_ev_path)

    return paths


def load_data_from_path(task : str = WoMAD_config.target_tasks[0],
                        run  : str = WoMAD_config.run_direction[0],
                        subject : str = WoMAD_config.dev_subjects[0],
                        subtask : str = WoMAD_config.target_subtasks["WM"][0]):
    """
    Reads the contents of each subject's files.

    Arguments:
        task    (str): The target task from HCP -> [W]orking [M]emory as default
        run     (str): RL or LR -> RL as our arbitrary default
        subject (int): ID of specific target subject
        subtask (str): The target subtask in string format -> Example: "0bk_tools"

    Returns:
        Dictionary of {Subject: (Tuple of fMRI data)} and
        EV file contents assigned to the ev_file variable.
    """
    try:
        paths = generate_paths(task = task, run = run)
        bold_ts_path = paths[subject][0] + f"tfMRI_{task}_{run}/tfMRI_{task}_{run}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii"
        ev_file_path = paths[subject][1] + f"{subtask}.txt"

        with open(ev_file_path, "r") as ev:
            ev_file = ev.read()

        bold_ts = nib.load(bold_ts_path)
        bold_data = bold_ts.get_fdata()
        bold_header = bold_ts.header

        fmri_timeseries = {subject: (bold_ts, bold_header, bold_data)}

        return fmri_timeseries, ev_file

    except Exception as e:
        print("Error loading time series and EV files from path!")


# Preprocessing:
## Parse and Isolate Trials
def isolate_trials(fmri_ts, ev_file, TR : float = 0.72):
    """
    Parses through the data and isolates each task trial using EV files.

    Input: The fMRI dictionary and EV file from load_data_from_path() function.

    Returns:
        List of trials isolated using the ev_file.
    """
    trial_list = []

    for subject, (bold_ts, bold_header, bold_data) in fmri_ts.items():
        data_array = bold_data

        try:
            ev_data = np.loadtxt(io.StringIO(ev_file))
        except ValueError:
            print(f"Could not parse EV file for subject {subject}.")
            continue

        for onset, duration, _ in ev_data:
            start_idx = int(np.floor(onset / TR))
            end_idx = int(np.ceil((onset + duration) / TR))
            trial_data = data_array[:, start_idx:end_idx]

            trial_list.append({
                "subject" : subject,
                "onset" : onset,
                "duration" : duration,
                "data" : trial_data
            })

    return trial_list

## Normalization
def normalize_data(data : np.ndarray, norm_mode: str = "z_score"):
    """
    Normalizes a numpy array of fMRI time series data.

    Arguments:
        data (np.ndarray): The time series data with shape (voxels, time_points)
        norm_mode   (str): Method of normalization (Z score, min/max, etc.)

    Returns:
        Numpy array of normalized data.
    """
    data = np.array(data)

    if norm_mode == "z_score":
        ts_data_mean = np.mean(data, axis = 1, keepdims = True)
        ts_data_stdv = np.std(data , axis = 1, keepdims = True)

        ts_data_stdv[ts_data_stdv == 0] = 1.0

        normalized_ts_data = (data - ts_data_mean) / ts_data_stdv

        return normalized_ts_data

    elif norm_mode == "min_max":
        min_ts_data = np.min(data, axis = 1, keepdims = True)
        max_ts_data = np.max(data, axis = 1, keepdims = True)

        range_ts_data = max_ts_data - min_ts_data
        range_ts_data[range_ts_data == 0] = 1.0

        normalized_ts_data = (data - min_ts_data) / range_val

        return normalized_ts_data

    else: # For now ...
        print(f"Normalization mode '{norm_mode}' not defined.\nReturning data as is.")
        return data

## Save to Pandas DataFrame
def save_to_df(trial_list : List[Dict[str, Any]],
               file_name : str,
               output_dir : str = WoMAD_config.processed_path):
    """
    Converts the list of isolated trials to a Pandas DF and saves it to defined path.

    Arguments:
        trial_list (list): List of {"subject", "onset", "duration", "data"} dictionaries.
        file_name   (str): Name of output file.
        output_dir  (str): Directory for saving the output file.

    Saves the pd.DataFrame to output_dir.
    """
    df_from_trial_ts = pd.DataFrame(trial_list)

    os.makedirs(output_dir, exist_ok = True)
    save_path = os.path.join(output_dir, f"{file_name}.pkl")

    df_from_trial_ts.to_pickle(save_path)

    print(f"Data saved to {save_path}")

    return df_from_trial_ts


# Initial Processing (with the WoMAD_data class):
class WoMAD_data(Dataset):
    def __init__(self,
                 task : str,
                 runs : list = WoMAD_config.run_direction,
                 subjects : list = WoMAD_config.dev_subjects,
                 output_dir : str = WoMAD_config.processed_path):
        """
        Basic configuration of the dataset.

        Arguments:
            task (str): The target task ("WM")
            runs (list): Run directions
            subjects (list): List of target subject IDs
            output_dir (str): Directory for saving processed data
        """
        self.task = task
        self.runs = runs
        self.subjects = subjects
        self.output_dir = output_dir

        self.data = []

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

    def _load_data(self):
        """
        Load and parse the data using NIfTI and EV files.
        """
        parsed_data = []

        for subject in self.subject:
            for run in self.runs:
                paths = generate_paths(task = self.task, run = run,
                                       sub_list = [subject])

                subtasks = WoMAD_config.target_subtasks.get(self.task, [])

                for subtask in subtasks:
                    fmri_ts, ev_file = load_data_from_path(task = self.task,
                                                           run = run,
                                                           subject = subject,
                                                           subtask = subtask)
                    trial_list_subtask = isolate_trials(fmri_ts, ev_file)

                    for trial_dict in trial_list_subtask:
                        trial_dict["run"] = run
                        trial_dict["subtask"] = subtask
                        parsed_data.append(trial_dict)

        # TODO: Error handling inside the for loop.

        self.data = parsed_data

        return self.data

    def basic_processing(self, norm_mode : str = "z_score",
                         file_name : str = "processed_fMRI_data"):
        """
        Normalization and saving with normalize_data() and save_to_df().
        """
        processed_trials = []

        for trial in self.data:
            normalized_trial = normalize_data(trial["data"], norm_mode = norm_mode)

            trial["data"] = normalized_trial
            trial["norm_mode"] = norm_mode
            processed_trials.append(trial)

        file_to_save_processed_data = f"{file_name}_{self.task}_{norm_mode}"
        self.processed_df = save_to_df(processed_trials,
                                       file_to_save_processed_data,
                                       self.output_dir)

        self.data = processed_trials

        return self.processed_df

    def initial_processing():
        """
        Basic statistical analysis of the task-based activity.
        """
        # TODO: Create the functions for analysis steps:
        # Basic analysis, Correlation matrices + functional connectivity
        # Visualize the whole-brain, network-level, and voxel-wise activity

# TODO: Add function for "validation set processing" which can process non-HCP data.
