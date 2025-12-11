"""
    This file contains all global variables used throughout the WoMAD project.
    (Variables, paths, and MISC code snippets.)
"""

import os

# Project Paths:
project_root = os.path.abspath("../")

unprocessed_path = os.path.join(project_root, "data", "HCP_zipped")
processed_path   = os.path.join(project_root, "data", "processed")
model_ready_path = os.path.join(project_root, "data", "model_ready")

sub_list_txt_path = os.path.join(project_root, "data", "full_3T_task_subjects.txt")

# WoMAD-specific variables:
target_tasks = ["WM", "EMOTION", "LANGUAGE"]

target_subtasks = {
    "WM"      : ["0bk_body", "0bk_faces", "0bk_places", "0bk_tools",
                 "2bk_body", "2bk_faces", "2bk_places", "2bk_tools"],
    "EMOTION" : ["fear", "neut"],
    "LANGUAGE": ["math", "story"],
}

TR = 0.72
target_parcellation = 360   # Glasser
target_timepoints   = 20    # Average trial length

rest_tasks    = ["REST1", "REST2"]
run_direction = ["LR"   , "RL"]

# Subjects with full 3T imaging protocol completed:
full_3T_task_subjects = []

with open(sub_list_txt_path, "r") as file:
    raw_list = file.read()
    str_list = raw_list.strip().split(",")
    num_list = [int(subID.strip()) for subID in str_list if subID.strip()]

full_3T_task_subjects = num_list

# Environment Setup Variables and Parameters:
## Dictionary for the model config and other global variables
lstm_config = {
    "hidden_size" : 128,
    "num_layers"  :   2,
    "dropout"     : 0.2
}

fusion_config = {
    "total_input_feats" : 512,      # TODO: Calculate total input based on lstm and conv4d output shapes.
    "hidden_size"       : 128
}

# Temporary variables for development:
pilot_subjects = [283543, 180937, 379657, 145632, 100206,
                  270332, 707749, 454140, 194847, 185038]

dev_subjects = [100206, 100408]
