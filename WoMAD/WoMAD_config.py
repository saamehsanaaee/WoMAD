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

# WoMAD-specific variables:
target_tasks = ["WM", "EMOTION", "LANGUAGE"]

target_subtasks = {
    "WM"      : ["0bk_body", "0bk_faces", "0bk_places", "0bk_tools",
                 "2bk_body", "2bk_faces", "2bk_places", "2bk_tools"],
    "EMOTION" : ["fear", "neut"],
    "LANGUAGE": ["math", "story"],
}

TR = 0.72

rest_tasks    = ["REST1", "REST2"]
run_direction = ["LR"   , "RL"]

# Subjects with full 3T imaging protocol completed:
full_3T_task_subjects = []

with open("../data/full_3T_task_subjects.txt", "r") as file:
    raw_list = file.read()
    str_list = raw_list.strip().split(",")
    num_list = [int(subID.strip()) for subID in str_list if subID.strip()]

full_3T_task_subjects = num_list

# Environment Setup Variables and Parameters:

# Temporary variables for development:
dev_subjects = ["100206", "100408"]
