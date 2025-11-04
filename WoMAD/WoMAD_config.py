"""
    This file contains all global variables used throughout the WoMAD project.
    (Variables, paths, and MISC code snippets.)
"""

import os

# Project Paths:
project_root = os.path.abspath("../WoMAD_main.py")

unprocessed_path = os.path.join(project_root, "data", "HCP_zipped")
processed_path = os.path.join(project_root, "data", "processed")
model_ready_path = os.path.join(project_root, "data", "model_ready")

# Subject Paths:
# These paths haven't been created yet. Please stand by! Thank.
subject_task_path = os.path.join(project_root, "")
subject_fmri_path = os.path.join(project_root, "")
subject_evfile_path = os.path.join(project_root, "")

# WoMAD-specific variables:
target_tasks = ["EMOTION"   , "GAMBLING"  , "LANGUAGE", "MOTOR",
                "RELATIONAL", "SOCIAL"    , "WM"]

rest_tasks    = ["REST1", "REST2"]
run_direction = ["LR"   , "RL"]

# Environment Setup Variables and Parameters:

# Temporary variables for development:
dev_subjects = ["100206", "100307", "100408", "100610", "101006",
                "101107", "101309", "101915", "102109", "102311"]
