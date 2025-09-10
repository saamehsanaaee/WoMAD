"""
    This file contains all global variables used throughout the WoMAD project.
    (Variables, paths, and MISC code snippets.)
"""

import os

# PATHS
## Absolute path to the root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## Relative paths based on the project root
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_READY_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "model-ready")

# WoMAD-specific variables
target_tasks  = ["EMOTION"   , "GAMBLING"  , "LANGUAGE", "MOTOR",
                 "RELATIONAL", "SOCIAL"    , "WM"]

rest_tasks    = ["REST1", "REST2"]
run_direction = ["LR"   , "RL"]
