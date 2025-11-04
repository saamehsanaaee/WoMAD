"""
    This is where the magic happens.
    This module will use all submodules of the `./WoMAD` directory
    to create the FINAL VERSION of WoMAD.

    This module manages the entire pipeline and loades the config,
    sets up the data loaders, WoMAD modules, and then allows us to
    execute the training and validation loops.

    The best-performing model is saved and figures are created using this module.
"""

import os
import time
import torch
from typing import Dict, Any

from WoMAD import WoMAD_config
from WoMAD import data_module
from WoMAD import model_setup_module
from WoMAD import model_train_module
from WoMAD import model_valid_module
from WoMAD import result_interp_module

# Terminal functions and UI
## print status, success, or error
## clear screen
## Welcome/Completion

def run_WoMAD(config):
    # Environment setup
    # Data and initial processing
    # Model setup
    # Training (and hyperparameter search)
    # Post-training: Analysis, Visualization, and Interpretation

if __name__ == "__main__":
    config = WoMAD_config.load_config()
    run_WoMAD(config)
