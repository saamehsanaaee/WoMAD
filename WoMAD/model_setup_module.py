"""
    This module is designed to create the WoMAD modules and submodules as follows:
    - Module 1: A 3D U-Net that labels voxels that are active (in WM tasks)
    - Module 2: Includes 3 parallel submodules
        - Submodule A: An LSTM that handels temporal information
        - Submodule B: A 4D network that handels spatiotemporal information
        - Submodule C: A GNN that handels "Information Flow" during the WM task
    - Module 3: A fusion layer that combines the outputs of previous modules and generates the final outputs
"""

import torch
import torch.nn as nn

from . import WoMAD_config

class WoMAD_model(nn.Module):
    def __init__(self, config: dict):
        """
        Sets up the complete WoMAD model with all modules and submodules.
        """
        # Module 1: 3D-UNet
        # Module 2: Parallel Submodules
        ## Submodule A: LSTM (for temporal features)
        ## Submodule B: ConvNet4D (for spatiotemporal features)
        ## Submodule C: GNN (for information flow)
        # Module 3: Fusion Layer

    def forward(self, input: torch.Tensor):
        """
        The forward pass that manages how the data passes through modules.
        """
        # Module 1 input
        # Module 2 inputs
        ## Creating 4D data
        ## Submodule inputs
        # Module 3
        return outputs

    def _prepare_4d_data(self, input: torch.Tensor) -> torch.Tensor:
        """
        Helper method to create the 4D network for the second module.
        """
        return four_dim_data

def model_config(config: dict) -> WoMAD_model:
    """
    Initialized WoMAD and moves it to the device.

    Argument:
        config (dict): WoMAD config dictionary

    Returns:
        WoMAD: Model ready to be trained.
    """
    model = WoMAD_model(config)

    if config["system"]["use_gpu"] and torch.cuda.is_available():
        model.cuda()

    return model
