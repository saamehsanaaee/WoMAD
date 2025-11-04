"""
    This module is designed to create the WoMAD modules and submodules as follows:
    - Information Flow Module:
        - Granger Causality Analysis (GCA) for Effective Connectivity (EC)
        - Hidden Markov Model (HMM) for Dynamic Functional Connectivity
        - Supervised learning of info flow by Temporal GNN (TGNN)
    - Core Module:
        - Submodule A: A 3D-UNet that labels active nodes (voxels or parcels)
        - Parallel Submodules:
            - B-1: An LSTM that handels temporal information
            - B-2: A 4D convolutional network which handels spatiotemporal information
        - Submodule C: A fusion layer that combines the outputs of previous module,
                       generating the final output
"""

import torch
import torch.nn as nn

from . import WoMAD_config

class WoMAD_model(nn.Module):
    def __init__(self, config: dict):
        """
        Sets up the complete WoMAD model with all modules and submodules.
        (Each module and submodule includes a dynamic input layer that matches the size of input.)
        """
        # Information Flow Module
        ## Effective Connectivity: GCA
        ## Dynamic Functional Connectivity: HMM
        ## Final info-flow manifold: Temporal GNN

        # Core Module
        ## Submodule A: 3D-UNet
        ## Parallel submodule B-1: LSTM (Temporal features)
        ## Parallel sunmodule B-2: ConvNet4D (Spatiotemporal features)
        ## Submodule C: Fusion Layer

    def forward(self, input: torch.Tensor, module_selection: str):
        """
        The forward pass that manages how the data passes through modules.
        Path of data passage is based on selected module: "info-flow" or "core"
        """
        # Info-flow input
        # 3D-UNet input
        # LSTM input
        # ConvNet4D input
        # Fusion input
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
