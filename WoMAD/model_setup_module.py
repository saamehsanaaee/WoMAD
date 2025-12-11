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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import WoMAD_config

# Dynamic Adapter
TARGET_NODE_COUNT = 360 # 360 parcels based on HCP and Glasser parcellation

class DynamicInput(nn.Module):
    """
    This module handles input data with different number of voxels
    and adapts it for the modules (Info flow or Core) of the WoMAD model.
    """
    def __init__(self, target_nodes: int = TARGET_NODE_COUNT):
        super().__init__()
        self.target_nodes = target_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamically adapts the input data to defined dimension.

        Input:
            x : 3D tensor with shape (batch, current_nodes, timepoints)

        Output:
            The dynamically adapted 3D tensor with shape (batch, target_nodes, timepoints)
        """
        batch, current_nodes, timepoints = x.shape

        if current_nodes == self.target_nodes:
            return x

        elif current_nodes > self.target_nodes:
            # TODO: Add voxel-to-parcel downsampling (requires a parcellation map)

            # Current solution for downsampling: Adaptive Pooling (linear projection)
            x_reshaped = x.transpose(1, 2)

            x_pooled = F.adaptive_avg_pool1d(x_reshaped, self.target_nodes)

            x_pooled_out = x_pooled.transpose(1, 2)

            return x_pooled_out

        else: # current_nodes < self.target_nodes
            required_padding = self.target_nodes - current_nodes

            x_padded = F.pad(0, 0, 0, required_padding)

            return x_padded

# Information flow module
class WoMAD_info_flow(nn.Module):
    def __init__(self, config: dict):
        """
        Sets up the complete WoMAD model with all modules and submodules.
        (Each module and submodule includes a dynamic input layer that matches the size of input.)
        """
        # Information Flow Module
        ## Effective Connectivity: GCA
        ## Dynamic Functional Connectivity: HMM
        ## Final info-flow manifold: Temporal GNN

    def forward(self, input: torch.Tensor, module_selection: str):
        """
        The forward pass that manages how the data passes through modules.
        """

        return outputs

    def _prepare_4d_data(self, input: torch.Tensor) -> torch.Tensor:
        """
        Helper method to create the 4D network for the second module.
        """
        return four_dim_data

def model_config(config: dict) -> WoMAD_info_flow:
    """
    Initialized WoMAD and moves it to the device.

    Argument:
        config (dict): WoMAD config dictionary

    Returns:
        WoMAD: Model ready to be trained.
    """
    model = WoMAD_info_flow(config)

    if config["system"]["use_gpu"] and torch.cuda.is_available():
        model.cuda()

    return model

# Core module
class WoMAD_core(nn.Module):
    def __init__(self, config: dict):
        """
        Sets up the complete WoMAD model with all modules and submodules.
        (Each module and submodule includes a dynamic input layer that matches the size of input.)
        """
        super().__init__()

        # Dynamic Adapter
        self.dyn_input_adapter = DynamicInput(target_nodes = 360)
        # TODO: Define target_nodes in WoMAD_config.

        target_nodes = 360 # Temporary. Should be defined in config.

        timepoints = 20 # Temporary. Should be defined using other functions.

        # Core Module
        ## Submodule A: 3D-UNet
        self.unet_input = nn.Linear(T, T)
        self.segment_and_label = nn.Sequential(
            # TODO: Add the nn.Conv3d() function
            nn.Indentity()
        )

        ## Parallel submodule B-1: LSTM (Temporal features)
        self.lstm_input = nn.Linear(target_nodes, target_nodes)
        self.temporal_lstm = nn.LSTM(input_size  = timepoints,
                                     hidden_size = config["lstm"]["hidden_size"],
                                     batch_first = TRUE)

        ## Parallel sunmodule B-2: ConvNet4D (Spatiotemporal features)
        # TODO: Prepare data for the ConvNet4D.
        self.spatiotemporal_cnv4d = nn.Sequential(
            # TODO: Add the nn.Conv3d() function
            nn.Identity()
        )

        ## Submodule C: Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, config["data"]["num_classes"])
        )

    def _prepare_4d_data(self, input: torch.Tensor) -> torch.Tensor:
        """
        Helper method to create the 4D network for the second module.

        Input:
            Tensor with shape (batch, target_nodes, timepoints).
            target_nodes is the flat spatial dimension.

        Output:
            5D tensor for the 4D ConvNet with
            shape (batch, C=1, timepoints, X, Y, Z)
        """
        batch, nodes, timepoints = input.shape
        # TODO: Create the mapping array to place nodes into a X*Y*Z grid.

        four_dim_data = input.unsqueeze(1)

        return four_dim_data

    def forward(self, input: torch.Tensor, module_selection: str):
        """
        The forward pass that manages how the data passes through modules.

        Input:
            Tensor with shape (batch, current_nodes, timepoints)
        """
        # Dynamic Input Adaption
        x_dynamically_adapted = self.dyn_input_adapter(input)

        # 3D-UNet
        # TODO: Create the required 5D input for UNet (with spatial reconstruction).
        x_ready_for_unet = self.segment_and_label(x_dynamically_adapted.mean(dim=2))

        # LSTM
        x_for_lstm = x_dynamically_adapted.transpose(1, 2)          # NOTE: Shape is now (batch, timepoints, target_nodes)
        _, (h_n, c_n) = self.temporal_lstm(x_for_lstm)
        lstm_out = h_n[-1]

        # ConvNet4D
        x_for_conv4d = self._prepare_4d_data(x_dynamically_adapted)
        conv4d_out = self.spatiotemporal_cnv4d(x_for_conv4d).mean(dim = [1, 2, 3])

        # Fusion Layer
        fused_feats = torch.cat([lstm_out, conv4d_out], dim = 1)

        core_module_out = self.fusion_layer(fused_feats)

        return core_module_out


def model_config(config: dict) -> WoMAD_core:
    """
    Initialized WoMAD and moves it to the device.

    Argument:
        config (dict): WoMAD config dictionary

    Returns:
        WoMAD: Model ready to be trained.
    """
    model = WoMAD_core(config)

    if config["system"]["use_gpu"] and torch.cuda.is_available():
        model.cuda()

    return model
