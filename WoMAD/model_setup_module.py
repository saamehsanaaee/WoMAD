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

            x_padded = F.pad(x, (0, 0, 0, required_padding))

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

        target_nodes = WoMAD_config.target_parcellation         # 360, Temporary.
        timepoints = WoMAD_config.target_timepoints             # 20, Temporary.

        lstm_h_size = WoMAD_config.lstm_config["hidden_size"]   # 128

        conv4d_out_size = 64                                    # From simplified config

        # Dynamic Adapter
        self.dyn_input_adapter = DynamicInput(target_nodes = 360)

        # Core Module
        ## Submodule A: 3D-UNet
        ## Input shape = (batch, target_nodes, timepoints)
        self.segment_and_label = nn.Sequential(
            nn.Conv1d(in_channels = target_nodes,
                      out_channels = target_nodes,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Identity()          # FIX: Placeholder should be replaced with the 3D-UNet.
        )

        ## Parallel submodule B-1: LSTM (Temporal features)
        self.temporal_lstm = nn.LSTM(input_size  = target_nodes,
                                     hidden_size = lstm_h_size,
                                     num_layers  = WoMAD_config.lstm_config["num_layers"],
                                     dropout     = WoMAD_config.lstm_config["dropout"],
                                     batch_first = True)

        ## Parallel submodule B-2: ConvNet4D (Spatiotemporal features)
        self.spatiotemporal_cnv4d = nn.Sequential(
            nn.Conv2d(in_channels  = 1,
                      out_channels = 32,
                      kernel_size  = (3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(in_channels  = 32,
                      out_channels = conv4d_out_size,
                      kernel_size  = (3, 3), padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        ## Submodule C: Fusion Layer
        fusion_input_size = lstm_out_size + conv4d_out_size     # 192
        self.fusion_block = nn.Sequential(
            nn.Linear(fusion_input_size, WoMAD_config.fusion_config["hidden_size"]),
            nn.ReLU()
        )

        ### Overall, WM-based activity score:
        self.overall_activity_score = nn.Linear(WoMAD_config.fusion_config["hidden_size"], 1)

        ### Node-based (voxel-based or parcel-based) activity scores:
        self.node_wise_activity_scores = nn.Linear(WoMAD_config.fusion_config["hidden_size"], target_nodes)

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The forward pass that manages how the data passes through modules.

        Sequence for forward pass:
            Input -> Segmentation (3D-UNet) -> (LSTM, Conv4D) -> Fusion

        Input:
            Tensor with shape (batch, current_nodes, timepoints)
        """
        # Dynamic Input Adaption
        x_dynamically_adapted = self.dyn_input_adapter(input)

        # 3D-UNet
        unet_out_timeseries = self.segment_and_label(x_dynamically_adapted)

        # LSTM
        x_for_lstm = unet_out_timeseries.transpose(1, 2)
        _, (h_n, _) = self.temporal_lstm(x_for_lstm)
        lstm_out = h_n[-1]

        # ConvNet4D
        x_for_conv4d = self._prepare_4d_data(unet_out_timeseries)
        conv4d_out = self.spatiotemporal_cnv4d(x_for_conv4d)

        # Fusion Layer
        fused_feats = torch.cat([lstm_out, conv4d_out], dim = 1)
        shared_features = self.shared_fusion_block(fused_feats)

        overall_score = self.overall_activity_score(shared_features)
        node_scores   = self.node_wise_activity_scores(shared_features)

        return overall_score, node_scores


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
