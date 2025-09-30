"""
    Create the training loop that works alongside the `model-setup.py` functions.
    This training loop will include K-fold cross-validation.

    Additionally, the train module will produce training stats that will be used in the test and evaluation modules.
"""

import torch
import torch.nn as nn

from . import WoMAD_config

def WoMAD_optimizer(model, config):
    """
    Configures the optimizer for WoMAD model.
    """
    return torch.optim.Adam(model.parameters(), lr = config["training"]["learning_rate"])

def WoMAD_loss_function(config):
    """
    Create a dictionary of loss functions.
    """
    loss_func_dict = {
        "UNet_loss"    : nn.BCEWithLogitsLoss(),
        "four_dim_loss": nn.MSELoss(),
        "graph_loss"   : nn.MSELoss()
    }

def run_training_epoch(model, data_loader, optimizer, loss_funcs, epoch, config):
    """
    Function to run a single training epoch.
    """
    model.train()
    total_train_loss = 0

    # For each batch in the data_loader:
    ## Forward pass
    ## Calculate loss
    ## Backpropogation
    ## Add the loss to total_loss

    avg_loss = total_loss / len(data_loader)
          print(f"Epoch {epoch+1} ---> Training Loss: {avg_loss: .4f}")
    return avg_loss
