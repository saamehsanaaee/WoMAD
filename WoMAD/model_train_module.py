"""
    Create the training loop that works alongside the `model-setup.py` functions.
    This training loop will include K-fold cross-validation.

    Additionally, the train module will produce training stats that will be used in the test and evaluation modules.
"""

import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold

from . import WoMAD_config
from .model_setup_module import DynamicInput, WoMAD_core

def WoMAD_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Configures the optimizer for WoMAD model.
    """
    lr = WoMAD_config.training_config["learning_rate"]
    return torch.optim.Adam(model.parameters(), lr = lr)

def WoMAD_loss_function(config: dict) -> Dict[str, nn.Module]:
    """
    Create a dictionary of loss functions.
    """
    loss_func_dict = {
        "overall_score_loss": nn.MSELoss(),
        "node_score_loss"   : nn.MSELoss()
    }
    return loss_func_dict

def run_training_epoch(model: WoMAD_core, data_loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       loss_funcs: Dict[str, nn.Module],
                       epoch: int, config: dict):
    """
    Function to run a single training epoch.
    """
    model.train()
    total_train_loss = 0
    total_samples = 0

    overall_weight = WoMAD_config.training_loss_weights["overall_loss_weight"]
    node_weight    = WoMAD_config.training_loss_weights["node_loss_weight"]

    overall_loss_fn = loss_funcs["overall_score_loss"]
    node_loss_fn    = loss_funcs["node_score_loss"]

    for batch_indx, (data, overall_target, node_target) in enumerate(data_loader):
        overall_target = overall_target.float()
        node_target    = node_target.float()

        optimizer.zero_grad()

        # Forward pass to return (overall and node-wise prediction)
        overall_pred, node_pred = model(data)

        # Calculate losses
        loss_overall = overall_loss_fn(overall_pred.squeeze(), overall_target)
        loss_nodes   = node_loss_fn(node_pred, node_target)

        combined_loss = (overall_weight * loss_overall) + (node_weight * loss_nodes)

        # Backpropagate
        combined_loss.backward()
        optimizer.step()

        total_train_loss += combined_loss.item() * data.size(0)
        total_samples += data.size(0)

    avg_loss = total_train_loss / total_samples
    print(f"Epoch {epoch+1:02d} | Training Loss: {avg_loss: .6f}")
    return avg_loss

def run_kfold_training(dataset, config: dict):
    """
    Executes K-fold cross validation for training.

    Arguments:
        dataset (Dataset): The WoMAD data which contains all target subject data.
        config     (dict): Configuration dictionary.

    Returns:
        List of dictionaries with training stats for each training fold.
    """
    k_folds = WoMAD_config.training_config["k_folds"]
    num_epochs = WoMAD_config.training_config["num_epochs"]
    batch_size = WoMAD_config.training_config["batch_size"]

    kfold = KFold(n_splits = k_folds, shuffle = True, random_state = 42)
    all_kfold_train_stats = []

    loss_funcs = WoMAD_loss_function(config)

    print(f"K-fold cross-validation for {k_folds} folds over {len(dataset)} trials:")

    for fold, (train_indx, valid_indx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{k_folds}:")

        train_subset = Subset(dataset, train_indx)
        valid_subset = Subset(dataset, valid_indx)

        train_loader = DataLoader(train_subset, batch_size = batch_size, shuffle = True)
        valid_loader = DataLoader(valid_subset, batch_size = batch_size, shuffle = False)

        print(f"Train samples: {len(train_subset)}, Validation samples: {len(valid_subset)}")

        # Model initiation and setup
        model = WoMAD_core(config)
        # TODO: Add the device logic (model.cuda())
        optimizer = WoMAD_optimizer(model, config)

        fold_history = {"train_loss"  : [],
                        "valid_loss"  : [],
                        "val_metrics" : []}

        for epoch in range(num_epochs):
            train_loss = run_training_epoch(model, train_loader, optimizer, loss_funcs, epoch, config)
            fold_history["train_loss"].append(train_loss)

            valid_loss = run_validation_epoch(model, val_loader, loss_funcs, epoch, config)
            fold_history["valid_loss"].append(valid_loss)

        all_kfold_train_stats.append({"fold": fold + 1, "history": fold_history})

        print("\nK-fold training complete.")

    return all_kfold_train_stats
