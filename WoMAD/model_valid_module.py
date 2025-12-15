"""
    The model evaluation and validation module is set to:
    - Evaluate model performance on the validation dataset
    - Calculate metrics for the outputs

    This will allow us to select the best config for the model and avoid overfitting.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from typing import Dict, Tuple, Any

from . import WoMAD_config

def calc_graph_overlap():
    """
    TO DO: Create function to calculate graph overlap for Info-Flow module.
    """
    # Create graph network with the info-flow output.
    # Calculate graph overlap:
    ## Step 1: Normalization (Nodes and weights)
    ## Step 2: Calculate overlap (Edge-wise percentage with thresholding and Jaccard Indx
    ##         and Weighted correlation with vectorization and Pearson's
    ## Step 3: Analyze topological similarity (Compare architecture using key topological metrics)

def calc_dice_coeff():
    """
    TO DO: Create function to calculate Dice coefficient.
    """
    dice_coeff = 0

    return dice_coeff

def calc_r_sqrd():
    """
    TO DO: Create function to calculate R^2 score.
    """
    r_sqrd = 0

    return r_sqrd

def calc_all_metrics():         # NOTE: This is for the information flow module.
    """
    TO DO: Create the function to calculate all metrics (Dice, MSE, R^2, overall score)
    """
    dice_score = calc_dice_coeff()
    mean_sqrd_err = calc_mse()
    r_sqrd = calc_r_sqrd()

    overall_score = (dice_score + r_sqrd) / 2

    metrics_dict = {
        "Dice_coefficient": dice_score,
        "Mean_Sqrd_Error" : mean_sqrd_err,
        "R_squared"       : r_sqrd,
        "Overall_score"   : overall_score
    }

    return metrics_dict

def run_valid_epoch(model, data_loader: DataLoader,
                    loss_funcs: Dict[str, nn.Module],
                    epoch: int, config: dict
                    ) -> Tuple[float, Dict[str, float]]:
    """
    Runs one validation epochs and calculates loss and metrics.
    """
    model.eval()
    total_val_loss = 0
    total_samples = 0

    all_overall_pred = []
    all_overall_target = []
    all_node_pred = []
    all_node_target = []

    overall_weight = WoMAD_config.training_loss_weights["overall_loss_weight"]
    node_weight    = WoMAD_config.training_loss_weights["node_loss_weight"]

    overall_loss_fn = loss_funcs["overall_score_loss"]
    node_loss_fn    = loss_funcs["node_score_loss"]

    with torch.no_grad():
        for data, overall_target, node_target in data_loader:
            overall_target = overall_target.float()
            node_target = node_target.float()

            overall_pred, node_pred = model(data)

            loss_overall = overall_loss_fn(overall_pred.squeeze(), overall_target)
            loss_node    = node_loss_fn(node_pred, node_target)
            combined_loss = (overall_weight  * loss_overall) + (node_weight * loss_node)

            total_val_loss += combined_loss.item() * data.size(0)
            total_samples  += data.size(0)

            all_overall_pred.append(overall_pred)
            all_node_pred.append(node_pred)

            all_overall_target.append(overall_target)
            all_node_target.append(node_target)

    avg_loss = total_val_loss / total_samples

    print(f"Epoch {epoch+1:02d} | Validation Loss: {avg_loss: .6g}")

    final_overall_pred   = torch.cat(all_overall_pred).squeeze()
    final_overall_target = torch.cat(all_overall_target)

    final_node_pred   = torch.cat(all_node_pred)
    final_node_target = torch.cat(all_node_target)

    metrics = calc_all_metrics(finall_overall_pred,
                               final_node_pred,
                               final_overall_target,
                               final_node_target)

    print(f"Validation metrics: {metrics}")

    return avg_loss, metrics
