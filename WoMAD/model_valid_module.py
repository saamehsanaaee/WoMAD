"""
    The model evaluation and validation module is set to:
    - Evaluate model performance on the validation dataset
    - Calculate metrics for the outputs

    This will allow us to select the best config for the model and avoid overfitting.
"""

import torch
from typing import Dict, Tuple, Any

from . import myProject_config

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

def calc_mse():
    """
    TO DO: Create function to calculate Mean Squared Error.
    """
    final_mse = 0

    return final_mse

def calc_r_sqrd():
    """
    TO DO: Create function to calculate R^2 score.
    """
    r_sqrd = 0

    return r_sqrd

def calc_all_metrics():
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

def run_valid_epoch():
    """
    TO DO: Create the function for running the validation epoch.
    """
    model.eval()
    total_val_loss = 0

    outputs = {}
    targets = {}

    avg_loss = total_val_loss / len("validation set")
    print(f"Epoch {epoch + 1} ---> Validation Loss: {avg_loss: .4f}")

    metrics = calc_all_metrics(outputs, targets)
    print(f"Validation Metrics: {metrics}")

    return avg_loss, metrics
