"""
   This module will allow us to get a better understanding of
   WoMAD's behaviour and prediction. We use these functions to
   create our visualizations.

   Essentially, the outputs of this module will let us form
   our conclusions and discussion.
"""

import numpy as np
import shap
import torch
from typing import Dict, Any

from . import WoMAD_config

def predict():
    """
    TO DO: Create the functions that allows us to use the model for inference.
    """
    return prediction

def visualize_and_interpret():
    """
    TO DO: Create the function that generates figures for each trial.
    """
    model.eval()

    # Visualize output
    # Generate Saliency map and other interpretation tools/metrics
    # Visualize the graph outputs
    # Save visualizations, maps, and metrics
    return 0

def run_test_set():
    """
    TO DO: Create the function that runs full test and analysis after training.
    """
    # Iterate through test set
    # Process test set
    # Visualize

    print("Test run complete. Test outputs saved to output directory.")
