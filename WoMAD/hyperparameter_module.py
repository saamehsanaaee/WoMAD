"""
    This module will act as the optimization module and allow us to select the best hyperparameters for WoMAD.
    It will test different combinations for the defined architecture of WoMAD and help us select the best results based on the training trials on a validation subset and evaluating the performance of each set of parameters.
    This will allow us to train the final WoMAD model using the most effective parameters.
"""

import optuna
from typing import Dict, Any, Callable

from . import WoMAD_main, WoMAD_config

def define_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the search space for hyperparameter optimization.

    Arguments:
        trial (optuna.Trial): The trial object that suggests parameters

    Returns:
        Dict[str, Any]: A dictionary of suggested hyperparameters.
    """
    # Main parameters (Learning rate, batch size, number of epochs)
    learning_rate = 0
    batch_size = 0
    epochs = 0
    # Model-specific parameters (Hidden layers, dropout rates)
    hidden_layers = 0
    dropout_rate = 0

    suggested_parameters = {
        "learning_rate": learning_rate,
        "batch_size"   : batch_size,
        "epochs"       : epochs,
        "hidden_layers": hidden_layers,
        "dropout_rate" : dropout_rate
    }

def objective(trial: optuna.Trial) -> float:
    """
    TO DO: Define the objective function for Optuna.
    """
    hyperparameters = define_search_space(trial)

    config = WoMAD_config.load_config()
    config["training"].update(hyperparameters)

    final_valid_metric = WoMAD_main.run_pipeline(config)

    return final_valid_metric

def run_hyperparameter_optim():
    """
    TO DO: Define the main hyperparameter search function.
    """
    # Define target for optimization (min loss, min MSE, etc.)

    # Print and save the results (best trial, best parameters, best target metric)

    # Save to file as well

if __name__ == "__main__":
    run_hyperparameter_optim()
