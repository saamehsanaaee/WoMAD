"""
    This module uses a small, randomly generated subset of data to test the functionality of each module.
    We test the modules separately and in combination to ensure the pipeline is working as we designed it to do. (Data, WoMAD modules, train, and evaluation loops.)
"""

import pytest
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

from WoMAD import WoMAD_config
from WoMAD import data_module
from WoMAD import model_setup_module
from WoMAD import model_train_module
from WoMAD import model_valid_module
from WoMAD import result_interp_module

# TODO: Create dummy data OR sample a tiny subset of the actual dataset
# TODO: Create tests for each and every single function or method.

# TESTS:
def test_data_module():

def test_model_setup():

def test_model_training():

def test_model_validation():

def test_result_interpretation():
