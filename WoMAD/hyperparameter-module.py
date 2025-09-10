"""
    This module will allow us to test different versions of WoMAD using
    the training loop with a (test) subset of data. These different versions
    will be evaluated during each loop, their scores will be assessed, and
    the best result (the best performing weights) will go through the actual
    training loop to create the final version of the model.
    
    This module will work along the model evaluation and validation module
    and allow us to save the best version of the WoMAD model.
"""
