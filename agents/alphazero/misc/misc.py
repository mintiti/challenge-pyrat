'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math

__all__ = ['AverageMeter','AddNoiseToRoot']

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AddNoiseToRoot:
    """Context manager to add and remove add_noise to the root node"""

    def __init__(self, node, add_noise=True, dir_epsilon=0.25, dir_noise=2.5):
        self.node = node
        self.original_priors = node.child_priors.copy()
        self.add_noise = add_noise
        self.dir_epsilon = dir_epsilon
        self.dir_noise = dir_noise

    def __enter__(self):
        # Add dirichlet add_noise to the root node
        if self.add_noise:
            self.node.child_priors = (1 - self.dir_epsilon) * self.node.child_priors
            self.node.child_priors += self.dir_epsilon * np.random.dirichlet(
                [self.dir_noise] * self.node.child_priors.size)
        return self.node

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Revert the dirichlet add_noise
        self.node.child_priors = self.original_priors