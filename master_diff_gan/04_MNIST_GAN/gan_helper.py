import math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def blend_samples(dataset, indices, distances, reference=None, epsilon=1e-10):
    """
    Blend samples using inverse distance weighting and optionally compare to reference
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Dataset containing samples to blend, shape [n_samples, ...] 
        Can be either activations [60000, features] or images [60000, 1, 28, 28]
    indices : numpy.ndarray
        Indices of samples to blend
    distances : numpy.ndarray
        Distances to selected samples, matching indices
    reference : numpy.ndarray or torch.Tensor, optional
        Reference sample to compare blended result against
    epsilon : float, default=1e-10
        Small constant to avoid division by zero
        
    Returns:
    --------
    blended : numpy.ndarray
        Blended sample with same shape as a single item from dataset
    ref_distance : float or None
        L2 distance to reference sample if provided, else None
    """
    # Ensure inputs are numpy arrays
    dataset = np.asarray(dataset)
    distances = np.asarray(distances)
    
    # Calculate inverse distance weights
    weights = 1 / (distances + epsilon)
    weights = weights / np.sum(weights)
    
    # Select samples to blend
    samples = dataset[indices]
    
    # For image inputs, reshape to [n_samples, -1]
    if len(samples.shape) > 2:
        samples_flat = samples.reshape(len(samples), -1)
    else:
        samples_flat = samples
        
    weights = weights.reshape(-1, 1)  # Shape: [n_samples, 1]
    
    # Calculate weighted average
    blended_flat = np.sum(samples_flat * weights, axis=0)
    
    # Reshape back to original shape
    if len(dataset.shape) > 2:
        # For images, reshape to [1, 28, 28] or [1, 1, 28, 28]
        if len(dataset.shape) == 4:
            blended = blended_flat.reshape(1, dataset.shape[2], dataset.shape[3])
        else:
            blended = blended_flat.reshape(dataset.shape[1:])
    else:
        # For activations, keep flat
        blended = blended_flat
    
    # Calculate distance to reference if provided
    ref_distance = None
    if reference is not None:
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(reference):
            reference = reference.cpu().numpy()
        reference = np.asarray(reference).flatten()
        ref_distance = np.sqrt(np.sum((blended_flat - reference.flatten()) ** 2))
    
    return blended, ref_distance