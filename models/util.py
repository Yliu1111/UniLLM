import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import nn


# Load data from CSV file (used for SZ grid-based data)
def loaddata(data_name):
    """
    Load CSV data and reshape to (days, regions, width, height).
    """
    path = data_name
    data = pd.read_csv(path, header=None)
    return data.values.reshape(-1, 63, 10, 10)


# Normalize data to range [-1, 1]
def normalize(data):
    """
    Normalize data to the range [-1, 1].
    """
    min_val = torch.min(data)
    max_val = torch.max(data)

    if max_val - min_val == 0:
        return torch.zeros_like(data)

    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data


def process_d(speed, demand, inflow):
    """
    Process SZ data (speed, demand, inflow) with thresholding and normalization.
    Original data is SZ with 63 regions; output is reshaped to (N, 63, 100).
    """

    speed = torch.clamp(speed, max=140)     # Speed threshold (SZ)
    demand = torch.clamp(demand, max=100)   # Demand threshold (SZ)
    inflow = torch.clamp(inflow, max=100)   # Inflow threshold (SZ)

    # Normalize each modality
    normalized_speed = normalize(speed)
    normalized_demand = normalize(demand)
    normalized_inflow = normalize(inflow)

    # Reshape for later concatenation: 10x10 → 100
    res_speed = normalized_speed.unsqueeze(1).reshape(-1, 63, 100).float()
    res_demand = normalized_demand.unsqueeze(1).reshape(-1, 63, 100).float()
    res_inflow = normalized_inflow.unsqueeze(1).reshape(-1, 63, 100).float()

    return res_speed, res_demand, res_inflow


def process_small(speed, demand, inflow):
    """
    Process Xi'an/Chengdu data (speed, demand, inflow) with thresholding and normalization.
    Output shape is (days, 12, 1, 4, 10, 10), where 4 is the number of regions.
    """

    speed = torch.clamp(speed, max=140)
    demand = torch.clamp(demand, max=100)
    inflow = torch.clamp(inflow, max=100)

    normalized_speed = normalize(speed)
    normalized_demand = normalize(demand)
    normalized_inflow = normalize(inflow)

    res_speed = normalized_speed.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10).float()
    res_demand = normalized_demand.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10).float()
    res_inflow = normalized_inflow.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10).float()

    return res_speed, res_demand, res_inflow


def loaddata_city(city, prefix):
    """
    Load city-level data (Xi'an / Chengdu) for three modalities.
    Returns: speed, inflow, demand (all torch.Tensor), shape [162, 12, 4, 10, 10].
    """
    assert city in ['XA', 'CD'], f"Unsupported city: {city}"

    speed = np.load(f'{prefix}speed_{city}.npy')
    inflow = np.load(f'{prefix}inflow_{city}.npy')
    demand = np.load(f'{prefix}demand_{city}.npy')

    speed = torch.tensor(speed).float()
    inflow = torch.tensor(inflow).float()
    demand = torch.tensor(demand).float()

    return speed, inflow, demand


def process_d_city(speed, inflow, demand):
    """
    For Xi'an/Chengdu: apply clamp and [-1, 1] normalization to each modality.
    Train (first 3 regions) and test (last region) are normalized separately.
    Returns processed_speed, processed_inflow, processed_demand with shape [162, 12, 4, 10, 10].
    """

    def normalize(data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        if max_val - min_val == 0:
            return torch.zeros_like(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def process_tensor(tensor, max_val):
        tensor = torch.clamp(tensor, max=max_val)
        train = tensor[:, :, :3]   # First 3 regions (train)
        test = tensor[:, :, 3:]    # Last region (test)
        train_norm = normalize(train)
        test_norm = normalize(test)
        return torch.cat([train_norm, test_norm], dim=2)

    speed = process_tensor(speed, 140)
    inflow = process_tensor(inflow, 100)
    demand = process_tensor(demand, 100)

    return speed, inflow, demand
