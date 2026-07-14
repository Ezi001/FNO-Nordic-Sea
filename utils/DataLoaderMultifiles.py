from DataLoader import NordicSeaCurrentDataset
from torch.utils.data.distributed import DistributedSampler
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

def get_data_loader(params, files_pattern, distributed, train):

  dataset = NordicSeaCurrentDataset(params, files_pattern, train)
  sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size),
                          num_workers=params.num_data_workers,
                          shuffle=False, #(sampler is None),
                          sampler=sampler if train else None,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset