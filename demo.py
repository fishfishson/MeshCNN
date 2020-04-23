from options.train_options import TrainOptions
from data.dataloader import MSDSurfTrainDataset

opts = TrainOptions().parse()
dataset = MSDSurfTrainDataset(opts)
import torch
import numpy as np


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})
    return meta


dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=2,
                                         shuffle=False,
                                         num_workers=2,
                                         collate_fn=collate_fn)
for i, data in enumerate(dataloader):
    if i == 0:
        break
print(data['img_patch'].shape)
