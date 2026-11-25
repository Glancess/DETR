import torch

from moxie import MyDataset
from torch.utils.data import random_split

if __name__ == "__main__":
    dataset = MyDataset("/opt/data/private/datasets/yangzhuohan/coco128")

    val_num = round(len(dataset) * 0.2)
    train_num = len(dataset) - val_num

    train_dataset, val_dataset = random_split(
        dataset, [train_num, val_num], generator=torch.Generator().manual_seed(42)
    )
    for i in train_dataset.indices:
        print(i)
