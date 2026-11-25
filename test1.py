import torch

from moxie import MyDataset
from torch.utils.data import random_split


def Getrealimage():
    dataset = MyDataset("/opt/data/private/datasets/yangzhuohan/coco128")

    val_num = round(len(dataset) * 0.2)
    train_num = len(dataset) - val_num

    train_dataset, val_dataset = random_split(
        dataset, [train_num, val_num], generator=torch.Generator().manual_seed(42)
    )
    for i in train_dataset.indices:
        print(i)
    count = 0
    show = []
    for filename in dataset.files:

        index = dataset.files.index(filename)
        if index in train_dataset.indices and (count < 10):
            print(filename)
            count += 1
            show.append(filename)
