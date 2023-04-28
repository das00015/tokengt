"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Optional
# from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
import torch
from .pcqm4mv2_pyg import PygPCQM4Mv2Dataset, UPFD
from torch_geometric.data import Dataset
import torch_geometric.data as data
from ..pyg_datasets import TokenGTPYGDataset
import torch.distributed as dist
import os


class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class OGBDatasetLookupTable:
    @staticmethod
    def GetOGBDataset(dataset_name: str, seed: int) -> Optional[Dataset]:
        inner_dataset = None
        train_idx = None
        valid_idx = None
        test_idx = None
        if dataset_name == "pcqm4mv2":   
            
            os.system("mkdir -p dataset/pcqm4m-v2/")
            os.system("touch dataset/pcqm4m-v2/RELEASE_v1.txt")
            inner_dataset = UPFD(root=".", name="gossipcop", feature="content", split="train")
            #idx_split = inner_dataset.get_idx_split()
           
            idx_split={"train":torch.LongTensor(range(len(inner_dataset))),"valid":torch.LongTensor(range(len(inner_dataset))),"test-dev":torch.LongTensor(range(len(inner_dataset)))}
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]

        if dataset_name == "upfd":
            
            print("now downloading upfd dataset")
            train_dataset = UPFD(root=".", name="gossipcop", feature="content", split="train")
            test_dataset = UPFD(root=".", name="gossipcop", feature="content", split="test")
            val_dataset = UPFD(root=".", name="gossipcop", feature="content", split="val")
            # concatenate all three together to form inner ; currently we are using train for all three 
            train_range  = len(train_dataset)
            test_range  = len(test_dataset)
            val_range  = len(val_dataset)

            # Combine the three splits into one dataset
            class CombinedUPFD(data.InMemoryDataset):
                def __init__(self, root, transform=None, pre_transform=None):
                    super().__init__(root, transform, pre_transform)
                    self.data, self.slices = self.collate([train_dataset[i] for i in range(len(train_dataset))] +
                                                        [test_dataset[i] for i in range(len(test_dataset))] +
                                                        [val_dataset[i] for i in range(len(val_dataset))]
                                                        )

            # Create the combined dataset
            inner_dataset = CombinedUPFD(root=".", transform=None, pre_transform=None)

            
            # this gets the ids of the train, test and val
            idx_split={"train":torch.LongTensor(range(0,len(train_dataset))),
                       "valid":torch.LongTensor(range(len(train_dataset), len(train_dataset)+len(test_dataset))),
                       "test-dev":torch.LongTensor(range(len(train_dataset)+len(test_dataset), len(train_dataset)+len(test_dataset)+len(val_dataset)))}
            
            
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]
            

        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for ogb source.")
        return (
            None
            if inner_dataset is None
            else TokenGTPYGDataset(
                inner_dataset, seed, train_idx, valid_idx, test_idx
            )
        )
