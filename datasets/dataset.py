from abc import ABC, abstractmethod
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_split):
        assert data_split in ['train', 'valid', 'test']
        self.data_split = data_split

    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
