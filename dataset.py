import os
import torch

from torch_geometric.data import InMemoryDataset
class DessinsDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform = None, pre_transform = None):
        """
        :param root: where dataset is stored
        :param transform: nm
        :param pre_transform: nm
        """

        self.data_list = data_list
        super(DessinsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root,'Dessins','raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'Dessins', 'processed')

    @property
    def raw_file_names(self):
        return 'data.pt'

    @property
    def processed_file_names(self):
        return 'data_processed.pt'

    def process(self):
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    print("hi")
    dataset = DessinsDataset('.data')
    print(dataset.__dict__)
