from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(self,
                dataset,
                batch_size,
                num_workers=0,
                # num_workers=self.num_workers,
                pin_memory=True):
        
        super().__init__()
        self.dataet = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory