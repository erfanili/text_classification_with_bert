from lightning import LightningDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch


class AmazonReviewDataModule(LightningDataModule):
    def __init__(self,dataset_name, subset, tokenizer, batch_size = 64):
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
    def prepare_data(self):
        load_dataset(self.dataset_name, self.subset, trust_remote_code=True)
    
    def setup(self, stage=None):
        full = load_dataset(self.dataset_name, self.subset, split='full')
        split = full.train_test_split(test_size=0.2, seed = 42)
        val_test = split['test'].train_test_split(test_size=0.5)
        self.train_data = split['train']
        self.val_data = val_test['train']
        self.test_data = val_test['test']

    def collate_fn(self,batch):
        texts = [x['title'] if x['title'] else "" for x in batch]
        labels = [int(x['rating']-1) for x in batch]
        encodings = self.tokenizer(texts, padding = True, return_tensors = 'pt')
        return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, collate_fn=self.collate_fn, num_workers=4)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size, collate_fn=self.collate_fn, num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.test_data,batch_size = self.batch_size, collate_fn=self.collate_fn, num_workers=4)
    