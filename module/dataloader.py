from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CNSDataset(Dataset):
    def __init__(self,
                 path,
                 tokenizer,
                 n_data):
        self.tokenizer = tokenizer

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        input_ids, labels = [], []

        t = lines[:int(n_data)] if len(lines) > n_data else lines
        for line in t:
            input, output = line.split(',')
            input, output = input.replace('"', '').strip(), output.strip()
            input_ids.append(input)
            labels.append(output)
        self.data = list(zip(input_ids, labels))

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
class CNSDataLoader:
    def __init__(self,
                 config,
                 tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.prefix = config.prefix
        self.train_path = config.train_data_path
        self.test_path = config.test_data_path
        self.batch_size = config.batch_size
        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.prefetch_n_workers = config.prefetch_n_workers
        self.verbose = config.verbose
        self.train_n_data = config.train_n_data
        self.test_n_data = config.test_n_data

    def collate_fn(self, batch):
        input_ids, labels = [], []

        for data in batch:
            inp, outp = data
            inp = self.tokenizer(self.prefix + inp, max_length = self.max_input_length, truncation=True, padding='longest')
            # with self.tokenizer.as_target_tokenizer():
            outp = self.tokenizer(outp, max_length = self.max_target_length, truncation=True, padding='longest')

            input_ids.append(torch.tensor(inp['input_ids']))
            labels.append(torch.tensor(outp['input_ids']))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(dtype=torch.long)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).to(dtype=torch.long)

        return input_ids, labels

    def get_dataloader(self, is_train=True):
        test_dataset = CNSDataset(self.test_path, self.tokenizer, self.test_n_data)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 collate_fn=self.collate_fn,
                                 num_workers=self.prefetch_n_workers,
                                 persistent_workers=True)
        if not is_train:
            return test_loader
        
        train_dataset = CNSDataset(self.train_path, self.tokenizer, self.train_n_data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  collate_fn=self.collate_fn,
                                  num_workers=self.prefetch_n_workers,
                                  persistent_workers=True)
        
        if self.verbose > 0:
            print('''[Dataset]
                  - Training: {}
                  - Test: {}'''.format(len(train_dataset), len(test_dataset)))
            
        return train_loader, test_loader

