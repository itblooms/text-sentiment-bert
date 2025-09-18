from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from typing import Tuple
import torch
import transformers


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

def load_data(dataset_name: str, 
                        model_name: str,
                        batch_size: int,
                        num_workers: int
                        ) -> Tuple[torch.utils.data.DataLoader, 
                                   torch.utils.data.DataLoader,
                                   torch.utils.data.DataLoader,
                                   transformers.PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    if isinstance(tokenized_dataset, (Dataset, DatasetDict)):
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        raise ValueError("The dataset_name must refers to dataset with type Dataset | DatasetDict.\n"
                         "Check if your dataset is of type IterableDataset | IterableDatasetDict, "
                         "which are currently not supported")
    
    train_dataset = HFDatasetWrapper(tokenized_dataset["train"])
    val_dataset = HFDatasetWrapper(tokenized_dataset["validation"])
    test_dataset = HFDatasetWrapper(tokenized_dataset["test"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=data_collator)
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=data_collator)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=data_collator)
    
    return train_loader, val_loader, test_loader, tokenizer
