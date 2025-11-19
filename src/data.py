from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from typing import Tuple
import torch
import transformers


def load_data(
    dataset_name: str, 
    model_name: str,
    batch_size: int,
    num_workers: int
) -> Tuple[
    torch.utils.data.DataLoader, 
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    transformers.PreTrainedTokenizerBase
]:
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        dataset=tokenized_dataset["train"], # type: ignore
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        dataset=tokenized_dataset["validation"], # type: ignore
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    test_loader = DataLoader(
        dataset=tokenized_dataset["test"], # type: ignore
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    return train_loader, val_loader, test_loader, tokenizer
