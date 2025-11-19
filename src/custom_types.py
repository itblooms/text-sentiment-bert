from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import TypeAlias, Union

DatasetType: TypeAlias = Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]