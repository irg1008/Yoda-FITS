from os import path

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def get_dataset() -> DatasetDict:
    data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
    splits_path = path.join(data_path, "splits")

    dataset = load_dataset(
        "csv",
        data_files={
            "train": path.join(splits_path, "train.csv"),
            "test": path.join(splits_path, "test.csv"),
            "validation": path.join(splits_path, "val.csv"),
        },
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError("'data' is not a Dataset Dictionary")

    return dataset


def tokenize_dataset(
    tokenizer: PreTrainedTokenizerBase, dataset: DatasetDict
) -> DatasetDict:
    def preprocess_function(examples):
        prefix = ""
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)
