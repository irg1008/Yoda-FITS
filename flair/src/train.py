from os import path

from datasets.dataset_dict import DatasetDict
from transformers.data.data_collator import DataCollator, DataCollatorForSeq2Seq
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments

from config import get_arguments
from dataset import get_dataset, tokenize_dataset
from model import get_model_and_tokenizer


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    args: TrainingArguments,
    data_collator: DataCollator,
    tokenized_dataset: DatasetDict,
):
    return Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )


def get_data_collator(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def train():
    models_path = path.abspath(path.join(path.dirname(__file__), "../models/"))
    model_path = path.join(models_path, "test")

    # Hyperparameters.
    args = get_arguments(models_path)

    # Model, tokenizer and collator.
    model, tokenizer = get_model_and_tokenizer()
    data_collator = get_data_collator(model, tokenizer)

    # Dataset.
    dataset = get_dataset()
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)

    # Trainer and training.
    trainer = get_trainer(model, tokenizer, args, data_collator, tokenized_dataset)
    trainer.train()

    # Save model.
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    train()
