from os import path

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from transformers.data.data_collator import DataCollator, DataCollatorForSeq2Seq
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments, OptimizerNames
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments


def get_tokenizer(checkpoint: str):
    return AutoTokenizer.from_pretrained(checkpoint)


def get_model(checkpoint: str):
    return AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


def get_arguments(output_dir: str) -> Seq2SeqTrainingArguments:
    batch_size = 4
    return Seq2SeqTrainingArguments(
        output_dir=path.join(output_dir, "logs"),
        evaluation_strategy=IntervalStrategy.EPOCH,
        learning_rate=1e-4,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        report_to=["tensorboard"],
        optim=OptimizerNames.ADAMW_TORCH,
    )


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


def tokenize_dataset(
    tokenizer: PreTrainedTokenizerBase, dataset: DatasetDict
) -> DatasetDict:
    def preprocess_function(examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)


def get_dataset() -> DatasetDict:
    dataset = load_dataset("billsum", split="ca_test")
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset


def train():
    models_path = path.abspath(path.join(path.dirname(__file__), "../models/"))
    model_path = path.join(models_path, "test")

    checkpoint = "t5-small"

    model = get_model(checkpoint)
    tokenizer = get_tokenizer(checkpoint)
    args = get_arguments(models_path)

    data_collator = get_data_collator(model, tokenizer)

    dataset = get_dataset()
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)

    trainer = get_trainer(model, tokenizer, args, data_collator, tokenized_dataset)
    trainer.train()

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    train()
