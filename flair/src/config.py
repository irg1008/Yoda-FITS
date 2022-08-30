from os import path

from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments


def get_model_path(model_name="mt5"):
    return path.join(path.dirname(__file__), "../models", model_name)


def get_checkpoint():
    # model_name = "nbroad/mt5-small-qgen"
    # model_name = "csebuetnlp/mT5_multilingual_XLSum"
    # model_name = "LeoCordoba/mt5-small-cc-news-es-titles"
    # model_name = "ELiRF/mbart-large-cc25-dacsa-es"
    # model_name = "facebook/bart-base"
    # model_name = "t5-base"
    model_name = "josmunpen/mt5-small-spanish-summarization"
    return model_name


def get_arguments(output_dir: str) -> Seq2SeqTrainingArguments:
    batch_size = 1
    return Seq2SeqTrainingArguments(
        output_dir=path.join(output_dir, "logs"),
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        eval_steps=200,
        save_steps=400,
        learning_rate=2e-4,
        num_train_epochs=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        report_to=["tensorboard"],
        optim=OptimizerNames.ADAMW_TORCH,
        load_best_model_at_end=True,
    )
