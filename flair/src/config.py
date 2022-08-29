from os import path

from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments


def get_arguments(output_dir: str) -> Seq2SeqTrainingArguments:
    batch_size = 1
    return Seq2SeqTrainingArguments(
        output_dir=path.join(output_dir, "logs"),
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        eval_steps=500,
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        report_to=["tensorboard"],
        optim=OptimizerNames.ADAMW_TORCH,
        load_best_model_at_end=True,
    )
