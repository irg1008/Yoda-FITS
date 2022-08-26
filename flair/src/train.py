import pytorch_lightning as pl
from transformers.models.auto.tokenization_auto import AutoTokenizer

from lightning_transformers.task.nlp.summarization import (
    SummarizationTransformer,
    SummarizationDataModule,
)

from os import path


def train():
    data_path = path.join(path.dirname(__file__), "../data/splits/")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")

    model = SummarizationTransformer(
        pretrained_model_name_or_path="t5-base",
        use_stemmer=True,
        val_target_max_length=142,
        num_beams=None,
        compute_generate_metrics=True,
    )

    dm = SummarizationDataModule(
        batch_size=1,
        max_source_length=128,
        max_target_length=128,
        tokenizer=tokenizer,
        test_file=data_path + "test.csv",
        train_file=data_path + "train.csv",
        validation_file=data_path + "val.csv",
    )

    out_path = path.join(path.dirname(__file__), "../models/")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        default_root_dir=out_path,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
