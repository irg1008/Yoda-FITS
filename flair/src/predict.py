from transformers.models.auto.tokenization_auto import AutoTokenizer
from lightning_transformers.task.nlp.summarization import SummarizationTransformer
import pytorch_lightning as pl

from os import path


def main():
    models_path = path.join(path.dirname(__file__), "../models/lightning_logs")
    version_path = path.join(models_path, "version_0")

    model_path = path.join(version_path, "checkpoints/epoch=0-step=297.ckpt")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")

    model = SummarizationTransformer.load_from_checkpoint(
        model_path, tokenizer=tokenizer
    )

    model.hf_predict(
        "The results found significant improvements over all tasks evaluated",
        min_length=2,
        max_length=12,
    )


if __name__ == "__main__":
    main()
