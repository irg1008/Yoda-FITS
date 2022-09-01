from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from config import get_checkpoint, get_model_path


def get_model_and_tokenizer(use_checkpoint=False):
    checkpoint = get_checkpoint() if use_checkpoint else get_model_path()
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer
