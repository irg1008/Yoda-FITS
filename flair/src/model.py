from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM


def get_model_and_tokenizer():
    checkpoint = "LeoCordoba/mt5-small-cc-news-es-titles"
    # josmunpen/mt5-small-spanish-summarization
    # nbroad/mt5-small-qgen
    # LeoCordoba/mt5-small-cc-news-es-titles
    # csebuetnlp/mT5_multilingual_XLSum
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer
