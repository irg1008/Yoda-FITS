import sys
from os import path

from onnx_config import get_onnx_config

sys.path.append(path.join(path.dirname(__file__), ".."))

from config import get_model_path, get_onnx_path
from model import get_model_and_tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.pipelines import pipeline


def infer():
    model_path = get_model_path()
    onnx_config = get_onnx_config(model_path)

    # onnx_dir, onnx_path = get_onnx_path()
    _, tokenizer = get_model_and_tokenizer()
    model_path = get_model_path()

    model = ORTModelForSeq2SeqLM.from_pretrained(model_path, from_transformers=True, use_cache=False)

    engine = pipeline("summarization", model, tokenizer)

    text = "Hello world sdf sdfs dfs dfsd fsdf"
    output = engine(text)
    print(output)


if __name__ == "__main__":
    infer()
