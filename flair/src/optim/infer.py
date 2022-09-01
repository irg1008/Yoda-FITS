import sys
from os import path

from onnxruntime import InferenceSession
from onnx_config import get_onnx_config

sys.path.append(path.join(path.dirname(__file__), ".."))

from config import get_onnx_path
from model import get_model_and_tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch


def get_session(onnx_path: str):
    ort_session = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return ort_session


def get_inputs_for_text(text: str, tokenizer: PreTrainedTokenizerBase):
    inputs = tokenizer(text, return_tensors="pt")

    inputs["decoder_input_ids"] = torch.ones_like(inputs["input_ids"]).cpu().numpy()
    inputs["decoder_attention_mask"] = (
        torch.ones_like(inputs["attention_mask"]).cpu().numpy()
    )
    inputs["input_ids"] = inputs["input_ids"].cpu().numpy()
    inputs["attention_mask"] = inputs["attention_mask"].cpu().numpy()

    return inputs


def inference(text: str, session: InferenceSession, tokenizer: PreTrainedTokenizerBase):
    inputs = get_inputs_for_text(text, tokenizer)
    outputs = session.run(output_names=["logits"], input_feed=dict(inputs))

    # TODO: Proccess output with beam searhc or similar to fetch summarized text

    print(outputs[0][0])
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


def infer():
    _, onnx_path = get_onnx_path()

    _, tokenizer = get_model_and_tokenizer()
    session = get_session(onnx_path)

    text = "Hello world"
    output = inference(text, session, tokenizer)

    print(output)


if __name__ == "__main__":
    infer()
