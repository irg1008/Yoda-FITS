from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.mt5.configuration_mt5 import MT5OnnxConfig


def get_onnx_config(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    onnx_config = MT5OnnxConfig(config, task="seq2seq-lm")
    return onnx_config
