from os import path
from pathlib import Path
from typing import Mapping

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.onnx.config import OnnxSeq2SeqConfigWithPast
from transformers.onnx.convert import export

from config import get_model_path
from model import get_model_and_tokenizer


class MT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {
                0: "batch",
                1: "past_decoder_sequence + sequence",
            }
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {
                0: "batch",
                1: "decoder_sequence",
            }

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13


def export_onnx():

    model_path = get_model_path()
    onnx_path = Path(path.join(model_path), "onnx")

    config = AutoConfig.from_pretrained(model_path)
    onnx_config = MT5OnnxConfig(config)

    model, tokenizer = get_model_and_tokenizer()

    onnx_inputs, onnx_outputs = export(
        tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path
    )


if __name__ == "__main__":
    export_onnx()
