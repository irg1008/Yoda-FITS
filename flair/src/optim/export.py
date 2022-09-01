import sys
from os import mkdir, path
from pathlib import Path

from transformers.onnx.convert import export

from onnx_config import get_onnx_config

sys.path.append(path.join(path.dirname(__file__), ".."))

from config import get_model_path, get_onnx_path
from model import get_model_and_tokenizer
from transformers.onnx import validate_model_outputs


def export_onnx():

    model_path = get_model_path()
    onnx_dir, onnx_path = get_onnx_path()

    # Create dir if not exists
    if not path.exists(onnx_dir):
        mkdir(onnx_dir)

    onnx_config = get_onnx_config(model_path)

    model, tokenizer = get_model_and_tokenizer()

    onnx_inputs, onnx_outputs = export(
        tokenizer,
        model,
        onnx_config,
        onnx_config.default_onnx_opset,
        Path(onnx_path),
    )

    validate_model_outputs(
        onnx_config,
        tokenizer,
        model,
        Path(onnx_path),
        onnx_outputs,
        atol=onnx_config.atol_for_validation,
    )

    print(f"Inputs: {onnx_inputs}")
    print(f"Outputs: {onnx_outputs}")


if __name__ == "__main__":
    export_onnx()
