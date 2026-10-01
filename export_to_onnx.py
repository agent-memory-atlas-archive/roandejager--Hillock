"""Export all-MiniLM-L6-v2 to a dynamically shaped FP16 ONNX model."""

import argparse
import tempfile
from pathlib import Path


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 256


def export_model(output_path: str = "minilm.onnx", model_id: str = MODEL_ID) -> None:
    try:
        import onnx
        import torch
        from onnxconverter_common import float16
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Export requires torch, transformers, onnx, and onnxconverter-common."
        ) from exc

    class MiniLMTokenEncoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).last_hidden_state

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id)
    encoder.eval()
    model = MiniLMTokenEncoder(encoder).eval()

    sample = tokenizer(
        ["ONNX export sample."],
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    )
    inputs = (
        sample["input_ids"],
        sample["attention_mask"],
        sample.get("token_type_ids", torch.zeros_like(sample["input_ids"]))
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    dynamic_axes = {
        name: {0: "batch_size", 1: "sequence_length"}
        for name in input_names
    }
    dynamic_axes["token_embeddings"] = {0: "batch_size", 1: "sequence_length"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        fp32_path = Path(temp_dir) / "minilm_fp32.onnx"
        torch.onnx.export(
            model,
            inputs,
            str(fp32_path),
            input_names=input_names,
            output_names=["token_embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False
        )
        onnx_model = onnx.load(str(fp32_path))
        fp16_model = float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=False,
            disable_shape_infer=True
        )
        onnx.checker.check_model(fp16_model)
        onnx.save(fp16_model, str(output_path))

    print(f"Exported FP16 ONNX model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="minilm.onnx", help="Output ONNX model path.")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model identifier.")
    args = parser.parse_args()
    export_model(output_path=args.output, model_id=args.model)


if __name__ == "__main__":
    main()
