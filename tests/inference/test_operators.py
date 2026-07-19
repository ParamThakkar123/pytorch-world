import pytest
import torch
from world_models.inference.operators import (
    OperatorABC,
    TensorSpec,
    DreamerOperator,
    JEPAOperator,
    IrisOperator,
    PlaNetOperator,
)


class _SpecOperator(OperatorABC):
    input_specs = {"x": TensorSpec(shape=(None, 2), dtype=torch.float32)}
    output_specs = {"y": TensorSpec(shape=(None, 2), dtype=torch.float32)}

    def preprocess(self, inputs):
        return {"x": torch.as_tensor(inputs, dtype=torch.float32)}

    def forward(self, inputs):
        return {"y": inputs["x"] + 1.0}

    def postprocess(self, outputs):
        outputs["z"] = outputs["y"] * 2.0
        return outputs


class _MissingKeyOperator(OperatorABC):
    def preprocess(self, inputs):
        if inputs == "missing_b":
            return {"a": torch.tensor([1.0])}
        return {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}


class _BadSpecOperator(OperatorABC):
    input_specs = {"x": TensorSpec(shape=(2,), dtype=torch.float32)}

    def preprocess(self, inputs):
        return {"x": inputs}


class TestOperators:
    def test_dreamer_operator(self):
        operator = DreamerOperator()

        # Test with dummy tensor
        inputs = {
            "image": torch.randn(3, 64, 64),
            "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        result = operator.process(inputs)
        assert "obs" in result
        assert "action" in result
        assert result["obs"].shape == (1, 3, 64, 64)
        assert result["action"].shape == (1, 6)

    def test_jepa_operator(self):
        operator = JEPAOperator()

        # Test with dummy images
        dummy_img = torch.randn(3, 224, 224)
        inputs = {"images": [dummy_img]}
        result = operator.process(inputs)
        assert "images" in result
        assert "mask" in result
        assert result["images"].shape == (1, 3, 224, 224)

    def test_iris_operator(self):
        operator = IrisOperator()

        # Test with dummy tokens
        inputs = {"tokens": [1, 2, 3, 4, 5]}
        result = operator.process(inputs)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 512)
        assert result["attention_mask"].shape == (1, 512)

    def test_planet_operator(self):
        operator = PlaNetOperator()

        # Test with dummy inputs
        inputs = {
            "obs": torch.randn(32),
            "action": [0.1, 0.2],
            "reward": 1.0,
            "done": False,
        }
        result = operator.process(inputs)
        assert "obs" in result
        assert "action" in result
        assert "reward" in result
        assert "done" in result

    def test_operator_lifecycle_and_batch_interface(self):
        operator = IrisOperator(seq_length=4)

        operator.eval()
        assert operator.training is False
        operator.train()
        assert operator.training is True
        assert operator.to("cpu") is operator

        result = operator({"tokens": [1, 2]})
        assert result["input_ids"].device.type == "cpu"
        assert result["input_ids"].shape == (1, 4)

        batch = operator.batch([{"tokens": [1, 2]}, {"tokens": [3, 4]}])
        assert batch["input_ids"].shape == (2, 1, 4)
        assert batch["attention_mask"].shape == (2, 1, 4)

    def test_operator_pipeline_runs_forward_postprocess_and_call(self):
        operator = _SpecOperator()

        result = operator([[1.0, 2.0]])

        assert torch.equal(result["y"], torch.tensor([[2.0, 3.0]]))
        assert torch.equal(result["z"], torch.tensor([[4.0, 6.0]]))

    def test_operator_validation_reports_shape_dtype_and_tensor_errors(self):
        operator = _BadSpecOperator()

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            operator.process([1.0, 2.0])

        with pytest.raises(TypeError, match="must have dtype"):
            operator.process(torch.tensor([1, 2], dtype=torch.long))

        with pytest.raises(ValueError, match="must have 1 dims"):
            operator.process(torch.ones(1, 2))

    def test_operator_batch_rejects_empty_or_inconsistent_outputs(self):
        spec_operator = _SpecOperator()
        with pytest.raises(ValueError, match="empty input sequence"):
            spec_operator.batch([])

        missing_key_operator = _MissingKeyOperator()
        with pytest.raises(ValueError, match="must share keys"):
            missing_key_operator.batch(["ok", "missing_b"])
