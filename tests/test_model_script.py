import pytest
from models.model_architecture import build_model, compile_model

def test_build_model():
    """
    test if the model's architecture is built correctly with the correct input shape
    """
    input_shape = (224, 224, 3)
    model = build_model(input_shape)

    # check if the input shape of the model matches the required shape
    assert model.input_shape == (None, *input_shape), "model input shape is incorrect"

def test_compile_model():
    """
    Test if the model compiles without any errors.
    """
    model = build_model()
    compiled_model = compile_model(model)

    # models should have an optimiser to compile correctly
    assert compiled_model.optimizer is not None, "model did not compile correctly"
