import pytest
from models.model_architecture import SignetRingModel

def test_build_model():
    """
    test if the model's architecture is built correctly with the correct input shape
    """
    input_shape = (224, 224, 3)
    signet_model = SignetRingModel(input_shape=input_shape)

    # check if the input shape of the model matches the required shape
    assert signet_model.model.input_shape == (None, *input_shape), "model input shape is incorrect"

def test_compile_model():
    """
    test if the model compiles without any errors.
    """
    signet_model = SignetRingModel()
    signet_model.compile()

    # models should have an optimiser to compile correctly
    assert signet_model.model.optimizer is not None, "model did not compile correctly"
