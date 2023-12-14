import os
import pytest
from src.preprocessing import Preprocessing
import pandas as pd

def test_generator_output():
    """ test that the data generator outputs images of the correct shape """
    target_size = (224, 224)
    batch_size = 2  # size 2 for test purpose only
    preprocessor = Preprocessing(target_size=target_size, batch_size=batch_size)

    # determine the root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # use paths to retrieve test images
    df = pd.DataFrame({
        'image_path': [os.path.join(project_root, 'tests', 'test_images', 'test_image1.jpeg'),
                       os.path.join(project_root, 'tests', 'test_images', 'test_image2.jpeg')],
        'label': [0, 1]
    })

    train_gen = preprocessor.create_generator(df, 'image_path', 'label', train=True)

    # fetch data
    images, labels = next(train_gen)

    # check if the images are in the correct shape
    assert images.shape == (batch_size, *target_size, 3), "Oops! Images don't have the correct shape :("
