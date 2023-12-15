import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

class SignetRingPredictor:
    def __init__(self, model_path, target_size=(224, 224)):
        self.model = load_model(model_path)
        self.target_size = target_size
        self.datagen = ImageDataGenerator(rescale=1./255)

    def prepare_data(self, image_paths):
        """
        prepare the image data for prediction using a generator
        """
        # Create a DataFrame with image paths
        df = pd.DataFrame({
            'image_path': image_paths
        })

        generator = self.datagen.flow_from_dataframe(
            dataframe=df,
            x_col='image_path',
            y_col=None,
            target_size=self.target_size,
            class_mode=None,
            batch_size=1,
            shuffle=False
        )
        return generator

    def predict(self, img_path):
        """
        make predictions on new data and returns a probability

        close to 1: it suggests the image contains signet ring cells
        close to 0: it suggests the image does not contains signet ring cells
        around 0.5: it indicate uncertainty
        """
        # check if img_path is a directory or a file
        if os.path.isdir(img_path):
            # if directory, list all img files
            image_paths = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            # if a single img, wrap it in a list
            image_paths = [img_path]

        # prepare data
        data_generator = self.prepare_data(image_paths)

        # predict
        predictions = self.model.predict(data_generator, steps=len(image_paths))
        return predictions

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'saved_model', 'signet_ring_model.keras')
    test_images_dir = os.path.join(current_dir, '..', 'tests', 'test_images')

    predictor = SignetRingPredictor(model_path)
    predictions = predictor.predict(test_images_dir)

    # Output predictions
    for img_path, pred in zip(os.listdir(test_images_dir), predictions):
        print(f'Prediction for {img_path}: {pred[0]} ✅')
