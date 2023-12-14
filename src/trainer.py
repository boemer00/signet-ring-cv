import os
import numpy as np
import tensorflow as tf
import random
from src.data_loader import load_image_paths
from src.preprocessing import Preprocessing
from models.model_architecture import SignetRingModel
from keras.callbacks import ModelCheckpoint, TensorBoard

# random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)

def train_model():
    # load data and preprocessing
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, 'data')
    sub_dirs = [('sig-train-neg', 0), ('sig-train-pos', 1)]
    train_df = load_image_paths(base_dir, sub_dirs)

    preprocessor = Preprocessing()
    train_generator = preprocessor.create_generator(train_df, 'image_path', 'label', train=True)

    # initialise model
    model = SignetRingModel()
    model.compile()

    # callbacks
    checkpoint_path = os.path.join(project_root, 'models', 'checkpoints', 'model-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join(project_root, 'logs'))

    # train the model
    model.model.fit(train_generator,
                    epochs=50,
                    callbacks=[checkpoint, tensorboard])

    # save the final model
    model_save_path = os.path.join(project_root, 'models', 'saved_model', 'signet_ring_model.keras')
    model.model.save(model_save_path)
    print('model trained and saved 🙌')

if __name__ == '__main__':
    train_model()
