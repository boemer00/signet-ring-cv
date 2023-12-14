from keras.applications import VGG19
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import Recall

def build_model(input_shape=(224, 224, 3), trainable=False):
    """
    builds a image classification model based on a pre-trained VGG19 architecture.

    :param input_shape: tuple, shape of the input images
    :param trainable: bool, whether the VGG19 layers should be trainable
    :return: Compiled Keras model.
    """
    # load pre-trained VGG19 model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # freeze base model layers
    for layer in base_model.layers:
        layer.trainable = trainable

    # add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # binary 1 or 0

    # define the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def compile_model(model, learning_rate=0.001):
    """
    compiles the model
    """
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', Recall()])
    return model

def main():
    # build and compile the model
    model = build_model()
    compiled_model = compile_model(model)
    model.summary()

    return compiled_model

if __name__ == "__main__":
    model = main()
