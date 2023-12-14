from keras.applications import VGG19
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import Recall

class SignetRingModel:
    def __init__(self, input_shape=(224, 224, 3), trainable=False):
        self.input_shape = input_shape
        self.trainable = trainable
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = self.trainable

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy', Recall()])

def main():
    signet_model = SignetRingModel()
    signet_model.compile()
    signet_model.model.summary()

    return signet_model

if __name__ == "__main__":
    model_instance = main()
