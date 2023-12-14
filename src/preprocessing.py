from keras.preprocessing.image import ImageDataGenerator

class Preprocessing:
    def __init__(self, target_size=(224, 224), batch_size=32):
        self.target_size = target_size
        self.batch_size = batch_size
        self.train_datagen = self._create_train_datagen()
        self.val_datagen = self._create_val_datagen()

    def _create_train_datagen(self):
        """ define data augmentation parameters for the training set """
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

    def _create_val_datagen(self):
        """ for the validation generator, usually, we only rescale """
        return ImageDataGenerator(rescale=1./255)

    def create_generator(self, dataframe, x_col, y_col, train=True):
        """
        create a data generator from a df

        :param dataframe: df, the source data
        :param x_col: str, column name for the image paths
        :param y_col: str, column name for the labels
        :param train: bool, whether to create a training or validation generator
        :return: a data generator
        """
        temp_df = dataframe.copy()
        temp_df[y_col] = temp_df[y_col].astype(str)

        datagen = self.train_datagen if train else self.val_datagen

        return datagen.flow_from_dataframe(
            dataframe=temp_df,
            x_col=x_col,
            y_col=y_col,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
