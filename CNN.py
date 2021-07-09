import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tools import *
from config import Config


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: 
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
  except RuntimeError as e:
    print(e)


class CNN:
    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = load_data()

        self.X_train = self.X_train.reshape(list(self.X_train.shape)+[1])
        self.X_test = self.X_test.reshape(list(self.X_test.shape)+[1])

        #show_tsne(self.X_test[:1000], self.y_test[:1000])

        model_input = tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2], 1))
        z = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(model_input)
        z = tf.keras.layers.MaxPooling2D((2, 2))(z)
        z = tf.keras.layers.Dropout(0.5)(z)
        z = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(z)
        z = tf.keras.layers.MaxPooling2D((2, 2))(z)
        z = tf.keras.layers.Dropout(0.5)(z)
        z = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(z)
        z = tf.keras.layers.Flatten()(z)
        dense = tf.keras.layers.Dense(16, activation='relu')(z)
        if Config.isMC:
            model_output = tf.keras.layers.Dense(5, activation='softmax')(dense)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
            loss = tf.keras.losses.BinaryCrossentropy()
        
        self.model = tf.keras.models.Model(model_input, model_output)

        self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss=loss,
                    metrics=['accuracy'],
        )

        print(self.model.summary())

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                validation_split=0.2,
                                batch_size=Config.BATCH_SIZE,
                                epochs=Config.EPOCHS,)
        #Config.MODEL_NAME = f"models/test_train_D.h5"
        self.model.save(Config.MODEL_NAME)

        show_train_result(hist)

    def test(self):
        self.model = tf.keras.models.load_model(Config.MODEL_NAME)
        y_pred_prob = self.model.predict(self.X_test)

        if Config.isMC:
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:
            y_pred = np.around(y_pred_prob)
        y_true = self.y_test

        show_test_result(y_true, y_pred)

        return y_true, y_pred_prob


if __name__ == "__main__":
    cnn = CNN()
    cnn.train()
    cnn.test()