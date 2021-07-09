import os
import numpy as np
import tensorflow as tf

from tools import *
from config import Config


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8196)])
  except RuntimeError as e:
    print(e)


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()

        self.X_train, self.y_train, self.X_test, self.y_test = load_data()
        
        model_input = tf.keras.layers.Input(shape=(Config.UNIT_TIMESTEP, Config.N_ID, 2))
        lstm1 = tf.keras.layers.LSTM(32, activation='relu', return_sequences=False)(model_input[:, :, :, 0])
        lstm2 = tf.keras.layers.LSTM(32, activation='relu', return_sequences=False)(model_input[:, :, :, 1])
        lstm1 = tf.keras.layers.Dropout(0.5)(lstm1)
        lstm2 = tf.keras.layers.Dropout(0.5)(lstm2)
        dense = tf.keras.layers.Dense(16, activation='relu')(tf.keras.layers.Concatenate()([lstm1, lstm2]))
        #dense = tf.keras.layers.Dense(16, activation='relu')(lstm1)

        if Config.isMC:
            model_output = tf.keras.layers.Dense(5, activation='softmax')(dense)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
            loss = tf.keras.losses.BinaryCrossentropy()
        self.model = tf.keras.models.Model(model_input, model_output)
        
        self.model.compile(loss=loss,
                        optimizer="adam",
                        metrics=['accuracy'])

        print(self.model.summary())

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                validation_split=0.2,
                                epochs=Config.EPOCHS,)
        Config.MODEL_NAME = f"models/lstm_train_D.h5"
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
    lstm = LSTM()
    lstm.train()
    lstm.test()
